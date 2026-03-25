from __future__ import annotations

from collections import OrderedDict
from functools import partial
import copy
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers import DDIMScheduler, SchedulerMixin
from diffusers.training_utils import EMAModel
from gymnasium import spaces

from robobase.backends.torch.method.bc import BC
from robobase.backends.torch.models.diffusion import (
    ConditionalUnet1D,
    replace_bn_with_gn,
)
from robobase.backends.torch.models.fully_connected import FullyConnectedModule
from robobase.method.diffusion import DiffusionModelSpec
from robobase.replay_buffer.vision_feature_cache import (
    cached_feature_observation_key,
)
from robobase.method.utils import (
    extract_from_spec,
    extract_many_from_spec,
    flatten_time_dim_into_channel_dim,
    sequence_loss_mean,
    stack_tensor_dictionary,
)


class SafeEMAModel(EMAModel):
    @torch.no_grad()
    def step(self, parameters):
        if isinstance(parameters, torch.nn.Module):
            parameters = parameters.parameters()

        parameters = list(parameters)
        self.optimization_step += 1

        decay = self.get_decay(self.optimization_step)
        self.cur_decay_value = decay
        one_minus_decay = 1 - decay

        for shadow_param, param in zip(self.shadow_params, parameters):
            if param.requires_grad:
                shadow_param.sub_(one_minus_decay * (shadow_param - param))
            else:
                shadow_param.copy_(param)


class Actor(nn.Module):
    def __init__(
        self,
        action_space: spaces.Box,
        actor_model: FullyConnectedModule,
        noise_scheduler: SchedulerMixin,
        num_diffusion_iters: int,
        use_ema: bool,
    ):
        super().__init__()
        assert len(action_space.shape) == 2
        self.action_space = action_space
        self.actor = actor_model
        self.noise_scheduler = noise_scheduler
        self.num_diffusion_iters = num_diffusion_iters
        self.use_ema = use_ema
        self.sequence_length = action_space.shape[0]
        self.action_dim = action_space.shape[1]
        self.ema = None
        self.ema_actor = None
        if self.use_ema:
            self.ema = SafeEMAModel(
                parameters=self.actor.parameters(),
                power=0.75,
            )
            self.ema_actor = copy.deepcopy(self.actor)

    @property
    def preferred_optimiser(self) -> callable:
        return getattr(
            self.actor,
            "preferred_optimiser",
            partial(torch.optim.Adam, self.actor.parameters()),
        )

    def _combine(self, low_dim_obs, fused_view_feats):
        flat_feats = []
        if low_dim_obs is not None:
            flat_feats.append(low_dim_obs)
        if fused_view_feats is not None:
            flat_feats.append(fused_view_feats)
        obs_features = torch.cat(flat_feats, dim=-1)
        return obs_features

    def forward(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        action_pad_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_features = self._combine(low_dim_obs, fused_view_feats)

        batch_size = obs_features.shape[0]
        noise = torch.randn(
            (batch_size,) + self.action_space.shape,
            device=obs_features.device,
        )
        timesteps = torch.randint(
            0,
            self.noise_scheduler.config.num_train_timesteps,
            (batch_size,),
            device=obs_features.device,
        ).long()

        noisy_actions = self.noise_scheduler.add_noise(action, noise, timesteps)
        if action_pad_mask is not None:
            action_pad_mask = action_pad_mask.to(
                dtype=torch.bool,
                device=noisy_actions.device,
            )
            noisy_actions = noisy_actions.masked_fill(action_pad_mask.unsqueeze(-1), 0.0)
            noise = noise.masked_fill(action_pad_mask.unsqueeze(-1), 0.0)

        net_ins = {
            "actions": noisy_actions,
            "features": obs_features,
            "timestep": timesteps,
        }
        noise_pred = self.actor(net_ins)
        return noise_pred, noise

    def infer(
        self,
        low_dim_obs,
        fused_view_feats,
        use_ema: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        obs_features = self._combine(low_dim_obs, fused_view_feats)
        with torch.no_grad():
            actor = self.actor
            if use_ema:
                if self.ema is None or self.ema_actor is None:
                    raise RuntimeError("EMA inference requested but EMA is disabled.")
                actor = self.ema_actor
                self.ema.copy_to(actor.parameters())

            batch_size = obs_features.shape[0]
            noisy_action = torch.randn(
                (batch_size, self.sequence_length, self.action_dim),
                device=obs_features.device,
            )

            self.noise_scheduler.set_timesteps(self.num_diffusion_iters)
            for timestep in self.noise_scheduler.timesteps:
                net_ins = {
                    "actions": noisy_action,
                    "features": obs_features,
                    "timestep": timestep,
                }
                noise_pred = actor(net_ins)
                noisy_action = self.noise_scheduler.step(
                    model_output=noise_pred,
                    timestep=timestep,
                    sample=noisy_action,
                ).prev_sample

        return noise_pred, noisy_action


class Diffusion(BC):
    _EMA_STATE_PREFIX = "_actor_ema_state."
    _EMA_SCALAR_FIELDS = {
        "decay": float,
        "min_decay": float,
        "optimization_step": int,
        "update_after_step": int,
        "use_ema_warmup": bool,
        "inv_gamma": float,
        "power": float,
    }

    def __init__(
        self,
        num_diffusion_iters: int,
        use_ema: bool = False,
        model: DiffusionModelSpec | None = None,
        *args,
        **kwargs,
    ):
        if not kwargs["frame_stack_on_channel"]:
            raise NotImplementedError(
                "frame_stack_on_channel must be true for diffusion policies."
            )
        self.num_diffusion_iters = num_diffusion_iters
        self.use_ema = use_ema
        kwargs["use_ema"] = use_ema
        self.noise_scheduler = DDIMScheduler(
            num_train_timesteps=num_diffusion_iters,
            beta_schedule="squaredcos_cap_v2",
            clip_sample=True,
            prediction_type="epsilon",
        )
        super().__init__(*args, model=model, **kwargs)
        if self.use_pixels:
            self._cached_pixel_feature_key = cached_feature_observation_key(
                "diffusion", "torch"
            )

    @classmethod
    def _ema_state_key(cls, suffix: str) -> str:
        return f"{cls._EMA_STATE_PREFIX}{suffix}"

    @staticmethod
    def _has_ema_actor_state(state_dict: OrderedDict) -> bool:
        return any(key.startswith("actor.ema_actor.") for key in state_dict)

    @staticmethod
    def _drop_ema_actor_state(state_dict: OrderedDict):
        for key in [k for k in list(state_dict.keys()) if k.startswith("actor.ema_actor.")]:
            state_dict.pop(key)

    @classmethod
    def _serialize_ema_scalar(cls, value):
        return torch.tensor(float(value), dtype=torch.float64)

    @classmethod
    def _deserialize_ema_scalar(cls, field: str, value: torch.Tensor):
        scalar = value.item()
        caster = cls._EMA_SCALAR_FIELDS[field]
        if caster is bool:
            return bool(int(scalar))
        return caster(scalar)

    @classmethod
    def _pop_ema_state(cls, state_dict: OrderedDict) -> dict | None:
        ema_keys = [k for k in list(state_dict.keys()) if k.startswith(cls._EMA_STATE_PREFIX)]
        if not ema_keys:
            return None

        ema_state = {}
        for field in cls._EMA_SCALAR_FIELDS:
            key = cls._ema_state_key(field)
            value = state_dict.pop(key, None)
            if value is not None:
                ema_state[field] = cls._deserialize_ema_scalar(field, value)

        shadow_prefix = cls._ema_state_key("shadow_params.")
        shadow_param_keys = sorted(
            [k for k in ema_keys if k.startswith(shadow_prefix)],
            key=lambda k: int(k.rsplit(".", 1)[-1]),
        )
        ema_state["shadow_params"] = [state_dict.pop(k) for k in shadow_param_keys]
        return ema_state

    def _sync_ema_actor(self):
        if self.actor.ema is None or self.actor.ema_actor is None:
            return
        self.actor.ema.copy_to(self.actor.ema_actor.parameters())

    def state_dict(self, *args, destination=None, prefix="", keep_vars=False):
        state = super().state_dict(
            *args,
            destination=destination,
            prefix=prefix,
            keep_vars=keep_vars,
        )
        if not self.use_ema or self.actor.ema is None:
            return state

        self._sync_ema_actor()
        ema_actor_prefix = f"{prefix}actor.ema_actor."
        for key in [k for k in list(state.keys()) if k.startswith(ema_actor_prefix)]:
            del state[key]
        ema_state = self.actor.ema.state_dict()
        for field in self._EMA_SCALAR_FIELDS:
            state[f"{prefix}{self._ema_state_key(field)}"] = self._serialize_ema_scalar(
                ema_state[field]
            )
        for idx, shadow_param in enumerate(ema_state["shadow_params"]):
            key = f"{prefix}{self._ema_state_key(f'shadow_params.{idx}')}"
            tensor = shadow_param.clone() if keep_vars else shadow_param.detach().clone()
            state[key] = tensor
        return state

    def load_state_dict(self, state_dict, strict=True):
        state_dict = OrderedDict(state_dict)
        ema_state = self._pop_ema_state(state_dict)
        has_ema_actor_state = self._has_ema_actor_state(state_dict)
        if not self.use_ema:
            self._drop_ema_actor_state(state_dict)
            return super().load_state_dict(state_dict, strict=strict)

        incompatible_keys = super().load_state_dict(state_dict, strict=False)
        missing_keys = [
            key
            for key in incompatible_keys.missing_keys
            if not key.startswith("actor.ema_actor.")
        ]
        unexpected_keys = incompatible_keys.unexpected_keys
        if strict and (missing_keys or unexpected_keys):
            error_messages = []
            if missing_keys:
                error_messages.append(f"Missing key(s): {missing_keys}")
            if unexpected_keys:
                error_messages.append(f"Unexpected key(s): {unexpected_keys}")
            raise RuntimeError(
                "Error(s) in loading state_dict for Diffusion:\n\t"
                + "\n\t".join(error_messages)
            )

        if ema_state is not None and ema_state["shadow_params"]:
            self.actor.ema.load_state_dict(ema_state)
        else:
            logging.warning(
                "Diffusion snapshot is missing EMA state; rebuilding EMA weights "
                "from the stored %s parameters.",
                "ema_actor" if has_ema_actor_state else "actor",
            )
            params = (
                self.actor.ema_actor.parameters()
                if has_ema_actor_state
                else self.actor.actor.parameters()
            )
            self.actor.ema.shadow_params = [param.detach().clone() for param in params]

        self._sync_ema_actor()
        return incompatible_keys

    def build_actor(self):
        if self.action_space.shape[-2] % 4 != 0:
            raise ValueError(
                "Action sequence length has to be a multiple of 4 for diffusion model."
            )
        if self.actor_model is None:
            if self.model_spec is None:
                raise ValueError("Diffusion requires either actor_model or model spec.")
            actor_spec = self.model_spec.actor_model
            if actor_spec.type != "conditional_unet1d":
                raise NotImplementedError(
                    "Unsupported torch Diffusion actor model type "
                    f"'{actor_spec.type}'."
                )
            if actor_spec.sequence_length != int(self.action_space.shape[0]):
                raise ValueError(
                    "Diffusion actor model sequence_length does not match the action space."
                )
            self.actor_model = partial(
                ConditionalUnet1D,
                sequence_length=actor_spec.sequence_length,
                diffusion_step_embed_dim=actor_spec.diffusion_step_embed_dim,
                down_dims=list(actor_spec.down_dims),
                kernel_size=actor_spec.kernel_size,
                n_groups=actor_spec.n_groups,
            )
        self.actor_model = self.actor_model(
            input_shapes=self.get_fully_connected_inputs(),
            output_shape=self.action_space.shape[-1],
        ).to(self.device)
        self.actor = Actor(
            self.action_space,
            self.actor_model,
            self.noise_scheduler,
            self.num_diffusion_iters,
            self.use_ema,
        ).to(self.device)
        self.actor_opt = (self.actor.preferred_optimiser)(lr=self.lr)

    def build_encoder(self):
        super().build_encoder()
        rgb_spaces = extract_many_from_spec(
            self.observation_space, r"rgb.*", missing_ok=True
        )
        if len(rgb_spaces) > 0:
            self.encoder = replace_bn_with_gn(self.encoder)
            self.encoder.to(self.device)
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)

    def get_fully_connected_inputs(self) -> dict[str, tuple]:
        input_shapes = super().get_fully_connected_inputs()
        input_shapes["actions"] = self.action_space.shape[-1:]
        return input_shapes

    def _act(self, observations: dict[str, torch.Tensor], eval_mode: bool):
        low_dim_obs = fused_rgb_feats = None
        if self.low_dim_size > 0:
            low_dim_obs = flatten_time_dim_into_channel_dim(
                extract_from_spec(observations, "low_dim_state")
            )
        if self.use_pixels:
            fused_rgb_feats = self._extract_cached_pixel_features(observations)
            if fused_rgb_feats is None:
                rgb_obs = flatten_time_dim_into_channel_dim(
                    stack_tensor_dictionary(
                        {key: observations[key] for key in self.obs_layout.rgb_keys},
                        1,
                    ),
                    has_view_axis=True,
                )
                with torch.no_grad():
                    multi_view_rgb_feats = self.encoder(rgb_obs.float())
                    fused_rgb_feats = self.view_fusion(multi_view_rgb_feats)
        _, noisy_action = self.actor.infer(
            low_dim_obs,
            fused_rgb_feats,
            use_ema=self.use_ema and eval_mode,
        )
        return noisy_action.detach()

    def update_actor(
        self,
        low_dim_obs,
        fused_view_feats,
        action,
        loss_coeff,
        action_pad_mask: torch.Tensor | None = None,
        used_cached_pixel_features: bool = False,
    ):
        metrics = dict()
        noise_pred, noise = self.actor(
            low_dim_obs,
            fused_view_feats,
            action,
            action_pad_mask=action_pad_mask,
        )
        mse_loss = sequence_loss_mean(
            F.mse_loss(noise_pred, noise, reduction="none"),
            action_pad_mask,
        ).unsqueeze(1)
        actor_loss = (mse_loss * loss_coeff.unsqueeze(1)).mean()

        new_pri = torch.sqrt(mse_loss + 1e-10)
        self._new_priority = (new_pri / torch.max(new_pri)).cpu().detach().numpy()

        if self.use_pixels and self.encoder_opt is not None and not used_cached_pixel_features:
            self.encoder_opt.zero_grad(set_to_none=True)
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.zero_grad(set_to_none=True)
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        if self.actor_grad_clip:
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.actor_grad_clip)
        self.actor_opt.step()
        if self.use_pixels and self.encoder is not None and not used_cached_pixel_features:
            if self.actor_grad_clip:
                nn.utils.clip_grad_norm_(
                    self.encoder.parameters(),
                    self.actor_grad_clip,
                )
            self.encoder_opt.step()
            if self.use_multicam_fusion and self.view_fusion_opt is not None:
                self.view_fusion_opt.step()

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()

        if self.use_ema and self.actor.ema is not None:
            self.actor.ema.step(self.actor.actor.parameters())

        if self.logging:
            metrics["actor_loss"] = actor_loss.item()
        return metrics
