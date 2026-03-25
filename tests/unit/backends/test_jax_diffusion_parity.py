from __future__ import annotations

import numpy as np
import pytest
import torch
from diffusers import DDIMScheduler
from diffusers.optimization import get_scheduler

from robobase.backends.jax.method.diffusion import _cosine_betas
from robobase.backends.jax.models.diffusion import JaxConditionalUnet1D
from robobase.models.diffusion_models import ConditionalUnet1D

jax = pytest.importorskip("jax")
pytest.importorskip("optax")


def _to_jax_conv_weight(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().transpose(2, 1, 0)


def _to_jax_conv_transpose_weight(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy()[:, :, ::-1].transpose(2, 0, 1)


def _to_jax_linear_weight(weight: torch.Tensor) -> np.ndarray:
    return weight.detach().cpu().numpy().transpose(1, 0)


def _copy_conv1d_block(torch_block, jax_params: dict):
    conv = torch_block.block[0]
    norm = torch_block.block[1]
    jax_params["conv"]["w"] = _to_jax_conv_weight(conv.weight)
    jax_params["conv"]["b"] = conv.bias.detach().cpu().numpy()
    jax_params["norm"]["scale"] = norm.weight.detach().cpu().numpy()
    jax_params["norm"]["bias"] = norm.bias.detach().cpu().numpy()


def _copy_residual_block(torch_block, jax_params: dict):
    _copy_conv1d_block(torch_block.blocks[0], jax_params["block1"])
    _copy_conv1d_block(torch_block.blocks[1], jax_params["block2"])
    linear = torch_block.cond_encoder[1]
    jax_params["cond"]["w"] = _to_jax_linear_weight(linear.weight)
    jax_params["cond"]["b"] = linear.bias.detach().cpu().numpy()
    if jax_params["residual"] is not None:
        conv = torch_block.residual_conv
        jax_params["residual"]["w"] = _to_jax_linear_weight(conv.weight[..., 0])
        jax_params["residual"]["b"] = conv.bias.detach().cpu().numpy()


def _copy_torch_diffusion_to_jax(torch_model: ConditionalUnet1D, jax_params: dict):
    dense1 = torch_model.diffusion_step_encoder[1]
    dense2 = torch_model.diffusion_step_encoder[3]
    jax_params["diffusion_step_encoder"]["dense1"]["w"] = _to_jax_linear_weight(
        dense1.weight
    )
    jax_params["diffusion_step_encoder"]["dense1"]["b"] = (
        dense1.bias.detach().cpu().numpy()
    )
    jax_params["diffusion_step_encoder"]["dense2"]["w"] = _to_jax_linear_weight(
        dense2.weight
    )
    jax_params["diffusion_step_encoder"]["dense2"]["b"] = (
        dense2.bias.detach().cpu().numpy()
    )

    for torch_block, jax_block in zip(torch_model.mid_modules, jax_params["mid_modules"]):
        _copy_residual_block(torch_block, jax_block)

    for torch_block, jax_block in zip(
        torch_model.down_modules,
        jax_params["down_modules"],
    ):
        _copy_residual_block(torch_block[0], jax_block["res1"])
        _copy_residual_block(torch_block[1], jax_block["res2"])
        if jax_block["downsample"] is not None:
            downsample = torch_block[2].conv
            jax_block["downsample"]["w"] = _to_jax_conv_weight(downsample.weight)
            jax_block["downsample"]["b"] = downsample.bias.detach().cpu().numpy()

    for torch_block, jax_block in zip(torch_model.up_modules, jax_params["up_modules"]):
        _copy_residual_block(torch_block[0], jax_block["res1"])
        _copy_residual_block(torch_block[1], jax_block["res2"])
        if jax_block["upsample"] is not None:
            upsample = torch_block[2].conv
            jax_block["upsample"]["w"] = _to_jax_conv_transpose_weight(
                upsample.weight
            )
            jax_block["upsample"]["b"] = upsample.bias.detach().cpu().numpy()

    _copy_conv1d_block(torch_model.final_conv[0], jax_params["final_conv"]["block"])
    final_conv = torch_model.final_conv[1]
    jax_params["final_conv"]["out"]["w"] = _to_jax_conv_weight(final_conv.weight)
    jax_params["final_conv"]["out"]["b"] = final_conv.bias.detach().cpu().numpy()
    return jax.tree_util.tree_map(jax.numpy.asarray, jax_params)


def _make_models():
    torch_model = ConditionalUnet1D(
        input_shapes={"actions": (3,), "features": (5,)},
        output_shape=3,
        sequence_length=8,
        diffusion_step_embed_dim=32,
        down_dims=[32, 64],
        kernel_size=3,
        n_groups=4,
    )
    jax_model = JaxConditionalUnet1D(
        action_shape=(8, 3),
        feature_dim=5,
        diffusion_step_embed_dim=32,
        down_dims=(32, 64),
        kernel_size=3,
        n_groups=4,
    )
    jax_params = jax_model.init_params(jax.random.PRNGKey(0))
    jax_params = _copy_torch_diffusion_to_jax(torch_model, jax_params)
    return torch_model.eval(), jax_model, jax_params


def test_jax_diffusion_schedule_matches_torch_warmup_cosine():
    parameter = torch.nn.Parameter(torch.tensor(1.0))
    optimizer = torch.optim.Adam([parameter], lr=1e-4)
    torch_scheduler = get_scheduler(
        "cosine",
        optimizer=optimizer,
        num_warmup_steps=100,
        num_training_steps=1000,
    )

    import optax

    jax_scheduler = optax.warmup_cosine_decay_schedule(
        init_value=0.0,
        peak_value=1e-4,
        warmup_steps=100,
        decay_steps=1000,
        end_value=0.0,
    )

    torch_values = [optimizer.param_groups[0]["lr"]]
    for _ in range(1000):
        optimizer.step()
        torch_scheduler.step()
        torch_values.append(optimizer.param_groups[0]["lr"])

    jax_values = [float(jax_scheduler(step)) for step in range(1001)]
    assert np.max(np.abs(np.asarray(torch_values) - np.asarray(jax_values))) < 1e-10


def test_jax_diffusion_model_parameter_count_matches_torch():
    torch_model, jax_model, jax_params = _make_models()
    torch_count = sum(param.numel() for param in torch_model.parameters())
    leaves, _ = jax.tree_util.tree_flatten(jax_params)
    jax_count = sum(int(np.prod(np.asarray(leaf).shape)) for leaf in leaves)
    assert jax_count == torch_count


def test_jax_diffusion_forward_and_ddim_step_match_torch():
    torch.manual_seed(0)
    np.random.seed(0)

    torch_model, jax_model, jax_params = _make_models()
    actions = torch.randn(2, 8, 3)
    features = torch.randn(2, 5)
    timesteps = torch.tensor([17, 63], dtype=torch.long)

    with torch.no_grad():
        torch_out = torch_model(
            {"actions": actions, "features": features, "timestep": timesteps}
        )

    jax_out = jax_model.apply(
        jax_params,
        jax.numpy.asarray(actions.numpy()),
        jax.numpy.asarray(timesteps.numpy()),
        jax.numpy.asarray(features.numpy()),
    )
    jax_out_np = np.asarray(jax.device_get(jax_out))

    assert np.max(np.abs(torch_out.numpy() - jax_out_np)) < 1e-3

    scheduler = DDIMScheduler(
        num_train_timesteps=100,
        beta_schedule="squaredcos_cap_v2",
        clip_sample=True,
        prediction_type="epsilon",
    )
    scheduler.set_timesteps(100)
    sample = torch.randn(1, 8, 3)
    torch_prev = scheduler.step(
        torch_out[:1],
        int(timesteps[0].item()),
        sample,
    ).prev_sample

    alphas_cumprod = np.cumprod(1.0 - _cosine_betas(100))
    timestep = int(timesteps[0].item())
    alpha_t = alphas_cumprod[timestep]
    alpha_prev = alphas_cumprod[timestep - 1] if timestep > 0 else 1.0
    pred_original_sample = (
        sample[0].numpy() - np.sqrt(1.0 - alpha_t) * jax_out_np[0]
    ) / np.sqrt(alpha_t)
    pred_original_sample = np.clip(pred_original_sample, -1.0, 1.0)
    jax_prev = np.sqrt(alpha_prev) * pred_original_sample + np.sqrt(
        max(1.0 - alpha_prev, 0.0)
    ) * jax_out_np[0]

    assert np.max(np.abs(torch_prev[0].numpy() - jax_prev)) < 2e-4
