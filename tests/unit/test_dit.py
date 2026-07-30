from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from gymnasium import spaces
import numpy as np
import pytest

from robobase.method.diffusion import (
    Diffusion,
    DiffusionModelSpec,
    diffusion_spec_from_cfg,
)
from robobase.method.flow_matching import (
    FlowMatching,
    FlowMatchingModelSpec,
    flow_matching_spec_from_cfg,
)
from robobase.models.backbone import DiffusionBackboneSpec, build_diffusion_backbone


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


def _clean_spec(**overrides) -> DiffusionBackboneSpec:
    values = {
        "type": "dit",
        "sequence_length": 8,
        "diffusion_step_embed_dim": 32,
        "d_model": 32,
        "n_heads": 4,
        "depth": 2,
        "dropout": 0.0,
        "timestep_embedding_type": "fourier",
        "operator_variant": "torch",
        "compatibility_mode": "clean_diffuser",
        "condition_adapter": "clean_mlp",
        "condition_hidden_dims": (32,),
        "condition_dropout": 0.25,
    }
    values.update(overrides)
    return DiffusionBackboneSpec(**values)


def test_clean_diffuser_dit_is_plug_and_play_with_raw_condition_features():
    model = build_diffusion_backbone(
        _clean_spec(),
        action_dim=3,
        sequence_length=8,
        condition_dim=11,
    )
    actions = jnp.zeros((2, 8, 3), dtype=jnp.float32)
    timesteps = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    condition = jnp.ones((2, 11), dtype=jnp.float32)
    variables = model.init(
        {"params": jax.random.PRNGKey(0), "dropout": jax.random.PRNGKey(1)},
        actions,
        timesteps,
        condition,
        train=True,
    )

    output = model.apply(
        variables,
        actions,
        timesteps,
        condition,
        train=True,
        rngs={"dropout": jax.random.PRNGKey(2)},
    )

    assert output.shape == actions.shape
    assert model.output_shape == (8, 3)
    frequencies = variables["params"]["time_embedding"]["fourier_frequencies"]
    base_params = variables["params"]
    gradient = jax.grad(
        lambda value: model.apply(
            {
                "params": {
                    **base_params,
                    "time_embedding": {
                        **base_params["time_embedding"],
                        "fourier_frequencies": value,
                    },
                }
            },
            actions,
            timesteps,
            condition,
            train=False,
        ).sum()
    )(frequencies)
    np.testing.assert_array_equal(gradient, np.zeros_like(gradient))


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"operator_variant": "legacy"}, "operator_variant must be torch"),
        ({"timestep_embedding_type": "campose"}, "timestep_embedding_type"),
        ({"condition_adapter": "linear"}, "condition_adapter"),
        ({"d_model": 30}, "divisible by n_heads"),
        (
            {"diffusion_step_embed_dim": 30},
            "positive multiple of 8",
        ),
    ],
)
def test_clean_diffuser_dit_rejects_silent_mismatches(overrides, message):
    with pytest.raises(ValueError, match=message):
        build_diffusion_backbone(
            _clean_spec(**overrides),
            action_dim=3,
            sequence_length=8,
            condition_dim=11,
        )


def test_clean_diffuser_dit_backbone_group_composes_for_diffusion():
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
            job_name="test_clean_diffuser_dit_config",
        ):
            cfg = compose(
                config_name="robobase_config",
                overrides=[
                    "backend=jax",
                    "method=diffusion",
                    "backbone=clean_diffuser_dit",
                    "env=dmc/cartpole_balance",
                    "pixels=false",
                    "action_sequence=8",
                ],
            )
        spec = diffusion_spec_from_cfg(cfg).model.resolved_backbone
    finally:
        GlobalHydra.instance().clear()

    assert spec.type == "dit"
    assert spec.d_model == 384
    assert spec.n_heads == 12
    assert spec.depth == 6
    assert spec.timestep_embedding_type == "fourier"
    assert spec.operator_variant == "torch"
    assert spec.compatibility_mode == "clean_diffuser"
    assert spec.condition_adapter == "clean_mlp"
    assert spec.condition_hidden_dims == (256,)
    assert spec.condition_dropout == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("launch", "method_name"),
    [
        ("clean_diffuser_dit_ddpm_state_robomimic", "diffusion"),
        ("clean_diffuser_dit_fm_state_robomimic", "flow_matching"),
    ],
)
def test_clean_diffuser_dit_launches_are_complete(launch, method_name):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(
            version_base=None,
            config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
            job_name=f"test_{launch}",
        ):
            cfg = compose(
                config_name="robobase_config",
                overrides=[
                    "backend=jax",
                    f"launch={launch}",
                    "env=robomimic_clean/lift",
                    "env.dataset_path=/tmp/low_dim_abs.hdf5",
                ],
            )
        if method_name == "diffusion":
            parsed = diffusion_spec_from_cfg(cfg)
            backbone = parsed.model.resolved_backbone
        else:
            parsed = flow_matching_spec_from_cfg(cfg)
            backbone = parsed.model.backbone
    finally:
        GlobalHydra.instance().clear()

    assert cfg.method.name == method_name
    assert cfg.action_sequence == 10
    assert cfg.execution_length == 8
    assert backbone.type == "dit"
    assert backbone.timestep_embedding_type == "fourier"
    assert backbone.operator_variant == "torch"
    assert backbone.condition_adapter == "clean_mlp"


@pytest.mark.parametrize("method_type", ["diffusion", "flow_matching"])
def test_clean_diffuser_dit_updates_and_samples_in_jax_methods(method_type):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(1, 5),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4, 2),
        dtype=np.float32,
    )
    backbone = _clean_spec(sequence_length=4, depth=1)
    common = dict(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=False,
        seed=0,
        use_ema=False,
        weight_decay=0.1,
    )
    if method_type == "diffusion":
        agent = Diffusion(
            model=DiffusionModelSpec(
                actor_model=backbone,
                encoder_model=None,
                view_fusion_model=None,
            ),
            num_diffusion_iters=4,
            **common,
        )
    else:
        agent = FlowMatching(
            model=FlowMatchingModelSpec(
                backbone=backbone,
                encoder_model=None,
                view_fusion_model=None,
            ),
            num_flow_steps=2,
            **common,
        )

    frequency_path = agent.params["params"]["time_embedding"]
    frequencies_before = np.asarray(frequency_path["fourier_frequencies"]).copy()
    batch = {
        "low_dim_state": np.ones((2, 1, 5), dtype=np.float32),
        "action": np.zeros((2, 4, 2), dtype=np.float32),
    }

    assert agent.update(iter([batch]), step=0) == {}
    sampled = agent.act(batch, step=0, eval_mode=False)
    frequencies_after = np.asarray(
        agent.params["params"]["time_embedding"]["fourier_frequencies"]
    )

    np.testing.assert_array_equal(frequencies_after, frequencies_before)
    assert sampled.shape == (2, 4, 2)
    assert np.isfinite(sampled).all()
