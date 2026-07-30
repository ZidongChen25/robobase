from pathlib import Path

import numpy as np
import pytest
from gymnasium import spaces
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

pytest.importorskip("jax")
pytest.importorskip("optax")

from robobase.factory import create_agent, method_name_from_cfg
from robobase.method.a2a import A2A
from robobase.method.flow_matching import flow_matching_spec_from_cfg
from robobase.method.legato import Legato
from robobase.workspace import _create_default_replay_buffer


CONFIG_DIR = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())


def _compose_launch(launch: str, *overrides: str):
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            return compose(
                config_name="robobase_config",
                overrides=[
                    f"launch={launch}",
                    "env=bigym/flip_cutlery",
                    *overrides,
                ],
            )
    finally:
        GlobalHydra.instance().clear()


@pytest.mark.parametrize(
    ("launch", "method_name", "source_type", "backbone_horizon", "policy_label"),
    [
        ("flow_extensions_bigym", "flow_matching", "gaussian", 8, "flow_matching"),
        ("a2a_flow_extensions_bigym", "a2a", "a2a", 1, "a2a"),
        ("a2a_noise_flow_extensions_bigym", "a2a", "a2a_noise", 1, "a2a_noise"),
        ("legato_flow_extensions_bigym", "legato", "legato", 8, "legato"),
    ],
)
def test_flow_extension_launch_contract(
    launch: str,
    method_name: str,
    source_type: str,
    backbone_horizon: int,
    policy_label: str,
):
    cfg = _compose_launch(launch)
    spec = flow_matching_spec_from_cfg(cfg)

    assert cfg.method.name == method_name
    assert cfg.method.policy_label == policy_label
    assert f"_{policy_label}_" in cfg.wandb.name
    assert spec.flow_source.type == source_type
    assert cfg.method.backbone.type == "transformer"
    assert cfg.method.backbone.sequence_length == backbone_horizon
    assert cfg.action_sequence == 8
    assert cfg.execution_length == 4


def test_repaired_pixel_profile_resolves_split_bigym_roots():
    cfg = _compose_launch(
        "a2a_flow_extensions_bigym",
        "profile=bigym_repaired_pixels",
    )

    home = Path.home()
    assert Path(cfg.env.dataset_root) == home / ".bigym"
    assert Path(cfg.env.pixel_dataset_root) == home / ".bigym_reset_aligned"
    assert Path(cfg.env.state_dataset_root) == home / ".bigym"


def test_repaired_200_epoch_launch_matches_baseline_checkpoint_protocol():
    cfg = _compose_launch("a2a_flip_cutlery_repaired_200e")

    assert cfg.num_pretrain_epochs == 200
    assert cfg.eval_every_epochs == 100
    assert cfg.num_eval_episodes == 50
    assert cfg.snapshot_every_epochs == 20
    assert cfg.action_sequence == 20
    assert cfg.execution_length == 20
    assert cfg.method.num_train_steps == 1_000_000
    assert cfg.method.num_flow_steps == 10
    assert cfg.method.adaptive_lr is True
    assert cfg.method.lang_feature_source == "precomputed"
    assert cfg.method.lang_description == "reach the target"
    assert cfg.method.flow_source.history_horizon == 20
    assert cfg.method.encoder_model.pretrained_weights_path == str(
        Path.home()
        / ".cache"
        / "robobase_jaxflat"
        / "resnet18_a1_in1k_jax_resnet.npz"
    )
    assert cfg.method.backbone.dropout == 0.0
    assert cfg.backend.fused_update_steps == 1
    assert cfg.backend.update_block_every_steps == 8


def test_repaired_200_epoch_launch_is_self_contained():
    GlobalHydra.instance().clear()
    try:
        with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
            cfg = compose(
                config_name="robobase_config",
                overrides=["launch=a2a_flip_cutlery_repaired_200e"],
            )
    finally:
        GlobalHydra.instance().clear()

    assert cfg.env.env_name == "bigym"
    assert cfg.env.task_name == "flip_cutlery"


def test_repaired_200_epoch_legato_launch_keeps_valid_overlap():
    cfg = _compose_launch("legato_flip_cutlery_repaired_200e")
    spec = flow_matching_spec_from_cfg(cfg)

    assert cfg.action_sequence == 20
    assert cfg.execution_length == 10
    assert (
        cfg.action_execution_start
        + spec.flow_source.delay_max_steps
        + spec.flow_source.ramp_max_steps
        <= cfg.action_sequence - cfg.execution_length
    )


@pytest.mark.parametrize(
    ("launch", "target", "expected_method", "expected_source"),
    [
        (
            "a2a_flow_extensions_bigym",
            "robobase.method.a2a.A2A",
            "a2a",
            "a2a",
        ),
        (
            "legato_flow_extensions_bigym",
            "robobase.method.legato.Legato",
            "legato",
            "legato",
        ),
    ],
)
def test_target_only_flow_extension_defaults_match_factory_dispatch(
    launch: str,
    target: str,
    expected_method: str,
    expected_source: str,
):
    composed = _compose_launch(launch)
    raw_cfg = OmegaConf.to_container(composed, resolve=False)
    method_cfg = raw_cfg["method"]
    method_cfg.pop("name")
    method_cfg["_target_"] = target
    method_cfg["flow_source"].pop("type")
    cfg = OmegaConf.create(raw_cfg)

    assert method_name_from_cfg(cfg) == expected_method
    assert flow_matching_spec_from_cfg(cfg).flow_source.type == expected_source


def test_legato_factory_rejects_replay_execution_offset_mismatch():
    cfg = _compose_launch(
        "legato_flow_extensions_bigym",
        "replay.action_sequence_start_offset=1",
    )

    with pytest.raises(
        ValueError,
        match="Legato requires replay.action_sequence_start_offset",
    ):
        create_agent(cfg, observation_space=None, action_space=None)


def test_legato_factory_rejects_full_chunk_execution_before_model_build():
    cfg = _compose_launch(
        "legato_flow_extensions_bigym",
        "action_sequence=20",
        "execution_length=20",
    )

    with pytest.raises(ValueError, match="previous chunk overlap"):
        create_agent(cfg, observation_space=None, action_space=None)


def test_flow_extensions_are_available_from_method_package():
    from robobase import method

    assert method.A2A is A2A
    assert method.Legato is Legato
    assert {"A2A", "Legato"}.issubset(method.__all__)


@pytest.mark.parametrize(
    ("launch", "expected_class", "source_type"),
    [
        ("a2a_flow_extensions_bigym", A2A, "a2a"),
        ("a2a_noise_flow_extensions_bigym", A2A, "a2a_noise"),
        ("legato_flow_extensions_bigym", Legato, "legato"),
    ],
)
def test_flow_extension_factory_creates_state_only_agent(
    launch: str,
    expected_class: type,
    source_type: str,
):
    overrides = [
        "pixels=false",
        "action_sequence=4",
        "execution_length=2",
        "method.use_lang_cond=false",
        "method.encoder_model=null",
        "method.view_fusion_model=null",
        "method.backbone.d_model=16",
        "method.backbone.n_heads=2",
        "method.backbone.num_layers=1",
        "method.backbone.n_cond_layers=0",
        "method.backbone.dropout=0.0",
        "backend.jit=false",
        "backend.platform=cpu",
    ]
    if launch.startswith("a2a"):
        overrides.extend(
            [
                "method.flow_source.history_horizon=4",
                "method.flow_source.latent_dim=8",
                "method.flow_source.hidden_dim=8",
                "method.flow_source.encoder_layers=1",
                "method.flow_source.decoder_layers=1",
                "method.flow_source.kernel_size=3",
                "method.flow_source.noise_exclude_last_n=0",
                ]
            )
    elif launch.startswith("legato"):
        overrides.extend(
            [
                "method.flow_source.delay_min_steps=1",
                "method.flow_source.delay_max_steps=1",
                "method.flow_source.ramp_min_steps=1",
                "method.flow_source.ramp_max_steps=1",
                "method.flow_source.eval_delay_steps=1",
                "method.flow_source.eval_ramp_steps=1",
            ]
        )
    cfg = _compose_launch(launch, *overrides)
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 4),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(4, 2),
        dtype=np.float32,
    )

    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    assert isinstance(agent, expected_class)
    assert agent._flow_source_type == source_type
    assert agent.action_sequence == 4
    assert agent.execution_length == 2


def test_a2a_standard_replay_builds_episode_local_history(tmp_path):
    cfg = _compose_launch(
        "a2a_flow_extensions_bigym",
        "pixels=false",
        "demos=0",
        "action_sequence=4",
        "execution_length=2",
        "method.flow_source.history_horizon=3",
        "method.flow_source.history_source=commanded_action",
        "method.flow_source.history_padding=zero",
    )
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf,
                np.inf,
                shape=(1, 2),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        -1.0,
        1.0,
        shape=(4, 2),
        dtype=np.float32,
    )
    replay = _create_default_replay_buffer(
        cfg,
        observation_space,
        action_space,
        save_dir=str(tmp_path),
    )
    for step in range(4):
        replay.add(
            {"low_dim_state": np.full((2,), step, dtype=np.float32)},
            np.full((2,), step + 1, dtype=np.float32),
            0.0,
            step == 3,
            False,
        )
    replay.add_final({"low_dim_state": np.full((2,), 4, dtype=np.float32)})

    batch = replay.sample_batch_indices(np.asarray([0, 3]))

    np.testing.assert_array_equal(
        batch["action_history"][:, :, 0],
        [[0, 0, 0], [1, 2, 3]],
    )
    np.testing.assert_array_equal(
        batch["action_history_pad_mask"],
        [[True, True, True], [False, False, False]],
    )


def test_a2a_standard_replay_can_use_current_executed_feedback(tmp_path):
    cfg = _compose_launch(
        "a2a_flow_extensions_bigym",
        "pixels=false",
        "demos=0",
        "action_sequence=4",
        "execution_length=2",
        "method.flow_source.history_horizon=3",
    )
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                -np.inf, np.inf, shape=(1, 2), dtype=np.float32
            ),
            "executed_action_feedback": spaces.Box(
                -np.inf, np.inf, shape=(1, 2), dtype=np.float32
            ),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, shape=(4, 2), dtype=np.float32)
    replay = _create_default_replay_buffer(
        cfg, observation_space, action_space, save_dir=str(tmp_path)
    )
    for step in range(4):
        replay.add(
            {
                "low_dim_state": np.zeros((2,), dtype=np.float32),
                "executed_action_feedback": np.full(
                    (2,), step, dtype=np.float32
                ),
            },
            np.full((2,), 100 + step, dtype=np.float32),
            0.0,
            step == 3,
            False,
        )
    replay.add_final(
        {
            "low_dim_state": np.zeros((2,), dtype=np.float32),
            "executed_action_feedback": np.full((2,), 4, dtype=np.float32),
        }
    )

    batch = replay.sample_batch_indices(np.asarray([0, 3]))

    np.testing.assert_array_equal(
        batch["action_history"][:, :, 0], [[0, 0, 0], [1, 2, 3]]
    )
    np.testing.assert_array_equal(batch["action_history_pad_mask"], False)
