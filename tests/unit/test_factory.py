from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf

from robobase.factory import create_agent, method_name_from_cfg


LEGACY_TARGETS = {
    "robobase.method.alix.ALIX": "alix",
    "robobase.method.dreamerv3.DreamerV3": "dreamerv3",
    "robobase.method.drm.DrM": "drm",
    "robobase.method.edp.DiffusionRL": "edp",
    "robobase.method.iql_drqv2.IQLDrQV2": "iql_drqv2",
    "robobase.method.mwm.MaskedWorldModel": "mwm",
    "robobase.method.sac_lix.SACLix": "sac_lix",
    "robobase.method.value_based.ValueBased": "value_based",
}


@pytest.mark.parametrize(
    ("target", "expected_name"),
    [
        ("robobase.method.cqn.CQN", "cqn"),
        ("robobase.method.cqn_as.CQNAS", "cqn_as"),
        ("robobase.method.cqn_flow.CQNFlowAS", "cqn_flow"),
        ("robobase.method.ppo.PPO", "ppo"),
        ("robobase.method.q_chunking.QChunking", "q_chunking"),
        ("robobase.method.sac.SAC", "sac"),
        ("robobase.method.drqv2.DrQV2", "drqv2"),
    ],
)
def test_pure_jax_rl_method_targets_are_available(target, expected_name):
    cfg = OmegaConf.create({"method": {"_target_": target}})
    assert method_name_from_cfg(cfg) == expected_name


@pytest.mark.parametrize(("target", "expected_name"), LEGACY_TARGETS.items())
def test_legacy_method_configs_fail_with_jax_only_error(target, expected_name):
    cfg = OmegaConf.create({"method": {"_target_": target}})

    assert method_name_from_cfg(cfg) == expected_name
    with pytest.raises(NotImplementedError, match="historical Torch configuration"):
        create_agent(cfg, observation_space=None, action_space=None)


@pytest.mark.parametrize(
    ("launch", "method", "backbone", "encoder", "fusion", "horizon"),
    [
        ("campose_dp_bigym", "diffusion", "unet1d", "dp_resnet", "dp_early", 32),
        ("campose_act_bigym", "act", None, "resnet", "act_late", 30),
        (
            "campose_fm_bigym",
            "flow_matching",
            "unet1d",
            "dp_resnet",
            "dp_early",
            32,
        ),
    ],
)
def test_campose_launches_are_complete_plug_and_play_configs(
    launch, method, backbone, encoder, fusion, horizon
):
    GlobalHydra.instance().clear()
    config_dir = str((Path(__file__).parents[2] / "robobase" / "cfgs").resolve())
    with initialize_config_dir(config_dir=config_dir, version_base=None):
        cfg = compose(
            config_name="robobase_config",
            overrides=[f"launch={launch}", "env=bigym/move_plate"],
        )

    assert cfg.method.name == method
    assert cfg.method.get("backbone", {}).get("type") == backbone
    assert cfg.method.encoder_model.type == encoder
    assert cfg.method.encoder_model.plucker_fusion_mode == fusion
    assert cfg.action_sequence == horizon
    assert cfg.execution_length == horizon
    assert cfg.is_imitation_learning
    assert cfg.num_pretrain_steps > 0
    assert cfg.num_train_frames == 0
    assert cfg.replay.nstep == 1
    assert cfg.lazy_replay.observation_timing == "pre_action"
