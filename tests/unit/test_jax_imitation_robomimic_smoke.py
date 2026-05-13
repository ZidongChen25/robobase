import json
from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from hydra.core.global_hydra import GlobalHydra

from robobase.envs.robomimic import RobomimicEnvFactory
from robobase.workspace import Workspace

h5py = pytest.importorskip("h5py")
pytest.importorskip("jax")
pytest.importorskip("optax")


def _write_low_dim_robomimic_dataset(path):
    with h5py.File(path, "w") as dataset_file:
        data_group = dataset_file.create_group("data")
        data_group.attrs["env_args"] = json.dumps(
            {
                "env_name": "Lift",
                "env_version": "1.4.1",
                "type": 1,
                "env_kwargs": {
                    "robots": ["Panda"],
                    "controller_configs": {"type": "OSC_POSE"},
                    "reward_shaping": False,
                },
            }
        )

        episode = data_group.create_group("demo_0")
        length = 5
        episode.attrs["num_samples"] = length
        episode.create_dataset(
            "actions",
            data=np.linspace(-0.5, 0.5, length * 2, dtype=np.float32).reshape(length, 2),
        )
        rewards = np.zeros(length, dtype=np.float32)
        rewards[-1] = 1.0
        episode.create_dataset("rewards", data=rewards)
        dones = np.zeros(length, dtype=np.uint8)
        dones[-1] = 1
        episode.create_dataset("dones", data=dones)

        obs = episode.create_group("obs")
        next_obs = episode.create_group("next_obs")
        base = np.arange(length * 4, dtype=np.float32).reshape(length, 4)
        obs.create_dataset("robot0_eef_pos", data=base)
        next_obs.create_dataset("robot0_eef_pos", data=base + 1)
        obj = np.arange(length * 3, dtype=np.float32).reshape(length, 3)
        obs.create_dataset("object", data=obj)
        next_obs.create_dataset("object", data=obj + 1)

        mask_group = dataset_file.create_group("mask")
        mask_group.create_dataset("train", data=np.asarray([b"demo_0"]))


@pytest.mark.parametrize("method_name", ["flow_matching", "act"])
def test_jax_imitation_methods_pretrain_on_robomimic_low_dim_dataset(
    tmp_path,
    method_name,
):
    dataset_path = tmp_path / "robomimic_low_dim.hdf5"
    _write_low_dim_robomimic_dataset(dataset_path)

    overrides = [
        "backend=jax",
        f"method={method_name}",
        "env=robomimic/lift",
        f"env.dataset_path={dataset_path}",
        "env.use_live_env=false",
        "pixels=false",
        "demos=1",
        "num_pretrain_steps=1",
        "num_train_frames=0",
        "num_train_envs=1",
        "num_eval_envs=1",
        "num_eval_episodes=0",
        "batch_size=1",
        "replay.size=16",
        "replay.nstep=1",
        "replay.num_workers=0",
        "replay.pin_memory=false",
        "replay.prioritization=false",
        "replay_size_before_train=0",
        "action_sequence=3",
        "execution_length=1",
        "use_min_max_normalization=false",
        "use_standardization=false",
        "norm_obs=false",
        "log_pretrain_every=1",
        "log_eval_video=false",
        "save_snapshot=true",
        "snapshot_every_n=1",
        "wandb.use=false",
    ]
    if method_name == "flow_matching":
        overrides.extend(
            [
                "method.adaptive_lr=false",
                "method.num_flow_steps=2",
                "method.backbone.hidden_dims=[16]",
            ]
        )
    else:
        overrides.extend(
            [
                "method.adaptive_lr=false",
                "method.actor_model.hidden_dim=32",
                "method.actor_model.enc_layers=1",
                "method.actor_model.dec_layers=1",
                "method.actor_model.dim_feedforward=64",
                "method.actor_model.nheads=4",
                "method.actor_model.dropout=0.0",
            ]
        )

    GlobalHydra.instance().clear()
    with initialize_config_dir(
        version_base=None,
        config_dir=str(Path(__file__).resolve().parents[2] / "robobase/cfgs"),
        job_name=f"test_robomimic_{method_name}",
    ):
        cfg = compose(config_name="robobase_config", overrides=overrides)

    workspace = Workspace(
        cfg,
        env_factory=RobomimicEnvFactory(),
        work_dir=tmp_path / method_name,
    )
    try:
        workspace.train()
    finally:
        GlobalHydra.instance().clear()

    assert (tmp_path / method_name / "snapshots" / "latest_snapshot.pkl").exists()
