from __future__ import annotations

import gzip
import json
import math
import pickle
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.official_bigym.a2a_official_entrypoint import (
    patch_eval_policy_cfg_joint_pos,
    patch_eval_trajectory,
)
from benchmarks.official_roboverse.audit_proxy_data import HASH_SPEC, SCHEMA
from benchmarks.official_roboverse.eval import build_eval_command
from benchmarks.official_roboverse.random_initialization import (
    generate_close_box_random_initializations,
    validate_random_initialization_manifest,
)


def _source_trajectory(path: Path, count: int = 8) -> Path:
    demos = []
    for index in range(count):
        yaw = -0.2 + 0.4 * index / (count - 1)
        demos.append(
            {
                "init_state": {
                    "box_base": {
                        "pos": [0.2 + 0.01 * index, 0.1 + 0.01 * index, 0.075],
                        "rot": [math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)],
                        "dof_pos": {"box_joint": 2.3},
                    },
                    "franka": {
                        "pos": [0.0, 0.0, 0.0],
                        "rot": [1.0, 0.0, 0.0, 0.0],
                        "dof_pos": {"joint": 0.0},
                    },
                }
            }
        )
    with gzip.open(path, "wb") as file:
        pickle.dump({"franka": demos}, file)
    return path


def test_eval_policy_cfg_joint_pos_patch_corrects_loaded_checkpoint_metadata():
    class EvalRunner:
        def _init_policy(self, *args, **kwargs):
            del args, kwargs
            self.policy_cfg = SimpleNamespace(
                obs_config=SimpleNamespace(obs_type="ee"),
                action_config=SimpleNamespace(action_type="ee", delta=True),
            )

    patch_eval_policy_cfg_joint_pos(EvalRunner)
    runner = EvalRunner()
    runner._init_policy()

    assert runner.policy_cfg.obs_config.obs_type == "joint_pos"
    assert runner.policy_cfg.action_config.action_type == "joint_pos"
    assert runner.policy_cfg.action_config.delta is False


def test_generates_unique_disjoint_reproducible_states(tmp_path):
    source = _source_trajectory(tmp_path / "source_v2.pkl.gz")
    first = generate_close_box_random_initializations(
        source_trajectory=source,
        output_trajectory=tmp_path / "first_v2.pkl.gz",
        output_manifest=tmp_path / "first.json",
        training_source_indices=list(range(8)),
        count=5,
        seed=7,
        position_jitter_m=0.005,
        yaw_jitter_degrees=2.0,
        near_duplicate_position_threshold_m=0.0001,
        near_duplicate_rotation_threshold_degrees=0.01,
    )
    second = generate_close_box_random_initializations(
        source_trajectory=source,
        output_trajectory=tmp_path / "second_v2.pkl.gz",
        output_manifest=tmp_path / "second.json",
        training_source_indices=list(range(8)),
        count=5,
        seed=7,
        position_jitter_m=0.005,
        yaw_jitter_degrees=2.0,
        near_duplicate_position_threshold_m=0.0001,
        near_duplicate_rotation_threshold_degrees=0.01,
    )

    assert first["exact_training_overlap_count"] == 0
    assert first["unique_generated_state_count"] == 5
    assert first["minimum_normalized_pose_distance"] >= 1.0
    assert [row["position_offset_m"] for row in first["states"]] == [
        row["position_offset_m"] for row in second["states"]
    ]
    assert validate_random_initialization_manifest(
        tmp_path / "first.json", expected_count=5
    )["count"] == 5


def test_manifest_rejects_trajectory_drift(tmp_path):
    source = _source_trajectory(tmp_path / "source_v2.pkl.gz")
    generate_close_box_random_initializations(
        source_trajectory=source,
        output_trajectory=tmp_path / "eval_v2.pkl.gz",
        output_manifest=tmp_path / "manifest.json",
        training_source_indices=list(range(8)),
        count=2,
        near_duplicate_position_threshold_m=0.0001,
        near_duplicate_rotation_threshold_degrees=0.01,
    )
    (tmp_path / "eval_v2.pkl.gz").write_bytes(b"changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_random_initialization_manifest(tmp_path / "manifest.json", expected_count=2)


def test_entrypoint_patches_only_supported_task(tmp_path):
    trajectory = _source_trajectory(tmp_path / "eval_v2.pkl.gz")
    task_class = SimpleNamespace(traj_filepath="original")
    assert patch_eval_trajectory(trajectory, task_class=task_class) == trajectory.resolve()
    assert task_class.traj_filepath == str(trajectory.resolve())
    with pytest.raises(ValueError, match="Unsupported"):
        patch_eval_trajectory(trajectory, task="stack_cube", task_class=task_class)


def test_eval_command_binds_random_cohort_and_equalized_flow_steps(tmp_path):
    source = _source_trajectory(tmp_path / "source_v2.pkl.gz")
    dataset = tmp_path / "train.zarr"
    random_manifest = tmp_path / "random.json"
    generate_close_box_random_initializations(
        source_trajectory=source,
        output_trajectory=tmp_path / "heldout_v2.pkl.gz",
        output_manifest=random_manifest,
        training_source_indices=list(range(8)),
        count=50,
        seed=9,
        position_jitter_m=0.005,
        yaw_jitter_degrees=2.0,
        near_duplicate_position_threshold_m=0.0001,
        near_duplicate_rotation_threshold_degrees=0.01,
    )
    provenance = tmp_path / "provenance.json"
    provenance.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "pass",
                "hash_spec": HASH_SPEC,
                "episodes": 8,
                "dataset": str(dataset.resolve()),
                "logical_content_sha256": "a" * 64,
                "selected_source_indices": list(range(8)),
                "errors": [],
                "raw_exact_match": True,
            }
        )
    )

    command, manifest = build_eval_command(
        task_key="close_box",
        dataset=dataset,
        checkpoint=tmp_path / "30.ckpt",
        output=tmp_path / "eval",
        method="fm_unet",
        checkpoint_epoch=30,
        expected_episodes=8,
        dataset_provenance=provenance,
        random_initialization_manifest=random_manifest,
        flow_steps=6,
        fm_solver="euler",
    )

    assert "policy_config.num_inference_steps=6" in command
    assert (
        "policy_config._target_=benchmarks.official_bigym.fm_unet_euler_policy."
        "EulerFlowMatchingUnetImagePolicy"
    ) in command
    assert manifest["flow_steps"] == 6
    assert manifest["solver"] == "euler"
    assert manifest["model_calls_per_replan"] == 6
    assert manifest["evaluation_split"] == "heldout_random_initialization"
    assert manifest["random_initialization"]["exact_training_overlap_count"] == 0
    assert manifest["dataset_provenance"][
        "random_initialization_source_ids_match"
    ] is True
    assert manifest["environment_overrides"][
        "ROBOBASE_OFFICIAL_EVAL_TRAJECTORY"
    ] == str((tmp_path / "heldout_v2.pkl.gz").resolve())
