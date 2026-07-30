from __future__ import annotations

import json
from pathlib import Path
import subprocess

import numpy as np
import pytest

from benchmarks.official_roboverse.eval import build_eval_command
from benchmarks.official_roboverse.audit_proxy_data import HASH_SPEC, SCHEMA
from benchmarks.official_roboverse.preflight import (
    audit_zarr_dataset,
    run_preflight,
    validate_paper_checkout,
)
from benchmarks.official_roboverse.protocol import (
    PAPER_DATA_REVISION,
    PAPER_SOURCE_COMMIT,
    PAPER_TASKS,
    paper_protocol_manifest,
)
from benchmarks.official_roboverse.train import build_train_command


def _make_zarr(path: Path, *, episodes: int) -> Path:
    zarr = pytest.importorskip("zarr")
    root = zarr.group(str(path))
    data = root.create_group("data")
    meta = root.create_group("meta")
    frames = episodes * 2
    data.create_dataset(
        "head_camera",
        data=np.zeros((frames, 3, 4, 4), dtype=np.uint8),
        chunks=(16, 3, 4, 4),
    )
    data.create_dataset(
        "state", data=np.zeros((frames, 9), dtype=np.float32), chunks=(16, 9)
    )
    data.create_dataset(
        "action", data=np.zeros((frames, 9), dtype=np.float32), chunks=(16, 9)
    )
    meta.create_dataset(
        "episode_ends",
        data=np.arange(2, frames + 1, 2, dtype=np.int64),
        chunks=(16,),
    )
    return path


def _git(*args: str, cwd: Path) -> str:
    return subprocess.run(
        ["git", *args], cwd=cwd, check=True, capture_output=True, text=True
    ).stdout.strip()


def _write_provenance(
    path: Path, *, dataset: Path, source_indices: list[int]
) -> Path:
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "pass",
                "hash_spec": HASH_SPEC,
                "episodes": len(source_indices),
                "dataset": str(dataset.resolve()),
                "logical_content_sha256": "a" * 64,
                "selected_source_indices": source_indices,
                "errors": [],
                "raw_exact_match": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_protocol_distinguishes_exact_tasks_from_blocked_proxies():
    exact = [task.key for task in PAPER_TASKS.values() if task.is_exact]
    proxies = [task.key for task in PAPER_TASKS.values() if not task.is_exact]
    assert exact == ["close_box", "pick_cube", "stack_cube"]
    assert proxies == ["open_drawer", "pick_place_bowl"]
    manifest = paper_protocol_manifest()
    assert manifest["source_commit"] == PAPER_SOURCE_COMMIT
    assert manifest["data_revision"] == PAPER_DATA_REVISION
    assert manifest["demonstrations"] == 100
    assert manifest["paper_epochs"] == 30
    assert manifest["methods"] == {
        "a2a": {"flow_steps": 6, "source_variant": "initial_release_ot"},
        "a2a_current": {
            "flow_steps": 6,
            "source_variant": "current_main_conditional",
            "paper_target_comparable": False,
        },
        "fm_unet": {"flow_steps": 10},
    }


def test_dataset_audit_reads_real_episode_boundaries(tmp_path):
    dataset = _make_zarr(tmp_path / "paper.zarr", episodes=100)
    audit = audit_zarr_dataset(dataset, image_size=4)
    assert audit.episodes == 100
    assert audit.frames == 200
    assert audit.action_shape == (200, 9)


def test_dataset_audit_rejects_silent_demo_shortfall(tmp_path):
    dataset = _make_zarr(tmp_path / "short.zarr", episodes=99)
    with pytest.raises(ValueError, match="has 99 episodes; expected exactly 100"):
        audit_zarr_dataset(dataset, image_size=4)


def test_dataset_audit_rejects_non_finite_actions(tmp_path):
    dataset = _make_zarr(tmp_path / "nan.zarr", episodes=100)
    zarr = pytest.importorskip("zarr")
    root = zarr.open_group(str(dataset), mode="a")
    root["data/action"][17, 3] = np.nan
    with pytest.raises(ValueError, match="data/action contains non-finite"):
        audit_zarr_dataset(dataset, image_size=4)


def test_source_preflight_rejects_tracked_drift(tmp_path):
    checkout = tmp_path / "upstream"
    policy = checkout / "roboverse_learn/il/policies/a2a/a2a_policy.py"
    policy.parent.mkdir(parents=True)
    policy.write_text("# pinned\n", encoding="utf-8")
    _git("init", "-q", cwd=checkout)
    _git("config", "user.email", "test@example.com", cwd=checkout)
    _git("config", "user.name", "Test", cwd=checkout)
    _git("add", ".", cwd=checkout)
    _git("commit", "-qm", "fixture", cwd=checkout)
    commit = _git("rev-parse", "HEAD", cwd=checkout)
    assert validate_paper_checkout(checkout, expected_commit=commit)[1] == commit
    policy.write_text("# modified\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="tracked modifications"):
        validate_paper_checkout(checkout, expected_commit=commit)


def test_proxy_preflight_is_blocked_before_launch(tmp_path):
    with pytest.raises(RuntimeError, match="no exact public task mapping"):
        run_preflight(
            task_key="open_drawer",
            dataset=tmp_path / "not-needed.zarr",
            checkout=tmp_path / "not-needed",
        )


def test_simulator_proxy_is_not_mislabeled_as_exact(tmp_path):
    _, manifest = build_train_command(
        task_key="stack_cube",
        dataset=tmp_path / "proxy.zarr",
        output=tmp_path / "run",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        simulator="mujoco",
    )
    assert "eval_config.eval_args.sim=mujoco" in manifest["command"]
    assert manifest["simulator_matches_paper"] is False
    assert manifest["exact_paper_protocol"] is False


def test_skip_latent_visualization_is_explicit_in_manifest(tmp_path):
    _, manifest = build_train_command(
        task_key="stack_cube",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "run",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        skip_latent_visualization=True,
    )
    assert manifest["skip_latent_visualization"] is True
    assert manifest["latent_visualization_mode"] == "rng_preserving_no_plot"
    assert manifest["environment_overrides"]["ROBOBASE_OFFICIAL_SKIP_LATENT_VIZ"] == "1"


@pytest.mark.parametrize(
    ("method", "flow_override"),
    [
        ("a2a", "policy_config.flow_matcher.num_sampling_steps=6"),
        ("a2a_current", "policy_config.flow_matcher.num_sampling_steps=6"),
        ("fm_unet", "policy_config.num_inference_steps=10"),
    ],
)
def test_long200_train_command_keeps_e30_and_e200(method, flow_override, tmp_path):
    command, manifest = build_train_command(
        task_key="stack_cube",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "run",
        method=method,
        arm="long200",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        device="cuda:2",
    )
    assert "train_config.training_params.num_epochs=200" in command
    assert "train_config.training_params.checkpoint_every=30" in command
    assert "train_config.training_params.seed=42" in command
    assert "train_config.dataloader.batch_size=32" in command
    assert "shape_meta.action.shape=[9]" in command
    assert "horizon=16" in command
    assert "n_obs_steps=8" in command
    assert "n_action_steps=8" in command
    assert flow_override in command
    assert manifest["arm"]["saved_checkpoints"] == (
        30,
        60,
        90,
        120,
        150,
        180,
        200,
    )
    assert manifest["arm"]["comparison_checkpoints"] == (30, 200)
    assert manifest["lr_schedule_epoch_horizon"] == 200
    assert manifest["execution_steps"] == 8


def test_current_a2a_variant_is_explicit_and_not_claimed_as_paper_pin(tmp_path):
    command, manifest = build_train_command(
        task_key="stack_cube",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "run",
        method="a2a_current",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
    )
    assert (
        "policy_config.flow_matcher._target_="
        "roboverse_learn.il.utils.flow.flow_matchers.ConditionalFlowMatcher"
    ) in command
    assert manifest["upstream_policy_name"] == "a2a"
    assert manifest["source_variant"] == "current_main_conditional"
    assert manifest["exact_paper_protocol"] is False


def test_gaussian_latent_ablation_reuses_a2a_config_and_has_distinct_identity(
    tmp_path,
):
    train_command, train_manifest = build_train_command(
        task_key="close_box",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "train",
        method="a2a_gaussian_latent",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
    )
    target = (
        "policy_config._target_=benchmarks.official_bigym.a2a_gaussian_policy."
        "GaussianLatentA2AImagePolicy"
    )
    assert "policy_name=a2a_gaussian_latent" in train_command
    assert target in train_command
    assert (
        "policy_config.flow_matcher._target_="
        "roboverse_learn.il.utils.flow.flow_matchers.ConditionalFlowMatcher"
    ) in train_command
    assert train_manifest["source_variant"] == "gaussian_latent_source_ablation"
    assert train_manifest["upstream_policy_name"] == "a2a_gaussian_latent"
    assert train_manifest["environment_overrides"]["policy_name"] == "a2a"
    assert train_manifest["declared_paper_controls_match"] is False

    eval_command, eval_manifest = build_eval_command(
        task_key="close_box",
        dataset=tmp_path / "data.zarr",
        checkpoint=tmp_path / "30.ckpt",
        output=tmp_path / "eval",
        method="a2a_gaussian_latent",
        checkpoint_epoch=30,
        checkout=tmp_path / "source",
        python=tmp_path / "python",
    )
    assert "policy_name=a2a_gaussian_latent" in eval_command
    assert target in eval_command
    assert eval_manifest["solver"] == "euler"
    assert eval_manifest["model_calls_per_replan"] == 6
    assert eval_manifest["upstream_policy_name"] == "a2a_gaussian_latent"
    assert eval_manifest["environment_overrides"][
        "ROBOBASE_OFFICIAL_EVAL_FORCE_JOINT_POS"
    ] == "1"


def test_full_epoch_training_disables_the_250_batch_cutoff(tmp_path):
    command, manifest = build_train_command(
        task_key="close_box",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "train",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        full_epochs=True,
    )

    assert "train_config.training_params.max_train_steps=null" in command
    assert manifest["max_train_steps_per_epoch"] is None
    assert manifest["full_epochs"] is True
    assert manifest["declared_paper_controls_match"] is False


def test_fresh30_and_eval_commands_encode_controlled_protocol(tmp_path):
    train_command, train_manifest = build_train_command(
        task_key="close_box",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "train",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
    )
    assert "train_config.training_params.num_epochs=30" in train_command
    assert "train_config.training_params.checkpoint_every=30" in train_command
    assert train_manifest["lr_schedule_epoch_horizon"] == 30
    assert train_manifest["declared_paper_controls_match"] is True
    assert train_manifest["exact_paper_protocol"] is False
    assert train_manifest["exact_protocol_blockers"]

    eval_command, eval_manifest = build_eval_command(
        task_key="close_box",
        dataset=tmp_path / "data.zarr",
        checkpoint=tmp_path / "30.ckpt",
        output=tmp_path / "eval",
        method="a2a",
        checkpoint_epoch=30,
        checkout=tmp_path / "source",
        python=tmp_path / "python",
    )
    assert "+eval_config.eval_args.max_demo=50" in eval_command
    assert "+eval_config.eval_args.task_id_range_low=0" in eval_command
    assert "+eval_config.eval_args.task_id_range_high=50" in eval_command
    assert "eval_config.eval_args.max_step=300" in eval_command
    assert "eval_config.policy_runner.action.action_type=joint_pos" in eval_command
    assert eval_manifest["eval_trajectory_indices"] == [0, 49]
    assert eval_manifest["evaluation_set_id"] == "official_fixed:0-49"
    assert eval_manifest["dataset_provenance"] is None
    assert eval_manifest["prediction_steps"] == 8
    assert eval_manifest["execution_steps"] == 8
    assert eval_manifest["declared_paper_controls_match"] is True
    assert eval_manifest["exact_paper_protocol"] is False


def test_heldout_eval_command_binds_disjoint_training_provenance(tmp_path):
    dataset = tmp_path / "stack.zarr"
    provenance = _write_provenance(
        tmp_path / "provenance.json",
        dataset=dataset,
        source_indices=list(range(100)),
    )
    command, manifest = build_eval_command(
        task_key="stack_cube",
        dataset=dataset,
        checkpoint=tmp_path / "200.ckpt",
        output=tmp_path / "eval",
        method="a2a",
        checkpoint_epoch=200,
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        simulator="mujoco",
        eval_start_index=100,
        dataset_provenance=provenance,
    )

    assert "+eval_config.eval_args.task_id_range_low=100" in command
    assert "+eval_config.eval_args.task_id_range_high=150" in command
    assert manifest["eval_trajectory_indices"] == [100, 149]
    assert manifest["evaluation_split"] == "heldout_source_disjoint"
    assert manifest["evaluation_set_id"] == "heldout_source_disjoint:100-149"
    assert manifest["dataset_provenance"]["evaluation_overlap_count"] == 0
    assert manifest["dataset_provenance"]["selected_source_indices"] == list(
        range(100)
    )


def test_heldout_eval_rejects_missing_or_overlapping_provenance(tmp_path):
    dataset = tmp_path / "stack.zarr"
    with pytest.raises(ValueError, match="requires --dataset-provenance"):
        build_eval_command(
            task_key="stack_cube",
            dataset=dataset,
            checkpoint=tmp_path / "200.ckpt",
            output=tmp_path / "eval",
            method="a2a",
            checkpoint_epoch=200,
            eval_start_index=100,
        )

    provenance = _write_provenance(
        tmp_path / "provenance.json",
        dataset=dataset,
        source_indices=list(range(100)),
    )
    with pytest.raises(ValueError, match="overlaps training source indices"):
        build_eval_command(
            task_key="stack_cube",
            dataset=dataset,
            checkpoint=tmp_path / "200.ckpt",
            output=tmp_path / "eval",
            method="a2a",
            checkpoint_epoch=200,
            eval_start_index=50,
            dataset_provenance=provenance,
        )


def test_command_preserves_virtualenv_python_symlink(tmp_path):
    real_python = tmp_path / "system-python"
    real_python.write_text("", encoding="utf-8")
    venv_python = tmp_path / "venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.symlink_to(real_python)
    command, _ = build_train_command(
        task_key="pick_cube",
        dataset=tmp_path / "data.zarr",
        output=tmp_path / "run",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=venv_python,
    )
    assert command[0] == str(venv_python.absolute())


def test_proxy_demo_budget_is_not_mislabeled_as_paper_protocol(tmp_path):
    _, manifest = build_train_command(
        task_key="open_drawer",
        dataset=tmp_path / "proxy.zarr",
        output=tmp_path / "run",
        method="a2a",
        arm="fresh30",
        checkout=tmp_path / "source",
        python=tmp_path / "python",
        expected_episodes=50,
    )
    assert manifest["demonstrations_expected"] == 50
    assert manifest["exact_demo_budget"] is False
