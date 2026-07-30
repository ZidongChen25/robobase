from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks.official_roboverse import eval as eval_module
from benchmarks.official_roboverse.audit_proxy_data import HASH_SPEC, SCHEMA
from benchmarks.official_roboverse.eval import (
    build_eval_command,
    prepare_empty_output_directory,
    validate_checkpoint_ready,
    write_manifest_atomic,
)
from benchmarks.official_roboverse.results import (
    aggregate_results,
    load_evaluation,
    parse_final_stats,
    write_report,
)
from benchmarks.official_roboverse.train import build_train_command


def _write_stats(
    directory: Path,
    *,
    successes: int,
    completed: int = 50,
    demos_evaluated: int = 50,
    episode_records: int = 50,
    episode_start: int = 0,
) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    stats = directory / "final_stats.txt"
    rate = successes / completed
    stats.write_text(
        "\n".join(
            (
                "=== Success Statistics ===",
                f"  Average   Success Rate : {rate:.4f}",
                f"Total Completed: {completed}",
                f"Total Success : {successes}",
                "",
                "=== Overall Inference Time Statistics ===",
                "Total Inference Steps: 12345",
                f"Number of Demos Evaluated: {demos_evaluated}",
                "Average Inference Time: 1.25ms",
                "STD of Demo Avg Inference Time: 0.15ms",
                "Min Inference Time: 0.07ms",
                "Max Inference Time: 250.50ms",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    for index in range(episode_start, episode_start + episode_records):
        (directory / f"{index:04d}.txt").write_text(
            f"Demo Index: {index}\n"
            f"SuccessOnce: {index - episode_start < successes}\n"
            "SuccessEnd: False\n",
            encoding="utf-8",
        )
    return stats


def _write_evaluation(
    root: Path,
    *,
    task: str = "close_box",
    method: str = "a2a",
    arm: str,
    epoch: int,
    successes: int,
    expected_episodes: int = 100,
    eval_start_index: int = 0,
    dataset_provenance: Path | None = None,
) -> Path:
    dataset = root / "datasets" / f"{task}.zarr"
    train_output = root / "train" / task / method / arm
    train_output.mkdir(parents=True, exist_ok=True)
    _, train_manifest = build_train_command(
        task_key=task,
        dataset=dataset,
        output=train_output,
        method=method,
        arm=arm,
        checkout=root / "source",
        python=root / "python",
        expected_episodes=expected_episodes,
    )
    (train_output / "train_manifest.json").write_text(
        json.dumps(train_manifest), encoding="utf-8"
    )
    checkpoint = train_output / "checkpoints" / f"{epoch}.ckpt"
    checkpoint.parent.mkdir(exist_ok=True)
    checkpoint.write_bytes(f"{task}-{method}-{arm}-{epoch}".encode())

    eval_output = root / "eval_runs" / task / method / f"{arm}_e{epoch}"
    eval_output.mkdir(parents=True)
    _, eval_manifest = build_eval_command(
        task_key=task,
        dataset=dataset,
        checkpoint=checkpoint,
        output=eval_output,
        method=method,
        checkpoint_epoch=epoch,
        checkout=root / "source",
        python=root / "python",
        expected_episodes=expected_episodes,
        eval_start_index=eval_start_index,
        dataset_provenance=dataset_provenance,
    )
    eval_manifest["checkpoint_sha256"] = hashlib.sha256(
        checkpoint.read_bytes()
    ).hexdigest()
    manifest_path = eval_output / "eval_manifest.json"
    manifest_path.write_text(json.dumps(eval_manifest), encoding="utf-8")
    _write_stats(
        eval_output
        / "eval"
        / train_manifest["task"]["official_task_name"]
        / eval_manifest.get("upstream_policy_name", method)
        / "franka"
        / f"{epoch}.ckpt_fixture",
        successes=successes,
        episode_start=eval_start_index,
    )
    return manifest_path


def _write_triplet(
    root: Path,
    *,
    task: str = "close_box",
    method: str = "a2a",
    successes: tuple[int, int, int] = (44, 45, 49),
    expected_episodes: int = 100,
    eval_start_index: int = 0,
    dataset_provenance: Path | None = None,
) -> list[Path]:
    points = (("fresh30", 30), ("long200", 30), ("long200", 200))
    return [
        _write_evaluation(
            root,
            task=task,
            method=method,
            arm=arm,
            epoch=epoch,
            successes=count,
            expected_episodes=expected_episodes,
            eval_start_index=eval_start_index,
            dataset_provenance=dataset_provenance,
        )
        for (arm, epoch), count in zip(points, successes, strict=True)
    ]


def _write_provenance(path: Path, *, dataset: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            {
                "schema": SCHEMA,
                "status": "pass",
                "hash_spec": HASH_SPEC,
                "episodes": 100,
                "dataset": str(dataset.resolve()),
                "logical_content_sha256": "b" * 64,
                "selected_source_indices": list(range(100)),
                "errors": [],
                "raw_exact_match": True,
            }
        ),
        encoding="utf-8",
    )
    return path


def test_parse_final_stats_is_order_and_whitespace_independent(tmp_path):
    path = _write_stats(tmp_path / "stats", successes=37)
    stats = parse_final_stats(path)
    assert stats.total_success == 37
    assert stats.total_completed == 50
    assert stats.average_success_rate == pytest.approx(0.74)
    assert stats.demos_evaluated == 50
    assert stats.total_inference_steps == 12345
    assert stats.average_inference_time_ms == pytest.approx(1.25)
    assert stats.std_demo_average_inference_time_ms == pytest.approx(0.15)
    assert stats.min_inference_time_ms == pytest.approx(0.07)
    assert stats.max_inference_time_ms == pytest.approx(250.5)


def test_final_stats_requires_complete_millisecond_timing_evidence(tmp_path):
    path = _write_stats(tmp_path / "stats", successes=20)
    text = path.read_text(encoding="utf-8")
    path.write_text(
        text.replace("Min Inference Time: 0.07ms\n", ""), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="min_inference_time_ms"):
        parse_final_stats(path)

    path.write_text(
        text.replace("Average Inference Time: 1.25ms", "Average Inference Time: 1.25s"),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="finite value in milliseconds"):
        parse_final_stats(path)


def test_checkpoint_readiness_requires_a_complete_mapping(tmp_path, monkeypatch):
    checkpoint = tmp_path / "30.ckpt"
    checkpoint.write_bytes(b"complete")
    calls = []
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(
            load=lambda path, **kwargs: calls.append((path, kwargs))
            or {"cfg": {}, "state_dicts": {"model": {}}, "pickles": {}}
        ),
    )

    validate_checkpoint_ready(
        checkpoint, stable_polls=1, poll_interval_seconds=0
    )

    assert calls == [
        (
            checkpoint,
            {"map_location": "cpu", "weights_only": False},
        )
    ]


def test_checkpoint_readiness_rejects_non_mapping_payload(tmp_path, monkeypatch):
    checkpoint = tmp_path / "30.ckpt"
    checkpoint.write_bytes(b"incomplete")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: []),
    )

    with pytest.raises(ValueError, match="payload must be a mapping"):
        validate_checkpoint_ready(
            checkpoint, stable_polls=1, poll_interval_seconds=0
        )


@pytest.mark.parametrize(
    ("payload", "match"),
    [
        ({"cfg": {}, "state_dicts": {}}, "missing required fields.*pickles"),
        (
            {"cfg": [], "state_dicts": {}, "pickles": {}},
            "fields must be mappings.*cfg",
        ),
        (
            {"cfg": {}, "state_dicts": [], "pickles": {}},
            "fields must be mappings.*state_dicts",
        ),
        (
            {"cfg": {}, "state_dicts": {}, "pickles": []},
            "fields must be mappings.*pickles",
        ),
    ],
)
def test_checkpoint_readiness_validates_required_payload_fields(
    tmp_path, monkeypatch, payload, match
):
    checkpoint = tmp_path / "30.ckpt"
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setitem(
        sys.modules,
        "torch",
        SimpleNamespace(load=lambda *args, **kwargs: payload),
    )

    with pytest.raises(ValueError, match=match):
        validate_checkpoint_ready(
            checkpoint, stable_polls=1, poll_interval_seconds=0
        )


def test_eval_output_directory_must_be_empty(tmp_path):
    output = tmp_path / "new" / "eval"
    assert prepare_empty_output_directory(output) == output.resolve()
    assert output.is_dir()
    assert prepare_empty_output_directory(output) == output.resolve()

    (output / "stale.txt").write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="must be empty"):
        prepare_empty_output_directory(output)

    file_output = tmp_path / "not-a-directory"
    file_output.write_text("stale", encoding="utf-8")
    with pytest.raises(FileExistsError, match="not a directory"):
        prepare_empty_output_directory(file_output)


def test_atomic_manifest_write_refuses_overwrite(tmp_path):
    manifest_path = tmp_path / "eval_manifest.json"
    write_manifest_atomic(manifest_path, {"status": "complete"})
    assert json.loads(manifest_path.read_text(encoding="utf-8")) == {
        "status": "complete"
    }

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_manifest_atomic(manifest_path, {"status": "replacement"})


def _configure_eval_main(monkeypatch, tmp_path, output):
    checkpoint = tmp_path / "train" / "checkpoints" / "30.ckpt"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"checkpoint")
    monkeypatch.setattr(
        eval_module, "run_preflight", lambda **kwargs: {"status": "pass"}
    )
    monkeypatch.setattr(
        eval_module, "validate_checkpoint_ready", lambda *args, **kwargs: None
    )
    monkeypatch.setattr(eval_module, "file_sha256", lambda path: "checkpoint-sha256")
    monkeypatch.setattr(
        eval_module, "_subprocess_environment", lambda *args, **kwargs: {}
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "official-roboverse-eval",
            "--task",
            "close_box",
            "--dataset",
            str(tmp_path / "data.zarr"),
            "--checkpoint",
            str(checkpoint),
            "--checkpoint-epoch",
            "30",
            "--output",
            str(output),
            "--method",
            "a2a",
            "--official-checkout",
            str(tmp_path / "source"),
            "--python",
            str(tmp_path / "python"),
        ],
    )


def test_eval_main_publishes_manifest_after_complete_evidence(
    tmp_path, monkeypatch
):
    output = tmp_path / "eval-output"
    _configure_eval_main(monkeypatch, tmp_path, output)

    def run_evaluator(*args, **kwargs):
        del args, kwargs
        assert not (output / "eval_manifest.json").exists()
        _write_stats(
            output / "eval" / "close_box" / "a2a" / "franka" / "30.ckpt_test",
            successes=20,
        )

    monkeypatch.setattr(eval_module.subprocess, "run", run_evaluator)

    assert eval_module.main() == 0
    manifest_path = output / "eval_manifest.json"
    assert manifest_path.is_file()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["checkpoint_sha256"] == "checkpoint-sha256"


def test_eval_main_can_finalize_completed_upstream_output(tmp_path, monkeypatch):
    output = tmp_path / "eval-output"
    _configure_eval_main(monkeypatch, tmp_path, output)
    sys.argv.append("--finalize-existing")
    _write_stats(
        output / "eval" / "close_box" / "a2a" / "franka" / "30.ckpt_test",
        successes=20,
    )

    def unexpected_run(*args, **kwargs):
        del args, kwargs
        raise AssertionError("finalize-existing must not launch the evaluator")

    monkeypatch.setattr(eval_module.subprocess, "run", unexpected_run)
    assert eval_module.main() == 0
    assert (output / "eval_manifest.json").is_file()


def test_eval_main_does_not_publish_manifest_on_incomplete_output(
    tmp_path, monkeypatch
):
    output = tmp_path / "eval-output"
    _configure_eval_main(monkeypatch, tmp_path, output)
    monkeypatch.setattr(eval_module.subprocess, "run", lambda *args, **kwargs: None)

    with pytest.raises(ValueError, match="exactly one final_stats"):
        eval_module.main()
    assert not (output / "eval_manifest.json").exists()


def test_eval_main_does_not_publish_manifest_on_subprocess_failure(
    tmp_path, monkeypatch
):
    output = tmp_path / "eval-output"
    _configure_eval_main(monkeypatch, tmp_path, output)

    def fail_evaluator(*args, **kwargs):
        del args, kwargs
        raise eval_module.subprocess.CalledProcessError(1, ["upstream-evaluator"])

    monkeypatch.setattr(eval_module.subprocess, "run", fail_evaluator)

    with pytest.raises(eval_module.subprocess.CalledProcessError):
        eval_module.main()
    assert not (output / "eval_manifest.json").exists()


def test_eval_main_rejects_checkpoint_change_before_publishing_manifest(
    tmp_path, monkeypatch
):
    output = tmp_path / "eval-output"
    _configure_eval_main(monkeypatch, tmp_path, output)
    checkpoint_hashes = iter(("checkpoint-before", "checkpoint-after"))
    monkeypatch.setattr(
        eval_module, "file_sha256", lambda path: next(checkpoint_hashes)
    )

    def run_evaluator(*args, **kwargs):
        del args, kwargs
        _write_stats(
            output / "eval" / "close_box" / "a2a" / "franka" / "30.ckpt_test",
            successes=20,
        )

    monkeypatch.setattr(eval_module.subprocess, "run", run_evaluator)

    with pytest.raises(RuntimeError, match="Checkpoint changed"):
        eval_module.main()
    assert not (output / "eval_manifest.json").exists()


def test_aggregate_writes_paper_targets_counts_rates_and_deltas(tmp_path):
    _write_triplet(tmp_path)
    report = aggregate_results([tmp_path / "eval_runs"])
    assert report["evaluation_count"] == 3
    assert report["comparison_count"] == 1
    comparison = report["comparisons"][0]
    fresh_evaluation = report["evaluations"][0]
    assert fresh_evaluation["episode_successes"] == [True] * 44 + [False] * 6
    assert fresh_evaluation["total_inference_steps"] == 12345
    assert fresh_evaluation["average_inference_time_ms"] == pytest.approx(1.25)
    assert (
        fresh_evaluation["inference_timing_scope"]
        == "amortized_get_action_per_control_step"
    )
    assert fresh_evaluation["model_replan_interval_steps"] == 8
    assert comparison["paper_target"] == {
        "success_count": 46,
        "success_rate": 0.92,
        "success_pct": 92,
    }
    assert comparison["fresh30_e30"]["success_count"] == 44
    assert comparison["fresh30_e30"]["success_rate"] == pytest.approx(0.88)
    assert comparison["fresh30_e30"]["average_inference_time_ms"] == pytest.approx(
        1.25
    )
    assert comparison["inference_timing_scope"] == (
        "amortized_get_action_per_control_step"
    )
    assert comparison["model_replan_interval_steps"] == 8
    assert comparison["paper_target_comparable"] is False
    assert comparison["fresh30_e30"]["delta_vs_paper_success_count"] is None
    assert comparison["long200_e200"]["success_count"] == 49
    assert comparison["deltas"]["long200_e30_minus_fresh30_e30"] == {
        "success_count": 1,
        "success_rate": pytest.approx(0.02),
        "percentage_points": pytest.approx(2.0),
    }
    assert comparison["deltas"]["long200_e200_minus_long200_e30"] == {
        "success_count": 4,
        "success_rate": pytest.approx(0.08),
        "percentage_points": pytest.approx(8.0),
    }
    assert comparison["deltas"]["long200_e200_minus_fresh30_e30"] == {
        "success_count": 5,
        "success_rate": pytest.approx(0.10),
        "percentage_points": pytest.approx(10.0),
    }

    json_output = tmp_path / "reports" / "comparison.json"
    csv_output = tmp_path / "reports" / "comparison.csv"
    write_report(report, json_output=json_output, csv_output=csv_output)
    saved = json.loads(json_output.read_text(encoding="utf-8"))
    assert saved["schema"] == "official_a2a_roboverse_results_v1"
    with csv_output.open(encoding="utf-8", newline="") as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == 1
    assert rows[0]["paper_target_success_count"] == "46"
    assert rows[0]["demonstrations_expected"] == "100"
    assert rows[0]["inference_timing_scope"] == (
        "amortized_get_action_per_control_step"
    )
    assert rows[0]["model_replan_interval_steps"] == "8"
    assert rows[0]["fresh30_e30_success_count"] == "44"
    assert float(
        rows[0]["long200_e200_minus_long200_e30_percentage_points"]
    ) == pytest.approx(8.0)
    assert float(rows[0]["fresh30_e30_average_inference_time_ms"]) == pytest.approx(
        1.25
    )


def test_aggregate_computes_exact_paired_mcnemar_tests(tmp_path):
    _write_triplet(tmp_path, method="a2a", successes=(8, 12, 15))
    _write_triplet(tmp_path, method="fm_unet", successes=(0, 5, 9))

    paired = aggregate_results([tmp_path / "eval_runs"])["paired_tests"]

    assert len(paired["cross_method"]) == 3
    fresh = next(
        row
        for row in paired["cross_method"]
        if row["comparison_point"] == "fresh30_e30"
    )
    assert fresh["method_a"] == "a2a"
    assert fresh["method_b"] == "fm_unet"
    assert fresh["method_a_success_count"] == 8
    assert fresh["method_b_success_count"] == 0
    assert fresh["both_success_count"] == 0
    assert fresh["method_a_only_success_count"] == 8
    assert fresh["method_b_only_success_count"] == 0
    assert fresh["both_failure_count"] == 42
    assert fresh["discordant_count"] == 8
    assert fresh["episode_count"] == 50
    assert fresh["mcnemar_exact_two_sided_p_value"] == pytest.approx(0.0078125)

    assert len(paired["long200_e30_vs_e200"]) == 2
    a2a_epochs = next(
        row
        for row in paired["long200_e30_vs_e200"]
        if row["method"] == "a2a"
    )
    assert a2a_epochs["e30_success_count"] == 12
    assert a2a_epochs["e200_success_count"] == 15
    assert a2a_epochs["e30_only_success_count"] == 0
    assert a2a_epochs["e200_only_success_count"] == 3
    assert a2a_epochs["mcnemar_exact_two_sided_p_value"] == pytest.approx(0.25)


@pytest.mark.parametrize(
    ("completed", "demos_evaluated", "episode_records", "match"),
    [
        (49, 50, 50, "completed 49 episodes"),
        (50, 49, 50, "reports 49 demos evaluated"),
        (50, 50, 49, "does not contain the exact 50 episode records"),
    ],
)
def test_incomplete_50_episode_evidence_is_rejected(
    tmp_path, completed, demos_evaluated, episode_records, match
):
    manifest_path = _write_evaluation(
        tmp_path,
        arm="fresh30",
        epoch=30,
        successes=20,
    )
    stats_path = next((manifest_path.parent / "eval").rglob("final_stats.txt"))
    for path in stats_path.parent.glob("*.txt"):
        path.unlink()
    _write_stats(
        stats_path.parent,
        successes=20,
        completed=completed,
        demos_evaluated=demos_evaluated,
        episode_records=episode_records,
    )
    with pytest.raises(ValueError, match=match):
        load_evaluation(manifest_path)


def test_episode_success_count_must_match_final_stats(tmp_path):
    manifest_path = _write_evaluation(
        tmp_path, arm="fresh30", epoch=30, successes=20
    )
    stats_path = next((manifest_path.parent / "eval").rglob("final_stats.txt"))
    (stats_path.parent / "0000.txt").write_text(
        "Demo Index: 0\nSuccessOnce: False\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="episode records contain 19"):
        load_evaluation(manifest_path)


def test_training_and_evaluation_manifest_mismatch_is_rejected(tmp_path):
    manifest_path = _write_evaluation(
        tmp_path, arm="fresh30", epoch=30, successes=20
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["dataset"] = str(tmp_path / "different.zarr")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="mismatch for 'dataset'"):
        load_evaluation(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flow_steps", 99),
        ("observation_steps", 7),
        ("prediction_steps", 7),
        ("execution_steps", 1),
        ("exact_paper_protocol", True),
    ],
)
def test_eval_protocol_tampering_is_rejected(tmp_path, field, value):
    manifest_path = _write_evaluation(
        tmp_path, arm="fresh30", epoch=30, successes=20
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[field] = value
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_evaluation(manifest_path)


def test_checkpoint_hash_tampering_is_rejected(tmp_path):
    manifest_path = _write_evaluation(
        tmp_path, arm="fresh30", epoch=30, successes=20
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    Path(manifest["checkpoint"]).write_bytes(b"tampered")

    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_evaluation(manifest_path)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("seed", 0),
        ("batch_size", 16),
        ("horizon", 8),
        ("flow_steps", 1),
        ("lr_schedule_epoch_horizon", 999),
    ],
)
def test_training_protocol_tampering_is_rejected(tmp_path, field, value):
    manifest_path = _write_evaluation(
        tmp_path, arm="fresh30", epoch=30, successes=20
    )
    eval_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    train_manifest_path = (
        Path(eval_manifest["checkpoint"]).parent.parent / "train_manifest.json"
    )
    train_manifest = json.loads(train_manifest_path.read_text(encoding="utf-8"))
    train_manifest[field] = value
    train_manifest_path.write_text(json.dumps(train_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=field):
        load_evaluation(manifest_path)


def test_duplicate_and_partial_comparisons_are_rejected(tmp_path):
    manifests = _write_triplet(tmp_path / "complete")
    duplicate = _write_evaluation(
        tmp_path / "duplicate",
        arm="fresh30",
        epoch=30,
        successes=44,
    )
    with pytest.raises(ValueError, match="Duplicate evaluation"):
        aggregate_results([*manifests, duplicate])
    with pytest.raises(ValueError, match="Incomplete comparison"):
        aggregate_results([manifests[0]])


def test_official_and_heldout_evaluation_sets_aggregate_separately(tmp_path):
    official = _write_triplet(tmp_path / "official", task="pick_cube")
    heldout_root = tmp_path / "heldout"
    provenance = _write_provenance(
        heldout_root / "provenance.json",
        dataset=heldout_root / "datasets" / "pick_cube.zarr",
    )
    heldout = _write_triplet(
        heldout_root,
        task="pick_cube",
        eval_start_index=125,
        dataset_provenance=provenance,
    )

    report = aggregate_results([*official, *heldout])

    assert report["evaluation_count"] == 6
    assert report["comparison_count"] == 2
    assert report["evaluation_set_ids"] == [
        "heldout_source_disjoint:125-174",
        "official_fixed:0-49",
    ]
    heldout_comparison = next(
        row
        for row in report["comparisons"]
        if row["evaluation_split"] == "heldout_source_disjoint"
    )
    assert heldout_comparison["eval_trajectory_indices"] == [125, 174]
    assert heldout_comparison["dataset_provenance"]["evaluation_overlap_count"] == 0
    heldout_evaluation = next(
        row
        for row in report["evaluations"]
        if row["evaluation_split"] == "heldout_source_disjoint"
    )
    assert heldout_evaluation["episode_indices"] == list(range(125, 175))


def test_heldout_result_rejects_provenance_drift(tmp_path):
    dataset = tmp_path / "datasets" / "pick_cube.zarr"
    provenance = _write_provenance(tmp_path / "provenance.json", dataset=dataset)
    manifest_path = _write_evaluation(
        tmp_path,
        task="pick_cube",
        arm="fresh30",
        epoch=30,
        successes=20,
        eval_start_index=125,
        dataset_provenance=provenance,
    )
    payload = json.loads(provenance.read_text(encoding="utf-8"))
    payload["logical_content_sha256"] = "c" * 64
    provenance.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset_provenance does not match"):
        load_evaluation(manifest_path)


def test_long200_checkpoints_must_share_one_training_run(tmp_path):
    fresh = _write_evaluation(
        tmp_path / "run_a", arm="fresh30", epoch=30, successes=40
    )
    long30 = _write_evaluation(
        tmp_path / "run_a", arm="long200", epoch=30, successes=42
    )
    long200 = _write_evaluation(
        tmp_path / "run_b", arm="long200", epoch=200, successes=48
    )

    with pytest.raises(ValueError, match="different uninterrupted training runs"):
        aggregate_results([fresh, long30, long200])


def test_proxy_rows_are_explicit_and_paper_target_is_not_comparable(tmp_path):
    _write_triplet(
        tmp_path,
        task="open_drawer",
        method="fm_unet",
        successes=(10, 12, 18),
        expected_episodes=50,
    )
    comparison = aggregate_results([tmp_path / "eval_runs"])["comparisons"][0]
    assert comparison["mapping_status"] == "proxy_blocked"
    assert comparison["mapping_is_proxy"] is True
    assert comparison["protocol_is_proxy"] is True
    assert comparison["simulator"] == "mujoco"
    assert comparison["exact_paper_protocol"] is False
    assert comparison["paper_target_comparable"] is False
    assert comparison["paper_target"]["success_count"] == 17
    assert comparison["fresh30_e30"]["delta_vs_paper_success_count"] is None
    assert comparison["fresh30_e30"]["delta_vs_paper_percentage_points"] is None


def test_cross_method_comparison_requires_same_dataset(tmp_path):
    a2a = _write_triplet(tmp_path / "a2a", method="a2a")
    fm = _write_triplet(tmp_path / "fm", method="fm_unet")

    with pytest.raises(ValueError, match="Cross-method.*dataset"):
        aggregate_results([*a2a, *fm])


def test_cross_method_provenance_comparison_ignores_dict_key_order(tmp_path):
    provenance = _write_provenance(
        tmp_path / "provenance.json",
        dataset=tmp_path / "datasets" / "pick_cube.zarr",
    )
    a2a = _write_triplet(
        tmp_path,
        task="pick_cube",
        method="a2a",
        eval_start_index=125,
        dataset_provenance=provenance,
    )
    fm = _write_triplet(
        tmp_path,
        task="pick_cube",
        method="fm_unet",
        eval_start_index=125,
        dataset_provenance=provenance,
    )
    for manifest_path in fm:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        provenance_fields = manifest["dataset_provenance"]
        manifest["dataset_provenance"] = {
            key: provenance_fields[key] for key in reversed(provenance_fields)
        }
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = aggregate_results([*a2a, *fm])

    assert report["comparison_count"] == 2
    assert report["evaluation_set_ids"] == [
        "heldout_source_disjoint:125-174"
    ]


def test_current_a2a_variant_is_aggregated_separately(tmp_path):
    _write_triplet(tmp_path, method="a2a_current")

    report = aggregate_results([tmp_path / "eval_runs"])
    comparison = report["comparisons"][0]

    assert comparison["method"] == "a2a_current"
    assert comparison["source_variant"] == "current_main_conditional"
    assert comparison["paper_target_comparable"] is False
    assert comparison["fresh30_e30"]["delta_vs_paper_success_count"] is None
    assert report["evaluation_set_ids"] == ["official_fixed:0-49"]
    assert report["full_five_task_two_method_matrix"] is False
    with pytest.raises(ValueError, match="Incomplete five-task/two-method matrix"):
        aggregate_results(
            [tmp_path / "eval_runs"], require_full_matrix=True
        )


@pytest.mark.parametrize(
    ("manifest_kind", "mutation", "match"),
    [
        ("eval", {"source_variant": "initial_release_ot"}, "source_variant"),
        ("train", {"upstream_policy_name": "fm_unet"}, "upstream_policy_name"),
        ("eval", {"command": []}, "matcher override"),
        (
            "train",
            {
                "command": [
                    "policy_config.flow_matcher._target_=example.WrongMatcher"
                ]
            },
            "matcher override",
        ),
    ],
)
def test_current_a2a_method_identity_tampering_is_rejected(
    tmp_path, manifest_kind, mutation, match
):
    manifest_path = _write_evaluation(
        tmp_path,
        method="a2a_current",
        arm="fresh30",
        epoch=30,
        successes=20,
    )
    eval_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    target_path = manifest_path
    if manifest_kind == "train":
        target_path = (
            Path(eval_manifest["checkpoint"]).parent.parent / "train_manifest.json"
        )
    target_manifest = json.loads(target_path.read_text(encoding="utf-8"))
    target_manifest.update(mutation)
    target_path.write_text(json.dumps(target_manifest), encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        load_evaluation(manifest_path)


def test_legacy_primary_manifests_infer_method_identity(tmp_path):
    manifests = _write_triplet(tmp_path, method="a2a")
    train_manifests = set()
    for manifest_path in manifests:
        eval_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        eval_manifest.pop("source_variant")
        eval_manifest.pop("upstream_policy_name")
        manifest_path.write_text(json.dumps(eval_manifest), encoding="utf-8")
        train_manifests.add(
            Path(eval_manifest["checkpoint"]).parent.parent / "train_manifest.json"
        )
    for manifest_path in train_manifests:
        train_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        train_manifest.pop("source_variant")
        train_manifest.pop("upstream_policy_name")
        manifest_path.write_text(json.dumps(train_manifest), encoding="utf-8")

    comparison = aggregate_results(manifests)["comparisons"][0]
    assert comparison["source_variant"] == "initial_release_ot"


def test_triplet_requires_identical_demonstration_budget(tmp_path):
    manifests = [
        _write_evaluation(
            tmp_path,
            arm="fresh30",
            epoch=30,
            successes=20,
            expected_episodes=50,
        ),
        _write_evaluation(
            tmp_path,
            arm="long200",
            epoch=30,
            successes=20,
            expected_episodes=99,
        ),
        _write_evaluation(
            tmp_path,
            arm="long200",
            epoch=200,
            successes=20,
            expected_episodes=99,
        ),
    ]

    with pytest.raises(ValueError, match="inconsistent 'demonstrations_expected'"):
        aggregate_results(manifests)


def test_full_matrix_check_rejects_a_single_task_method(tmp_path):
    _write_triplet(tmp_path)
    with pytest.raises(ValueError, match="Incomplete five-task/two-method matrix"):
        aggregate_results(
            [tmp_path / "eval_runs"], require_full_matrix=True
        )
