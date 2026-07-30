#!/usr/bin/env python3
"""Validate and aggregate official RoboVerse A2A/FM evaluation results.

The upstream evaluator writes one ``final_stats.txt`` plus one text record per
episode.  This module treats both as required evidence and joins them with the
evaluation and training manifests before producing comparison tables.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence

from benchmarks.official_bigym.a2a_upstream import file_sha256
from benchmarks.official_roboverse.heldout import (
    validate_disjoint_dataset_provenance,
)
from benchmarks.official_roboverse.protocol import (
    GLOBAL_EXACT_PROTOCOL_BLOCKERS,
    PAPER_ACTION_DIM,
    PAPER_ACTION_STEPS,
    PAPER_BATCH_SIZE,
    PAPER_EVAL_EPISODES,
    PAPER_DEMONSTRATIONS,
    PAPER_HORIZON,
    PAPER_IMAGE_SIZE,
    PAPER_MAX_EVAL_STEPS,
    PAPER_MAX_TRAIN_STEPS_PER_EPOCH,
    PAPER_OBSERVATION_STEPS,
    PAPER_SEED,
    PAPER_SOURCE_COMMIT,
    PAPER_TASKS,
    TaskProtocol,
    get_task,
)
from benchmarks.official_roboverse.train import (
    CURRENT_A2A_MATCHER,
    METHOD_FLOW_STEPS,
    TRAIN_ARMS,
    UPSTREAM_POLICY_NAMES,
)


METHODS = ("a2a", "a2a_current", "fm_unet")
PRIMARY_METHODS = ("a2a", "fm_unet")
COMPARISON_POINTS = (
    ("fresh30", 30),
    ("long200", 30),
    ("long200", 200),
)
INFERENCE_TIMING_SCOPE = "amortized_get_action_per_control_step"


def _source_variant(method: str) -> str:
    return {
        "a2a": "initial_release_ot",
        "a2a_current": "current_main_conditional",
        "fm_unet": "pinned_fm_unet",
    }[method]


def _validate_method_identity(
    manifest: Mapping[str, Any], manifest_path: Path, method: str
) -> None:
    """Ensure a manifest's method label matches the policy it actually launches."""

    expected_source_variant = _source_variant(method)
    source_variant = manifest.get("source_variant")
    if source_variant is None:
        # The first primary-method runs predate this manifest field. Current A2A
        # was introduced with it and has no legitimate legacy representation.
        if method == "a2a_current":
            raise ValueError(
                f"{manifest_path} is missing required field 'source_variant'."
            )
    elif source_variant != expected_source_variant:
        raise ValueError(
            f"{manifest_path} source_variant={source_variant!r} does not match "
            f"method {method!r} ({expected_source_variant!r})."
        )

    expected_policy_name = UPSTREAM_POLICY_NAMES[method]
    upstream_policy_name = manifest.get("upstream_policy_name")
    if upstream_policy_name is None:
        if method == "a2a_current":
            raise ValueError(
                f"{manifest_path} is missing required field 'upstream_policy_name'."
            )
    elif upstream_policy_name != expected_policy_name:
        raise ValueError(
            f"{manifest_path} upstream_policy_name={upstream_policy_name!r} does "
            f"not match method {method!r} ({expected_policy_name!r})."
        )

    environment = _require_value(
        manifest, "environment_overrides", dict, source=manifest_path
    )
    if environment.get("policy_name") != expected_policy_name:
        raise ValueError(
            f"{manifest_path} environment_overrides.policy_name="
            f"{environment.get('policy_name')!r} does not match method {method!r} "
            f"({expected_policy_name!r})."
        )

    command = _require_value(manifest, "command", list, source=manifest_path)
    if not all(isinstance(argument, str) for argument in command):
        raise ValueError(f"{manifest_path} field 'command' must contain only strings.")
    matcher_prefix = "policy_config.flow_matcher._target_="
    matcher_overrides = [
        argument for argument in command if argument.startswith(matcher_prefix)
    ]
    expected_matcher_overrides = (
        [f"{matcher_prefix}{CURRENT_A2A_MATCHER}"]
        if method == "a2a_current"
        else []
    )
    if matcher_overrides != expected_matcher_overrides:
        raise ValueError(
            f"{manifest_path} matcher override {matcher_overrides!r} does not match "
            f"method {method!r} ({expected_matcher_overrides!r})."
        )


@dataclass(frozen=True)
class FinalStats:
    total_success: int
    total_completed: int
    average_success_rate: float
    demos_evaluated: int
    total_inference_steps: int
    average_inference_time_ms: float
    std_demo_average_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float


def validate_evaluation_outputs(
    output: str | Path,
    *,
    task: TaskProtocol,
    upstream_policy_name: str,
    checkpoint_epoch: int,
    episode_index_start: int = 0,
) -> tuple[Path, FinalStats]:
    """Validate the native evaluator's complete 50-episode output tree."""

    output = Path(output).expanduser().resolve()
    stats_candidates = sorted((output / "eval").rglob("final_stats.txt"))
    if len(stats_candidates) != 1:
        raise ValueError(
            f"{output} must contain exactly one final_stats.txt below eval/, found "
            f"{len(stats_candidates)}."
        )
    stats_path = stats_candidates[0].resolve()
    stats_relative = stats_path.relative_to(output / "eval")
    expected_prefix = (task.official_task_name, upstream_policy_name, "franka")
    path_matches = stats_relative.parts[:3] == expected_prefix
    checkpoint_matches = stats_relative.parent.name.startswith(
        f"{checkpoint_epoch}.ckpt_"
    )
    if not path_matches or not checkpoint_matches:
        raise ValueError(
            f"{stats_path} is not under the expected evaluation identity "
            f"{task.official_task_name}/{upstream_policy_name}/franka/"
            f"{checkpoint_epoch}.ckpt_*."
        )
    stats = parse_final_stats(stats_path)
    if stats.total_completed != PAPER_EVAL_EPISODES:
        raise ValueError(
            f"{stats_path} completed {stats.total_completed} episodes; expected "
            f"exactly {PAPER_EVAL_EPISODES}."
        )
    if stats.demos_evaluated != PAPER_EVAL_EPISODES:
        raise ValueError(
            f"{stats_path} reports {stats.demos_evaluated} demos evaluated; expected "
            f"exactly {PAPER_EVAL_EPISODES}."
        )
    episode_indices, episode_successes = _parse_episode_records(stats_path.parent)
    if episode_index_start < 0:
        raise ValueError("episode_index_start must be non-negative.")
    expected_indices = list(
        range(episode_index_start, episode_index_start + PAPER_EVAL_EPISODES)
    )
    if episode_indices != expected_indices:
        missing = sorted(set(expected_indices) - set(episode_indices))
        unexpected = sorted(set(episode_indices) - set(expected_indices))
        raise ValueError(
            f"{stats_path.parent} does not contain the exact 50 episode records; "
            f"missing={missing}, unexpected={unexpected}."
        )
    episode_success_count = sum(episode_successes)
    if episode_success_count != stats.total_success:
        raise ValueError(
            f"{stats_path} reports {stats.total_success} successes but episode "
            f"records contain {episode_success_count}."
        )
    return stats_path, stats


def _read_json_object(path: Path, *, description: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing {description}: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed {description} {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ValueError(f"{description.capitalize()} {path} must contain an object.")
    return value


def _require_value(
    mapping: Mapping[str, Any], key: str, expected_type: type, *, source: Path
) -> Any:
    if key not in mapping:
        raise ValueError(f"{source} is missing required field {key!r}.")
    value = mapping[key]
    if expected_type is bool:
        valid = type(value) is bool
    else:
        valid = isinstance(value, expected_type) and not (
            expected_type is int and isinstance(value, bool)
        )
    if not valid:
        raise ValueError(
            f"{source} field {key!r} must be {expected_type.__name__}, "
            f"got {type(value).__name__}."
        )
    return value


def _parse_int(value: str, *, label: str, path: Path) -> int:
    if not re.fullmatch(r"[+-]?\d+", value.strip()):
        raise ValueError(f"{path} field {label!r} is not an integer: {value!r}.")
    return int(value)


def _parse_float(value: str, *, label: str, path: Path) -> float:
    try:
        result = float(value.strip())
    except ValueError:
        raise ValueError(
            f"{path} field {label!r} is not a float: {value!r}."
        ) from None
    if not math.isfinite(result):
        raise ValueError(f"{path} field {label!r} must be finite, got {result}.")
    return result


def _parse_milliseconds(value: str, *, label: str, path: Path) -> float:
    match = re.fullmatch(
        r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)\s*ms",
        value.strip(),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(
            f"{path} field {label!r} must be a finite value in milliseconds, "
            f"got {value!r}."
        )
    return _parse_float(match.group(1), label=label, path=path)


def parse_final_stats(path: str | Path) -> FinalStats:
    """Parse the pinned evaluator's labeled statistics without relying on order."""

    path = Path(path).expanduser().resolve()
    labels = {
        "total success": "total_success",
        "total completed": "total_completed",
        "average success rate": "average_success_rate",
        "total inference steps": "total_inference_steps",
        "number of demos evaluated": "demos_evaluated",
        "average inference time": "average_inference_time_ms",
        "std of demo avg inference time": "std_demo_average_inference_time_ms",
        "min inference time": "min_inference_time_ms",
        "max inference time": "max_inference_time_ms",
    }
    values: dict[str, str] = {}
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError:
        raise FileNotFoundError(f"Missing upstream final statistics: {path}") from None
    for line in lines:
        if ":" not in line:
            continue
        raw_label, raw_value = line.split(":", 1)
        canonical = labels.get(" ".join(raw_label.strip().lower().split()))
        if canonical is None:
            continue
        if canonical in values:
            raise ValueError(
                f"{path} contains duplicate field {raw_label.strip()!r}; "
                "the result is ambiguous."
            )
        values[canonical] = raw_value.strip()
    missing = sorted(set(labels.values()) - values.keys())
    if missing:
        raise ValueError(f"{path} is missing required statistics: {', '.join(missing)}.")

    stats = FinalStats(
        total_success=_parse_int(
            values["total_success"], label="Total Success", path=path
        ),
        total_completed=_parse_int(
            values["total_completed"], label="Total Completed", path=path
        ),
        average_success_rate=_parse_float(
            values["average_success_rate"],
            label="Average Success Rate",
            path=path,
        ),
        demos_evaluated=_parse_int(
            values["demos_evaluated"],
            label="Number of Demos Evaluated",
            path=path,
        ),
        total_inference_steps=_parse_int(
            values["total_inference_steps"],
            label="Total Inference Steps",
            path=path,
        ),
        average_inference_time_ms=_parse_milliseconds(
            values["average_inference_time_ms"],
            label="Average Inference Time",
            path=path,
        ),
        std_demo_average_inference_time_ms=_parse_milliseconds(
            values["std_demo_average_inference_time_ms"],
            label="STD of Demo Avg Inference Time",
            path=path,
        ),
        min_inference_time_ms=_parse_milliseconds(
            values["min_inference_time_ms"],
            label="Min Inference Time",
            path=path,
        ),
        max_inference_time_ms=_parse_milliseconds(
            values["max_inference_time_ms"],
            label="Max Inference Time",
            path=path,
        ),
    )
    if stats.total_completed <= 0:
        raise ValueError(f"{path} Total Completed must be positive.")
    if not 0 <= stats.total_success <= stats.total_completed:
        raise ValueError(
            f"{path} Total Success {stats.total_success} is outside "
            f"[0, {stats.total_completed}]."
        )
    expected_rate = stats.total_success / stats.total_completed
    if not math.isclose(
        stats.average_success_rate, expected_rate, rel_tol=0.0, abs_tol=5.1e-5
    ):
        raise ValueError(
            f"{path} success rate {stats.average_success_rate} disagrees with "
            f"{stats.total_success}/{stats.total_completed}={expected_rate:.4f}."
        )
    if stats.total_inference_steps <= 0:
        raise ValueError(f"{path} Total Inference Steps must be positive.")
    inference_times = {
        "Average Inference Time": stats.average_inference_time_ms,
        "STD of Demo Avg Inference Time": stats.std_demo_average_inference_time_ms,
        "Min Inference Time": stats.min_inference_time_ms,
        "Max Inference Time": stats.max_inference_time_ms,
    }
    for label, value in inference_times.items():
        if value < 0:
            raise ValueError(f"{path} {label} must be non-negative, got {value}.")
    if stats.min_inference_time_ms > stats.max_inference_time_ms:
        raise ValueError(
            f"{path} Min Inference Time {stats.min_inference_time_ms}ms exceeds "
            f"Max Inference Time {stats.max_inference_time_ms}ms."
        )
    if not (
        stats.min_inference_time_ms
        <= stats.average_inference_time_ms
        <= stats.max_inference_time_ms
    ):
        raise ValueError(
            f"{path} Average Inference Time {stats.average_inference_time_ms}ms "
            "must fall between Min and Max Inference Time."
        )
    return stats


def _parse_episode_records(stats_dir: Path) -> tuple[list[int], list[bool]]:
    records: dict[int, bool] = {}
    for path in sorted(stats_dir.glob("*.txt")):
        if re.fullmatch(r"\d{4}", path.stem) is None:
            continue
        fields: dict[str, str] = {}
        for line in path.read_text(encoding="utf-8").splitlines():
            if ":" not in line:
                continue
            label, value = line.split(":", 1)
            normalized = " ".join(label.strip().lower().split())
            if normalized in {"demo index", "successonce"}:
                if normalized in fields:
                    raise ValueError(f"{path} contains duplicate field {label!r}.")
                fields[normalized] = value.strip()
        if set(fields) != {"demo index", "successonce"}:
            raise ValueError(
                f"{path} must contain exactly Demo Index and SuccessOnce evidence."
            )
        index = _parse_int(fields["demo index"], label="Demo Index", path=path)
        if path.stem != f"{index:04d}":
            raise ValueError(
                f"{path} filename does not match its Demo Index {index}."
            )
        if index in records:
            raise ValueError(f"Duplicate episode record for demo index {index}.")
        success_text = fields["successonce"].lower()
        if success_text not in {"true", "false"}:
            raise ValueError(
                f"{path} SuccessOnce must be True or False, got "
                f"{fields['successonce']!r}."
            )
        records[index] = success_text == "true"
    indices = sorted(records)
    return indices, [records[index] for index in indices]


def discover_eval_manifests(inputs: Iterable[str | Path]) -> list[Path]:
    """Find evaluation manifests below input paths and return unique paths."""

    discovered: set[Path] = set()
    for raw_path in inputs:
        path = Path(raw_path).expanduser().resolve()
        if path.is_file():
            if path.name != "eval_manifest.json":
                raise ValueError(
                    f"Input file must be named eval_manifest.json, got {path}."
                )
            discovered.add(path)
        elif path.is_dir():
            discovered.update(item.resolve() for item in path.rglob("eval_manifest.json"))
        else:
            raise FileNotFoundError(f"Result input does not exist: {path}")
    if not discovered:
        raise ValueError("No eval_manifest.json files were found.")
    return sorted(discovered)


def _paper_target(task: TaskProtocol, method: str) -> int:
    if method in ("a2a", "a2a_current"):
        return task.paper_a2a_success_pct
    if method == "fm_unet":
        return task.paper_fm_unet_success_pct
    raise ValueError(f"Unknown method {method!r}.")


def _manifest_evaluation_identity(
    manifest: Mapping[str, Any], manifest_path: Path, task: TaskProtocol
) -> dict[str, Any]:
    indices = _require_value(
        manifest, "eval_trajectory_indices", list, source=manifest_path
    )
    if len(indices) != 2 or any(
        not isinstance(index, int) or isinstance(index, bool) for index in indices
    ):
        raise ValueError(
            f"{manifest_path} eval_trajectory_indices must contain two integers."
        )
    start = manifest.get("eval_start_index", indices[0])
    if not isinstance(start, int) or isinstance(start, bool) or start < 0:
        raise ValueError(
            f"{manifest_path} eval_start_index must be a non-negative integer."
        )
    stop = start + PAPER_EVAL_EPISODES
    expected_indices = [start, stop - 1]
    if indices != expected_indices:
        raise ValueError(
            f"{manifest_path} eval_trajectory_indices={indices!r}; expected "
            f"{expected_indices!r}."
        )
    if stop > task.public_unique_trajectories:
        raise ValueError(
            f"{manifest_path} evaluation range [{start}, {stop}) exceeds the "
            f"{task.public_unique_trajectories} public task states."
        )
    command = _require_value(manifest, "command", list, source=manifest_path)
    expected_overrides = (
        f"+eval_config.eval_args.task_id_range_low={start}",
        f"+eval_config.eval_args.task_id_range_high={stop}",
        f"+eval_config.eval_args.max_demo={PAPER_EVAL_EPISODES}",
    )
    for override in expected_overrides:
        if command.count(override) != 1:
            raise ValueError(
                f"{manifest_path} command must contain exactly one {override!r}."
            )
    split = manifest.get("evaluation_split", "official_fixed" if start == 0 else None)
    if split not in {"official_fixed", "heldout_source_disjoint"}:
        raise ValueError(f"{manifest_path} has invalid evaluation_split {split!r}.")
    expected_set_id = f"{split}:{start}-{stop - 1}"
    set_id = manifest.get("evaluation_set_id", expected_set_id)
    if set_id != expected_set_id:
        raise ValueError(
            f"{manifest_path} evaluation_set_id={set_id!r}; expected "
            f"{expected_set_id!r}."
        )
    provenance = manifest.get("dataset_provenance")
    if split == "official_fixed":
        if start != 0 or provenance is not None:
            raise ValueError(
                f"{manifest_path} official_fixed evaluation must use indices 0..49 "
                "without held-out provenance."
            )
    else:
        if not isinstance(provenance, dict):
            raise ValueError(
                f"{manifest_path} heldout_source_disjoint evaluation requires "
                "dataset_provenance."
            )
        provenance_path = provenance.get("path")
        if not isinstance(provenance_path, str):
            raise ValueError(
                f"{manifest_path} dataset_provenance.path must be a string."
            )
        expected_episodes = _require_value(
            manifest, "demonstrations_expected", int, source=manifest_path
        )
        verified = validate_disjoint_dataset_provenance(
            provenance_path,
            dataset=_require_value(manifest, "dataset", str, source=manifest_path),
            expected_episodes=expected_episodes,
            eval_start_index=start,
        )
        if provenance != verified:
            raise ValueError(
                f"{manifest_path} dataset_provenance does not match the current "
                "audited provenance evidence."
            )
    return {
        "eval_start_index": start,
        "eval_trajectory_indices": expected_indices,
        "evaluation_split": split,
        "evaluation_set_id": set_id,
        "dataset_provenance": provenance,
    }


def _validate_manifest_protocol(
    manifest: Mapping[str, Any],
    manifest_path: Path,
    task: TaskProtocol,
    method: str,
) -> dict[str, Any]:
    if manifest.get("schema") != "official_a2a_roboverse_eval_v1":
        raise ValueError(f"{manifest_path} has an unsupported evaluation schema.")
    if manifest.get("source_commit") != PAPER_SOURCE_COMMIT:
        raise ValueError(f"{manifest_path} does not use the pinned paper source commit.")
    _validate_method_identity(manifest, manifest_path, method)
    task_metadata = _require_value(manifest, "task", dict, source=manifest_path)
    expected_task_fields = {
        "key": task.key,
        "paper_name": task.paper_name,
        "benchmark": task.benchmark,
        "official_task_name": task.official_task_name,
        "mapping_status": task.mapping_status,
    }
    for key, expected in expected_task_fields.items():
        if task_metadata.get(key) != expected:
            raise ValueError(
                f"{manifest_path} task.{key}={task_metadata.get(key)!r} does not "
                f"match protocol value {expected!r}."
            )
    if manifest.get("eval_episodes") != PAPER_EVAL_EPISODES:
        raise ValueError(
            f"{manifest_path} must declare exactly {PAPER_EVAL_EPISODES} eval episodes."
        )
    evaluation_identity = _manifest_evaluation_identity(
        manifest, manifest_path, task
    )
    if manifest.get("max_eval_steps") != PAPER_MAX_EVAL_STEPS:
        raise ValueError(
            f"{manifest_path} max_eval_steps does not match the paper protocol."
        )
    expected_eval_fields = {
        "flow_steps": METHOD_FLOW_STEPS[method],
        "observation_steps": PAPER_OBSERVATION_STEPS,
        "prediction_steps": PAPER_ACTION_STEPS,
        "execution_steps": PAPER_ACTION_STEPS,
    }
    for key, expected in expected_eval_fields.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"{manifest_path} field {key!r}={manifest.get(key)!r} does not "
                f"match protocol value {expected!r}."
            )
    demonstrations_expected = _require_value(
        manifest, "demonstrations_expected", int, source=manifest_path
    )
    if demonstrations_expected < 1:
        raise ValueError(
            f"{manifest_path} demonstrations_expected must be positive."
        )
    expected_demo_budget = demonstrations_expected == PAPER_DEMONSTRATIONS
    expected_simulator_match = manifest.get("simulator") == task.simulator
    expected_declared_controls = (
        task.is_exact
        and expected_demo_budget
        and expected_simulator_match
        and method != "a2a_current"
        and evaluation_identity["evaluation_split"] == "official_fixed"
    )
    expected_flags = {
        "exact_demo_budget": expected_demo_budget,
        "simulator_matches_paper": expected_simulator_match,
        "exact_paper_protocol": False,
    }
    for key, expected in expected_flags.items():
        actual = _require_value(manifest, key, bool, source=manifest_path)
        if actual is not expected:
            raise ValueError(
                f"{manifest_path} field {key!r}={actual} disagrees with "
                f"derived value {expected}."
            )
    if "declared_paper_controls_match" in manifest:
        declared_controls = _require_value(
            manifest,
            "declared_paper_controls_match",
            bool,
            source=manifest_path,
        )
        if declared_controls is not expected_declared_controls:
            raise ValueError(
                f"{manifest_path} field 'declared_paper_controls_match'="
                f"{declared_controls} disagrees with derived value "
                f"{expected_declared_controls}."
            )
    if "exact_protocol_blockers" in manifest:
        blockers = _require_value(
            manifest, "exact_protocol_blockers", list, source=manifest_path
        )
        if blockers != list(GLOBAL_EXACT_PROTOCOL_BLOCKERS):
            raise ValueError(
                f"{manifest_path} field 'exact_protocol_blockers' disagrees "
                "with the frozen global blockers."
            )
    return evaluation_identity


def _load_train_manifest(
    eval_manifest: Mapping[str, Any], eval_manifest_path: Path
) -> tuple[dict[str, Any], Path, str]:
    checkpoint_value = _require_value(
        eval_manifest, "checkpoint", str, source=eval_manifest_path
    )
    checkpoint = Path(checkpoint_value).expanduser().resolve()
    checkpoint_epoch = _require_value(
        eval_manifest, "checkpoint_epoch", int, source=eval_manifest_path
    )
    if checkpoint.name != f"{checkpoint_epoch}.ckpt":
        raise ValueError(
            f"{eval_manifest_path} checkpoint filename {checkpoint.name!r} "
            f"does not match epoch {checkpoint_epoch}."
        )
    train_manifest_path = checkpoint.parent.parent / "train_manifest.json"
    train_manifest = _read_json_object(
        train_manifest_path, description="training manifest"
    )
    if train_manifest.get("schema") != "official_a2a_roboverse_train_v1":
        raise ValueError(f"{train_manifest_path} has an unsupported training schema.")
    arm_metadata = _require_value(
        train_manifest, "arm", dict, source=train_manifest_path
    )
    arm = _require_value(arm_metadata, "name", str, source=train_manifest_path)
    if arm not in TRAIN_ARMS:
        raise ValueError(f"{train_manifest_path} has unknown training arm {arm!r}.")
    expected_arm = TRAIN_ARMS[arm]
    expected_arm_fields = {
        "epochs": expected_arm.epochs,
        "checkpoint_every": expected_arm.checkpoint_every,
        "saved_checkpoints": list(expected_arm.saved_checkpoints),
        "comparison_checkpoints": list(expected_arm.comparison_checkpoints),
    }
    for key, expected in expected_arm_fields.items():
        if arm_metadata.get(key) != expected:
            raise ValueError(
                f"{train_manifest_path} arm.{key}={arm_metadata.get(key)!r} does "
                f"not match protocol value {expected!r}."
            )
    if (arm, checkpoint_epoch) not in COMPARISON_POINTS:
        raise ValueError(
            f"{eval_manifest_path} arm/epoch ({arm}, {checkpoint_epoch}) is not one "
            "of fresh30/E30, long200/E30, or long200/E200."
        )
    comparison_checkpoints = arm_metadata.get("comparison_checkpoints")
    if not isinstance(
        comparison_checkpoints, (list, tuple)
    ) or checkpoint_epoch not in comparison_checkpoints:
        raise ValueError(
            f"{train_manifest_path} does not declare epoch {checkpoint_epoch} as a "
            "comparison checkpoint."
        )
    expected_checkpoint = (
        Path(_require_value(train_manifest, "output", str, source=train_manifest_path))
        .expanduser()
        .resolve()
        / "checkpoints"
        / f"{checkpoint_epoch}.ckpt"
    )
    if checkpoint != expected_checkpoint:
        raise ValueError(
            f"{eval_manifest_path} checkpoint {checkpoint} does not match training "
            f"manifest checkpoint {expected_checkpoint}."
        )
    if not checkpoint.is_file():
        raise FileNotFoundError(f"Missing evaluated checkpoint: {checkpoint}")
    declared_hash = _require_value(
        eval_manifest, "checkpoint_sha256", str, source=eval_manifest_path
    )
    actual_hash = file_sha256(checkpoint)
    if declared_hash != actual_hash:
        raise ValueError(
            f"{eval_manifest_path} checkpoint SHA-256 mismatch: declared "
            f"{declared_hash}, actual {actual_hash}."
        )
    expected_train_fields = {
        "seed": PAPER_SEED,
        "batch_size": PAPER_BATCH_SIZE,
        "horizon": PAPER_HORIZON,
        "observation_steps": PAPER_OBSERVATION_STEPS,
        "action_steps": PAPER_ACTION_STEPS,
        "execution_steps": PAPER_ACTION_STEPS,
        "action_dim": PAPER_ACTION_DIM,
        "image_size": PAPER_IMAGE_SIZE,
        "flow_steps": METHOD_FLOW_STEPS[eval_manifest["method"]],
        "max_train_steps_per_epoch": PAPER_MAX_TRAIN_STEPS_PER_EPOCH,
        "lr_schedule_epoch_horizon": expected_arm.epochs,
    }
    for key, expected in expected_train_fields.items():
        if train_manifest.get(key) != expected:
            raise ValueError(
                f"{train_manifest_path} field {key!r}={train_manifest.get(key)!r} "
                f"does not match protocol value {expected!r}."
            )
    for key in (
        "source_commit",
        "task_key",
        "method",
        "dataset",
        "demonstrations_expected",
        "simulator",
        "exact_demo_budget",
        "simulator_matches_paper",
        "exact_paper_protocol",
        "flow_steps",
        "observation_steps",
        "execution_steps",
    ):
        if train_manifest.get(key) != eval_manifest.get(key):
            raise ValueError(
                f"Training/evaluation manifest mismatch for {key!r}: "
                f"{train_manifest.get(key)!r} != {eval_manifest.get(key)!r}."
            )
    _require_value(
        train_manifest, "demonstrations_expected", int, source=train_manifest_path
    )
    _validate_method_identity(
        train_manifest, train_manifest_path, eval_manifest["method"]
    )
    return train_manifest, train_manifest_path, arm


def load_evaluation(manifest_path: str | Path) -> dict[str, Any]:
    """Load one evaluation and reject incomplete or mismatched evidence."""

    manifest_path = Path(manifest_path).expanduser().resolve()
    manifest = _read_json_object(manifest_path, description="evaluation manifest")
    task_key = _require_value(manifest, "task_key", str, source=manifest_path)
    task = get_task(task_key)
    method = _require_value(manifest, "method", str, source=manifest_path)
    if method not in METHODS:
        raise ValueError(f"{manifest_path} has unknown method {method!r}.")
    evaluation_identity = _validate_manifest_protocol(
        manifest, manifest_path, task, method
    )
    train_manifest, train_manifest_path, arm = _load_train_manifest(
        manifest, manifest_path
    )
    checkpoint_epoch = int(manifest["checkpoint_epoch"])

    output = Path(
        _require_value(manifest, "output", str, source=manifest_path)
    ).expanduser().resolve()
    if output != manifest_path.parent:
        raise ValueError(
            f"{manifest_path} declares output {output}, which does not match its "
            f"containing directory {manifest_path.parent}."
        )
    upstream_policy_name = UPSTREAM_POLICY_NAMES[method]
    stats_path, stats = validate_evaluation_outputs(
        output,
        task=task,
        upstream_policy_name=upstream_policy_name,
        checkpoint_epoch=checkpoint_epoch,
        episode_index_start=evaluation_identity["eval_start_index"],
    )
    episode_indices, episode_successes = _parse_episode_records(stats_path.parent)

    success_rate = stats.total_success / PAPER_EVAL_EPISODES
    paper_target_pct = _paper_target(task, method)
    paper_target_count = paper_target_pct * PAPER_EVAL_EPISODES / 100
    if not paper_target_count.is_integer():
        raise ValueError(
            f"Paper target {paper_target_pct}% is not an integral count over "
            f"{PAPER_EVAL_EPISODES} episodes."
        )
    exact_protocol = bool(manifest["exact_paper_protocol"])
    declared_controls_match = (
        task.is_exact
        and bool(manifest["exact_demo_budget"])
        and bool(manifest["simulator_matches_paper"])
        and method != "a2a_current"
        and evaluation_identity["evaluation_split"] == "official_fixed"
    )
    return {
        "task_key": task_key,
        "paper_task": task.paper_name,
        "benchmark": task.benchmark,
        "official_task_name": task.official_task_name,
        "method": method,
        "arm": arm,
        "checkpoint_epoch": checkpoint_epoch,
        "comparison_point": f"{arm}_e{checkpoint_epoch}",
        "mapping_status": task.mapping_status,
        "mapping_is_proxy": not task.is_exact,
        "demonstrations_expected": manifest["demonstrations_expected"],
        "exact_demo_budget": bool(manifest["exact_demo_budget"]),
        "simulator_matches_paper": bool(manifest["simulator_matches_paper"]),
        "declared_paper_controls_match": declared_controls_match,
        "exact_paper_protocol": exact_protocol,
        "exact_protocol_blockers": list(GLOBAL_EXACT_PROTOCOL_BLOCKERS),
        "protocol_is_proxy": not exact_protocol,
        **evaluation_identity,
        "simulator": manifest["simulator"],
        "dataset": manifest["dataset"],
        "seed": train_manifest["seed"],
        "batch_size": train_manifest["batch_size"],
        "flow_steps": manifest["flow_steps"],
        "source_variant": _source_variant(method),
        "observation_steps": manifest["observation_steps"],
        "prediction_steps": manifest["prediction_steps"],
        "execution_steps": manifest["execution_steps"],
        "paper_target_comparable": exact_protocol,
        "paper_target_success_count": int(paper_target_count),
        "paper_target_success_rate": paper_target_pct / 100,
        "paper_target_success_pct": paper_target_pct,
        "success_count": stats.total_success,
        "completed_count": stats.total_completed,
        "success_rate": success_rate,
        "success_pct": success_rate * 100,
        "delta_vs_paper_success_count": (
            stats.total_success - int(paper_target_count) if exact_protocol else None
        ),
        "delta_vs_paper_percentage_points": (
            success_rate * 100 - paper_target_pct if exact_protocol else None
        ),
        "reported_success_rate": stats.average_success_rate,
        "inference_timing_scope": INFERENCE_TIMING_SCOPE,
        "model_replan_interval_steps": manifest["execution_steps"],
        "total_inference_steps": stats.total_inference_steps,
        "average_inference_time_ms": stats.average_inference_time_ms,
        "std_demo_average_inference_time_ms": (
            stats.std_demo_average_inference_time_ms
        ),
        "min_inference_time_ms": stats.min_inference_time_ms,
        "max_inference_time_ms": stats.max_inference_time_ms,
        "episode_successes": episode_successes,
        "episode_indices": episode_indices,
        "eval_manifest": str(manifest_path),
        "train_manifest": str(train_manifest_path),
        "final_stats": str(stats_path),
        "checkpoint": str(Path(manifest["checkpoint"]).expanduser().resolve()),
        "checkpoint_sha256": manifest["checkpoint_sha256"],
    }


def _delta(new: Mapping[str, Any], old: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success_count": new["success_count"] - old["success_count"],
        "success_rate": new["success_rate"] - old["success_rate"],
        "percentage_points": new["success_pct"] - old["success_pct"],
    }


def _score(result: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "success_count": result["success_count"],
        "success_rate": result["success_rate"],
        "success_pct": result["success_pct"],
        "total_inference_steps": result["total_inference_steps"],
        "average_inference_time_ms": result["average_inference_time_ms"],
        "std_demo_average_inference_time_ms": result[
            "std_demo_average_inference_time_ms"
        ],
        "min_inference_time_ms": result["min_inference_time_ms"],
        "max_inference_time_ms": result["max_inference_time_ms"],
        "delta_vs_paper_success_count": result["delta_vs_paper_success_count"],
        "delta_vs_paper_percentage_points": result[
            "delta_vs_paper_percentage_points"
        ],
    }


def _build_comparisons(evaluations: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[
        tuple[str, str, str], dict[tuple[str, int], Mapping[str, Any]]
    ] = {}
    logical_keys: set[tuple[str, str, str, str, int]] = set()
    for result in evaluations:
        logical_key = (
            str(result["task_key"]),
            str(result["method"]),
            str(result["evaluation_set_id"]),
            str(result["arm"]),
            int(result["checkpoint_epoch"]),
        )
        if logical_key in logical_keys:
            raise ValueError(
                "Duplicate evaluation for "
                f"{logical_key[0]}/{logical_key[1]}/{logical_key[2]}/"
                f"{logical_key[3]}/E{logical_key[4]}."
            )
        logical_keys.add(logical_key)
        group = groups.setdefault(logical_key[:3], {})
        group[logical_key[3:]] = result

    comparisons: list[dict[str, Any]] = []
    for (task_key, method, evaluation_set_id), points in groups.items():
        missing = [point for point in COMPARISON_POINTS if point not in points]
        if missing:
            formatted = ", ".join(f"{arm}/E{epoch}" for arm, epoch in missing)
            raise ValueError(
                f"Incomplete comparison for {task_key}/{method}/{evaluation_set_id}; "
                f"missing {formatted}."
            )
        fresh30 = points[("fresh30", 30)]
        long30 = points[("long200", 30)]
        long200 = points[("long200", 200)]
        if long30["train_manifest"] != long200["train_manifest"]:
            raise ValueError(
                f"Comparison {task_key}/{method} mixes long200 checkpoints from "
                "different uninterrupted training runs."
            )
        invariant_fields = (
            "mapping_status",
            "mapping_is_proxy",
            "dataset",
            "simulator",
            "seed",
            "batch_size",
            "flow_steps",
            "source_variant",
            "observation_steps",
            "prediction_steps",
            "execution_steps",
            "demonstrations_expected",
            "exact_demo_budget",
            "simulator_matches_paper",
            "declared_paper_controls_match",
            "exact_paper_protocol",
            "exact_protocol_blockers",
            "eval_start_index",
            "eval_trajectory_indices",
            "evaluation_split",
            "evaluation_set_id",
            "dataset_provenance",
            "paper_target_success_count",
            "paper_target_success_rate",
        )
        for field in invariant_fields:
            if not (fresh30[field] == long30[field] == long200[field]):
                raise ValueError(
                    f"Comparison {task_key}/{method} has inconsistent {field!r}."
                )
        comparisons.append(
            {
                "task_key": task_key,
                "paper_task": fresh30["paper_task"],
                "benchmark": fresh30["benchmark"],
                "method": method,
                "eval_start_index": fresh30["eval_start_index"],
                "eval_trajectory_indices": fresh30["eval_trajectory_indices"],
                "evaluation_split": fresh30["evaluation_split"],
                "evaluation_set_id": fresh30["evaluation_set_id"],
                "dataset_provenance": fresh30["dataset_provenance"],
                "mapping_status": fresh30["mapping_status"],
                "mapping_is_proxy": fresh30["mapping_is_proxy"],
                "protocol_is_proxy": fresh30["protocol_is_proxy"],
                "simulator": fresh30["simulator"],
                "simulator_matches_paper": fresh30["simulator_matches_paper"],
                "declared_paper_controls_match": fresh30[
                    "declared_paper_controls_match"
                ],
                "exact_protocol_blockers": fresh30["exact_protocol_blockers"],
                "dataset": fresh30["dataset"],
                "seed": fresh30["seed"],
                "batch_size": fresh30["batch_size"],
                "flow_steps": fresh30["flow_steps"],
                "source_variant": fresh30["source_variant"],
                "observation_steps": fresh30["observation_steps"],
                "prediction_steps": fresh30["prediction_steps"],
                "execution_steps": fresh30["execution_steps"],
                "inference_timing_scope": fresh30["inference_timing_scope"],
                "model_replan_interval_steps": fresh30[
                    "model_replan_interval_steps"
                ],
                "demonstrations_expected": fresh30["demonstrations_expected"],
                "exact_demo_budget": fresh30["exact_demo_budget"],
                "exact_paper_protocol": fresh30["exact_paper_protocol"],
                "paper_target_comparable": fresh30["paper_target_comparable"],
                "paper_target": {
                    "success_count": fresh30["paper_target_success_count"],
                    "success_rate": fresh30["paper_target_success_rate"],
                    "success_pct": fresh30["paper_target_success_pct"],
                },
                "fresh30_e30": _score(fresh30),
                "long200_e30": _score(long30),
                "long200_e200": _score(long200),
                "deltas": {
                    "long200_e30_minus_fresh30_e30": _delta(long30, fresh30),
                    "long200_e200_minus_long200_e30": _delta(long200, long30),
                    "long200_e200_minus_fresh30_e30": _delta(long200, fresh30),
                },
            }
        )
    task_order = {key: index for index, key in enumerate(PAPER_TASKS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    comparisons.sort(
        key=lambda item: (
            task_order[item["task_key"]],
            item["evaluation_set_id"],
            method_order[item["method"]],
        )
    )
    return comparisons


def _validate_cross_method_controls(
    comparisons: Sequence[Mapping[str, Any]],
) -> None:
    by_task: dict[tuple[str, str], list[Mapping[str, Any]]] = {}
    for comparison in comparisons:
        key = (
            str(comparison["task_key"]),
            str(comparison["evaluation_set_id"]),
        )
        by_task.setdefault(key, []).append(comparison)
    fields = (
        "dataset",
        "simulator",
        "seed",
        "batch_size",
        "observation_steps",
        "prediction_steps",
        "execution_steps",
        "demonstrations_expected",
        "exact_demo_budget",
        "simulator_matches_paper",
        "eval_start_index",
        "eval_trajectory_indices",
        "evaluation_split",
        "dataset_provenance",
    )
    for (task_key, evaluation_set_id), rows in by_task.items():
        if len(rows) < 2:
            continue
        for field in fields:
            reference = rows[0][field]
            if any(row[field] != reference for row in rows[1:]):
                raise ValueError(
                    f"Cross-method comparison for {task_key}/{evaluation_set_id} "
                    f"has inconsistent {field!r}."
                )


def _paired_binary_counts(
    first: Sequence[bool], second: Sequence[bool]
) -> dict[str, Any]:
    if len(first) != len(second):
        raise ValueError(
            "Paired success vectors must have the same length, got "
            f"{len(first)} and {len(second)}."
        )
    if any(type(value) is not bool for value in (*first, *second)):
        raise ValueError("Paired success vectors must contain only booleans.")

    both_success = sum(a and b for a, b in zip(first, second, strict=True))
    first_only = sum(a and not b for a, b in zip(first, second, strict=True))
    second_only = sum(not a and b for a, b in zip(first, second, strict=True))
    both_failure = len(first) - both_success - first_only - second_only
    discordant = first_only + second_only
    if discordant == 0:
        p_value = 1.0
    else:
        lower_tail = sum(
            math.comb(discordant, count)
            for count in range(min(first_only, second_only) + 1)
        ) / (2**discordant)
        p_value = min(1.0, 2.0 * lower_tail)
    return {
        "episode_count": len(first),
        "both_success_count": both_success,
        "first_only_success_count": first_only,
        "second_only_success_count": second_only,
        "both_failure_count": both_failure,
        "discordant_count": discordant,
        "mcnemar_exact_two_sided_p_value": p_value,
    }


def _build_paired_tests(
    evaluations: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    method_order = {method: index for index, method in enumerate(METHODS)}
    task_order = {task: index for index, task in enumerate(PAPER_TASKS)}
    point_order = {
        f"{arm}_e{epoch}": index
        for index, (arm, epoch) in enumerate(COMPARISON_POINTS)
    }

    by_task_point: dict[
        tuple[str, str, str], dict[str, Mapping[str, Any]]
    ] = {}
    by_task_method: dict[
        tuple[str, str, str], dict[str, Mapping[str, Any]]
    ] = {}
    for result in evaluations:
        task_key = str(result["task_key"])
        method = str(result["method"])
        point = str(result["comparison_point"])
        evaluation_set_id = str(result["evaluation_set_id"])
        by_task_point.setdefault((task_key, evaluation_set_id, point), {})[
            method
        ] = result
        by_task_method.setdefault((task_key, method, evaluation_set_id), {})[
            point
        ] = result

    cross_method: list[dict[str, Any]] = []
    for (task_key, evaluation_set_id, point), methods in sorted(
        by_task_point.items(),
        key=lambda item: (
            task_order[item[0][0]],
            item[0][1],
            point_order[item[0][2]],
        ),
    ):
        ordered_methods = sorted(methods, key=method_order.__getitem__)
        for first_index, method_a in enumerate(ordered_methods):
            for method_b in ordered_methods[first_index + 1 :]:
                result_a = methods[method_a]
                result_b = methods[method_b]
                if result_a["episode_indices"] != result_b["episode_indices"]:
                    raise ValueError(
                        "Cannot pair cross-method evaluations with different "
                        f"episode indices for {task_key}/{evaluation_set_id}/{point}."
                    )
                counts = _paired_binary_counts(
                    result_a["episode_successes"], result_b["episode_successes"]
                )
                cross_method.append(
                    {
                        "task_key": task_key,
                        "paper_task": result_a["paper_task"],
                        "evaluation_set_id": evaluation_set_id,
                        "evaluation_split": result_a["evaluation_split"],
                        "eval_trajectory_indices": result_a[
                            "eval_trajectory_indices"
                        ],
                        "comparison_point": point,
                        "method_a": method_a,
                        "method_b": method_b,
                        "method_a_success_count": result_a["success_count"],
                        "method_b_success_count": result_b["success_count"],
                        "both_success_count": counts["both_success_count"],
                        "method_a_only_success_count": counts[
                            "first_only_success_count"
                        ],
                        "method_b_only_success_count": counts[
                            "second_only_success_count"
                        ],
                        "both_failure_count": counts["both_failure_count"],
                        "discordant_count": counts["discordant_count"],
                        "episode_count": counts["episode_count"],
                        "mcnemar_exact_two_sided_p_value": counts[
                            "mcnemar_exact_two_sided_p_value"
                        ],
                    }
                )

    long200_e30_vs_e200: list[dict[str, Any]] = []
    for (task_key, method, evaluation_set_id), points in sorted(
        by_task_method.items(),
        key=lambda item: (
            task_order[item[0][0]],
            item[0][2],
            method_order[item[0][1]],
        ),
    ):
        e30 = points["long200_e30"]
        e200 = points["long200_e200"]
        if e30["episode_indices"] != e200["episode_indices"]:
            raise ValueError(
                "Cannot pair E30/E200 evaluations with different episode indices "
                f"for {task_key}/{method}/{evaluation_set_id}."
            )
        counts = _paired_binary_counts(
            e30["episode_successes"], e200["episode_successes"]
        )
        long200_e30_vs_e200.append(
            {
                "task_key": task_key,
                "paper_task": e30["paper_task"],
                "method": method,
                "evaluation_set_id": evaluation_set_id,
                "evaluation_split": e30["evaluation_split"],
                "eval_trajectory_indices": e30["eval_trajectory_indices"],
                "arm": "long200",
                "epoch_a": 30,
                "epoch_b": 200,
                "e30_success_count": e30["success_count"],
                "e200_success_count": e200["success_count"],
                "both_success_count": counts["both_success_count"],
                "e30_only_success_count": counts["first_only_success_count"],
                "e200_only_success_count": counts["second_only_success_count"],
                "both_failure_count": counts["both_failure_count"],
                "discordant_count": counts["discordant_count"],
                "episode_count": counts["episode_count"],
                "mcnemar_exact_two_sided_p_value": counts[
                    "mcnemar_exact_two_sided_p_value"
                ],
            }
        )

    return {
        "cross_method": cross_method,
        "long200_e30_vs_e200": long200_e30_vs_e200,
    }


def aggregate_results(
    inputs: Iterable[str | Path], *, require_full_matrix: bool = False
) -> dict[str, Any]:
    """Aggregate complete fresh30/long200 triplets from result directories."""

    manifests = discover_eval_manifests(inputs)
    evaluations = [load_evaluation(path) for path in manifests]
    comparisons = _build_comparisons(evaluations)
    _validate_cross_method_controls(comparisons)
    paired_tests = _build_paired_tests(evaluations)
    expected = {
        (task, method) for task in PAPER_TASKS for method in PRIMARY_METHODS
    }
    evaluation_set_ids = {
        str(row["evaluation_set_id"]) for row in comparisons
    }
    actual_by_set: dict[str, set[tuple[str, str]]] = {
        evaluation_set_id: set() for evaluation_set_id in evaluation_set_ids
    }
    for row in comparisons:
        if row["method"] in PRIMARY_METHODS:
            actual_by_set[str(row["evaluation_set_id"])].add(
                (str(row["task_key"]), str(row["method"]))
            )
    if require_full_matrix:
        missing_by_set = {
            evaluation_set_id: sorted(expected - actual)
            for evaluation_set_id, actual in actual_by_set.items()
            if expected - actual
        }
        if missing_by_set:
            details = "; ".join(
                f"{evaluation_set_id}: "
                + ", ".join(f"{task}/{method}" for task, method in missing)
                for evaluation_set_id, missing in sorted(missing_by_set.items())
            )
            raise ValueError(
                "Incomplete five-task/two-method matrix by evaluation set; "
                + details
                + "."
            )
    comparison_rank = {point: index for index, point in enumerate(COMPARISON_POINTS)}
    task_order = {key: index for index, key in enumerate(PAPER_TASKS)}
    method_order = {method: index for index, method in enumerate(METHODS)}
    evaluations.sort(
        key=lambda row: (
            task_order[row["task_key"]],
            row["evaluation_set_id"],
            method_order[row["method"]],
            comparison_rank[(row["arm"], row["checkpoint_epoch"])],
        )
    )
    return {
        "schema": "official_a2a_roboverse_results_v1",
        "paper_source_commit": PAPER_SOURCE_COMMIT,
        "eval_episodes_per_checkpoint": PAPER_EVAL_EPISODES,
        "evaluation_count": len(evaluations),
        "comparison_count": len(comparisons),
        "evaluation_set_ids": sorted(evaluation_set_ids),
        "full_five_task_two_method_matrix": bool(evaluation_set_ids)
        and all(actual == expected for actual in actual_by_set.values()),
        "evaluations": evaluations,
        "comparisons": comparisons,
        "paired_tests": paired_tests,
    }


def _comparison_csv_rows(report: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for comparison in report["comparisons"]:
        paper = comparison["paper_target"]
        fresh = comparison["fresh30_e30"]
        long30 = comparison["long200_e30"]
        long200 = comparison["long200_e200"]
        deltas = comparison["deltas"]
        row = {
            "task_key": comparison["task_key"],
            "paper_task": comparison["paper_task"],
            "benchmark": comparison["benchmark"],
            "method": comparison["method"],
            "evaluation_set_id": comparison["evaluation_set_id"],
            "evaluation_split": comparison["evaluation_split"],
            "eval_start_index": comparison["eval_start_index"],
            "eval_trajectory_indices": comparison["eval_trajectory_indices"],
            "dataset_provenance_logical_sha256": (
                comparison["dataset_provenance"]["logical_content_sha256"]
                if comparison["dataset_provenance"] is not None
                else None
            ),
            "evaluation_overlap_count": (
                comparison["dataset_provenance"]["evaluation_overlap_count"]
                if comparison["dataset_provenance"] is not None
                else None
            ),
            "mapping_status": comparison["mapping_status"],
            "mapping_is_proxy": comparison["mapping_is_proxy"],
            "protocol_is_proxy": comparison["protocol_is_proxy"],
            "simulator": comparison["simulator"],
            "simulator_matches_paper": comparison["simulator_matches_paper"],
            "declared_paper_controls_match": comparison[
                "declared_paper_controls_match"
            ],
            "dataset": comparison["dataset"],
            "seed": comparison["seed"],
            "batch_size": comparison["batch_size"],
            "flow_steps": comparison["flow_steps"],
            "source_variant": comparison["source_variant"],
            "observation_steps": comparison["observation_steps"],
            "prediction_steps": comparison["prediction_steps"],
            "execution_steps": comparison["execution_steps"],
            "inference_timing_scope": comparison["inference_timing_scope"],
            "model_replan_interval_steps": comparison[
                "model_replan_interval_steps"
            ],
            "demonstrations_expected": comparison["demonstrations_expected"],
            "exact_demo_budget": comparison["exact_demo_budget"],
            "exact_paper_protocol": comparison["exact_paper_protocol"],
            "paper_target_comparable": comparison["paper_target_comparable"],
            "paper_target_success_count": paper["success_count"],
            "paper_target_success_rate": paper["success_rate"],
            "paper_target_success_pct": paper["success_pct"],
        }
        for prefix, score in (
            ("fresh30_e30", fresh),
            ("long200_e30", long30),
            ("long200_e200", long200),
        ):
            for key, value in score.items():
                row[f"{prefix}_{key}"] = value
        for prefix, delta in deltas.items():
            for key, value in delta.items():
                row[f"{prefix}_{key}"] = value
        rows.append(row)
    return rows


def write_report(
    report: Mapping[str, Any], *, json_output: str | Path, csv_output: str | Path
) -> None:
    json_path = Path(json_output).expanduser().resolve()
    csv_path = Path(csv_output).expanduser().resolve()
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    rows = _comparison_csv_rows(report)
    if not rows:
        raise ValueError("Cannot write an empty comparison CSV.")
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        action="append",
        required=True,
        help="Evaluation root or eval_manifest.json; may be repeated.",
    )
    parser.add_argument("--json-output", type=Path, required=True)
    parser.add_argument("--csv-output", type=Path, required=True)
    parser.add_argument(
        "--require-full-matrix",
        action="store_true",
        help="Require all five tasks, both methods, and all three comparison points.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    report = aggregate_results(
        args.input, require_full_matrix=args.require_full_matrix
    )
    write_report(report, json_output=args.json_output, csv_output=args.csv_output)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "COMPARISON_POINTS",
    "FinalStats",
    "aggregate_results",
    "discover_eval_manifests",
    "load_evaluation",
    "parse_final_stats",
    "validate_evaluation_outputs",
    "write_report",
]
