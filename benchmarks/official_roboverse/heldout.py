"""Evidence checks for source-disjoint RoboVerse evaluation ranges."""

from __future__ import annotations

import json
from pathlib import Path

from benchmarks.official_bigym.a2a_upstream import file_sha256
from benchmarks.official_roboverse.audit_proxy_data import (
    HASH_SPEC as PROVENANCE_HASH_SPEC,
    SCHEMA as PROVENANCE_SCHEMA,
)
from benchmarks.official_roboverse.protocol import PAPER_EVAL_EPISODES


def validate_disjoint_dataset_provenance(
    provenance_path: str | Path,
    *,
    dataset: str | Path,
    expected_episodes: int,
    eval_start_index: int,
) -> dict[str, object]:
    """Bind a held-out range to an audited training-source index set."""

    provenance_path = Path(provenance_path).expanduser().resolve()
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise FileNotFoundError(
            f"Missing dataset provenance manifest: {provenance_path}"
        ) from None
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Malformed dataset provenance manifest {provenance_path}: {exc}"
        ) from exc
    if not isinstance(provenance, dict):
        raise ValueError(f"Dataset provenance must be an object: {provenance_path}")
    expected_fields = {
        "schema": PROVENANCE_SCHEMA,
        "status": "pass",
        "hash_spec": PROVENANCE_HASH_SPEC,
        "episodes": expected_episodes,
    }
    for key, expected in expected_fields.items():
        if provenance.get(key) != expected:
            raise ValueError(
                f"{provenance_path} field {key!r}={provenance.get(key)!r}; "
                f"expected {expected!r}."
            )
    if provenance.get("errors") != [] or provenance.get("raw_exact_match") is not True:
        raise ValueError(
            f"{provenance_path} must have no errors and raw_exact_match=true."
        )
    declared_dataset = Path(str(provenance.get("dataset", ""))).expanduser().resolve()
    dataset = Path(dataset).expanduser().resolve()
    if declared_dataset != dataset:
        raise ValueError(
            f"{provenance_path} audits {declared_dataset}, not requested dataset "
            f"{dataset}."
        )
    logical_sha256 = provenance.get("logical_content_sha256")
    if not isinstance(logical_sha256, str) or len(logical_sha256) != 64:
        raise ValueError(f"{provenance_path} has invalid logical_content_sha256.")
    source_indices = provenance.get("selected_source_indices")
    if not isinstance(source_indices, list) or any(
        not isinstance(index, int) or isinstance(index, bool) or index < 0
        for index in source_indices
    ):
        raise ValueError(
            f"{provenance_path} selected_source_indices must be non-negative integers."
        )
    if len(source_indices) != expected_episodes or len(set(source_indices)) != len(
        source_indices
    ):
        raise ValueError(
            f"{provenance_path} must declare exactly {expected_episodes} unique "
            "training source indices."
        )
    eval_indices = set(
        range(eval_start_index, eval_start_index + PAPER_EVAL_EPISODES)
    )
    overlap = sorted(eval_indices.intersection(source_indices))
    if overlap:
        raise ValueError(
            "Held-out evaluation range overlaps training source indices: "
            f"{overlap}."
        )
    return {
        "path": str(provenance_path),
        "file_sha256": file_sha256(provenance_path),
        "schema": PROVENANCE_SCHEMA,
        "hash_spec": PROVENANCE_HASH_SPEC,
        "logical_content_sha256": logical_sha256,
        "dataset": str(dataset),
        "training_source_count": len(source_indices),
        "selected_source_indices": source_indices,
        "training_source_index_min": min(source_indices),
        "training_source_index_max": max(source_indices),
        "evaluation_overlap_count": 0,
        "evaluation_overlap_indices": [],
    }


__all__ = ["validate_disjoint_dataset_provenance"]
