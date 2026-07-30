import json

import numpy as np
import pytest

from scripts.merge_cqn_branch_caches import CacheSplit, merge_caches
from scripts.resplit_cqn_branch_cache import resplit_cache


def _write_cache(path, *, train_seed, heldout_seed, source="snapshot.pkl"):
    def split(seed):
        return {
            "features": np.full((2, 3), seed, np.float32),
            "actions": np.full((2, 5, 2, 2), seed, np.float32),
            "returns": np.arange(10, dtype=np.float32).reshape(2, 5),
            "action_dimensions": np.asarray([0, 1], np.int32),
            "metadata": [
                {"eval_seed": seed, "anchor_step": 30},
                {"eval_seed": seed, "anchor_step": 75},
            ],
        }

    train = split(train_seed)
    heldout = split(heldout_seed)
    metadata = {
        "source_snapshot": source,
        "train_seeds": [train_seed],
        "heldout_seeds": [heldout_seed],
        "anchor_steps": [30, 75],
        "action_dimensions": [0, 1],
        "candidate_mode": "sibling_bins",
        "force_level": 1,
        "intervention_horizon": 4,
        "max_continuation_steps": 300,
        "gamma": 0.99,
    }
    arrays = {"cache_metadata": np.asarray(json.dumps(metadata))}
    for prefix, data in (("train", train), ("heldout", heldout)):
        for name in (
            "features",
            "actions",
            "returns",
            "action_dimensions",
        ):
            arrays[f"{prefix}_{name}"] = data[name]
        arrays[f"{prefix}_metadata"] = np.asarray(
            json.dumps(data["metadata"])
        )
    np.savez_compressed(path, **arrays)


def test_merge_branch_caches_promotes_disjoint_splits(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    output = tmp_path / "merged.npz"
    _write_cache(first, train_seed=10, heldout_seed=20)
    _write_cache(second, train_seed=30, heldout_seed=40)

    result = merge_caches(
        train_sources=[
            CacheSplit(first, "train"),
            CacheSplit(first, "heldout"),
            CacheSplit(second, "train"),
        ],
        heldout_sources=[CacheSplit(second, "heldout")],
        output=output,
    )

    assert result["train_seeds"] == [10, 20, 30]
    assert result["heldout_seeds"] == [40]
    assert result["num_train_records"] == 6
    with np.load(output, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["cache_metadata"].item()))
        assert metadata["train_seeds"] == [10, 20, 30]
        assert metadata["heldout_seeds"] == [40]
        assert payload["train_features"].shape == (6, 3)
        assert payload["heldout_features"].shape == (2, 3)
        assert payload["train_return_samples"].shape == (6, 5, 1)


def test_merge_branch_caches_rejects_protocol_mismatch(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    _write_cache(first, train_seed=10, heldout_seed=20)
    _write_cache(
        second,
        train_seed=30,
        heldout_seed=40,
        source="different_snapshot.pkl",
    )

    with pytest.raises(ValueError, match="protocol mismatch"):
        merge_caches(
            train_sources=[
                CacheSplit(first, "train"),
                CacheSplit(second, "train"),
            ],
            heldout_sources=[CacheSplit(first, "heldout")],
            output=tmp_path / "merged.npz",
        )


def test_resplit_branch_cache_partitions_records_by_seed(tmp_path):
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    merged = tmp_path / "merged.npz"
    output = tmp_path / "resplit.npz"
    _write_cache(first, train_seed=10, heldout_seed=20)
    _write_cache(second, train_seed=30, heldout_seed=40)
    merge_caches(
        train_sources=[
            CacheSplit(first, "train"),
            CacheSplit(first, "heldout"),
        ],
        heldout_sources=[
            CacheSplit(second, "train"),
            CacheSplit(second, "heldout"),
        ],
        output=merged,
    )

    result = resplit_cache(
        input_path=merged,
        train_seeds=[10, 30],
        heldout_seeds=[20, 40],
        output=output,
    )

    assert result["num_train_records"] == 4
    assert result["num_heldout_records"] == 4
    with np.load(output, allow_pickle=False) as payload:
        train_records = json.loads(str(payload["train_metadata"].item()))
        heldout_records = json.loads(
            str(payload["heldout_metadata"].item())
        )
        assert {record["eval_seed"] for record in train_records} == {10, 30}
        assert {record["eval_seed"] for record in heldout_records} == {20, 40}
