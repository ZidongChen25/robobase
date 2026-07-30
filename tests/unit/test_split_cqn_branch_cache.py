import json

import numpy as np
import pytest

from scripts.split_cqn_branch_cache import split_cache


def _write_source(path):
    records = [
        {"eval_seed": seed, "anchor_step": anchor}
        for seed in (10, 20, 30)
        for anchor in (30, 75)
    ]
    rows = len(records)
    metadata = {
        "source_snapshot": "snapshot.pkl",
        "train_seeds": [10, 20, 30],
        "heldout_seeds": [40],
        "anchor_steps": [30, 75],
        "action_dimensions": [13],
    }
    np.savez_compressed(
        path,
        cache_metadata=np.asarray(json.dumps(metadata)),
        train_features=np.arange(rows * 3, dtype=np.float32).reshape(rows, 3),
        train_actions=np.arange(
            rows * 5 * 2, dtype=np.float32
        ).reshape(rows, 5, 2),
        train_returns=np.arange(rows * 5, dtype=np.float32).reshape(rows, 5),
        train_return_samples=np.arange(
            rows * 5 * 2, dtype=np.float32
        ).reshape(rows, 5, 2),
        train_action_dimensions=np.full(rows, 13, np.int32),
        train_metadata=np.asarray(json.dumps(records)),
        heldout_features=np.zeros((1, 3), np.float32),
        heldout_actions=np.zeros((1, 5, 2), np.float32),
        heldout_returns=np.zeros((1, 5), np.float32),
        heldout_action_dimensions=np.asarray([13], np.int32),
        heldout_metadata=np.asarray(
            json.dumps([{"eval_seed": 40, "anchor_step": 30}])
        ),
    )


def test_split_cache_partitions_source_train_rows_by_seed(tmp_path):
    source = tmp_path / "source.npz"
    output = tmp_path / "output.npz"
    _write_source(source)

    result = split_cache(
        source,
        output,
        train_seeds=[10, 30],
        heldout_seeds=[20],
    )

    assert result["num_train_states"] == 4
    assert result["num_heldout_states"] == 2
    with np.load(output, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["cache_metadata"].item()))
        train_records = json.loads(str(payload["train_metadata"].item()))
        heldout_records = json.loads(
            str(payload["heldout_metadata"].item())
        )
        assert metadata["train_seeds"] == [10, 30]
        assert metadata["heldout_seeds"] == [20]
        assert {record["eval_seed"] for record in train_records} == {10, 30}
        assert {record["eval_seed"] for record in heldout_records} == {20}
        assert payload["train_features"].shape == (4, 3)
        assert payload["heldout_features"].shape == (2, 3)
        assert payload["train_return_samples"].shape == (4, 5, 2)
        assert payload["heldout_return_samples"].shape == (2, 5, 2)


def test_split_cache_rejects_overlap_and_missing_seed(tmp_path):
    source = tmp_path / "source.npz"
    _write_source(source)

    with pytest.raises(ValueError, match="disjoint"):
        split_cache(
            source,
            tmp_path / "overlap.npz",
            train_seeds=[10],
            heldout_seeds=[10],
        )
    with pytest.raises(ValueError, match="absent"):
        split_cache(
            source,
            tmp_path / "missing.npz",
            train_seeds=[10],
            heldout_seeds=[99],
        )
