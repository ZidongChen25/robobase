import json

import numpy as np

from scripts.compose_cqn_external_eval_cache import (
    compose_external_eval_cache,
)


def _cache(path, train_seeds, heldout_seeds):
    arrays = {}
    for split, seeds in (
        ("train", train_seeds),
        ("heldout", heldout_seeds),
    ):
        count = len(seeds)
        arrays[f"{split}_features"] = np.zeros((count, 3), np.float32)
        arrays[f"{split}_actions"] = np.zeros(
            (count, 5, 2, 1), np.float32
        )
        arrays[f"{split}_returns"] = np.zeros((count, 5), np.float32)
        arrays[f"{split}_action_dimensions"] = np.zeros(count, np.int32)
        arrays[f"{split}_metadata"] = np.asarray(
            json.dumps([{"eval_seed": seed} for seed in seeds])
        )
        arrays[f"{split}_return_samples"] = np.zeros(
            (count, 5, 1), np.float32
        )
    arrays["cache_metadata"] = np.asarray("{}")
    np.savez_compressed(path, **arrays)


def test_composes_original_train_with_both_external_splits(tmp_path):
    train_cache = tmp_path / "train.npz"
    evaluation_cache = tmp_path / "evaluation.npz"
    output = tmp_path / "composed.npz"
    _cache(train_cache, [1, 2], [3])
    _cache(evaluation_cache, [10, 11], [12])

    result = compose_external_eval_cache(
        train_cache=train_cache,
        evaluation_cache=evaluation_cache,
        output=output,
    )

    assert result["train_seeds"] == [1, 2]
    assert result["heldout_seeds"] == [10, 11, 12]
    with np.load(output, allow_pickle=False) as payload:
        assert payload["train_features"].shape[0] == 2
        assert payload["heldout_features"].shape[0] == 3
