import numpy as np
import pytest

from robobase.workspace_fast import WorkspaceFast, _DeviceMergedIterator


class _ListIterator:
    def __init__(self, batches):
        self._batches = iter(batches)
        self.closed = False

    def __next__(self):
        return next(self._batches)

    def close(self):
        self.closed = True


def test_device_merged_iterator_matches_host_concat():
    rng = np.random.default_rng(0)
    online = {
        "rgb": rng.integers(0, 255, (4, 3, 8, 8), dtype=np.uint8),
        "action": rng.normal(size=(4, 2)).astype(np.float32),
        "demo": np.zeros((4,), np.uint8),
    }
    demo = {
        "rgb": rng.integers(0, 255, (4, 3, 8, 8), dtype=np.uint8),
        "action": rng.normal(size=(4, 2)).astype(np.float32),
        "demo": np.zeros((4,), np.uint8),
    }
    base_online = _ListIterator([online])
    base_demo = _ListIterator([{k: v.copy() for k, v in demo.items()}])
    it = _DeviceMergedIterator(base_online, base_demo)
    merged = next(it)
    # Values identical to the host-side np.concatenate merge, demo flags
    # forced to one for the demo half.
    np.testing.assert_array_equal(
        np.asarray(merged["rgb"]),
        np.concatenate([online["rgb"], demo["rgb"]], axis=0),
    )
    np.testing.assert_array_equal(
        np.asarray(merged["action"]),
        np.concatenate([online["action"], demo["action"]], axis=0),
    )
    np.testing.assert_array_equal(
        np.asarray(merged["demo"]),
        np.concatenate([np.zeros(4), np.ones(4)]),
    )
    it.close()
    assert base_online.closed and base_demo.closed


def test_device_merged_iterator_materializes_before_returning():
    online = {
        "value": np.arange(8, dtype=np.float32).reshape(4, 2),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    demo = {
        "value": np.arange(8, 16, dtype=np.float32).reshape(4, 2),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    it = _DeviceMergedIterator(
        _ListIterator([online]),
        _ListIterator([{key: value.copy() for key, value in demo.items()}]),
    )

    real_block_until_ready = it._block_until_ready
    blocked = []

    def record_block(tree):
        result = real_block_until_ready(tree)
        blocked.append(result)
        return result

    it._block_until_ready = record_block
    merged = next(it)

    assert len(blocked) == 1
    assert blocked[0] is merged
    np.testing.assert_array_equal(
        np.asarray(merged["value"]),
        np.concatenate([online["value"], demo["value"]], axis=0),
    )


def test_device_merged_iterator_verifier_rejects_corruption(monkeypatch):
    online = {
        "value": np.arange(8, dtype=np.float32).reshape(4, 2),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    demo = {
        "value": np.arange(8, 16, dtype=np.float32).reshape(4, 2),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    monkeypatch.setenv("ROBOBASE_DEVICE_MERGE_VERIFY", "1")
    it = _DeviceMergedIterator(
        _ListIterator([online]),
        _ListIterator([{key: value.copy() for key, value in demo.items()}]),
    )
    assert it._verify_non_image
    real_block_until_ready = it._block_until_ready

    def corrupt_after_block(tree):
        ready = real_block_until_ready(tree)
        ready["value"] = ready["value"].at[0, 0].set(-999.0)
        return real_block_until_ready(ready)

    it._block_until_ready = corrupt_after_block

    with pytest.raises(RuntimeError, match="verification failed for 'value'"):
        next(it)


def test_device_merged_iterator_verifier_accepts_jax_dtype_canonicalization(
    monkeypatch,
):
    monkeypatch.setenv("ROBOBASE_DEVICE_MERGE_VERIFY", "1")
    online = {
        "discount": np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float64),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    demo = {
        "discount": np.array([0.5, 0.6, 0.7, 0.8], dtype=np.float64),
        "demo": np.zeros((4,), dtype=np.uint8),
    }
    it = _DeviceMergedIterator(
        _ListIterator([online]),
        _ListIterator([{key: value.copy() for key, value in demo.items()}]),
    )

    merged = next(it)

    expected = np.concatenate([online["discount"], demo["discount"]]).astype(
        np.asarray(merged["discount"]).dtype
    )
    np.testing.assert_array_equal(np.asarray(merged["discount"]), expected)


def test_workspace_fast_inherits_agent_rollout_diagnostics_for_training_logs():
    workspace = object.__new__(WorkspaceFast)

    class _Agent:
        @staticmethod
        def rollout_diagnostics():
            return {
                "episodic_twin_head_assignments": 7.0,
                "episodic_twin_head0_rate": 3.0 / 7.0,
                "episodic_twin_head1_rate": 4.0 / 7.0,
            }

    workspace.agent = _Agent()

    assert workspace._get_rollout_diagnostics() == {
        "episodic_twin_head_assignments": 7.0,
        "episodic_twin_head0_rate": 3.0 / 7.0,
        "episodic_twin_head1_rate": 4.0 / 7.0,
    }
