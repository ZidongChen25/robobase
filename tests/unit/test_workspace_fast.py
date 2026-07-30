import numpy as np

from robobase.workspace_fast import _DeviceMergedIterator


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
