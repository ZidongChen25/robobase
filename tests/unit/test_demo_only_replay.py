import numpy as np
from omegaconf import OmegaConf

from robobase.workspace import _DemoOnlyIterator
from robobase.workspace_fast import WorkspaceFast


class _CountingIterator:
    def __init__(self, batch):
        self.batch = batch
        self.count = 0
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self):
        self.count += 1
        return {key: value.copy() for key, value in self.batch.items()}

    def close(self):
        self.closed = True


def test_demo_only_iterator_ignores_online_values_and_stored_demo_flag():
    online = _CountingIterator(
        {
            "x": np.full((4, 2), 99.0, dtype=np.float32),
            "demo": np.zeros((4,), dtype=np.uint8),
        }
    )
    demo = _CountingIterator(
        {
            "x": np.arange(16, dtype=np.float32).reshape(8, 2),
            "demo": np.zeros((8,), dtype=np.uint8),
        }
    )
    iterator = _DemoOnlyIterator(online, demo)

    batch = next(iterator)

    assert online.count == 0
    assert demo.count == 1
    assert batch["x"].shape == (8, 2)
    np.testing.assert_array_equal(
        batch["x"],
        np.arange(16, dtype=np.float32).reshape(8, 2),
    )
    np.testing.assert_array_equal(batch["demo"], np.ones(8, dtype=np.uint8))

    iterator.close()
    assert online.closed
    assert demo.closed


def test_workspace_fast_demo_only_hook_keeps_demo_only_iterator():
    workspace = object.__new__(WorkspaceFast)
    workspace.cfg = OmegaConf.create(
        {"replay": {"demo_only_updates": True}}
    )
    workspace.demo_only_updates = True
    online = _CountingIterator(
        {"x": np.zeros((4, 1)), "demo": np.zeros(4, dtype=np.uint8)}
    )
    demo = _CountingIterator(
        {"x": np.ones((8, 1)), "demo": np.zeros(8, dtype=np.uint8)}
    )

    iterator = workspace._make_merged_replay_iter(online, demo)

    assert isinstance(iterator, _DemoOnlyIterator)
    batch = next(iterator)
    assert batch["x"].shape == (8, 1)
    assert online.count == 0
