"""WorkspaceFast device-side demo merge: wiring and value equivalence.

Regression tests for the bug where the device merge was injected by
overriding the ``replay_iter`` property and isinstance-checking the parent's
return value — which is the PrefetchReplayBatchIterator whenever
``backend.replay_prefetch_size > 0``, so the device merge never engaged
under the default jax backend config.
"""

import numpy as np
import pytest
from omegaconf import OmegaConf

from robobase.replay_buffer.iterator import PrefetchReplayBatchIterator
from robobase.workspace import _DemoMergedIterator
from robobase.workspace_fast import WorkspaceFast, _DeviceMergedIterator


class _FakeBuffer:
    batch_size = 4

    def __init__(self, demo_flag):
        self._demo_flag = demo_flag
        self._rng = np.random.default_rng(0 if demo_flag else 1)

    def sample(self, batch_size):
        return {
            "low_dim_state": self._rng.standard_normal(
                (batch_size, 3)
            ).astype(np.float32),
            "demo": np.full((batch_size,), self._demo_flag, dtype=np.uint8),
        }


class _FakeAgent:
    @staticmethod
    def prefetch_batch(batch):
        return batch


class _StubWorkspaceFast(WorkspaceFast):
    """WorkspaceFast with only the attributes the replay_iter property needs."""

    def __init__(self):  # deliberately no super().__init__
        self._replay_iter = None
        self.cfg = OmegaConf.create(
            {
                "backend": {
                    "replay_prefetch_size": 4,
                    "replay_device_prefetch": True,
                }
            }
        )
        self.replay_buffer = _FakeBuffer(demo_flag=0)
        self.demo_replay_buffer = _FakeBuffer(demo_flag=0)
        self.use_demo_replay = True
        self.replay_num_workers = 0
        self.demo_replay_num_workers = 0
        self.agent = _FakeAgent()

    def _should_use_epoch_style_replay(self):
        return False

    def _should_use_lazy_replay(self):
        return False


def test_device_merge_engages_under_prefetch_config():
    workspace = _StubWorkspaceFast()
    iterator = workspace.replay_iter
    try:
        assert isinstance(iterator, PrefetchReplayBatchIterator)
        assert isinstance(iterator._replay_iter, _DeviceMergedIterator)
        batch = next(iterator)
        assert batch["low_dim_state"].shape == (8, 3)
        merged_demo = np.asarray(batch["demo"])
        np.testing.assert_array_equal(merged_demo[:4], 0)
        np.testing.assert_array_equal(merged_demo[4:], 1)
    finally:
        iterator.close()


def test_host_merge_kill_switch(monkeypatch):
    monkeypatch.setenv("ROBOBASE_HOST_MERGE", "1")
    workspace = _StubWorkspaceFast()
    iterator = workspace.replay_iter
    try:
        assert isinstance(iterator, PrefetchReplayBatchIterator)
        assert isinstance(iterator._replay_iter, _DemoMergedIterator)
    finally:
        iterator.close()


@pytest.mark.parametrize("dtype", [np.float32, np.uint8, np.int8])
def test_device_and_host_merge_values_match(dtype):
    rng = np.random.default_rng(3)

    def make_batches():
        batch = {
            "x": (rng.standard_normal((4, 2, 3)) * 10).astype(dtype),
            "demo": np.zeros((4,), dtype=np.uint8),
        }
        demo_batch = {
            "x": (rng.standard_normal((4, 2, 3)) * 10).astype(dtype),
            "demo": np.zeros((4,), dtype=np.uint8),
        }
        return batch, demo_batch

    batch_a, demo_a = make_batches()
    batch_b = {k: v.copy() for k, v in batch_a.items()}
    demo_b = {k: v.copy() for k, v in demo_a.items()}

    host = _DemoMergedIterator(iter([batch_a]), iter([demo_a]))
    device = _DeviceMergedIterator(iter([batch_b]), iter([demo_b]))

    host_out = next(host)
    device_out = next(device)

    assert set(host_out.keys()) == set(device_out.keys())
    for key in host_out:
        device_np = np.asarray(device_out[key])
        np.testing.assert_array_equal(
            device_np, host_out[key].astype(device_np.dtype)
        )
