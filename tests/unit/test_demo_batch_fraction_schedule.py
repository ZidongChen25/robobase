"""Focused checks for demo_batch_fraction_schedule (cqn-rline.md wave-3 D2).

The schedule must (1) be an exact no-op when null, (2) keep the total batch
constant while moving the demo/online split per step, (3) clamp to at least
one sample per side, (4) reject unsupported configurations loudly.
"""

from types import SimpleNamespace

import pytest

from robobase.workspace import Workspace


class _FakeBuffer:
    def __init__(self, batch_size):
        self._batch_size = batch_size

    @property
    def batch_size(self):
        return self._batch_size

    def set_batch_size(self, batch_size):
        batch_size = int(batch_size)
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        self._batch_size = batch_size


class _FakeCfg(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _fake_workspace(schedule, step=0, num_workers=0, demo_buffer=True):
    ws = SimpleNamespace()
    ws.cfg = _FakeCfg(
        batch_size=256,
        demo_batch_size=256,
        demo_batch_fraction_schedule=schedule,
        replay=_FakeCfg(num_workers=num_workers),
    )
    ws.replay_buffer = _FakeBuffer(256)
    ws.demo_replay_buffer = _FakeBuffer(256) if demo_buffer else None
    ws.pretrain_steps = 0
    ws.main_loop_iterations = step
    ws._apply_demo_batch_fraction_schedule = (
        Workspace._apply_demo_batch_fraction_schedule.__get__(ws)
    )
    return ws


def test_null_schedule_is_exact_noop():
    ws = _fake_workspace(None)
    assert ws._apply_demo_batch_fraction_schedule() is None
    assert ws.replay_buffer.batch_size == 256
    assert ws.demo_replay_buffer.batch_size == 256


def test_linear_schedule_moves_split_at_constant_total():
    spec = "linear(0.75,0.25,1000)"
    start = _fake_workspace(spec, step=0)
    fraction = start._apply_demo_batch_fraction_schedule()
    assert fraction == pytest.approx(0.75)
    assert start.demo_replay_buffer.batch_size == 384
    assert start.replay_buffer.batch_size == 128
    assert (
        start.demo_replay_buffer.batch_size + start.replay_buffer.batch_size
        == 512
    )

    mid = _fake_workspace(spec, step=500)
    mid._apply_demo_batch_fraction_schedule()
    assert mid.demo_replay_buffer.batch_size == 256
    assert mid.replay_buffer.batch_size == 256

    end = _fake_workspace(spec, step=100000)
    fraction = end._apply_demo_batch_fraction_schedule()
    assert fraction == pytest.approx(0.25)
    assert end.demo_replay_buffer.batch_size == 128
    assert end.replay_buffer.batch_size == 384


def test_extreme_fractions_keep_one_sample_per_side():
    low = _fake_workspace("linear(0.0,0.0,1)", step=5)
    low._apply_demo_batch_fraction_schedule()
    assert low.demo_replay_buffer.batch_size == 1
    assert low.replay_buffer.batch_size == 511

    high = _fake_workspace("linear(1.0,1.0,1)", step=5)
    high._apply_demo_batch_fraction_schedule()
    assert high.demo_replay_buffer.batch_size == 511
    assert high.replay_buffer.batch_size == 1


def test_requires_demo_buffer():
    ws = _fake_workspace("linear(0.75,0.25,1000)", demo_buffer=False)
    with pytest.raises(ValueError, match="demo_batch_size"):
        ws._apply_demo_batch_fraction_schedule()


def test_requires_single_process_replay():
    ws = _fake_workspace("linear(0.75,0.25,1000)", num_workers=2)
    with pytest.raises(ValueError, match="num_workers"):
        ws._apply_demo_batch_fraction_schedule()
