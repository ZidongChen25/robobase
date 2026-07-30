import threading
import time

import numpy as np

from robobase.replay_buffer.iterator import (
    EpochReplayBatchIterator,
    PrefetchReplayBatchIterator,
)


class _IndexReplayBuffer:
    batch_size = 2
    frame_stack = 1
    action_seq = 1

    def episode_index_metadata(self):
        return [(0, 4), (4, 4)]

    def sample_batch_indices(self, indices):
        return {"indices": np.asarray(indices).copy()}


class _SlowEpochReplayBatchIterator(EpochReplayBatchIterator):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._test_sequence = 0
        self.completion_order = []
        self._completion_lock = threading.Lock()
        self._epoch_release_events = {}

    def _reserve_batch_request(self):
        indices = super()._reserve_batch_request()
        sequence_id = self._test_sequence
        self._test_sequence += 1
        epoch = sequence_id // self.batches_per_epoch
        release_event = self._epoch_release_events.setdefault(epoch, threading.Event())
        return sequence_id, indices, release_event

    def _materialize_batch_request(self, request):
        sequence_id, indices, release_event = request
        if sequence_id % self.batches_per_epoch == 0:
            assert release_event.wait(timeout=1.0)
        else:
            time.sleep(0.001)
        batch = self._sample_batch_indices(indices)
        with self._completion_lock:
            self.completion_order.append(sequence_id)
        release_event.set()
        return batch


def test_multithread_prefetch_preserves_epoch_batch_order_across_boundaries():
    replay_buffer = _IndexReplayBuffer()
    expected_iter = EpochReplayBatchIterator(
        replay_buffer,
        execution_length=1,
        shuffle=True,
        drop_last=True,
        seed=17,
    )
    expected = [
        next(expected_iter)["indices"]
        for _ in range(2 * expected_iter.batches_per_epoch)
    ]

    epoch_iter = _SlowEpochReplayBatchIterator(
        replay_buffer,
        execution_length=1,
        shuffle=True,
        drop_last=True,
        seed=17,
    )
    prefetch_iter = PrefetchReplayBatchIterator(
        epoch_iter,
        queue_size=8,
        worker_name="ordered_epoch_test",
        num_workers=4,
    )
    try:
        actual = [next(prefetch_iter) for _ in range(2 * epoch_iter.batches_per_epoch)]
    finally:
        prefetch_iter.close()

    assert [set(batch) for batch in actual] == [{"indices"}] * len(actual)
    for actual_batch, expected_indices in zip(actual, expected, strict=True):
        np.testing.assert_array_equal(actual_batch["indices"], expected_indices)

    expected_set = sorted(epoch_iter.sample_indices.tolist())
    for epoch_start in range(0, len(actual), epoch_iter.batches_per_epoch):
        epoch_batches = actual[epoch_start : epoch_start + epoch_iter.batches_per_epoch]
        epoch_indices = np.concatenate([batch["indices"] for batch in epoch_batches])
        assert len(np.unique(epoch_indices)) == len(epoch_indices)
        assert sorted(epoch_indices.tolist()) == expected_set

    assert epoch_iter.completion_order[0] != 0
    assert epoch_iter.completion_order != sorted(epoch_iter.completion_order)


def test_multithread_prefetch_orders_plain_finite_iterator_without_batch_metadata():
    completion_order = []
    completion_lock = threading.Lock()
    first_request_release = threading.Event()

    def slow_map(batch):
        value = int(batch["value"])
        if value == 0:
            assert first_request_release.wait(timeout=1.0)
        else:
            time.sleep(0.001)
        with completion_lock:
            completion_order.append(value)
        if value == 1:
            first_request_release.set()
        return batch

    source = iter({"value": np.asarray(value)} for value in range(8))
    prefetch_iter = PrefetchReplayBatchIterator(
        source,
        queue_size=8,
        worker_name="ordered_plain_test",
        map_fn=slow_map,
        num_workers=4,
    )

    batches = list(prefetch_iter)

    assert [set(batch) for batch in batches] == [{"value"}] * 8
    assert [int(batch["value"]) for batch in batches] == list(range(8))
    assert completion_order != list(range(8))
