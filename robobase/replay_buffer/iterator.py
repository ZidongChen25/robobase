from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import random
import threading
import traceback
from dataclasses import dataclass
from typing import Any, Iterator

import numpy as np

from robobase.replay_buffer.replay_buffer import ReplayBuffer

_REPLAY_WORKER_ID: int | None = None


def set_replay_worker_id(worker_id: int | None):
    global _REPLAY_WORKER_ID
    _REPLAY_WORKER_ID = worker_id


def get_replay_worker_id(default: int = 0) -> int:
    if _REPLAY_WORKER_ID is not None:
        return _REPLAY_WORKER_ID
    return default


class ReplayBatchIterator:
    def __iter__(self):
        return self

    def close(self):
        pass


class SingleProcessReplayBatchIterator(ReplayBatchIterator):
    def __init__(self, replay_buffer: ReplayBuffer):
        self._replay_buffer = replay_buffer

    def __next__(self):
        return self._replay_buffer.sample(self._replay_buffer.batch_size)


class EpochReplayBatchIterator(ReplayBatchIterator):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        *,
        execution_length: int,
        shuffle: bool = True,
        drop_last: bool = True,
        seed: int | None = None,
        load_all_episodes: bool = False,
        batch_chunk_size: int | None = None,
    ):
        if execution_length < 1:
            raise ValueError("execution_length must be >= 1.")

        self._replay_buffer = replay_buffer
        self._batch_size = int(replay_buffer.batch_size)
        self._shuffle = shuffle
        self._drop_last = drop_last
        self._rng = np.random.default_rng(seed)
        self._batch_chunk_size = (
            0 if batch_chunk_size is None else int(batch_chunk_size)
        )
        if self._batch_chunk_size < 0:
            raise ValueError("batch_chunk_size must be >= 0 or null.")
        if self._batch_chunk_size > self._batch_size:
            self._batch_chunk_size = self._batch_size

        if load_all_episodes:
            replay_buffer.load_all_episodes()
        else:
            logging.info(
                "Epoch replay iterator will stream replay episodes from disk "
                "instead of eagerly loading the full dataset into RAM."
            )
        self._episode_indices = self._build_episode_sample_indices(
            replay_buffer,
            execution_length=execution_length,
        )
        self._all_indices = (
            np.concatenate(self._episode_indices)
            if self._episode_indices
            else np.asarray([], dtype=np.int32)
        )
        if self._all_indices.size == 0:
            raise ValueError(
                "Epoch replay iterator found no valid samples in the replay buffer."
            )
        if self._drop_last and self._all_indices.size < self._batch_size:
            raise ValueError(
                "Epoch replay iterator requires at least one full batch, but "
                f"only found {self._all_indices.size} samples for batch_size="
                f"{self._batch_size}."
            )
        if self._drop_last and self._all_indices.size % self._batch_size != 0:
            logging.warning(
                "Epoch replay iterator is dropping the final incomplete batch: "
                "%d samples with batch_size=%d.",
                self._all_indices.size,
                self._batch_size,
            )
        if self._batch_chunk_size > 0:
            logging.info(
                "Epoch replay iterator will build locality-aware batches with "
                "batch_chunk_size=%d.",
                self._batch_chunk_size,
            )

        self._epoch_indices = self._all_indices.copy()
        self._epoch_batches = None
        self._cursor = 0
        self._cursor_lock = threading.Lock()
        self._reshuffle()

    @staticmethod
    def _build_episode_sample_indices(
        replay_buffer: ReplayBuffer,
        *,
        execution_length: int,
    ) -> list[np.ndarray]:
        frame_stack = int(getattr(replay_buffer, "frame_stack", 1))
        action_seq = int(getattr(replay_buffer, "action_seq", 1))
        episode_indices = []
        for global_start, episode_length in replay_buffer.episode_index_metadata():
            max_valid_index = min(
                episode_length - 1,
                episode_length - action_seq + execution_length + frame_stack - 2,
            )
            if max_valid_index < 0:
                continue
            episode_indices.append(
                np.arange(
                    global_start,
                    global_start + max_valid_index + 1,
                    dtype=np.int32,
                )
            )
        return episode_indices

    @staticmethod
    def _build_sample_indices(
        replay_buffer: ReplayBuffer,
        *,
        execution_length: int,
    ) -> np.ndarray:
        episode_indices = EpochReplayBatchIterator._build_episode_sample_indices(
            replay_buffer,
            execution_length=execution_length,
        )
        if not episode_indices:
            return np.asarray([], dtype=np.int32)
        return np.concatenate(episode_indices)

    def _reshuffle(self):
        if self._batch_chunk_size > 0:
            self._epoch_batches = self._build_locality_aware_batches()
            self._cursor = 0
            return

        self._epoch_batches = None
        self._epoch_indices = self._all_indices.copy()
        if self._shuffle:
            self._rng.shuffle(self._epoch_indices)
        self._cursor = 0

    def _build_locality_aware_batches(self) -> list[np.ndarray]:
        chunks = []
        for episode_indices in self._episode_indices:
            indices = episode_indices.copy()
            if self._shuffle:
                self._rng.shuffle(indices)
            for start in range(0, indices.size, self._batch_chunk_size):
                chunks.append(indices[start : start + self._batch_chunk_size])

        if self._shuffle and len(chunks) > 1:
            order = self._rng.permutation(len(chunks))
            chunks = [chunks[index] for index in order]

        batches = []
        current = []
        current_size = 0
        for chunk in chunks:
            offset = 0
            while offset < chunk.size:
                needed = self._batch_size - current_size
                take = min(needed, chunk.size - offset)
                current.append(chunk[offset : offset + take])
                current_size += take
                offset += take

                if current_size == self._batch_size:
                    batches.append(np.concatenate(current).astype(np.int32, copy=False))
                    current = []
                    current_size = 0

        if current_size and not self._drop_last:
            batches.append(np.concatenate(current).astype(np.int32, copy=False))
        return batches

    @property
    def sample_indices(self) -> np.ndarray:
        return self._all_indices.copy()

    @property
    def batches_per_epoch(self) -> int:
        if self._drop_last:
            return self._all_indices.size // self._batch_size
        return int(np.ceil(self._all_indices.size / self._batch_size))

    def __next__(self):
        request = self._reserve_batch_request()
        return self._materialize_batch_request(request)

    def _reserve_batch_request(self) -> np.ndarray:
        """Reserve the next batch's indices without doing replay I/O."""
        return self._next_batch_indices()

    def _materialize_batch_request(self, request: np.ndarray):
        return self._sample_batch_indices(request)

    def _next_batch_indices(self) -> np.ndarray:
        with self._cursor_lock:
            return self._next_batch_indices_unlocked()

    def _next_batch_indices_unlocked(self) -> np.ndarray:
        if self._epoch_batches is not None:
            if self._cursor >= len(self._epoch_batches):
                self._reshuffle()
            batch_indices = self._epoch_batches[self._cursor]
            self._cursor += 1
            return batch_indices

        if self._cursor >= self._epoch_indices.size:
            self._reshuffle()

        batch_end = self._cursor + self._batch_size
        if batch_end > self._epoch_indices.size:
            if self._drop_last:
                self._reshuffle()
                batch_end = self._batch_size
            else:
                batch_end = self._epoch_indices.size

        batch_indices = self._epoch_indices[self._cursor : batch_end]
        self._cursor = batch_end
        return batch_indices

    def _sample_batch_indices(self, batch_indices: np.ndarray):
        sample_batch_indices = getattr(
            self._replay_buffer, "sample_batch_indices", None
        )
        if callable(sample_batch_indices):
            return sample_batch_indices(batch_indices)
        return self._replay_buffer.sample(
            batch_size=batch_indices.shape[0],
            indices=batch_indices.tolist(),
        )


class _ReplayWorkerError:
    def __init__(self, worker_id: int, traceback_text: str):
        self.worker_id = worker_id
        self.traceback_text = traceback_text


@dataclass(frozen=True, slots=True)
class _PrefetchRequest:
    sequence_id: int
    payload: Any
    deferred: bool


@dataclass(frozen=True, slots=True)
class _PrefetchBatch:
    sequence_id: int
    batch: Any


@dataclass(frozen=True, slots=True)
class _PrefetchError:
    sequence_id: int
    worker_name: str
    traceback_text: str


@dataclass(frozen=True, slots=True)
class _PrefetchEnd:
    sequence_id: int


def _replay_worker_loop(
    replay_buffer: ReplayBuffer,
    worker_id: int,
    out_queue,
    stop_event,
    seed: int,
):
    set_replay_worker_id(worker_id)
    np.random.seed(seed)
    random.seed(seed)
    try:
        while not stop_event.is_set():
            batch = replay_buffer.sample(replay_buffer.batch_size)
            while not stop_event.is_set():
                try:
                    out_queue.put(batch, timeout=0.1)
                    break
                except queue.Full:
                    continue
    except Exception:
        out_queue.put(_ReplayWorkerError(worker_id, traceback.format_exc()))


class MultiProcessReplayBatchIterator(ReplayBatchIterator):
    def __init__(
        self,
        replay_buffer: ReplayBuffer,
        num_workers: int,
        queue_size: int | None = None,
        start_method: str = "fork",
    ):
        if num_workers < 1:
            raise ValueError("num_workers must be >= 1 for multi-process replay.")

        ctx = mp.get_context(start_method)
        self._queue = ctx.Queue(maxsize=queue_size or max(2, num_workers * 2))
        self._stop_event = ctx.Event()
        self._workers = []

        seed_base = int(np.random.get_state()[1][0])
        for worker_id in range(num_workers):
            process = ctx.Process(
                target=_replay_worker_loop,
                args=(
                    replay_buffer,
                    worker_id,
                    self._queue,
                    self._stop_event,
                    seed_base + worker_id,
                ),
            )
            process.daemon = True
            process.start()
            self._workers.append(process)

    def __next__(self):
        item = self._queue.get()
        if isinstance(item, _ReplayWorkerError):
            self.close()
            raise RuntimeError(
                f"Replay worker {item.worker_id} failed.\n{item.traceback_text}"
            )
        return item

    def close(self):
        if getattr(self, "_stop_event", None) is None:
            return
        self._stop_event.set()
        for worker in self._workers:
            worker.join(timeout=1.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=1.0)
        self._workers.clear()
        self._stop_event = None
        try:
            self._queue.close()
        except Exception:
            pass


class PrefetchReplayBatchIterator(ReplayBatchIterator):
    def __init__(
        self,
        replay_iter: Iterator[dict[str, Any]],
        queue_size: int,
        worker_name: str,
        map_fn=None,
        num_workers: int = 1,
    ):
        self._replay_iter = replay_iter
        self._map_fn = map_fn
        prefetch_capacity = max(2, queue_size)
        self._queue = queue.Queue(maxsize=prefetch_capacity)
        self._request_slots = threading.BoundedSemaphore(prefetch_capacity)
        self._stop_event = threading.Event()
        self._source_lock = threading.Lock()
        self._source_exhausted = False
        self._next_request_sequence = 0
        self._next_output_sequence = 0
        self._pending_results = {}
        self._closed = False
        self._finished = False
        reserve_batch = getattr(replay_iter, "_reserve_batch_request", None)
        materialize_batch = getattr(replay_iter, "_materialize_batch_request", None)
        if callable(reserve_batch) and callable(materialize_batch):
            self._reserve_batch = reserve_batch
            self._materialize_batch = materialize_batch
        else:
            self._reserve_batch = None
            self._materialize_batch = None
        self._threads = []
        for worker_idx in range(max(1, int(num_workers))):
            thread = threading.Thread(
                target=self._prefetch_loop,
                args=(f"{worker_name}:{worker_idx}",),
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _reserve_request(self, worker_name: str):
        with self._source_lock:
            if self._source_exhausted:
                return None

            sequence_id = self._next_request_sequence
            self._next_request_sequence += 1
            try:
                if self._reserve_batch is not None:
                    payload = self._reserve_batch()
                    return _PrefetchRequest(sequence_id, payload, deferred=True)
                batch = next(self._replay_iter)
                return _PrefetchRequest(sequence_id, batch, deferred=False)
            except StopIteration:
                self._source_exhausted = True
                return _PrefetchEnd(sequence_id)
            except Exception:
                self._source_exhausted = True
                return _PrefetchError(
                    sequence_id,
                    worker_name,
                    traceback.format_exc(),
                )

    def _put_result(self, result):
        while not self._stop_event.is_set():
            try:
                self._queue.put(result, timeout=0.1)
                return
            except queue.Full:
                continue

    def _acquire_request_slot(self) -> bool:
        while not self._stop_event.is_set():
            if self._request_slots.acquire(timeout=0.1):
                return True
        return False

    def _prefetch_loop(self, worker_name: str):
        while not self._stop_event.is_set():
            if not self._acquire_request_slot():
                return
            request = self._reserve_request(worker_name)
            if request is None:
                self._request_slots.release()
                return
            if isinstance(request, (_PrefetchEnd, _PrefetchError)):
                self._put_result(request)
                return

            try:
                batch = request.payload
                if request.deferred:
                    batch = self._materialize_batch(batch)
                if self._map_fn is not None:
                    batch = self._map_fn(batch)
                result = _PrefetchBatch(request.sequence_id, batch)
            except Exception:
                with self._source_lock:
                    self._source_exhausted = True
                result = _PrefetchError(
                    request.sequence_id,
                    worker_name,
                    traceback.format_exc(),
                )
            self._put_result(result)
            if isinstance(result, _PrefetchError):
                return

    def __next__(self):
        if self._finished or self._closed:
            raise StopIteration

        while self._next_output_sequence not in self._pending_results:
            item = self._queue.get()
            self._pending_results[item.sequence_id] = item

        item = self._pending_results.pop(self._next_output_sequence)
        self._next_output_sequence += 1
        self._request_slots.release()
        if isinstance(item, _PrefetchEnd):
            self._finished = True
            self.close()
            raise StopIteration
        if isinstance(item, _PrefetchError):
            self.close()
            raise RuntimeError(
                f"Replay worker {item.worker_name} failed.\n{item.traceback_text}"
            )
        return item.batch

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._stop_event.set()
        close = getattr(self._replay_iter, "close", None)
        if callable(close):
            close()
        for thread in getattr(self, "_threads", []):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads = []


def create_numpy_replay_iterator(
    replay_buffer: ReplayBuffer, *, num_workers: int, start_method: str = "fork"
) -> ReplayBatchIterator:
    if num_workers <= 0:
        return SingleProcessReplayBatchIterator(replay_buffer)
    return MultiProcessReplayBatchIterator(
        replay_buffer,
        num_workers=num_workers,
        start_method=start_method,
    )


def create_jax_replay_iterator(
    replay_buffer: ReplayBuffer,
    *,
    num_workers: int,
) -> ReplayBatchIterator:
    return create_numpy_replay_iterator(
        replay_buffer,
        num_workers=num_workers,
        start_method="spawn",
    )


def create_epoch_replay_iterator(
    replay_buffer: ReplayBuffer,
    *,
    execution_length: int,
    shuffle: bool = True,
    seed: int | None = None,
    load_all_episodes: bool = False,
    batch_chunk_size: int | None = None,
) -> ReplayBatchIterator:
    return EpochReplayBatchIterator(
        replay_buffer,
        execution_length=execution_length,
        shuffle=shuffle,
        drop_last=True,
        seed=seed,
        load_all_episodes=load_all_episodes,
        batch_chunk_size=batch_chunk_size,
    )


def normalize_replay_num_workers(backend_name: str, requested_num_workers: int) -> int:
    if backend_name == "jax":
        return requested_num_workers
    if requested_num_workers > 0:
        logging.warning(
            "backend=%s currently falls back to replay.num_workers=0 because "
            "it does not implement a backend-specific replay prefetch pipeline.",
            backend_name,
        )
    return 0
