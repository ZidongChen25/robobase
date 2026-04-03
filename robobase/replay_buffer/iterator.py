from __future__ import annotations

import logging
import multiprocessing as mp
import queue
import random
import threading
import traceback
from typing import Any, Iterator

import numpy as np

from robobase.replay_buffer.replay_buffer import ReplayBuffer

try:
    import torch as _torch
    _TORCH_AVAILABLE = True
except ImportError:
    _torch = None
    _TORCH_AVAILABLE = False


_REPLAY_WORKER_ID: int | None = None


def set_replay_worker_id(worker_id: int | None):
    global _REPLAY_WORKER_ID
    _REPLAY_WORKER_ID = worker_id


def get_replay_worker_id(default: int = 0) -> int:
    if _REPLAY_WORKER_ID is not None:
        return _REPLAY_WORKER_ID
    try:
        worker_info = _torch.utils.data.get_worker_info() if _TORCH_AVAILABLE else None
    except Exception:
        worker_info = None
    if worker_info is None:
        return default
    return int(worker_info.id)


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


class _ReplayWorkerError:
    def __init__(self, worker_id: int, traceback_text: str):
        self.worker_id = worker_id
        self.traceback_text = traceback_text


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


def _numpy_to_torch(value: Any, pin_memory: bool):
    if not _TORCH_AVAILABLE:
        raise RuntimeError("torch is required for TorchReplayBatchIterator")
    if _torch.is_tensor(value):
        tensor = value
    elif isinstance(value, np.ndarray):
        tensor = _torch.from_numpy(value)
    else:
        tensor = _torch.as_tensor(value)
    if pin_memory and tensor.device.type == "cpu":
        tensor = tensor.pin_memory()
    return tensor


class TorchReplayBatchIterator(ReplayBatchIterator):
    def __init__(self, replay_iter: Iterator[dict[str, Any]], pin_memory: bool):
        self._replay_iter = replay_iter
        self._pin_memory = pin_memory

    def __next__(self):
        batch = next(self._replay_iter)
        return {
            key: _numpy_to_torch(value, pin_memory=self._pin_memory)
            for key, value in batch.items()
        }

    def close(self):
        close = getattr(self._replay_iter, "close", None)
        if callable(close):
            close()


class PrefetchReplayBatchIterator(ReplayBatchIterator):
    def __init__(
        self,
        replay_iter: Iterator[dict[str, Any]],
        queue_size: int,
        worker_name: str,
    ):
        self._replay_iter = replay_iter
        self._queue = queue.Queue(maxsize=max(2, queue_size))
        self._stop_event = threading.Event()
        self._thread = threading.Thread(
            target=self._prefetch_loop,
            args=(worker_name,),
            daemon=True,
        )
        self._thread.start()

    def _prefetch_loop(self, worker_name: str):
        try:
            while not self._stop_event.is_set():
                batch = next(self._replay_iter)
                while not self._stop_event.is_set():
                    try:
                        self._queue.put(batch, timeout=0.1)
                        break
                    except queue.Full:
                        continue
        except Exception:
            if self._stop_event.is_set():
                return
            error = _ReplayWorkerError(worker_name, traceback.format_exc())
            while not self._stop_event.is_set():
                try:
                    self._queue.put(error, timeout=0.1)
                    break
                except queue.Full:
                    continue

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
        close = getattr(self._replay_iter, "close", None)
        if callable(close):
            close()
        if getattr(self, "_thread", None) is not None and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._stop_event = None


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


def create_torch_replay_iterator(
    replay_buffer: ReplayBuffer,
    *,
    num_workers: int,
    pin_memory: bool,
) -> ReplayBatchIterator:
    return TorchReplayBatchIterator(
        create_numpy_replay_iterator(
            replay_buffer,
            num_workers=num_workers,
            start_method="fork",
        ),
        pin_memory=pin_memory,
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


def normalize_replay_num_workers(
    backend_name: str, requested_num_workers: int
) -> int:
    if backend_name in {"torch", "jax"}:
        return requested_num_workers
    if requested_num_workers > 0:
        logging.warning(
            "backend=%s currently falls back to replay.num_workers=0 because "
            "it does not implement a backend-specific replay prefetch pipeline.",
            backend_name,
        )
    return 0
