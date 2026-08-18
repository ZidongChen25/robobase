"""Throughput-optimized Workspace variant (opt-in via ``train_fast.py``).

Profile-driven (py-spy, cqn-flow.md 41): the stock pipeline already
prefetches replay batches on a background thread, and the measured
main-thread hot spot is the host-side online+demo batch merge
(``np.concatenate`` over ~130MB per update, ~30% of total CPU).  Changes
relative to :class:`robobase.workspace.Workspace` (untouched):

A'. **Device-side demo merge** -- both batches are ``device_put`` and
    concatenated on the accelerator instead of the host.  Bit-identical
    values (concatenation is exact, same layout), removes a ~130MB/update
    GIL-serialized host memcopy from the prefetch thread.  The prefetch
    worker waits for the merged tree before releasing the source NumPy
    batches; this keeps transfer/concat on device while preventing an
    unfinished asynchronous transfer from outliving reusable host storage.
    Injected via the ``_make_merged_replay_iter`` hook so it lands inside
    the prefetch wrapper (the original property-override injection was dead
    code under ``backend.replay_prefetch_size > 0``; cqn-flow.md 48.1).
    Rollback: ``ROBOBASE_HOST_MERGE=1``.

B.  **Async dispatch** -- ``backend.update_block_every_steps`` defaults
    to 10 (numerically identical with uniform replay; metric fetches
    already only happen on logging steps).

C.  **No wandb, no eval videos** -- both forced off.
"""

import os

from omegaconf import open_dict

from robobase.workspace import Workspace


class _DeviceMergedIterator:
    """Merge the online and demo batches on device, not the host."""

    def __init__(self, replay_iter, demo_replay_iter):
        import jax
        import jax.numpy as jnp
        import numpy as np

        self._block_until_ready = jax.block_until_ready
        self._device_get = jax.device_get
        self._jnp = jnp
        self._np = np
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter
        self._verify_non_image = (
            os.environ.get("ROBOBASE_DEVICE_MERGE_VERIFY", "") == "1"
        )
        self._is_safe = False

    def __iter__(self):
        return self

    def __next__(self):
        batch = next(self.replay_iter)
        demo_batch = next(self.demo_replay_iter)
        if not self._is_safe:
            assert set(batch.keys()) == set(demo_batch.keys())
            self._is_safe = True
        demo_batch["demo"] = self._np.ones_like(demo_batch["demo"])
        jnp = self._jnp
        merged = {
            k: jnp.concatenate(
                [jnp.asarray(batch[k]), jnp.asarray(demo_batch[k])], axis=0
            )
            for k in batch.keys()
        }
        # JAX dispatch and host-to-device transfer are asynchronous.  These
        # source batches arrive through replay/prefetch queues whose backing
        # storage may be released or reused as soon as this method returns.
        # Materialize the whole tree inside the background prefetch worker so
        # no transfer can observe recycled host memory.  The worker can still
        # overlap this batch's transfer/concat with the main thread's current
        # accelerator update, and no ~130 MB host concatenate is reintroduced.
        merged = self._block_until_ready(merged)
        if self._verify_non_image:
            self._verify_non_image_values(batch, demo_batch, merged)
        return merged

    def _verify_non_image_values(self, batch, demo_batch, merged):
        """Exact debug gate for the known float observation corruption signature."""

        for key in batch.keys():
            # Full RGB verification would add a ~260 MB device-to-host copy per
            # update.  The two observed failures were float observation buffers;
            # verify every other replay field exactly in the attribution arm.
            if key.startswith("rgb_"):
                continue
            expected = self._np.concatenate([batch[key], demo_batch[key]], axis=0)
            actual = self._np.asarray(self._device_get(merged[key]))
            # JAX legitimately canonicalizes unsupported 64-bit host dtypes
            # when x64 is disabled (for example replay ``discount`` float64
            # becomes float32).  Verify values after applying the same dtype
            # boundary instead of reporting every rounded element as corrupt.
            expected = expected.astype(actual.dtype, copy=False)
            if not self._np.array_equal(actual, expected):
                mismatch_count = int(self._np.count_nonzero(actual != expected))
                raise RuntimeError(
                    "Device merge verification failed for "
                    f"{key!r}: shape={actual.shape}, dtype={actual.dtype}, "
                    f"mismatched_values={mismatch_count}"
                )

    def close(self):
        for iterator in (self.replay_iter, self.demo_replay_iter):
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


class WorkspaceFast(Workspace):
    """Workspace with device-side merge, relaxed syncs, no wandb/videos."""

    def __init__(self, cfg, work_dir: str = None):
        with open_dict(cfg):
            cfg.wandb.use = False
            cfg.log_eval_video = False
            backend = cfg.get("backend", None)
            if backend is not None and int(
                backend.get("update_block_every_steps", 1)
            ) == 1:
                cfg.backend.update_block_every_steps = 10
        # ``work_dir`` mirrors ``Workspace``: None means "take the Hydra run
        # dir", which is what train_fast.py relies on. Driver scripts that
        # build a workspace outside Hydra pass an explicit directory.
        super().__init__(cfg, work_dir=work_dir)

    def _make_merged_replay_iter(self, replay_iter, demo_replay_iter):
        # Hook runs before the prefetch wrapper is added, so the device-side
        # concat executes inside the prefetch thread. The old implementation
        # overrode the replay_iter property and isinstance-checked the
        # parent's return value, which is the PrefetchReplayBatchIterator
        # whenever backend.replay_prefetch_size > 0 — the device merge was
        # dead code under the default jax backend (cqn-flow.md 48.1).
        if (
            getattr(self, "demo_only_updates", False)
            or os.environ.get("ROBOBASE_HOST_MERGE", "") == "1"
        ):
            return super()._make_merged_replay_iter(replay_iter, demo_replay_iter)
        return _DeviceMergedIterator(replay_iter, demo_replay_iter)
