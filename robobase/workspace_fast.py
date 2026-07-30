"""Throughput-optimized Workspace variant (opt-in via ``train_fast.py``).

Profile-driven (py-spy, cqn-flow.md 41): the stock pipeline already
prefetches replay batches on a background thread, and the measured
main-thread hot spot is the host-side online+demo batch merge
(``np.concatenate`` over ~130MB per update, ~30% of total CPU).  Changes
relative to :class:`robobase.workspace.Workspace` (untouched):

A'. **Device-side demo merge** -- both batches are ``device_put`` and
    concatenated on the accelerator instead of the host.  Bit-identical
    values (concatenation is exact, same layout), removes a ~130MB/update
    GIL-serialized host memcopy from the prefetch thread.  Injected via
    the ``_make_merged_replay_iter`` hook so it lands inside the prefetch
    wrapper (the original property-override injection was dead code under
    ``backend.replay_prefetch_size > 0``; cqn-flow.md 48.1).  Rollback:
    ``ROBOBASE_HOST_MERGE=1``.

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
        import jax.numpy as jnp
        import numpy as np

        self._jnp = jnp
        self._np = np
        self.replay_iter = replay_iter
        self.demo_replay_iter = demo_replay_iter
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
        return {
            k: jnp.concatenate(
                [jnp.asarray(batch[k]), jnp.asarray(demo_batch[k])], axis=0
            )
            for k in batch.keys()
        }

    def close(self):
        for iterator in (self.replay_iter, self.demo_replay_iter):
            close = getattr(iterator, "close", None)
            if callable(close):
                close()


class WorkspaceFast(Workspace):
    """Workspace with device-side merge, relaxed syncs, no wandb/videos."""

    def __init__(self, cfg):
        with open_dict(cfg):
            cfg.wandb.use = False
            cfg.log_eval_video = False
            backend = cfg.get("backend", None)
            if backend is not None and int(
                backend.get("update_block_every_steps", 1)
            ) == 1:
                cfg.backend.update_block_every_steps = 10
        super().__init__(cfg)

    def _make_merged_replay_iter(self, replay_iter, demo_replay_iter):
        # Hook runs before the prefetch wrapper is added, so the device-side
        # concat executes inside the prefetch thread. The old implementation
        # overrode the replay_iter property and isinstance-checked the
        # parent's return value, which is the PrefetchReplayBatchIterator
        # whenever backend.replay_prefetch_size > 0 — the device merge was
        # dead code under the default jax backend (cqn-flow.md 48.1).
        if os.environ.get("ROBOBASE_HOST_MERGE", "") == "1":
            return super()._make_merged_replay_iter(replay_iter, demo_replay_iter)
        return _DeviceMergedIterator(replay_iter, demo_replay_iter)
