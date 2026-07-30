"""Throughput-optimized Workspace variant (opt-in via ``train_fast.py``).

Profile-driven (py-spy, cqn-flow.md 41): the stock pipeline already
prefetches replay batches on a background thread, and the measured
main-thread hot spot is the host-side online+demo batch merge
(``np.concatenate`` over ~130MB per update, ~30% of total CPU).  Changes
relative to :class:`robobase.workspace.Workspace` (untouched):

A'. **Device-side demo merge** -- both batches are ``device_put`` and
    concatenated on the accelerator instead of the host.  Bit-identical
    values (concatenation is exact, same layout), removes the largest
    GIL-serialized memcopy from the main thread.

B.  **Async dispatch** -- ``backend.update_block_every_steps`` defaults
    to 10 (numerically identical with uniform replay; metric fetches
    already only happen on logging steps).

C.  **No wandb, no eval videos** -- both forced off.
"""

from omegaconf import open_dict

from robobase.workspace import Workspace, _DemoMergedIterator


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

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            base_iterator = Workspace.replay_iter.fget(self)
            if isinstance(base_iterator, _DemoMergedIterator):
                self._replay_iter = _DeviceMergedIterator(
                    base_iterator.replay_iter,
                    base_iterator.demo_replay_iter,
                )
        return self._replay_iter
