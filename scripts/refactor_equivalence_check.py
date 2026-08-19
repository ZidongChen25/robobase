"""CPU-only smoke harness for the R1 CQN/CQN-AS decoupling.

Phase R1 restores ``robobase/method/cqn.py`` and ``robobase/method/cqn_as.py``
to the pristine official JAX port and moves the research implementation to
``cqn_research.py`` / ``cqn_as_research.py``.  This harness proves the split is
wired correctly end to end:

* ``method=cqn_as`` / ``method=cqn`` compose and route to the **research**
  classes (so every existing experiment config keeps behaving as before);
* ``method=cqn_as_official`` / ``method=cqn_official`` compose and route to the
  **pristine** classes;
* every one of those four agents can ``act()`` on a synthetic observation batch
  and run one ``update()`` on a synthetic replay batch, producing finite
  outputs with the expected shapes.

Research-vs-official numeric equality is deliberately NOT asserted: the
research default path adds a NaN guard, extra diagnostics and a ulp-level
dueling reassociation on top of the pristine math.  This is a shape/finite/
routing smoke test only.

Run with::

    JAX_PLATFORMS=cpu CUDA_VISIBLE_DEVICES="" \
        PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor \
        /home/zc1525/robobase_jaxflat/.venv/bin/python \
        scripts/refactor_equivalence_check.py
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from gymnasium import spaces  # noqa: E402
from hydra import compose, initialize_config_dir  # noqa: E402
from hydra.core.global_hydra import GlobalHydra  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from robobase.factory import create_agent  # noqa: E402

CONFIG_DIR = str((REPO_ROOT / "robobase" / "cfgs").resolve())

ACTION_SEQUENCE = 4
ACTION_DIM = 8
LOW_DIM = 5
RGB_KEY = "rgb_head"
# BiGym-shaped pixel observation: [time=1, frame_stack * 3 = 12, 84, 84].
RGB_SHAPE = (1, 12, 84, 84)
BATCH = 2


def _compose(method: str, *overrides: str):
    GlobalHydra.instance().clear()
    with initialize_config_dir(config_dir=CONFIG_DIR, version_base=None):
        return compose(
            config_name="robobase_config",
            overrides=[
                f"method={method}",
                "num_train_envs=2",
                "num_eval_envs=2",
                "num_explore_steps=0",
                "backend.jit=false",
                "backend.platform=cpu",
                "method.model.hidden_dims=[32,32]",
                "method.atoms=11",
                *overrides,
            ],
        )


def _spaces(*, action_sequence: int, pixels: bool):
    obs = {
        "low_dim_state": spaces.Box(
            -np.inf, np.inf, shape=(1, LOW_DIM), dtype=np.float32
        )
    }
    if pixels:
        obs[RGB_KEY] = spaces.Box(0, 255, shape=RGB_SHAPE, dtype=np.uint8)
    action_space = spaces.Box(
        -1.0, 1.0, shape=(action_sequence, ACTION_DIM), dtype=np.float32
    )
    return spaces.Dict(obs), action_space


def _observation(*, pixels: bool):
    rng = np.random.default_rng(3)
    obs = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
    }
    if pixels:
        obs[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return obs


def _batch(*, action_sequence: int, pixels: bool):
    rng = np.random.default_rng(7)
    batch = {
        "low_dim_state": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(np.float32),
        "low_dim_state_tp1": rng.normal(size=(BATCH, 1, LOW_DIM)).astype(
            np.float32
        ),
        "action": rng.uniform(
            -1.0, 1.0, size=(BATCH, action_sequence, ACTION_DIM)
        ).astype(np.float32),
        "reward": rng.normal(size=(BATCH,)).astype(np.float32),
        "discount": np.full((BATCH,), 0.99, dtype=np.float32),
        "terminal": np.zeros((BATCH,), dtype=bool),
        "truncated": np.zeros((BATCH,), dtype=bool),
        "demo": np.ones((BATCH,), dtype=np.uint8),
    }
    if pixels:
        batch[RGB_KEY] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
        batch[f"{RGB_KEY}_tp1"] = rng.integers(
            0, 256, size=(BATCH, *RGB_SHAPE), dtype=np.uint8
        )
    return batch


def _assert_finite(name: str, value) -> None:
    array = np.asarray(value, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise AssertionError(f"{name} contains non-finite values: {array}")


def _check_agent(
    *,
    method: str,
    expected_module: str,
    expected_class: str,
    action_sequence: int,
    pixels: bool,
    overrides: tuple[str, ...] = (),
) -> None:
    started = time.perf_counter()
    cfg = _compose(method, *overrides)
    observation_space, action_space = _spaces(
        action_sequence=action_sequence, pixels=pixels
    )
    agent = create_agent(
        cfg,
        observation_space=observation_space,
        action_space=action_space,
    )

    actual_module = type(agent).__module__
    actual_class = type(agent).__name__
    if actual_module != expected_module or actual_class != expected_class:
        raise AssertionError(
            f"method={method} routed to {actual_module}.{actual_class}, "
            f"expected {expected_module}.{expected_class}"
        )

    observation = _observation(pixels=pixels)
    for eval_mode in (True, False):
        action = agent.act(observation, step=100, eval_mode=eval_mode)
        action = np.asarray(action)
        if action.shape[0] != BATCH:
            raise AssertionError(
                f"method={method} act(eval_mode={eval_mode}) returned batch "
                f"{action.shape[0]}, expected {BATCH}"
            )
        if action.shape[-1] != ACTION_DIM:
            raise AssertionError(
                f"method={method} act(eval_mode={eval_mode}) returned action "
                f"dim {action.shape[-1]}, expected {ACTION_DIM}"
            )
        _assert_finite(f"{method} act(eval_mode={eval_mode})", action)
        act_shape = action.shape

    agent.logging = True
    batch = _batch(action_sequence=action_sequence, pixels=pixels)
    metrics = agent.update(iter([batch]), step=1)
    if "critic_loss" not in metrics:
        raise AssertionError(
            f"method={method} update() returned no critic_loss: "
            f"{sorted(metrics)}"
        )
    for key, value in metrics.items():
        _assert_finite(f"{method} update()[{key}]", value)

    elapsed = time.perf_counter() - started
    print(
        f"  OK  method={method:<17s} -> {actual_module}.{actual_class}  "
        f"act{act_shape}  critic_loss={float(metrics['critic_loss']):.6g}  "
        f"metrics={len(metrics)}  ({elapsed:.1f}s)"
    )


def main() -> int:
    print("R1 refactor equivalence/routing smoke (CPU only)")
    print("-- CQN-AS: pixels + low-dim, action_sequence=4, action_dim=8 --")
    _check_agent(
        method="cqn_as_official",
        expected_module="robobase.method.cqn_as",
        expected_class="CQNAS",
        action_sequence=ACTION_SEQUENCE,
        pixels=True,
        overrides=("pixels=true", f"action_sequence={ACTION_SEQUENCE}"),
    )
    _check_agent(
        method="cqn_as",
        expected_module="robobase.method.cqn_as_research",
        expected_class="CQNAS",
        action_sequence=ACTION_SEQUENCE,
        pixels=True,
        overrides=("pixels=true", f"action_sequence={ACTION_SEQUENCE}"),
    )

    print("-- CQN: low-dim only, action_sequence=1 --")
    _check_agent(
        method="cqn_official",
        expected_module="robobase.method.cqn",
        expected_class="CQN",
        action_sequence=1,
        pixels=False,
        overrides=("action_sequence=1",),
    )
    _check_agent(
        method="cqn",
        expected_module="robobase.method.cqn_research",
        expected_class="CQN",
        action_sequence=1,
        pixels=False,
        overrides=("action_sequence=1",),
    )

    print("All R1 routing/act/update smoke checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
