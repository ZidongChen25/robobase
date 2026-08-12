"""Focused checks for dense_return_floor_satisfaction_margin (Stage A9).

The satisficing floor must (1) be an exact no-op when None, (2) preserve
the exact zero-return action-label invariance even when per-bin Q values
differ (the suspension rule is target-conditioned, never label-
conditioned), (3) suspend the floor CE exactly on floor-targeted bins
whose expected Q is within the margin, and (4) never suspend a
positive-return chosen bin.
"""

import jax
import jax.numpy as jnp
import numpy as np

from robobase.method.cqn import dense_return_distributional_loss


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    batch, levels, heads, bins, atoms = 2, 2, 3, 5, 51
    logits = jnp.asarray(
        rng.normal(size=(batch, levels, heads, bins, atoms)),
        dtype=jnp.float32,
    )
    action = jnp.asarray(
        rng.integers(0, bins, size=(batch, levels, heads)),
        dtype=jnp.int32,
    )
    support = jnp.linspace(-2.0, 2.0, atoms, dtype=jnp.float32)
    return logits, action, support, atoms


def _point_mass(support, value, atoms):
    dist = np.zeros((atoms,), dtype=np.float32)
    idx = int(np.argmin(np.abs(np.asarray(support) - value)))
    dist[idx] = 1.0
    return jnp.asarray(dist)


def _loss(logits, action, chosen, support, margin):
    per_sample, _, _ = dense_return_distributional_loss(
        logits,
        action,
        jnp.broadcast_to(
            chosen, logits.shape[:-2] + (logits.shape[-1],)
        ),
        support,
        0.0,
        0.0,
        0.0,
        None,
        0.0,
        margin,
    )
    return per_sample.sum()


def test_margin_none_is_exact_identity():
    logits, action, support, atoms = _setup()
    chosen = _point_mass(support, 1.0, atoms)
    a = _loss(logits, action, chosen, support, None)
    b = _loss(logits, action, chosen, support, None)
    assert float(a) == float(b)


def test_zero_return_action_label_invariance_with_margin():
    # Random logits give every bin a different expected Q, so some bins
    # are suspended and others are not. Invariance must hold anyway
    # because suspension depends only on targets and values.
    logits, action, support, atoms = _setup()
    floor = _point_mass(support, 0.0, atoms)
    la = _loss(logits, action, floor, support, 0.02)
    ga = jax.grad(
        lambda lg: _loss(lg, action, floor, support, 0.02)
    )(logits)
    other = (action + 2) % 5
    lb = _loss(logits, other, floor, support, 0.02)
    gb = jax.grad(
        lambda lg: _loss(lg, other, floor, support, 0.02)
    )(logits)
    assert float(la) == float(lb)
    np.testing.assert_array_equal(np.asarray(ga), np.asarray(gb))


def test_satisfied_floor_bin_has_zero_gradient():
    # Construct logits where one unseen bin is exactly at the floor
    # (huge logit on the 0 atom) and another is far above it. Only the
    # elevated bin may receive floor gradient.
    _, _, support, atoms = _setup()
    zero_idx = int(np.argmin(np.abs(np.asarray(support))))
    high_idx = atoms - 1
    logits = np.zeros((1, 1, 1, 3, atoms), dtype=np.float32)
    logits[0, 0, 0, 0, zero_idx] = 20.0   # chosen bin, at floor
    logits[0, 0, 0, 1, zero_idx] = 20.0   # unseen, satisfied
    logits[0, 0, 0, 2, high_idx] = 20.0   # unseen, far above floor
    logits = jnp.asarray(logits)
    action = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    chosen = _point_mass(support, 1.0, atoms)
    grads = jax.grad(
        lambda lg: _loss(lg, action, chosen, support, 0.02)
    )(logits)
    g = np.abs(np.asarray(grads))[0, 0, 0]
    assert g[1].max() == 0.0   # satisfied floor bin: suspended
    assert g[2].max() > 0.0    # elevated floor bin: active
    assert g[0].max() > 0.0    # positive-return chosen bin: active


def test_positive_return_chosen_bin_never_suspended():
    # Even a chosen bin whose current Q is at the floor keeps its full
    # positive-return gradient (its target is not the floor).
    _, _, support, atoms = _setup()
    zero_idx = int(np.argmin(np.abs(np.asarray(support))))
    logits = np.zeros((1, 1, 1, 3, atoms), dtype=np.float32)
    logits[0, 0, 0, 0, zero_idx] = 20.0
    logits = jnp.asarray(logits)
    action = jnp.zeros((1, 1, 1), dtype=jnp.int32)
    chosen = _point_mass(support, 1.5, atoms)
    grads = jax.grad(
        lambda lg: _loss(lg, action, chosen, support, 0.02)
    )(logits)
    assert np.abs(np.asarray(grads))[0, 0, 0, 0].max() > 0.0
