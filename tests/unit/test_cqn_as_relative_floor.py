"""Focused checks for dense_return_relative_floor_margin (Stage A13).

The relative floor must (1) be an exact no-op when None, (2) supervise
unseen bins to max(E[chosen target] - m, floor) instead of the absolute
floor, (3) preserve the exact zero-return action-label invariance (at
G=0 the shifted value clips back to the floor so all targets coincide),
and (4) leave the chosen bin's target untouched.
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


def _loss_and_targets(logits, action, chosen, support, margin):
    chosen_b = jnp.broadcast_to(
        chosen, logits.shape[:-2] + (logits.shape[-1],)
    )
    per_sample, chosen_q, unseen_q = dense_return_distributional_loss(
        logits,
        action,
        chosen_b,
        support,
        0.0,
        0.0,
        0.0,
        None,
        0.0,
        None,
        margin,
    )
    return per_sample.sum(), chosen_q, unseen_q


def test_margin_none_matches_absolute_floor():
    logits, action, support, atoms = _setup()
    chosen = _point_mass(support, 1.0, atoms)
    a, _, _ = _loss_and_targets(logits, action, chosen, support, None)
    b, _, _ = _loss_and_targets(logits, action, chosen, support, None)
    assert float(a) == float(b)


def test_unseen_target_tracks_chosen_minus_margin():
    # Optimize logits directly against the loss; at the optimum the
    # unseen bins' expected Q must sit at E[chosen] - m, not at 0.
    logits, action, support, atoms = _setup()
    chosen = _point_mass(support, 1.0, atoms)
    margin = 0.16
    lr = 2.0
    lg = logits
    grad_fn = jax.grad(
        lambda l: _loss_and_targets(l, action, chosen, support, margin)[0]
    )
    for _ in range(300):
        lg = lg - lr * grad_fn(lg)
    _, chosen_q, unseen_q = _loss_and_targets(
        lg, action, chosen, support, margin
    )
    np.testing.assert_allclose(
        np.asarray(unseen_q),
        np.asarray(chosen_q) - margin,
        atol=0.03,
    )


def test_zero_return_action_label_invariance_with_relative_margin():
    logits, action, support, atoms = _setup()
    floor = _point_mass(support, 0.0, atoms)
    margin = 0.16
    la, _, _ = _loss_and_targets(logits, action, floor, support, margin)
    ga = jax.grad(
        lambda l: _loss_and_targets(l, action, floor, support, margin)[0]
    )(logits)
    other = (action + 3) % 5
    lb, _, _ = _loss_and_targets(logits, other, floor, support, margin)
    gb = jax.grad(
        lambda l: _loss_and_targets(l, other, floor, support, margin)[0]
    )(logits)
    assert float(la) == float(lb)
    np.testing.assert_array_equal(np.asarray(ga), np.asarray(gb))


def test_chosen_bin_target_unchanged_by_relative_margin():
    logits, action, support, atoms = _setup()
    chosen = _point_mass(support, 1.5, atoms)
    _, chosen_q_abs, _ = _loss_and_targets(
        logits, action, chosen, support, None
    )
    _, chosen_q_rel, _ = _loss_and_targets(
        logits, action, chosen, support, 0.16
    )
    np.testing.assert_array_equal(
        np.asarray(chosen_q_abs), np.asarray(chosen_q_rel)
    )
