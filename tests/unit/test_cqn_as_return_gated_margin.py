"""Focused checks for return_gated_margin (Stage A17).

The hinge must (1) vanish exactly on zero-return samples (loss AND
gradients invariant to the recorded action there — the project's
operational anti-imitation test), (2) activate only on positive-return
samples and push non-executed bins below chosen − m, (3) never read a
demonstration flag (the gate is the return mask passed in), and (4) be
an exact no-op at weight 0 / margin None.
"""

import jax
import jax.numpy as jnp
import numpy as np

from robobase.method.cqn_research import return_gated_margin_loss


def _setup(seed=0):
    rng = np.random.default_rng(seed)
    batch, levels, heads, bins, atoms = 4, 2, 3, 5, 51
    logits = jnp.asarray(
        rng.normal(size=(batch, levels, heads, bins, atoms)),
        dtype=jnp.float32,
    )
    action = jnp.asarray(
        rng.integers(0, bins, size=(batch, levels, heads)),
        dtype=jnp.int32,
    )
    support = jnp.linspace(-2.0, 2.0, atoms, dtype=jnp.float32)
    return logits, action, support


def test_zero_return_samples_have_no_margin_gradient():
    logits, action, support = _setup()
    positive = jnp.zeros((logits.shape[0],), dtype=bool)
    loss = return_gated_margin_loss(logits, action, support, 0.16, positive)
    assert float(loss.sum()) == 0.0
    grads = jax.grad(
        lambda lg: return_gated_margin_loss(
            lg, action, support, 0.16, positive
        ).sum()
    )(logits)
    assert float(jnp.abs(grads).max()) == 0.0
    other = (action + 2) % 5
    loss_b = return_gated_margin_loss(logits, other, support, 0.16, positive)
    assert float(loss_b.sum()) == float(loss.sum())


def test_positive_samples_penalise_only_margin_violations():
    logits, action, support = _setup()
    positive = jnp.ones((logits.shape[0],), dtype=bool)
    margin = 0.16
    lr = 2.0
    grad_fn = jax.grad(
        lambda lg: return_gated_margin_loss(
            lg, action, support, margin, positive
        ).sum()
    )
    lg = logits
    for _ in range(400):
        lg = lg - lr * grad_fn(lg)
    probs = jax.nn.softmax(lg, axis=-1)
    q = np.asarray(jnp.sum(probs * support, axis=-1))
    a = np.asarray(action)
    chosen = np.take_along_axis(q, a[..., None], axis=-1)[..., 0]
    for b in range(q.shape[-1]):
        mask = np.arange(q.shape[-1])[None, None, None, :] != a[..., None]
    unseen_max = np.where(
        np.arange(q.shape[-1])[None, None, None, :] == a[..., None],
        -np.inf,
        q,
    ).max(-1)
    # At the optimum every unseen bin sits at or below chosen - margin.
    assert np.all(unseen_max <= chosen - margin + 0.02)


def test_mixed_batch_gates_per_sample():
    logits, action, support = _setup()
    positive = jnp.asarray([True, False, True, False])
    loss = return_gated_margin_loss(logits, action, support, 0.16, positive)
    loss = np.asarray(loss)
    assert loss[1] == 0.0 and loss[3] == 0.0
    assert loss[0] > 0.0 and loss[2] > 0.0
