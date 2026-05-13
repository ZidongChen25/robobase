# JAX Backend Branch

## Branch Strategy

This branch is the JAX backend branch.

The project now uses two branches to support the two framework implementations:

1. A PyTorch branch keeps the original PyTorch implementation and PyTorch support.
2. This branch develops the JAX implementation.

This means the goal of this branch is no longer to maintain a fully symmetric
PyTorch/JAX backend abstraction inside one code tree. The JAX branch can simplify
or replace PyTorch-specific paths when that makes the JAX implementation cleaner.

## What This Branch Is

This branch is for developing RoboBase with JAX as the primary training backend.
The current default config selects JAX:

```yaml
defaults:
  - backend: jax
```

The JAX backend config lives in:

```text
robobase/cfgs/backend/jax.yaml
```

The current method implementations that matter for this branch are:

```text
robobase/method/bc.py
robobase/method/diffusion.py
robobase/method/jax_base.py
robobase/method/jax_utils.py
```

The JAX model components live under:

```text
robobase/models/diffusion.py
robobase/models/encoder.py
robobase/models/fully_connected.py
robobase/models/fusion.py
```

## What This Branch Is Not

This branch is not intended to preserve the original PyTorch implementation as a
first-class backend.

The previous plan was to restructure the repository into a single shared runtime
with parallel backend implementations such as:

```text
robobase/backends/torch/...
robobase/backends/jax/...
```

That is no longer the current branch strategy. PyTorch support should stay on
the PyTorch branch. JAX support should be developed here.

Some files may still contain compatibility code, old comments, or historical
PyTorch assumptions. Those should be treated as migration leftovers unless they
are still required by the JAX path.

## Current JAX Status

### Behaviour Cloning

`method=bc` is implemented with JAX/Flax/Optax.

The implementation is in:

```text
robobase/method/bc.py
```

Supported BC surfaces include:

- low-dimensional offline BC
- recurrent or MLP action-sequence output
- pixel BC through a frozen JAX ResNet encoder
- multi-camera feature fusion

The image encoder path uses:

```text
robobase/models/encoder.py
```

The current encoder is intentionally narrow:

- frozen pretrained ResNet feature extraction
- `resnet18` and `resnet34`
- weights are imported from `timm` at construction time and converted for the
  JAX/Flax feature model

### Diffusion Policy

`method=diffusion` is implemented with JAX/Flax/Optax.

The implementation is in:

```text
robobase/method/diffusion.py
robobase/models/diffusion.py
```

The current Diffusion Policy path includes:

- JAX conditional 1-D U-Net actor
- cosine diffusion schedule
- JIT-compiled update and sampling functions
- optional EMA parameters
- prioritized replay priority updates from diffusion loss
- low-dimensional observations
- pixel observations through the same frozen JAX ResNet feature path

The method config is:

```text
robobase/cfgs/method/diffusion.yaml
```

## Runtime Structure

The high-level runtime is still shared through:

```text
train.py
robobase/workspace.py
robobase/factory.py
```

The construction path is:

1. `train.py` loads Hydra config and applies GPU selection.
2. `Workspace` creates envs, replay buffers, logger, and agent.
3. `robobase/factory.py` maps `method=bc` or `method=diffusion` to the JAX method
   implementation.
4. JAX methods consume NumPy batches and move arrays into JAX at the method
   boundary.

Replay iteration is handled by:

```text
robobase/replay_buffer/iterator.py
```

The preferred JAX data boundary is:

```text
env / replay -> NumPy -> JAX method
```

Avoid adding new PyTorch tensor assumptions to replay or workspace code on this
branch unless they are needed for temporary compatibility.

## Development Rules For This Branch

Use JAX-native implementations for new method and model work.

Prefer:

- `jax`
- `jax.numpy`
- `flax.linen`
- `optax`
- NumPy at environment/replay boundaries

Avoid:

- adding new `torch.nn.Module` method implementations
- adding new PyTorch model targets to shared configs
- designing new code around a fake common tensor abstraction
- reintroducing a large in-branch PyTorch backend layer

If functionality exists only in the PyTorch branch, port the behavior to JAX
instead of importing the PyTorch implementation.

## Config Direction

Method configs on this branch should describe algorithm-level settings and JAX
model specs rather than PyTorch `_target_` classes.

For example, Diffusion Policy uses:

```yaml
method:
  name: diffusion
  actor_model:
    type: conditional_unet1d
  encoder_model:
    type: resnet
  view_fusion_model:
    type: multicam_feature
```

This is preferred over configs that point directly at PyTorch classes.

## Testing And Comparisons

JAX tests live mainly in:

```text
tests/unit/test_bc.py
tests/unit/test_diffusion.py
tests/unit/replay_buffer/
```

Backend comparison commands and historical experiment notes are kept in:

```text
test_comparison.md
```

Those comparison commands can still be useful for measuring the JAX branch
against the PyTorch branch, but PyTorch implementation work itself belongs on
the PyTorch branch.

## Known Migration Leftovers

There are still parts of the repository with historical PyTorch language or
compatibility code:

- README examples may still describe `torch.Tensor`.
- Some tests under `tests/unit/method/` may still reflect old PyTorch behavior.
- `robobase/utils.py` and logging utilities may still have optional torch imports.
- Some method configs for algorithms outside BC and Diffusion may still refer to
  old PyTorch-era structure.

These should be cleaned up incrementally when they block or confuse JAX backend
development.

## Immediate JAX Priorities

1. Keep BC and Diffusion Policy stable on real offline robomimic runs.
2. Improve JAX replay/data throughput for offline imitation learning.
3. Decide whether the image encoder should stay frozen for the first stable JAX
   path or gain a trainable JAX-native vision path.
4. Remove stale PyTorch assumptions from shared runtime code as the JAX path
   becomes the only supported path on this branch.
5. Keep experiment docs clear about which commands are run from this JAX branch
   and which comparisons require the separate PyTorch branch.

The detailed imitation-learning roadmap is tracked in
[`jax_imitation_development_plan.md`](jax_imitation_development_plan.md).
