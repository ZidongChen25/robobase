# JAX Imitation Learning Development Plan

## Purpose

This document tracks the JAX imitation-learning work planned for this branch.

The branch goal is to build the imitation-learning stack in JAX/Flax, with a
shared backbone structure that can support:

- Diffusion Policy
- Flow Matching / Rectified Flow
- ACT

The PyTorch implementation should remain on the PyTorch branch. This branch
should port model behavior and training semantics to JAX rather than importing
PyTorch implementations.

## CleanDiffuser Reference

Use [CleanDiffuser](https://github.com/CleanDiffuserTeam/CleanDiffuser) as the
main reference for model structure, training objectives, samplers, and backbone
interfaces.

Local reference checkout:

```text
/home/zc1525/CleanDiffuser
```

Useful reference files:

```text
/home/zc1525/CleanDiffuser/cleandiffuser/nn_diffusion/base_nn_diffusion.py
/home/zc1525/CleanDiffuser/cleandiffuser/nn_diffusion/mlps.py
/home/zc1525/CleanDiffuser/cleandiffuser/nn_diffusion/jannerunet.py
/home/zc1525/CleanDiffuser/cleandiffuser/nn_diffusion/chitransformer.py
/home/zc1525/CleanDiffuser/cleandiffuser/nn_diffusion/dit.py
/home/zc1525/CleanDiffuser/cleandiffuser/diffusion/ddpm.py
/home/zc1525/CleanDiffuser/cleandiffuser/diffusion/ddim.py
/home/zc1525/CleanDiffuser/cleandiffuser/diffusion/rectifiedflow.py
/home/zc1525/CleanDiffuser/pipelines/dp_robomimic.py
/home/zc1525/CleanDiffuser/pipelines/dp_robomimic_image.py
```

CleanDiffuser is a reference, not a dependency target. The implementation in
this branch should be native JAX/Flax/Optax.

## Target Architecture

The imitation-learning stack should be split into three layers:

1. Method layer
   - Owns RoboBase integration, replay batches, metrics, snapshots, action
     normalization, padding masks, and evaluation.
   - Examples: `Diffusion`, `FlowMatching`, `ACT`.

2. Generative objective / sampler layer
   - Owns training objective and sampling dynamics.
   - Examples: DDPM-style noise prediction, DDIM sampling, Rectified Flow /
     Flow Matching Euler sampling.

3. Backbone layer
   - Owns only neural network prediction.
   - Input/output contract should be shared by Diffusion and Flow Matching.

The backbone contract should be:

```text
backbone(x, time, condition) -> prediction
```

Where:

- `x`: action sequence, shape `(batch, action_horizon, action_dim)`
- `time`: diffusion/flow time, shape `(batch,)`
- `condition`: encoded observations or `None`
- `prediction`: same shape as `x`

Backbones should not know about replay buffers, optimizers, environments,
Hydra, action normalization, or snapshotting.

## Backbone Scope

The shared backbone registry should support:

1. Fully connected backbone
   - Baseline MLP implementation.
   - Useful for low-dimensional fast tests and ablations.

2. UNet backbone
   - Keep the current JAX 1-D conditional U-Net path working.
   - Refactor it behind the common backbone interface.
   - Use CleanDiffuser UNet variants as references for model details.

3. Transformer backbone
   - Add the Diffusion Policy author's Transformer-style action-sequence
     model.
   - Use CleanDiffuser `ChiTransformer` as the first reference point.
   - Support observation conditioning and causal action decoding behavior.

4. DiT backbone
   - Add a 1-D DiT-style action-sequence model.
   - Use CleanDiffuser `DiT1d` as the reference point.
   - Include adaLN-Zero style conditioning and output initialization.

Proposed implementation locations:

```text
robobase/models/backbone.py
robobase/models/backbones/fully_connected.py
robobase/models/backbones/unet1d.py
robobase/models/backbones/transformer.py
robobase/models/backbones/dit.py
```

The exact file layout can change, but the final structure should make the
backbone boundary obvious.

## Diffusion Policy Plan

Current JAX Diffusion Policy exists in:

```text
robobase/method/diffusion.py
robobase/models/diffusion.py
```

The current implementation should be refactored so `method=diffusion` chooses a
backbone through config instead of being tied to one actor model path.

Target config shape:

```yaml
method:
  name: diffusion
  objective:
    type: ddpm
    num_diffusion_iters: 100
    sampler: ddim
  backbone:
    type: unet1d
```

Supported backbone values should include:

```text
fully_connected
unet1d
transformer
dit
```

Diffusion milestones:

1. Extract the current U-Net actor into the backbone interface.
2. Add a small fully connected backbone for baseline and tests.
3. Add Transformer backbone.
4. Add DiT backbone.
5. Keep existing action padding mask behavior.
6. Keep optional EMA behavior.
7. Keep prioritized replay priority updates based on per-sample loss.

## Flow Matching Plan

Add a JAX Flow Matching method, using Rectified Flow as the first target.

Reference:

```text
/home/zc1525/CleanDiffuser/cleandiffuser/diffusion/rectifiedflow.py
```

The first implementation should match the CleanDiffuser convention unless we
explicitly decide to change notation:

```text
x0 = target action sequence
x1 = source sample, usually Gaussian noise
t  = sampled time in [0, 1]
xt = t * x1 + (1 - t) * x0
target velocity = x0 - x1
```

The Flow Matching model should use the same backbone contract as Diffusion:

```text
backbone(xt, t, condition) -> velocity
```

Initial sampler:

```text
Euler integration from source noise toward target actions
```

Flow Matching milestones:

1. Add `robobase/method/flow_matching.py`.
2. Add `robobase/cfgs/method/flow_matching.yaml`.
3. Reuse the shared backbone registry.
4. Add low-dimensional unit tests for loss, update, and action sampling.
5. Add smoke test through `Workspace`.
6. Add robomimic low-dimensional comparison command.
7. Add pixel-conditioning path after low-dimensional path is stable.

## ACT Plan

ACT should be supported as a separate imitation-learning method in JAX/Flax.

The ACT path should share the existing RoboBase action chunking semantics:

- `action_sequence`
- `execution_length`
- `action_pad_mask`
- temporal ensemble settings where applicable

ACT milestones:

1. Audit the current PyTorch branch ACT implementation and CleanDiffuser-style
   robomimic pipelines for expected data shapes.
2. Add Flax transformer encoder/decoder components.
3. Add `robobase/method/act.py` as a JAX method.
4. Add or convert `robobase/cfgs/method/act.yaml` to JAX-native config.
5. Support low-dimensional observations first.
6. Add pixel-conditioning path using the branch's JAX encoder/fusion components.
7. Add snapshot and workspace smoke coverage.

## Condition Encoding

BC, Diffusion, Flow Matching, and ACT should converge on the same observation
feature preparation path where possible.

Current reusable pieces:

```text
robobase/method/jax_base.py
robobase/method/bc_runtime.py
robobase/models/encoder.py
robobase/models/fusion.py
```

The desired direction is:

```text
observations -> low_dim features and/or pixel features -> condition -> method/backbone
```

Open design question:

```text
Should Transformer and ACT receive a flattened condition vector, a condition
token sequence, or both?
```

The first implementation can use the simplest shape that works with robomimic
low-dimensional data, then add richer token-based conditioning for image and
multi-view setups.

## Progress

| Area | Status | Notes |
| --- | --- | --- |
| JAX branch strategy documented | Done | See `backend.md`. |
| CleanDiffuser reference link added | Done | This file links to the GitHub project and local checkout. |
| JAX BC | In progress / usable | Low-dimensional and current pixel path exist. |
| JAX Diffusion Policy | In progress / usable | Uses the shared backbone registry while preserving old `actor_model` config compatibility. |
| Backbone abstraction | Done | `robobase/models/backbone.py` defines the registry/parser for shared action-sequence backbones. |
| Fully connected diffusion backbone | Done | Added as the low-dimensional baseline under the shared backbone API. |
| UNet backbone refactor | Done | Current JAX U-Net lives under `robobase/models/backbones/unet1d.py` and remains exported for compatibility. |
| Transformer backbone | Done | Added a Flax `ChiTransformer`-style action-sequence backbone. |
| DiT backbone | Done | Added a Flax `DiT1d`-style backbone with adaLN-Zero output initialization. |
| Flow Matching method | Done / low-dimensional covered | Added JAX Rectified Flow objective, Euler sampler, shared backbone reuse, act sampling, and snapshot tests. |
| ACT JAX method | Done / low-dimensional covered | Added JAX ACT transformer method with action padding masks and Workspace snapshot coverage. Pixel path uses the shared frozen encoder/fusion path but still needs real image-run validation. |
| Robomimic comparison commands | Done | `test_comparison.md` now includes Diffusion, Flow Matching, and ACT JAX comparison commands. |

## Implementation Phases

### Phase 1: Backbone Contract

- Define the JAX backbone call signature.
- Add a small registry/parser for `method.backbone.type`.
- Move current Diffusion U-Net into the backbone structure.
- Add a fully connected backbone.
- Add unit tests that instantiate each backbone and verify output shape.

### Phase 2: Diffusion Refactor

- Make `Diffusion` depend on the generic backbone interface.
- Keep current behavior for the existing U-Net config.
- Add config compatibility for the current `actor_model` fields if needed.
- Verify existing Diffusion tests still pass.

### Phase 3: Transformer And DiT

- Port CleanDiffuser `ChiTransformer` semantics to Flax.
- Port CleanDiffuser `DiT1d` semantics to Flax.
- Add low-dimensional tests for both backbones.
- Add short Workspace smoke runs.

### Phase 4: Flow Matching

- Implement Rectified Flow loss.
- Implement Euler sampler.
- Reuse all available backbones.
- Add unit tests for loss reduction, update step, sampling shape, and snapshot.
- Add robomimic low-dimensional smoke run.

### Phase 5: ACT

- Implement JAX ACT transformer components.
- Integrate action padding and chunking.
- Add low-dimensional robomimic smoke test.
- Add pixel path after low-dimensional behavior is stable.

### Phase 6: Experiment Tracking

- Add commands to `test_comparison.md` for:
  - Diffusion + FC/UNet/Transformer/DiT
  - Flow Matching + FC/UNet/Transformer/DiT
  - ACT
- Track:
  - training loss
  - action sampling speed
  - eval success
  - eval reward
  - GPU memory
  - replay/input throughput

## Acceptance Criteria

The backbone refactor is complete when:

- Diffusion and Flow Matching can share the same backbone modules.
- Backbones are selected through config.
- Each backbone has shape/unit tests.
- Existing JAX Diffusion behavior remains covered.

Flow Matching is complete when:

- It trains through `Workspace`.
- It samples action chunks through `act`.
- It supports the same observation feature path as Diffusion.
- It has snapshot coverage.

ACT is complete when:

- It trains through `Workspace`.
- It supports action chunking and padding masks.
- It has low-dimensional robomimic smoke coverage.
- Pixel support is either implemented or explicitly marked as unsupported.

## Open Decisions

1. The first Transformer backbone is a minimal Flax adaptation of CleanDiffuser
   `ChiTransformer` with the same public backbone contract.
2. DiT supports vector conditions first. Token-sequence conditioning remains a
   future extension.
3. Pixel encoders remain frozen for the first complete Diffusion/Flow/ACT stack.
4. Diffusion keeps backward compatibility with the old `actor_model` config
   fields while the preferred config uses `method.backbone`.
5. Flow Matching exposes Rectified Flow first; additional flow objectives can be
   added later behind `method.objective.type`.
