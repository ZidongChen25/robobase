# CamPose JAX Development and Validation Plan

## Current Baseline

- Runtime target: JAX/Flax/Optax only; Torch is isolated to the external
  CleanDiffuser benchmark worker.
- CamPose reference: `ripl/CamPoseOpensource@e0647105`.
- Torch performance baseline:
  `CleanDiffuser@05f17fc9dbeae7c19a5e264632c9ae9aaac5994e`.
- Existing latency evidence covers only the parameter-matched, state-only
  ChiUNet path. It is not evidence for pixel encoders or end-to-end replay.
- A fresh `.[jax,bigym]` install contained 87 distributions and none of Torch,
  torchvision, timm, or jax-resnet; core imports and a ResNet18 forward passed.

## P0: Correctness and Plug-in Contract

1. Move epoch-to-step resolution ahead of agent construction, then set the
   optimizer horizon from the resolved replay length. The complete CamPose
   launches currently use a fixed horizon; generic epoch-based custom launches
   should not need that workaround.
2. Add a composition test for every supported method/backbone/encoder/launch
   tuple and assert incompatible tuples fail before model initialization.
3. Version Torch-generated golden fixtures from the pinned official commits.
   Production and test collection must consume only NPZ/safetensors and never
   import Torch.
4. Keep CleanDiffuser local conditioning and exact DiT compatibility explicitly
   unsupported until their full topologies have golden tests. Do not silently
   approximate either path.
5. Add manifest-level corrupt-cache fixtures for missing keys, mismatched time
   dimensions, invalid matrix shapes, and non-finite values. Runtime validation
   for those cases is already in place.

## P1: Numerical Parity Gates

Test the following against the pinned golden fixtures:

- Camera rays for fixed intrinsics/C2W, both values and channel order.
- ACT ray CNN, RGB ResNet, late projection, learned positions, extrinsic tokens.
- DP early encoder and spatial keypoints, including the RGB-only ablation.
- Full UNet forward, DDPM training target/step, and rectified-flow pair/step.
- Fixed-seed one-update parameter deltas and short sampler trajectories.

Use per-module `max_abs <= 1e-5` in FP32 unless a documented reduction requires
a tighter or looser bound. Promote the successful manual fresh-environment
install check to CI; reject Torch, torchvision, timm, or jax-resnet there.

## P2: Reproducible Performance Gate

1. Reserve one GPU with no other compute processes.
2. Run five alternating JAX/Torch pairs after process-level compilation and
   allocator warmup. Report compile time separately.
3. Record p50, p95, coefficient of variation, peak device memory, parameters,
   dtype, batch, horizon, and solver steps.
4. Require JAX p50 and p95 to be at least 30% faster in every retained pair;
   otherwise profile the failed path before making a release claim.

Profiles: DDPM 100 steps, DDIM 100 steps where objectives match, and
rectified-flow Euler 10 steps. Keep EMA and optimizer work identical.

## P3: Pixel and Plucker Benchmarks

Add matched workers and gates for:

- DP `dp_early`: RGB+six ray channels, GroupNorm ResNet18, SpatialSoftmax.
- ACT `act_late`: ImageNet ResNet18 plus the five-layer ray CNN and transformer.
- End-to-end update: host batch to completed optimizer/EMA update.
- End-to-end sampling: host observation to host action, including ray creation.

Run 256 px, one frame, two cameras for strict CamPose ACT parity. Add a separate
three-camera BiGym result and label it as an extension, not official parity.

## P4: Training and Restore Gates

1. Overfit one 32-sample batch for each complete profile.
2. Assert finite gradients, decreasing loss, nonzero encoder updates, and frozen
   ACT BatchNorm affine/statistics where required.
3. Compare uninterrupted training with save/reload/resume at the same step.
4. Test fresh-home automatic weight download, offline override, bad SHA, and
   custom checkpoint fingerprints.
5. Run two identical seeds and establish deterministic tolerances.

## P5: Task Evaluation

Run matched JAX and CleanDiffuser seeds with the same normalized dataset split,
action horizon, execution length, camera inputs, and evaluation episodes.
Report success curves, wall-clock time to threshold, final success, and peak
memory. Promote a profile only when both correctness and performance gates pass.
