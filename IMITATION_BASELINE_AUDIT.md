# JAX Imitation Baseline Audit

Date: 2026-07-18

## Reference revisions

- CleanDiffuser: `05f17fc9dbeae7c19a5e264632c9ae9aaac5994e`
- CamPoseOpensource: `e0647105`
- Runtime contract: RoboBase imports and installed dependencies remain
  JAX/Flax/Optax-only. Torch is allowed only in external reference exporters and
  benchmark workers.

## Baseline status

| Entry point | Status | Reference contract |
|---|---|---|
| `method=bc` | Native baseline | Deterministic MLP/GRU action chunk with MSE. It is not the full robomimic BC family because Gaussian and GMM heads are missing. |
| `method=act` | Supported | JAX ACT, with separate CamPose late-Plucker profile. CleanDiffuser is not an ACT reference. |
| `launch=dp_state_robomimic` | Native legacy | Kept for old experiment/checkpoint compatibility. It is not a CleanDiffuser baseline. |
| `launch=clean_diffuser_dp_state_robomimic` | Matched training baseline | CleanDiffuser global ChiUNet, cosine DDPM, DDPM sampler, constant EMA, AdamW/cosine LR, absolute 6D actions and matched sequence protocol. |
| `launch=clean_diffuser_rf_state_robomimic` | Matched-core extension | CleanDiffuser ContinuousRectifiedFlow objective and Euler solver on the matched RoboMimic protocol. CleanDiffuser does not publish this as a RoboMimic recipe. |
| `launch=clean_diffuser_dit_ddpm_state_robomimic` | Matched architecture module | Pure-JAX CleanDiffuser `DiT1d` Fourier backbone plus RoboMimic `MLPCondition`, composed with RoboBase DDPM; the published RoboMimic DiT recipe uses EDM. |
| `launch=clean_diffuser_dit_fm_state_robomimic` | JAX extension | The matched DiT architecture composed with Rectified Flow; CleanDiffuser does not publish this RoboMimic recipe. |
| `launch=campose_dp_bigym` | CamPose baseline | CamPose DP early Plucker fusion and local-conditioned UNet; it is intentionally not CleanDiffuser local ChiUNet. |
| `launch=campose_fm_bigym` | JAX extension | CamPose visual stack with Rectified Flow. It is not an official CamPose method. |

## CleanDiffuser DDPM contract

The strict preset uses `compatibility_mode=clean_diffuser`. Model construction
fails before allocation unless all of these are true:

- global-conditioned ChiUNet, power-of-two action horizon;
- even 256-dimensional positional embedding and 256-dimensional condition
  projection;
- widths `[256, 512, 1024]`, GroupNorm-compatible groups, kernel 5;
- PyTorch-compatible Conv/ConvTranspose padding and initialization;
- FiLM `scale * x + bias`, not `(1 + scale) * x + bias`;
- cosine DDPM epsilon objective, full 50-step DDPM sampling;
- EMA `0.995` from the first update and AdamW weight decay `0.01`;
- two observation frames, horizon 16, execute indices `[1:9]`;
- action sequence start offset 1, edge padding, all demonstrations;
- absolute position plus 6D rotation actions and min/max normalization.

Task-specific observation contracts are separate environment modules:
`env=robomimic_clean/lift`, `can`, `square`, `tool_hang`, and `transport`.
Their episode limits match the pinned task recipes. This prevents the
CleanDiffuser key subset and absolute-action behavior from silently changing old
native RoboMimic runs.

## Verified parity

- Global ChiUNet topology and parameter count match. The pinned `[32, 64, 128]`
  fixture has 1,083,810 parameters; mapped-weight FP32 max-absolute errors are
  `6.33e-7` for the full forward pass, `5.36e-7` for the action VJP,
  `7.26e-8` for the timestep VJP and `1.13e-6` for the condition VJP.
- DiT fused QKV attention, CleanDiffuser residual ordering, adaLN-Zero,
  integer-dtype sequence position behavior, Fourier timestep embedding and
  final layer match. The 43,623-parameter mapped fixture has FP32 max-absolute
  errors of `8.94e-8` forward, `2.24e-8` action VJP, `1.21e-13` timestep VJP
  and `4.44e-15` condition VJP.
- Cosine betas and cumulative alphas match the pinned implementation.
- DDPM noise target, timestep distribution, posterior mean/variance, prediction
  clipping and MSE reduction match when padding and replay weights are disabled.
- Constant EMA and AdamW update semantics match when hyperparameters match.
- Continuous Rectified Flow matches for uniform `t` in `[0,1]`, reverse velocity
  `x0-x1`, uniform reverse schedule and Euler integration.
- Epoch replay produces the same valid anchor range as CleanDiffuser when
  `execution_length=8`, offset 1 and edge padding are composed.
- Sampling bounds now come from the action space. Standardized unbounded actions
  are no longer incorrectly clipped to `[-1,1]`.

The exporter is a development-only Torch tool. Its committed NPZ fixture and
the regression suite import only JAX/Flax/NumPy. The unit suite excluding the
unavailable optional RLBench module passed with `340 passed, 9 skipped` on CPU.

## Explicit non-parity

- CleanDiffuser local-conditioned ChiUNet is not implemented. The JAX local path
  is the CamPose topology and incompatible Clean presets fail fast.
- CleanDiffuser's published RoboMimic DiT recipe uses EDM. RoboBase currently
  exposes the matched DiT architecture under its DDPM and Rectified Flow
  objectives; neither composition is labeled as full recipe parity.
- RoboBase `sampler=ddim` is discrete cosine-DDPM eta-zero sampling. It is not
  CleanDiffuser's continuous `DDIM` class or its DPMSolver pipeline.
- The pixel benchmark is CamPose/DP visual conditioning, not CleanDiffuser
  `MultiImageObsCondition`; state-only timing cannot be generalized to pixels.
- JAX and Torch RNG streams are distribution-equivalent, not bitwise identical.
- Flow has finite-value guards that do not affect finite reference trajectories.

## Running the modules

Set the portable dataset root once when the datasets are outside this checkout:

```bash
export ROBOMIMIC_DATA_ROOT=/home/zc1525/robobase/third_party_datasets/robomimic
```

Then point the selected environment at that root:

```bash
python3 train.py launch=clean_diffuser_dp_state_robomimic \
  env=robomimic_clean/tool_hang \
  env.dataset_path=${ROBOMIMIC_DATA_ROOT}/tool_hang/ph/low_dim_v141.hdf5

python3 train.py launch=clean_diffuser_rf_state_robomimic \
  env=robomimic_clean/transport \
  env.dataset_path=${ROBOMIMIC_DATA_ROOT}/transport/ph/low_dim_v141.hdf5
```

## Development order

1. Overfit one fixed normalized batch in JAX and CleanDiffuser, then compare the
   loss curve and verify that save/restore produces the same next update and EMA
   parameters. The operator, DDPM posterior and RF trajectory golden gates are
   already complete.
2. Run matched CleanDiffuser and JAX training on the same normalized split for
   3-5 seeds. Compare success curves, time-to-threshold, final success and peak
   memory, not training loss alone.
3. Reserve an otherwise idle GPU and repeat five alternating JAX/Torch pairs at
   the official 50-step DDPM setting and the 10-step RF setting. Require both
   p50 and p95 update/sample latency to be at least 30% faster in every retained
   pair before making an unconditional speed claim.
4. Add a shared `ActionHead` interface with deterministic, diagonal Gaussian and
   GMM heads. Build BC-RNN-GMM first; it is the highest-value missing classic
   baseline and reuses every existing encoder.
5. Add VQ-BeT as an action-head/tokenizer module. Add IQL separately only for
   datasets with reward labels; it is offline RL, not pure imitation learning.
6. Add a shared DP3-style point-cloud encoder only when calibrated depth is
   available. Plucker rays encode camera geometry but not observed surfaces.
7. Add ManiFlow as the main new consistency-flow research path. Treat EDM,
   Consistency Policy and Shortcut as objective/sampler extensions rather than
   duplicating the full observation stack.

Primary method references: [robomimic BC baselines](https://proceedings.mlr.press/v164/mandlekar22a.html),
[VQ-BeT](https://arxiv.org/abs/2403.03181),
[IQL](https://arxiv.org/abs/2110.06169),
[DP3](https://arxiv.org/abs/2403.03954), and
[ManiFlow](https://proceedings.mlr.press/v305/yan25a.html).
