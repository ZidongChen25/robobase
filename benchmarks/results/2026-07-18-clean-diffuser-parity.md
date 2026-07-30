# RoboBase JAX vs CleanDiffuser UNet benchmark

Date: 2026-07-18

## Scope

- State-only, global-conditioned ChiUNet-compatible profile.
- Shape: batch 256, horizon 16, action dim 10, observation shape 2 x 23.
- FP32, `cond_predict_scale=true`, CleanDiffuser positional embedding.
- 68,906,250 trainable parameters on both backends.
- Update timing includes the full forward/backward pass, AdamW, and EMA update.
- Sample timing includes host observation input and host action output.
- JAX 0.6.2 versus CleanDiffuser commit
  `05f17fc9dbeae7c19a5e264632c9ae9aaac5994e` with Torch 2.11.0+cu130.
- CleanDiffuser was loaded from the clean worktree
  `/tmp/CleanDiffuser-baseline-05f17fc9`; `git status --short` was empty.
- Hardware: one NVIDIA GeForce RTX 5090. The GPU was not exclusive.

The primary gate is p50 latency. Speed advantage is
`(torch_p50 / jax_p50 - 1) * 100`. CV is computed across the measured calls.
The current harness also requires optimizer, learning rate, weight decay,
training schedule, loss reduction, temperature and bounds metadata. These rows
predate that serialized metadata gate, so they are historical performance
evidence rather than a current release certificate and must be rerun.
They also use 100 DDPM steps, while the pinned RoboMimic recipe uses 50.

## Recorded profiles

Each row is one paired run; the JAX latency shown is the JAX process from that
same pair. This avoids combining an advantage from one pair with a displayed
latency from another pair.

| Objective / Torch mode | JAX update p50 / CV | Clean update p50 / CV | Update advantage | JAX sample p50 / CV | Clean sample p50 / CV | Sample advantage |
|---|---:|---:|---:|---:|---:|---:|
| DDPM 100, eager | 12.527 ms / 0.91% | 18.851 ms / 0.80% | 50.48% | 141.895 ms / 9.93% | 396.937 ms / 12.47% | 179.74% |
| DDPM 100, `compile(default)` | 12.418 ms / 0.71% | 22.708 ms / 3.65% | 82.87% | 141.958 ms / 0.43% | 548.272 ms / 4.87% | 286.22% |
| Flow Euler 10, eager | 12.746 ms / 0.71% | 18.338 ms / 0.12% | 43.87% | 15.053 ms / 0.25% | 37.180 ms / 4.81% | 147.00% |
| Flow Euler 10, `compile(default)` | 12.597 ms / 0.68% | 23.148 ms / 2.26% | 83.75% | 15.283 ms / 1.80% | 51.249 ms / 3.57% | 235.34% |

All four recorded update and sample comparisons exceed 30% in these
paired runs. CleanDiffuser's unmodified ChiUNet triggers Inductor backend
failures at the scale/bias reshape (`aten._local_scalar_dense`) and graph
fallbacks, so `torch.compile(default)` is slower than eager here. Both modes
are reported rather than treating the compile result as an optimized upper
bound.

Raw reports: [DDPM paired runs](2026-07-18-ddpm100-raw.json),
[Flow paired runs](2026-07-18-fm10-raw.json), and
[DDPM three-repeat run](2026-07-18-ddpm100-r3-raw.json). The Flow report was
recorded before the JAX Flow EMA schedule was changed to CleanDiffuser's fixed
decay. Its latency remains informative, but it is not a semantic parity gate;
rerun it with the current harness before release.

Representative rerun commands:

```bash
.venv/bin/python benchmarks/compare_policy_backends.py --gpu 3 \
  --clean-root /tmp/CleanDiffuser-baseline-05f17fc9 \
  --objectives diffusion --ema-modes on --torch-modes eager default \
  --sample-steps 50 --pair-repeats 5 \
  --output benchmarks/results/ddpm50-release.json
.venv/bin/python benchmarks/compare_policy_backends.py --gpu 3 \
  --clean-root /tmp/CleanDiffuser-baseline-05f17fc9 \
  --objectives flow_matching --ema-modes on --torch-modes eager default \
  --sample-steps 10 --output benchmarks/results/fm10-current.json
```

## Repeated DDPM check

DDPM 100-step, EMA-on, `torch.compile(default)` was also run as three
alternating JAX/Torch pairs with 10 warmups and 30 measurements:

| Repeat | JAX update p50 | Torch update p50 | Update advantage | JAX sample p50 | Torch sample p50 | Sample advantage |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 12.815 ms | 22.606 ms | 76.41% | 141.348 ms | 533.408 ms | 277.37% |
| 1 | 17.532 ms | 21.908 ms | 24.96% | 172.673 ms | 483.208 ms | 179.84% |
| 2 | 16.736 ms | 25.231 ms | 50.76% | 178.660 ms | 504.936 ms | 182.62% |

The median update advantage is 50.76% (range 24.96-76.41%); the median sample
advantage is 182.62% (range 179.84-277.37%). Two unrelated GPU processes were
present throughout, and the final GPU snapshot showed 44% utilization. Since
one repeated update pair fell below 30%, an exclusive-GPU rerun remains
required before making an unconditional external claim that every run is at
least 30% faster.

## DDIM note

RoboBase JAX DDIM at 100 steps and EMA-on measured 12.529 ms update p50
(0.39% CV) and 142.068 ms sample p50 (0.58% CV). No cross-framework speed gate
is reported: RoboBase DDIM is the deterministic sampler for the discrete
cosine-DDPM objective, while CleanDiffuser's `DDIM` class uses a different
continuous-time noise process and training objective.
