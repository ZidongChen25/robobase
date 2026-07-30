# A2A and Legato Flow Policies

This repository exposes three pure-JAX flow policies through the same RoboBase
agent interface:

| Method config | Source/path | Rollout state |
|---|---|---|
| `flow_matching` | Gaussian to action chunk | none |
| `a2a` / `a2a_noise` | encoded prior actions to encoded future actions | executed command history |
| `legato` | schedule-guided Gaussian/reference mixture | previous generated chunk |

The observation encoder, language condition, action normalization, compatible
backbone, optimizer, EMA, checkpointing, and BiGym rollout path are shared.
Source/path math is isolated in `robobase/method/flow_sources.py`; the
environment-facing classes are `robobase.method.a2a.A2A` and
`robobase.method.legato.Legato`.

## Conventions

RoboBase uses reverse flow time `tau`: `tau=1` is the source and `tau=0` is the
target. Euler sampling evaluates `tau=1,...,1/N` and applies positive updates
of size `1/N`.

Gaussian FM uses:

```text
x_tau = tau * epsilon + (1 - tau) * action
v*    = action - epsilon
```

A2A encodes history `[B, P, A]` and the future chunk `[B, H, A]` into one
latent token `[B, 1, L]`, learns the same linear path between those tokens,
then decodes the final token to `[B, H, A]`. Its loss contains latent FM,
future reconstruction, and differentiable sampling-consistency terms. Both
encoders receive an explicit validity channel; masked action values are zeroed
before convolution so padding cannot alias a legitimate zero action. History
noise is applied after normalization, only on valid selected action dimensions.
Because A2A has one latent action token, its Transformer automatically uses
full cross-attention over every observation token. The ordinary FM Transformer
keeps its original causal memory mask.

Legato uses the paper convention `omega=1` for full continuation guidance. Its
model input is `[B, H, A+1]`: guided actions plus the public implementation's
`1-omega` schedule channel. For fixed Euler step `dt`, the v2 paper target in
RoboBase time is:

```text
v* = (1 - omega * tau / dt) * (action - epsilon)
```

`target_mode=paper_minus` is the default. The public Kinetix code's conflicting
plus-sign target remains available as `public_kinetix_plus` for a target-sign
ablation; it does not by itself select every public Kinetix schedule preset.
Guidance is reapplied before every solver evaluation; it is not post-hoc prefix
replacement.

## Alignment and rollout state

For target chunk starting at action `t`, lazy BiGym replay returns A2A history
`[a_(t-P), ..., a_(t-1)]`. Episode-boundary padding is always accompanied by a
mask and masked values cannot become valid source evidence.

The policy module is selected through the common factory. Lazy BiGym replay and
the standard Uniform/Prioritized replay buffers construct `action_history` and
`action_history_pad_mask` automatically from episode-local action windows.
Custom replay implementations must emit the same two fields.

At rollout, A2A appends the `execution_length` normalized actions sent through
the receding-horizon wrapper. BiGym does not currently feed measured
post-controller actions back into the agent, so this is commanded/executed-in-
simulation history, not hardware proprioceptive feedback. Temporal ensembling
is rejected until that feedback API exists.

Legato shifts the previous raw generated chunk by `execution_length`, then uses
PADLAST or zero padding to form the next reference. Train and eval states are
separate and `reset()` clears individual environment slots.
Continuation schedules start at `action_execution_start`, so their retained
prefix and ramp use the same chunk coordinates as the actions actually issued.
Static configuration must satisfy
`action_execution_start + delay + ramp <= action_sequence - execution_length`.
Configurations without any previous-chunk overlap fail during policy
construction instead of silently disabling continuation. Training also samples
an explicit no-guidance bootstrap path.

A2A's public representation transports one latent token. Transformer, DiT,
fully-connected backbones, and a single-scale UNet support that shape. A
multi-scale UNet requires a temporal length divisible by its downsampling
factor, so A2A rejects that combination during construction instead of silently
changing the public latent representation. Gaussian FM and Legato continue to
support the ordinary multi-scale UNet.

## Configuration

Important A2A fields are under `method.flow_source`:

```yaml
type: a2a                 # or a2a_noise
history_horizon: 8
history_padding: zero
latent_dim: 512
train_history_noise_std: 0.0
eval_history_noise_std: 0.0
noise_exclude_last_n: 2
consistency_steps: 1
```

Important Legato fields are:

```yaml
type: legato
delay_min_steps: 1
delay_max_steps: 2
ramp_min_steps: 1
ramp_max_steps: 2
schedule_profiles: [hard, linear, cosine]
no_guidance_probability: 0.1
target_mode: paper_minus
reference_padding: last
```

## BiGym matched screen

The supplied screen uses the same FlipCutlery data, visual encoder, Transformer,
`H=8`, `execution_length=4`, batch size, four Euler steps, update budget, and
evaluation seed policy for all methods:

```bash
CUDA_VISIBLE_DEVICES=0 MUJOCO_GL=egl .venv/bin/python train.py \
  launch=flow_extensions_bigym env=bigym/flip_cutlery \
  profile=bigym_repaired_pixels
CUDA_VISIBLE_DEVICES=1 MUJOCO_GL=egl .venv/bin/python train.py \
  launch=a2a_flow_extensions_bigym env=bigym/flip_cutlery \
  profile=bigym_repaired_pixels
CUDA_VISIBLE_DEVICES=2 MUJOCO_GL=egl .venv/bin/python train.py \
  launch=legato_flow_extensions_bigym env=bigym/flip_cutlery \
  profile=bigym_repaired_pixels
```

`a2a_noise_flow_extensions_bigym` enables the action-history robustness
adaptation with `0.02` noise on continuous dimensions. It is not the public
code's `0.1` all-dimension Exact-OT preset. The 30-epoch configs are screening
runs, not publication results. `consistency_steps=1` is deliberately a
single-step training regularizer while rollout uses four Euler steps; the
matched-step variant is a required ablation. A formal comparison should use at
least three seeds and a longer fixed budget, then sweep solver steps
`1,2,4,6,10` and Legato delay estimates.

Epoch replay reserves batch indices in the seeded order, materializes them in
parallel, and reorders completed I/O requests before consumption. Thus every
method sees the same batch-index sequence despite variable disk latency.
Rollout smoothness values are L2 diagnostics on clipped, normalized commanded
actions. Only `action_boundary_jump` crosses policy-call boundaries; the other
finite differences are within each executed chunk. They are not measured
proprioceptive trajectories or time-scaled physical jerk. Legato additionally
logs generated/reference RMSE over the hard prefix and every guided overlap
position.

The completed screen and its artifact paths are recorded in
`benchmarks/results/2026-07-19-flow-extensions-bigym.md`.

## FlipCutlery 200-epoch protocol

The checkpoint-matched long-run configs fully specify the training protocol:

```bash
.venv/bin/python train.py launch=a2a_flip_cutlery_repaired_200e
.venv/bin/python train.py launch=a2a_noise_flip_cutlery_repaired_200e
.venv/bin/python train.py launch=legato_flip_cutlery_repaired_200e
```

They use reset-aligned pixels, the original state/action cache, the exact
precomputed legacy language feature, and an explicit pure-JAX conversion of
`timm/resnet18.a1_in1k`. Those two artifact paths are host-local dependencies,
so the referenced files and their hashes must be archived with a released run.
Transformer `operator_variant=legacy` preserves the historical Flax GELU,
LayerNorm, positional embedding, zero embedding-dropout, and
conditional-encoder semantics. `torch` follows the original Torch RoboBase
Transformer's sinusoidal embedding order/frequencies, embedding and residual
dropout, exact GELU, and `epsilon=1e-5`; it is not an alias for every
CleanDiffuser Transformer release.

A2A uses `H=20/K=20`, giving 78 full batches per epoch, 15,600 updates over 200
epochs, eval at 7,800/15,600, and snapshots every 1,560 updates. Legato cannot
use `K=H` because no previous-chunk overlap remains; its long-run config uses
`H=20/K=10`, giving 75 batches per epoch, 15,000 updates, eval at 7,500/15,000,
and snapshots every 1,500 updates. A fair Legato comparison therefore requires
FM checkpoint evaluation at `K=10` in addition to the historical `K=20` result.

Standalone GPU evaluations must also lock XLA's fusion autotuning decisions.
Fixed environment and JAX PRNG seeds alone do not make independently compiled
GPU processes bitwise reproducible. Pass a persistent directory to the
checkpoint evaluator:

```bash
.venv/bin/python scripts/eval_flow_step_checkpoint.py \
  --run-dir <run-dir> --snapshot <snapshot.pkl> --flow-steps 10 \
  --output <result.json> --work-dir <work-dir> --gpu-id 0 \
  --num-eval-episodes 50 --xla-fusion-cache-dir <xla-cache-dir>
```

The option enables deterministic GPU flags, persists per-fusion autotuning
choices, and records both the resolved cache path and `XLA_FLAGS` in the result.
Populate the cache once, then reuse and archive it together with the JAX,
CUDA, driver, and GPU-model versions. A cache is only comparable on the same
software and hardware fingerprint.

The explicit A2A validity channel changes the first temporal convolution from
`action_dim` to `action_dim + 1`. Checkpoints created before this correction are
diagnostic artifacts only and cannot be resumed into the corrected model.

## Upstream differences kept explicit

- The A2A paper describes shared action representation learning, while the
  public policy has separate history/future temporal encoders.
- The public policy reads normalized `agent_pos` proprioception as its source.
  This BiGym adapter follows the paper/action-history protocol because BiGym's
  62-D proprioception is not isomorphic to its 16-D action space. Results must
  therefore be labeled as the action-history adaptation, not public-code parity.
- The public A2A code and paper text differ by one action in their prose-level
  time indexing. RoboBase tests the concrete preceding-history/current-target
  convention above.
- The public A2A policy consumes multiple visual observation frames. The
  matched BiGym screen uses `frame_stack=1` to hold the existing FM baseline
  configuration fixed; it is not a visual-context parity experiment.
- The Legato v2 derivation and public Kinetix target have opposite signs in the
  continuation correction. Both are named configs; the paper derivation is the
  default.
- Public Kinetix uses step guidance. Real-robot Legato recommends a ramp, so
  hard, linear, and cosine horizon schedules are selectable.
- Current BiGym evaluation emulates inference delay by issuing the retained
  prefix before the new suffix. It does not measure a concurrently running,
  wall-clock asynchronous executor.
