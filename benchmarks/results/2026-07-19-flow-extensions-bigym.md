# A2A and Legato BiGym screening report

Date: 2026-07-19

Audited upstream snapshots: A2A `a5792ecf4e7f8fa4d85fe66ea9a50618138f925c`,
Legato Kinetix `d302701268aa3a50ec7f07189cc3af3b31014f63`, and
CleanDiffuser `05f17fc9dbeae7c19a5e264632c9ae9aaac5994e`.

## Scope

This is a matched local screen, not a paper-level result. An initial screen was
invalidated after review found that multi-threaded replay returned batches in
I/O completion order and could cross epoch boundaries. The table below uses
only the full rerun after requests were assigned sequence IDs and reordered
before consumption.

All policies used the local default `~/.bigym` `flip_cutlery` replay cache: 43
successful demonstrations, 10,040 transitions, and 9,868 valid action windows.
The run used three 256x256 cameras, one observation frame, a trainable ResNet-18,
language conditioning, the same Transformer width/depth, normalized 16-D
actions, horizon 8, execution length 4, four Euler steps, batch size 128, seed 0,
and 30 replay epochs. Each epoch contained 77 full batches with `drop_last`; all
methods received the same seeded batch-index sequence. Evaluation used the same
10 environment seeds.

The serialized configs have `env.dataset_root: ''`, so this screen did **not**
use `/home/zc1525/.bigym_reset_aligned`. It is internally matched across the
five rows, but it is not a strict comparison with the repaired FM baseline that
was trained on reset-aligned pixels.

Runtime resolved 30 epochs to 2,310 optimizer updates. The serialized Hydra
config retains `method.num_train_steps=2340`, which is the cosine-schedule
horizon used to construct the optimizer. This 30-step mismatch is common to all
five runs but should be removed before the longer benchmark.

The A2A policies add separate history/future action encoders and a decoder.
Their 29,216,960 trainable parameters are 41.27% above FM's 20,681,552. Legato
adds one schedule channel and has 20,681,808 parameters, 256 more than FM.
Methods are matched on data and update count, not on parameter count, FLOPs, or
wall-clock budget.

## Results

| Policy | Success | Mean length | Boundary jump | First diff | Second diff | Jerk | Backend samples/s | First compile/update |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Gaussian FM | 1/10 | 11493.8 | 0.25016 | 0.22559 | 0.38546 | 0.70571 | 1374.4 | 28.8 s |
| A2A | 0/10 | 11262.5 | 0.09780 | 0.05798 | 0.08582 | 0.15097 | 1369.0 | 42.6 s |
| A2A noise 0.02 | 0/10 | 12500.0 | 0.09265 | 0.04562 | 0.06247 | 0.10914 | 1358.6 | 44.3 s |
| Legato paper-minus | 0/10 | 12500.0 | 0.31507 | 0.27400 | 0.45470 | 0.82150 | 1383.6 | 30.8 s |
| Legato public-plus | 0/10 | 12500.0 | 0.41964 | 0.87954 | 1.59034 | 2.99515 | 1354.8 | 31.5 s |

Relative to FM, deterministic A2A reduced commanded-action boundary jump by
60.91%, first difference by 74.30%, second difference by 77.73%, and third
difference by 78.61%. A2A noise reduced the same diagnostics by 62.96%, 79.78%,
83.79%, and 84.53%. Neither A2A policy completed a task. The deterministic A2A
run also produced one MuJoCo `BADQACC` health truncation, so its shorter mean
episode length is not task completion.

Legato paper-minus increased the four diagnostics by 25.95%, 21.46%, 17.96%,
and 16.41%. The public-code plus-sign target was worse: +67.75%, +289.88%,
+312.58%, and +324.42%. Both Legato targets tied at zero successes. The sign
disagreement therefore does not explain the local regression.

The only success was FM's 1/10. With one seed and ten episodes, this does not
establish a success-rate ranking. Low commanded-action differences can also
come from an inactive or hesitant policy; the zero-success A2A-noise result is
the clearest example. These results reject a claim that either extension is
already better on this BiGym setup, while identifying A2A command continuity as
a signal worth testing at a budget where policies reliably solve the task.

## Metric semantics

The four smoothness values are L2 diagnostics over clipped, normalized policy
commands. `action_boundary_jump` crosses policy-call boundaries. First, second,
and third differences are accumulated only inside each `execution_length=4`
chunk; `action_jerk` is a field name, not a time-scaled physical jerk. The
metrics do not use executed proprioceptive trajectories and are weighted by
policy calls rather than averaged per episode. Successes, early truncations,
and full-length failures therefore contribute different numbers of calls.

The backend rate is the median of `batch_size / backend_update_time` after the
compiling first update. It excludes replay fetching and observation preparation
and is not end-to-end training throughput. Logged wall time at step 2,300 was
381.3 s for FM, 424.4 s for A2A, 430.8 s for A2A noise, 377.8 s for Legato
paper-minus, and 384.5 s for Legato public-plus. The larger A2A compile and wall
times are consistent with its additional latent modules and parameters.

## Fidelity limits

- Public A2A uses normalized `agent_pos` proprioception as its source. BiGym has
  62-D proprioception and 16-D actions, so this adapter uses normalized prior
  commanded actions. It implements the action-history formulation but is not a
  line-for-line public-code reproduction.
- Public A2A uses multiple visual context frames. This screen held the existing
  FM baseline at `frame_stack=1`.
- A2A noise uses standard deviation 0.02 on selected continuous dimensions and
  no Exact OT. The public simulation preset uses 0.1 on all source dimensions
  with Exact OT.
- A2A transports one latent token. Transformer, DiT, fully-connected, and
  single-scale UNet backbones support it; multi-scale UNet is rejected rather
  than changing the public latent representation.
- The Legato paper and public Kinetix implementation disagree on the sign of
  the velocity correction. Both targets were screened above.
- BiGym emulates a one-control-step inference delay through the issued action
  prefix. It does not run policy inference concurrently with the environment,
  so this screen cannot validate Legato's wall-clock asynchronous advantage.

## Artifacts

- Aggregate CSV: `benchmarks/results/2026-07-19-flow-extensions-bigym.csv`
- FM: `exp_local/bigym_flip_cutlery_flow_matching_h8_k4_e0_f1_30e_seed0_ordered_20260719`
- A2A: `exp_local/bigym_flip_cutlery_a2a_h8_k4_e0_f1_30e_seed0_ordered_20260719`
- A2A noise: `exp_local/bigym_flip_cutlery_a2a_noise_h8_k4_e0_f1_30e_seed0_ordered_20260719`
- Legato paper-minus: `exp_local/bigym_flip_cutlery_legato_minus_h8_k4_e0_f1_30e_seed0_ordered_20260719`
- Legato public-plus: `exp_local/bigym_flip_cutlery_legato_plus_h8_k4_e0_f1_30e_seed0_ordered_20260719`
- Matched JAX/CleanDiffuser benchmark:
  `benchmarks/results/2026-07-19-fm10-release-final.json`

Every run has one eval row at iteration 2,310 and a
`snapshots/2310_snapshot.pkl` checkpoint.

## JAX speed gate

The matched state-only global UNet benchmark uses equal 68,906,250-parameter
models, batch size 256, horizon 16, 10 Euler steps, and EMA on an RTX 5090. In
three repeats against pinned CleanDiffuser Torch eager, JAX median sampling
speedup was 89.60% to 91.64% and median update speedup was 47.88% to 48.71%.
Every median and p95 comparison cleared the 30% gate. This measures the shared
FM policy core, not end-to-end BiGym visual rollout latency or A2A/Legato versus
their Torch implementations.

## Next experiment gate

1. Make the optimizer schedule horizon resolve to the actual epoch batch count
   before agent construction. Then train seeds 0, 1, and 2 for at least 100
   epochs and evaluate 50 episodes per seed with per-episode bootstrap intervals.
2. Establish a nonzero FM success floor before broad sweeps. Add parameter-
   matched A2A variants and report samples, FLOPs, compile, and wall time
   separately.
3. Sweep A2A solver/consistency steps `1,2,4,6,10`, history horizon, 2/4/8 visual
   frames, separate continuous/gripper heads, and public Exact-OT pairing. Add a
   learned 62-D proprio-to-action source and eventually measured executed-action
   feedback instead of commanded history.
4. Implement RTC and a wall-clock asynchronous executor before judging Legato.
   Ablate initialization-only, per-step-only, both targets, hard/linear/cosine
   schedules, fixed/random delay, and delays `0,1,2,4`, including wrong latency
   estimates.
5. Log per-episode commanded and executed trajectories, physical-time velocity,
   acceleration, and jerk, prefix error, failure categories, and completion time.
   Only after standalone methods pass these gates should an A2A-Legato hybrid be
   attempted.
