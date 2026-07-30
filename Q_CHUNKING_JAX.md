# Pure-JAX Q-Chunking

## Scope and upstream

- Paper: <https://arxiv.org/abs/2507.07969>
- Official repository: <https://github.com/ColinQiyangLi/qc>
- Audited upstream commit:
  `48283b4f662bd1c127ec9fa80647b49759f10653`
- Local method: `robobase/method/q_chunking.py`
- Hydra methods/launches:
  - `method=q_chunking`
  - `launch=q_chunking_state_dmc`
  - `launch=q_chunking_pixel_bigym`
- Standalone checkpoint evaluation:
  `scripts/eval_q_chunking_snapshot_sweep.py`

The port implements the official Best-of-N QC path: a conditional flow
behavior model proposes complete action chunks, two scalar Q functions score
the chunks, the selected next chunk is evaluated by a Polyak target critic,
and replay supplies the discounted K-step return and `gamma**K` bootstrap.
The selected chunk is executed open loop while RoboBase stores each primitive
action that was actually executed.

The MLP path matches the upstream defaults: four 512-wide GELU layers,
critic LayerNorm after each hidden activation, no actor LayerNorm,
`fan_avg/uniform` dense initialization, constant `3e-4` learning rates,
`tau=0.005`, 10 Euler steps, 32 candidates, and mean aggregation of two Qs.
The BiGym visual adapter uses this repository's three-camera JAX CQN CNN so it
is matched to the local CQN-AS baseline and does not expand every stacked frame
to 224x224 in two trainable ResNets. The generic method still supports the
shared JAX ResNet encoder. The upstream paper repository does not provide a
BiGym experiment, so this visual adapter is not claimed as upstream parity.

## Stage 0: Earlier partial QC-style CQN-AS

### 1. Previous-stage result

Artifact:
`exp_local/cqn_stage163_replan8/move_plate_replan8_seed1_gpu5_20260728151535/async_eval_5000.json`

- Snapshot: `5000_snapshot.pkl`
- Validation seeds: 400-449
- Episodes: 50
- Success: 25/50 = 50%
- Mean episode length: 228.44
- Configuration: CQN-AS, K=8 backup and replan interval 8

This was a single saved checkpoint, not a validation-selected best checkpoint.

### 2. Interpretation

The artifact establishes that the earlier K-step replay/replan wiring ran and
produced a nontrivial policy. It does not test Q-Chunking itself: it retains the
CQN-AS categorical sequence critic and policy, lacks the scalar chunk critic,
flow behavior policy, and Best-of-N selection, and uses K=8 rather than the
official QC default K=5. It therefore cannot support a claim about QC quality.

### 3. Next-stage decision

Hypothesis: the complete QC computation can be represented using the existing
primitive-action replay without storing commanded chunks, provided that
`action_sequence=execution horizon=K`, `execution_length=1`,
`replay.nstep=K`, and temporal ensembling is disabled.

Initial pass criterion:

- official actor/critic/target equations represented explicitly;
- invalid configuration rejected before training;
- JIT actor and critic updates are finite and change both parameter trees;
- open-loop execution samples only once per K primitive actions;
- checkpoints restore the target critic and optimizer state;
- a real replay episode returns `[B,K,A]`, discounted K-step reward, and
  `gamma**K`.

### 4. Execution

Implemented the method, factory registration, workspace multi-step allow-list,
Hydra configs, state-DMC and pixel-BiGym launches, checkpoint sweep evaluator,
and focused unit/integration tests.

## Stage 1: Real DMC/replay/checkpoint smoke

### 1. Previous-stage result

Run:
`exp_local/q_chunking/smoke_dmc_cartpole_balance_cpu_20260728`

Smoke-only overrides were K=5, execution length 1, n-step 5, batch 8,
two 32-wide hidden layers, two Euler steps, four candidates, CPU JIT, 1002
environment steps, and no task training claim.

Measured artifacts:

- Replay: 1000 primitive transitions after one complete DMC episode.
- Update 1: actor loss 1.699276, critic loss 6.020144, update time 0.718224 s.
- Update 2: actor loss 1.260669, critic loss 2.679330, update time 0.000517 s.
- Both updates: action-valid fraction 1.0 and chunk-valid fraction 1.0.
- Replay sample shape: `[2,5,1]`.
- Replay discount: `0.9509900499 = 0.99**5`.
- First replay return error versus a direct five-reward calculation:
  `7.77e-8`.
- Checkpoint: `snapshots/1002_snapshot.pkl` plus `latest_snapshot.pkl`.
- Independent fixed-seed checkpoint eval:
  `smoke_eval.csv` and `sweep_evals/eval_1002.json`.
- Eval seed 10000 return: 132.851613; DMC Cartpole does not expose the
  BiGym `episode_success` metric.

No Q-Chunking smoke process remained after completion.

### 2. Interpretation

This establishes end-to-end wiring across the real environment wrapper,
primitive replay storage, reconstructed action chunks, K-step TD target, JIT
updates, snapshot serialization, snapshot loading, and cached open-loop
inference. The finite losses are only wiring evidence. Two updates from random
data do not establish task quality, sample efficiency, or superiority over a
baseline.

### 3. Next-stage decision

The next safe gate is a three-camera BiGym `move_plate` visual smoke, followed
by a matched quality experiment.

Visual smoke pass criterion:

- all head/left-wrist/right-wrist inputs reach separate actor and critic JAX
  CQN encoder trees;
- one compiled update and one checkpoint eval complete without OOM or NaN;
- replay reports K=5 and every critic sample has a complete chunk.

Matched quality protocol:

- Task: `bigym/move_plate`.
- Data: identical local demonstrations and normalization statistics.
- Horizon: K=5; execution length 1; n-step 5.
- Cameras/frame stack: the same three cameras and frame stack 4.
- Training seeds: 1, 2, 3.
- Validation selection: 50 fixed episodes, seeds 400-449, at every saved
  checkpoint; select maximum success, tie-break by return, then earlier step.
- Held-out report: 200 episodes, seeds 800-999, evaluated only after checkpoint
  selection.
- Metrics: success, return, episode length, selected Q, actor/critic loss as
  diagnostics, compile-excluded update throughput, and plan inference latency.
- Baseline: a matched Best-of-N non-chunking arm must be added before making
  the paper's chunking attribution; the older CQN-AS run is secondary context,
  not that causal baseline.
- Quality pass: no held-out regression versus the matched non-chunking arm;
  an improvement claim additionally requires a positive paired bootstrap 95%
  confidence interval across held-out initializations.

### 4. Execution

The official-budget BiGym launch is configured for 1M offline updates followed
by 1M online environment steps. Because RoboBase includes offline updates in
`global_env_steps`, its upper bound is correctly set to 2M. A separate
`online_update_after_steps=5000` gate matches upstream collection before
online updates resume while retaining learned-policy actions.

The visual training launch was not started on 2026-07-28. At the final live
check, GPUs 0, 2, 3, and 5 held active training jobs; GPU 1 had a newly started
BiGym training process; and GPU 4 was assigned to evaluation daemons/watchers.
Starting another BiGym/EGL process would not be an isolated or safe
experiment. The state-DMC gate and standalone checkpoint evaluation completed
instead.
