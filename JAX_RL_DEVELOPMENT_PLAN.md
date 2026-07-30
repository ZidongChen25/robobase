# JAX RL Development and Test Plan

## Current implementation

- PPO: continuous Box policy, clipped surrogate/value objectives, GAE, timeout
  bootstrapping, advantage normalization, KL stopping, and a workspace-owned
  on-policy lifecycle.
- SAC: tanh-squashed Gaussian actor, twin critics, Polyak target, n-step replay
  discounts, terminal masking, and automatic entropy temperature.
- CQN: RoboBase coarse-to-fine action encoding, distributional dueling critic,
  C51 projection, centralized target option, demo FOSD/margin losses, and
  prioritized-replay TD priorities.
- State and trainable/frozen ResNet pixel inputs use the shared JAX observation,
  fusion, and Plucker adapter path. No first-party runtime module imports Torch.

References reviewed for this implementation:

- Stable-Baselines3 commit `4962054842c558dcf72975ac84503269455b0758`
  for PPO and SAC objective/update semantics.
- Local Torch RoboBase CQN/SAC code at HEAD
  `221cb2fe907690bb67103c17e9793cc1cbad7942`; that checkout was dirty, so
  fixture generation must record source-file hashes as well as the commit.

## Phase 1: numerical parity

1. Export deterministic Torch fixtures for SAC and CQN without adding Torch to
   the package dependency graph.
2. Compare actor parameters, log probabilities, Q values, CQN per-level logits,
   C51 targets, losses, gradients, and one optimizer step on identical tensors.
3. Add Stable-Baselines3 PPO fixtures for GAE, clipped policy/value losses,
   entropy, approximate KL, and rejected KL minibatches.

Acceptance: float32 forward/loss tolerances are documented per operator and all
one-step updates pass fixed golden tests on CPU and CUDA.

## Phase 2: pixel baseline parity

1. Port the Torch RoboBase multi-view stride CNN and its normalization exactly
   in Flax. Keep it as a plug-and-play `encoder_model.type`, alongside ResNet.
2. Match CQN visual bottleneck placement and SAC actor/critic encoder-gradient
   ownership. Add random-shift augmentation with a device-side JAX kernel.
3. Validate single-view, multi-view, frame-stack, cached frozen features, and
   Plucker-conditioned ResNet configurations separately.

Acceptance: matched parameter counts/shapes and Torch fixture parity for every
pixel baseline configuration used in DMC or RLBench.

## Phase 3: learning validation

1. Run five seeds on DMC state tasks: Cartpole Balance/Swingup, Finger Spin,
   Walker Walk, Cheetah Run, and Quadruped Run.
2. Compare return-vs-environment-step curves with Stable-Baselines3 PPO/SAC and
   Torch RoboBase CQN using identical wrappers, action repeat, seeds, and budgets.
3. Run CQN demo-driven RLBench only after the state DMC gate passes.

Acceptance: median final return and area under the learning curve are within the
predeclared tolerance of each reference; failures are investigated per seed.

## Phase 4: performance gate

1. Build isolated JAX and Torch workers with matched topology, batch, dtype,
   optimizer, replay location, update count, and observation preprocessing.
2. Exclude compilation and warmup. Synchronize the device around every measured
   region and report median, p95, coefficient of variation, memory, and XLA mode.
3. Measure act latency, update throughput, full environment throughput, and PPO
   rollout-update amortized throughput on an otherwise idle GPU.

Acceptance: JAX median steady-state throughput is at least 30% higher for each
declared baseline. No release claim is made from a non-exclusive GPU run or from
unmatched architectures.

## Phase 5: robustness and release

1. Add vector-env tests with 1, 4, and 16 environments, including mixed
   termination/truncation slots and snapshot resume mid-PPO rollout.
2. Add NaN/Inf guards, gradient/parameter norms, deterministic evaluation, and
   replay-priority integration tests.
3. Run long snapshot/resume jobs and the full first-party suite, then publish
   pinned launch profiles and benchmark artifacts.
