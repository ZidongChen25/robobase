# Research stage protocol

For research and experiment work in this repository, close the loop at every
meaningful stage. Do not stop after listing future work.

Each stage report must contain, in this order:

1. **Previous-stage result:** report the actual measured result from current
   artifacts, including the best-checkpoint comparison when applicable.
2. **Interpretation:** state what the result establishes, what it rules out,
   and what remains unresolved. Do not present training loss as policy quality.
3. **Next-stage decision:** define the next hypothesis, matched baselines,
   selection split, held-out split, metrics, and pass/fail criterion.
4. **Execution:** immediately implement or launch the safe in-scope next stage,
   then verify that it really started or completed from processes and output
   artifacts. If execution is blocked, report the concrete blocker.

Additional requirements:

- Compare each method at its validation-selected best checkpoint; do not create
  an advantage by comparing against an overtrained final checkpoint.
- Keep distinct research questions in separate experiments so a result has one
  clear interpretation.
- For running jobs, give an artifact-backed progress update and ETA. At the
  next meaningful milestone or completion, report the result before launching
  the following stage.
- Record the protocol, exact run paths, results, interpretation, and next
  decision in the relevant research Markdown file.
- A status answer must inspect live processes, logs, CSV/JSON outputs, and
  checkpoints rather than infer completion from an earlier launch.

# BiGym demo/MDP mismatch (settled 2026-08-03)

Live BiGym terminates on the first successful step and pays the sparse reward
once (`third_party/bigym/bigym/bigym_env.py:164,488`). Recorded demonstrations
keep a post-success tail, and the replay loader only marked the LAST frame
terminal — so **replayed demos were a different MDP from the one the agent acts
in**. Measured on move_plate (60 demos, 51 successful): median 24 consecutive
reward-1 steps per demo; discounted RTG (gamma 0.99) median 9.45, max 23.77.
**96.0% of demo transitions exceed the C51 `v_max=2.0` and clip to the same top
atom**, so the critic gets an identical label on almost every demo sample and
cannot learn temporal ordering; meanwhile the agent's own online successes are
worth <= 1.0, i.e. 10-20x *less* than a demo state.

`env.truncate_demo_at_success` is **true by default since 2026-08-03**. It sets
`terminal=True` at the first demo step with reward > 0.25 and drops the tail:
RTG becomes [0.044, 1.0] (0% clipped), demo transitions 9,287 -> 8,072 (-13.1%).

Measured effect (sealed 200-ep, seeds 800-999, fixed endpoint, no checkpoint
selection), all seed 1, official CQN-AS otherwise unchanged:

| | self-imitation on | self-imitation off |
|---|---:|---:|
| untruncated | 67.5 | 44.0 |
| **truncated** | **78.5** | 52.5 |

Truncation is worth +11.0 / +8.5; self-imitation is worth -23.5 / -26.0. The
two are additive with no interaction — truncation fixes the value SCALE, self
imitation supplies online-success COVERAGE.

**Consequences for the research lines:** every "+X pp over official 64.6" claim
was measured against a baseline that moves to ~74.5 once the data is corrected,
and every knob tuned on the saturated critic (`bc_lambda_schedule`,
`bin_explore_probs`, `q_reward_scale`, `unseen_return_floor_weight`, C51
support width) is off-optimum by construction on truncated data. Re-measure
before claiming. The official four-seed 64.625 reference exists ONLY at the
101k endpoint (its intermediate snapshots were deleted), so comparisons against
it must stay at 101k even though 100k is the preferred reporting point.

# Delayed-policy conditioning (`obs_delay`, added 2026-08-08)

`obs_delay: h` (top-level, default 0) makes the policy predict `a_t` from
`o_{t-h}` instead of `o_t`, for the non-Markov-demo hypothesis: the operator
reacted to what they perceived `h` steps earlier. Override on either entrypoint
with `obs_delay=2`; it is one knob for training AND evaluation, and
`scripts/eval_cqn_as_snapshot_sweep.py --obs-delay N` overrides it per sweep
(default: whatever the run trained with).

`h` counts **environment steps, not policy decisions** — with
`action_sequence=20, execution_length=20`, `obs_delay=1` is one env step, not
one chunk. Two mechanisms, each applying the shift exactly once:

- `ObservationDelay` (`robobase/envs/wrappers/observation_delay.py`) sits after
  `FrameStack` and inside `ActionSequence` in every `_wrap_env`, on live *and*
  demo envs. So online rollouts, eval, and demos imported through `DemoEnv`
  into `UniformReplayBuffer` all store/see `(o_{t-h}, a_t)`.
- `LazyBiGymReplayBuffer` reads demo files directly and never touches an env, so
  it shifts observation indices at sampling time instead.

Only observations move; actions, rewards and the n-step bootstrap keep their
timing. Episode starts repeat the reset frame in both mechanisms.

**Gotcha:** because the demo env bakes the delay in, `replay.demo_cache_dir`
caches cannot be shared across values of `h` — `demo_cache_key` includes
`obs_delay`, so sweeping `h` re-materializes the replay-formatted demos. The
lazy-replay path has no such cost.

# GPU allocation & infra protocol

Measured footprints (7/30, RTX 5090 32.6G, crown-scale CQN-AS config):
training peaks at ~13.1G with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
(JIT-compile inclusive; the default 75% preallocation pool is not real
usage); a 200-episode ne=25 eval peaks at ~1.6G.

- **Training may occupy at most TWO cards at a time** (user directive,
  8/03). This is a shared cluster; do not spread one run per card across
  every GPU. Two cards x two runs = **four concurrent training runs is the
  hard ceiling**. Anything beyond that waits in a queue and starts only as
  a slot frees — see `scripts/consolidate_two_cards.sh` for the pattern
  (stop at the next checkpoint, relaunch on a permitted card, poll for
  slots). A third run on one card would OOM: 3 x 13.1G > 32.6G.
- **Two training runs per card is the default for experiment waves**: add
  `xla_mem_fraction=0.45` to each launch (hard per-process slice, mutual
  OOM immunity; knob wired through `robobase/gpu.py`) and stagger the two
  starts by ~2 minutes so JIT compilation does not overlap. Measured
  (7/30, GPU5): peak 30.2G/32.6G, no OOM; each run 0.66x solo speed,
  1.31x total throughput per card (solo 10.7 -> 7.0 steps/s each).
- **Evaluation may use every otherwise-idle card** (user directive, 8/03):
  any GPU whose utilization/occupancy is below ~80% is fair game, and a
  single card can host several concurrent evals (a 200-ep ne=25 eval peaks
  at ~1.6G, so a free 32.6G card holds many). Evals are the one workload
  that should spread wide — they are short and the cards would otherwise
  sit idle. Still never co-locate an eval with training (7/28
  EGL-exhaustion incident), and stagger eval starts ~20 s so their env
  construction does not collide.
- `train_fast.py:39` auto-resumes from `snapshots/latest_snapshot.pkl` in
  the run dir, so relocating a run between cards costs only restart
  overhead (~4 min demo load + JIT), not training progress. Stop at a
  checkpoint, then relaunch the identical command with the same
  `hydra.run.dir`.
- Infra benchmarks run on GPU5 (user directive), via
  `scripts/bench_vecsample_equiv.sh` (ABBA order, steady-state steps/s
  over env_steps 1000->3000, cross-arm critic_loss drift judged against
  within-arm rerun drift).
- Infra rollback switches: `ROBOBASE_SCALAR_SAMPLE=1` (per-sample replay
  assembly), `ROBOBASE_HOST_MERGE=1` (host-side demo merge). Use them only
  for A/B attribution, not silently in experiments.
- Snapshot/resume: exploration RNG streams persist in checkpoints;
  persisted bin-explore windows deliberately do not (resume starts fresh
  episodes windowless; cqn-flow.md 48.2-48.3).
