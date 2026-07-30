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

# GPU allocation & infra protocol

Measured footprints (7/30, RTX 5090 32.6G, crown-scale CQN-AS config):
training peaks at ~13.1G with `XLA_PYTHON_CLIENT_PREALLOCATE=false`
(JIT-compile inclusive; the default 75% preallocation pool is not real
usage); a 200-episode ne=25 eval peaks at ~1.6G.

- **Two training runs per card is the default for experiment waves**: add
  `xla_mem_fraction=0.45` to each launch (hard per-process slice, mutual
  OOM immunity; knob wired through `robobase/gpu.py`) and stagger the two
  starts by ~2 minutes so JIT compilation does not overlap. Measured
  (7/30, GPU5): peak 30.2G/32.6G, no OOM; each run 0.66x solo speed,
  1.31x total throughput per card (solo 10.7 -> 7.0 steps/s each).
- Evals and probes never share a GPU with training (7/28 EGL-exhaustion
  incident). They fit on any card with ≥2G free.
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
