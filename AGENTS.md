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
