# HANDOFF — CQN-AS Value Research (robobase_jaxflat, branch `jaxflat`)

Last updated: 2026-07-30. Owner: Zidong Chen (zc1525@swirl04, Imperial).
Full chronological research log with pre-registrations and raw numbers:
**`cqn-flow.md` §22–§47** (the single source of truth). Advisor-facing
summaries auto-generated in `reports/daily/` + `reports/weekly/` (cron).
User-facing summary report: `CQNAS_VALUE_REPORT.md`.

## 1. What this project established

Research question: does CQN-AS learn a real value function, or is it
imitation disguised as RL — and how do we make it genuinely benefit from RL?

**Diagnosis (settled, multi-scale):** official CQN-AS's Q is
anti-calibrated against its own returns (ρ ≈ −0.45 at 100k, −0.54 at
10.5k) and has chance-level counterfactual knowledge (sibling probe
0.567 ≈ the imitation-prior baseline 0.58). Mechanisms: margin loss
broadcasts imitation onto all bins; failure trajectories are truncated
(never grounded) so TD constrains only the shape, not the level, of
their value; near-greedy data contains episode-level but not
action-level outcome variation.

**Two independent ~+10pp improvement paths (200-ep, seeds 800-999, final
checkpoints, selection-free):**

| Arm | Task | vs official 64.6 | Probe | Seeds |
|---|---|---|---|---|
| Combined = ε-bin explore + margin decay (linear 1→0.25) | **75.0** | +10.4 | **0.644** (only arm above imitation-prior) | 2 |
| Official + QC (replay.nstep=8 × replan-8 train&eval) | 74.0 | +9.4 | 0.467 (task-only mechanism) | 1 |
| Official + low-dim mask (keep floating base) | 72.5 | +7.9 | 0.616 | 1 |
| CCFF (coarse critic + bin-conditioned flow; 10.5k line) | 71.0 vs clean 60.7 | +10.3 | — | 3 |

Factorial structure: explore alone +4.2, decay alone +0.2 → the +10.4 is
a super-additive interaction, mirrored on the probe (each factor alone at
chance). Mechanism: exploration manufactures same-state/different-bin
contrast data; margin decay lets TD consume it.

Other settled negatives: mask on the *strong* recipe −5.5 (helps weak
base only); ε-annealing harmful (−8); Gaussian noise carries +7 task
(it's fine-level exploration, σ=0.01 crosses level-2 cells constantly,
coarse bins ~2%/dim/step but boundary-deep only); explore dose/leveling
insensitive (uniform=double=71.5); replan-8 at *eval time only* is a wash
(paired Δ +0.7pp over 12 checkpoints); CCFF's Bellman share is +1.0pp
(TD-off 70.0) — its gain is hierarchy-division architecture, not RL.

**Methodology rules (hard-won, enforce):** 25/50-ep validation spikes
never survive; all claims at 200 episodes (SE≈3.2pp), fresh eval-seed
family (800+ sealed, 400+ validation, 700-711 probes); ≥2 training seeds
before promoting any arm; runs are online but eval must be on a separate
GPU or post-hoc from snapshots (in-loop eval + co-located eval processes
caused a GPU crash + EGL exhaustion incident on 7/28).

## 2. In flight right now

- **Crown arm** (`stage164`, GPU0/2, seeds 1,2): combined × QC
  (explore [0.016,0.032,0.064] persist=2 + decay + nstep=8 + replan-8).
  Criteria: ≥80 → additivity, new flagship; ≈75 → shared credit-assignment
  bottleneck. Probe adjudicates whether QC's n-step erodes counterfactual
  knowledge (0.644 vs 0.467 tension). Auto-chains probe + 200-ep.
- Paused by user: 158b (combined seeds 3,4), 163 (combined × replan-8
  without nstep; latest snapshot kept), offqc8/offmask seed-2 replications.

## 3. Next steps (ranked)

1. Read crown results (tonight) → decide flagship.
2. **Second BiGym task** — the publication-critical external-validity gap;
   everything so far is MovePlate/51 demos. Prefer one long-horizon
   (saucepan_to_hob) + one mid-difficulty task; official vs combined
   (vs crown) × 2 seeds, new async protocol.
3. Seed-2/3 for single-seed cells (official+QC, official+mask, U/H/E/N).
4. Optional probes for CCFF line; sibling-vs-sibling-only probe metric
   (computable from stored records, makes anti-imitation claim airtight).

## 4. Code map (what was added on this branch)

- `robobase/method/cqn_as.py`: `coarse_flow` (+`_pure`), `bin_explore_probs`
  (+schedule, +persist_plans), `low_dim_mask_prob/keep_last`, `bin_flip_*`,
  flow EMA; `cqn.py`: canonical CFM loss, `bc_lambda_schedule` threading.
- Env: `AppendKeysToLowDim` wrapper (floating base → tail of low_dim_state),
  `env.append_floating_base_to_low_dim`.
- Infra: `train_fast.py`+`robobase/workspace_fast.py` (device-side demo
  merge — profile-driven, ~5-8%; equivalence within run-to-run variance),
  `scripts/async_eval_watcher.py` (separate-GPU 50-ep eval, deletes
  non-milestone snapshots), `scripts/eval_cqn_as_snapshot_sweep.py`
  (one-process multi-snapshot eval, `--num-eval-envs`, `--replan-interval`),
  `robobase/gpu.py` EGL setdefault fix.
- Instruments: `analyze_cqn_branch_counterfactual.py` (sibling probe:
  12 seeds × 3 anchors × 5 executed bins, state save/restore, greedy
  refinement inside forced bin), `analyze_cqn_value_fidelity.py`
  (needs `--offline-episode-count 51` on self-imitation runs).
- Automation: `scripts/daily_research_digest.sh` (cron 23:30) /
  `weekly_research_report.sh` (cron Sun 20:00) → `reports/`.
- Stage runners: `scripts/run_cqn_stage1*.sh` (each stage's exact commands).

## 5. Gotchas

- Snapshots are 429MB (full state incl. optimizer); graceful SIGINT saves
  an exact-step snapshot; resume = same `hydra.run.dir`.
- EGL enumeration ≠ CUDA order when a GPU degrades; `gpu.py` now respects
  external `MUJOCO_EGL_DEVICE_ID`. Vectorized eval (ne=25) is fast but
  nondeterministic (batched-kernel atomics) — sealed evals stay ne=1...
  (current 200-ep protocol uses ne=25; noise ±1 episode, accepted).
- Training itself is not bit-reproducible run-to-run (stock prefetch
  timing + GPU kernels): equivalence tests must be judged against
  rerun variance (~2.6% loss drift), not bit-identity.
- MovePlate has 51 successful demos (not 35); `demos=10` loads 9
  successful; `env.expected_successful_demos=null` needed when subsetting.
- pkill/pgrep self-match: always bracket patterns (`stage16[4]`).
- GPU1 crashed 7/28 (co-located evals) → fixed by reboot 7/29; keep evals
  off training GPUs.
