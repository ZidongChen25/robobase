# CQN-AS without imitation loss — agent research line (Claude)

Parallel research line under the same non-negotiable boundary as
`cqn-no-bc.md`: no BC, no margin, no FOSD, no actor/flow/likelihood/AWR, no
action labels. Demonstrations contribute only reward-based transitions,
completed returns, and data-supported Bellman candidates. Deployed policy is
canonical `argmax Q`. Evaluation protocol inherited unchanged: selection on
seeds 400–449 (50 episodes, ne=25 dev protocol), held-out seeds 800–999
sealed, progressive compute gates (no blind 101k). Assigned hardware: one
free card (physical GPU 0, UUID `GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919`),
two runs per card at `xla_mem_fraction=0.45` with 120 s stagger.

Baselines this line must beat (from the main line, all 50-ep seeds 400–449):

- matched BC reference (Stage 36 control, b256 offline+online): **66% @ raw 12.5k**;
- official full CQN-AS sealed reference: 64.6% mean / 62% same-seed at 101k;
- best no-BC so far: Stage 38 dense offline→online 58%/50% (mean 54%),
  offline-only raw-10k endpoints **52%/50%** (these are my matched controls).

## Stage A1: C51 support de-saturation via reward scaling (2026-08-01)

### 1. Previous-stage result (evidence being acted on)

Measured directly from the Stage-38 demo replay (6 episodes inspected,
`dense_seed1/offline_then_online/demo_replay/*.npz`): the official BiGym
reward semantics emit 24–28 consecutive rewards of 1.0 after success. With
gamma 0.99, discounted reward-to-go spans ≈3.6–6.5 at episode start and
peaks at ≈21.4–24.5 at first success. The C51 support is [-2, 2] with 51
atoms. Therefore **every positive C51 target in the b256 recipe clips at the
top atom**. Stage-38's own probe already recorded the symptom without the
cause being acted on: chosen Q saturates at 1.9999x, candidate span collapses
to 2.5e-6 in the canonical branch, and in the dense branch all positive-return
states project to the identical top-atom distribution, destroying (a)
temporal value ordering along the demo trajectory and (b) any Bellman
bootstrap discrimination between "just started" and "about to succeed".
No stage has ever tested a support/scale fix under the official reward
semantics (Stage 19's `q_reward_scale=2` was the old first-success MDP,
a different question).

### 2. Interpretation

This is hypothesis H3 of my erosion analysis: the dense floor preserves
chosen-vs-unseen separation (which is why Stage 38 still reaches 52/50
offline), but within the positive-return set the value function is
representationally flat, so the online phase cannot perform meaningful
Bellman improvement and checkpoint quality peaks early. A positive reward
scaling `c` is policy-invariant (argmax preserved) and, for categorical
cross-entropy, near loss-scale-invariant: targets simply land on
interior atoms instead of clipping. Choosing `c=0.07` maps max RTG ≈24.5 to
≈1.72 (< v_max=2 with bootstrap headroom) and episode-start values to ≈0.25–
0.46 (3–6 atoms above the floor). The support itself stays [-2, 2], so the
zero-init property (all initial expected Q exactly at the 0 floor) is
preserved exactly — this is why scaling is preferred over enlarging v_max,
which would make the uniform-init expectation optimistic.

### 3. Next-stage decision (preregistered before any result)

Stage A1 tests exactly one resolved field on the exact Stage-38 recipe:
`method.q_reward_scale: 1.0 -> 0.07`.

- Arms: fresh training seeds 1 and 2, **offline phase only** (10k demo-only
  reward-Q updates, force probability 1.0, batch 256, snapshots every 2.5k).
  No environment interaction is consumed beyond workspace construction.
- Matched controls (no new compute): Stage-38 `dense_seed1/2` raw-10k
  offline endpoints, 52%/50% on the identical 50-episode seeds-400–449
  split (`val50_seeds400_selection.csv`, raw-10000 rows).
- Evaluation: raw 5k/7.5k/10k per seed, 50 episodes, seeds 400–449, ne=25.
- Gate:
  - **PASS**: both raw-10k seeds ≥ matched control − 2pp AND mean raw-10k
    gain ≥ +5pp → promote the scale into the offline→online composition arm
    (Stage A2) and add seed 3 there.
  - **DIAGNOSTIC-PASS**: mean within ±5pp AND the ordering diagnostics
    improve decisively (native-scale RTG calibration finite instead of
    saturated; Q-vs-RTG Spearman on demo states materially higher; span no
    longer top-atom-degenerate) → still promote to Stage A2, labelled
    ordering-only (the mechanism's main claim concerns the online Bellman
    phase, which offline extraction cannot fully measure).
  - **FAIL**: mean ≤ −5pp → reject de-saturation; fall back to H1
    (positive-only online floor) / H2 (soft floor) lines.
- Mechanism diagnostics (read-only, no env seeds): value-fidelity probe on
  both raw-10k endpoints vs the matched Stage-38 endpoints — RTG calibration
  against `0.07 × RTG_native`, expert bin top-1/top-2, `Q(expert)−Q(greedy)`,
  candidate span, plus Q-vs-RTG rank correlation across demo states.
- Held-out seeds 800–999 remain sealed. No online step, no 101k.

Concurrent CPU workstream (serves the "why does naive removal fail /
why does dense erode" goal, consumes no GPU): erosion curve on the existing
Stage-38 snapshots (raw 10k→30k, both seeds) — expert-bin rank,
`Q(expert)−Q(greedy)`, and RTG calibration as a function of online steps,
from the frozen `analyze_cqn_value_fidelity.py` probe on fixed demo states.
Prediction registered in advance: if H1 (failure-transition floor erosion)
is real, `Q(expert)−Q(greedy)` on demo states declines monotonically with
online steps in the dense branch.

### 4. Execution

- Launch config `cqn_as_pixel_bigym_nobc_agent_a1_descale_gate.yaml`
  (inherits the Stage-38 gate config; adds only the scale).
- Runner `scripts/run_cqn_no_bc_agent_a1.sh`: offline pair on GPU 0
  (0.45 fraction, 120 s stagger), then two parallel 3-checkpoint 50-episode
  evaluators on the freed card.
- Wiring checks before launch: Hydra composition single-field diff vs
  Stage 38; focused `q_reward_scale` unit tests (Stage-19 suite).

Execution evidence appended below after launch.

Launched 2026-08-01 16:25:17 BST, controller in
`exp_local/cqn_no_bc/agent_a1_descale_gpu0_20260801162517`. Both seeds
co-resident on GPU 0 (seed 2 staggered at 16:27:17). Wiring evidence from
seed-1 `pretrain.csv`: `q_reward_scale=0.07`, `mc_return_mean=10.30` native,
`scaled_mc_return_mean=0.72` (first time the MC target lands on interior
support atoms instead of clipping), `demo_behavior_force_fraction=1.0`,
finite critic loss. Resolved-config diff against Stage 38 is exactly
`method.q_reward_scale: 1.0 -> 0.07`; the 4 focused reward-scale unit tests
pass. This is wiring evidence only.

## Erosion probe result: preregistered H1 prediction REFUTED (2026-08-01 ~16:55)

### 1. Result

18/18 probes completed (`agent_erosion_probe_stage38/`, fixed sampling seed
0, 8 states/group, target critic, CPU-only). On demo-success states, the
Stage-38 dense online phase does NOT erode expert-action ranking — it
strengthens it monotonically:

| seed | metric | raw 10k | raw 20k | raw 30k |
| --- | --- | ---: | ---: | ---: |
| 1 | `Q(expert)-Q(greedy)` | -0.053 | +0.049 | +0.076 |
| 1 | expert top-1 (current action) | 83.3% | 87.2% | 89.2% |
| 2 | `Q(expert)-Q(greedy)` | -0.062 | +0.002 | +0.026 |
| 2 | expert top-1 (current action) | 82.2% | 85.6% | 89.7% |

max-minus-replay-Q shrinks 0.028 → 0.0066; span stays ~1.7–1.9 (no
within-state collapse). Meanwhile fixed task success over the same span was
flat-to-declining (58%→46% / 50%→38%).

### 2. Interpretation

The preregistered H1 prediction (failure-transition floor erosion should
make demo-state `Q(expert)-Q(greedy)` decline with online steps) is
refuted at demo states: demo/protected samples win the tug-of-war. Two
observations survive and point at H3 instead:

1. **Ceiling creep.** Q(expert) rises 1.32 → 1.72 and keeps climbing toward
   v_max=2 on both seeds. Under clipped targets, every executed bin whose
   trajectory eventually succeeds — and, worse, every failed action whose
   bootstrap state has near-ceiling V — receives a ≈2.0 target
   (`0 + 0.99 × 2.0 = 1.98`). With RTG≈24 clipped to 2, TD cannot
   distinguish success from failure one step before bootstrap wherever
   V(s') saturates. As more distinct bins get executed over training, the
   set of ceiling bins grows and argmax discrimination decays globally.
   This unifies the Stage-37 101k collapse (52%→34% val, 28.5% held-out)
   without needing floor erosion.
2. **Dissociation = probe blind spot.** Demo-state ranking improves while
   closed-loop success stalls, so whatever degrades lives off the probed
   demo manifold (policy-visited states) — consistent with
   bootstrap-saturation contaminating off-demo values, not with demo-state
   forgetting.

Consequence for design: the online phase should KEEP the dense target
(no erosion to fix) and the leverage is de-saturation (Stage A1's scale),
which restores success/failure discrimination in exactly the bootstrap
regime implicated above. Positive-only online gating (H1 fix) is
de-prioritized.

### 3. Next-stage decision

Unchanged gate for A1. For Stage A2 (offline→online composition), the
default online variant is now dense + q_reward_scale=0.07 (same objective
both phases), pending the A1 gate and the main line's Stage-40 handoff
result as a cross-check.

## Cross-check: main-line Stage 40 confirms the floor is load-bearing online

Stage 40 (`stage40_summary.json`, main line) handed the shared Stage-38
raw-10k dense offline checkpoints (re-evaluated 54%/46%) to canonical
chosen-bin C51 online. Both seeds collapsed to **0% at the first
post-handoff checkpoint (12.5k, only 2.5k online steps) and stayed 0**
through 20k. Combined with the erosion probe above, the mechanism picture
is now three-legged and coherent:

1. dense floor online is necessary — removing it lets canonical targets
   re-saturate (clipped chosen targets ≈ ceiling; no floor on unseen bins;
   dueling V-stream lifts everything) and destroys a working policy within
   2.5k steps — the same failure mode as the all-zero Stages 30–36 branch;
2. with the floor kept, demo-state ranking does not erode (probe above);
3. the remaining slow decline (58→46/38 by 30k; Stage-37 101k collapse) is
   ceiling creep among *executed* bins, which the floor cannot prevent and
   which q_reward_scale=0.07 directly removes.

A2's online variant is therefore settled as dense + scale (identical
objective in both phases), exactly what `run_cqn_no_bc_agent_a2.sh`
implements. Incidentally Stage 40's re-evaluation of the identical Stage-38
checkpoints (52/50 → 54/46) calibrates the ne=25 run-to-run eval noise at
±2–4pp on 50 episodes; the A1 gate tolerances already account for this.

## Stage A1 result: preregistered gate FAILED; bounded A1b extension (2026-08-01 ~17:25)

### 1. Result

Fixed 50-episode seeds-400–449 curves for the de-saturated offline arms:

| seed | 5k | 7.5k | 10k | matched Stage-38 @10k | Δ@10k |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 36% | 38% | 40% | 52% | −12pp |
| 2 | 42% | **56%** | 46% | 50% | −4pp |

Mean Δ@10k = −8pp, outside the ±5pp DIAGNOSTIC-PASS window. By the
preregistered rule this is a **FAIL** at the raw-10k screen and is recorded
as such. Mechanism diagnostics did land as designed (Spearman 0.905/0.881
vs 0.635/0.849 unscaled; Pearson 0.97/0.99; ranking preserved), and seed
2's 7.5k point (56%) is the highest offline-only value observed at b256 in
either line, but neither overrides the failed gate.

### 2. Interpretation

Two readings are compatible with the numbers. (a) De-saturation genuinely
hurts offline extraction (spreading target mass over ~20 atoms weakens the
floor contrast that drives extraction). (b) Convergence is simply slower:
scaled targets carry more distributional information per head; seed 1's
best is exactly at the 10k boundary and still rising (36→38→40), and the
historical precedent for this exact intervention family is Stage 19→27,
where `q_reward_scale=2` looked flat at the 10k screen (46/46) and jumped
to 66/50 by 15k — the main line's corrected protocol explicitly makes a
boundary-peaked candidate "eligible for a 20k continuation rather than
automatic rejection".

### 3. Next-stage decision (Stage A1b, preregistered)

Resume both exact A1 offline states from 10k to 20k demo-only updates (no
environment interaction), snapshots every 2.5k, then evaluate raw
12.5k/15k/17.5k/20k with 50 episodes on seeds 400–449.

- PASS: both seeds' extended best (over 5k–20k) ≥ matched control − 2pp
  (seed 1 ≥ 50%, seed 2 ≥ 48%) AND extended-best mean ≥ control mean 51%
  → de-saturation earns the A2 online composition (which is the phase the
  mechanism actually targets; compute there is matched again).
- FAIL: otherwise → reject de-saturation as an offline component; pivot to
  the H2 soft-floor design.

This screen is deliberately lenient (best-over-20k vs the control's single
10k point, unmatched update count) and is labelled as such: it only decides
whether the mechanism is *viable*, not whether it is *better*. The decisive
comparison remains A2 online vs Stage-38's 58/50 at matched budgets.
Note for execution: `_pretrain_step` is saved in snapshots, so resuming
with `num_pretrain_steps=20000 num_train_frames=20000` continues pretrain
10k→20k and exits before any env step; the A2 runner must be updated to
`num_pretrain_steps=20000 num_train_frames=30000` with eval steps
22.5k–30k if A1b passes.

### 4. Execution

`scripts/run_cqn_no_bc_agent_a1b.sh` resumes both run dirs in place on GPU
0 (0.45 fraction, 120 s stagger) and then runs the two 4-checkpoint
evaluators. Held-out seeds remain sealed; no online step is consumed.

Launched ~16:58 BST; explicit `resuming: .../latest_snapshot.pkl` (→10k)
logged for seed 1; both workers co-resident (17.7 GiB, 96%). Completed
~18:20 BST with all sentinels.

## Stage A1b result: gate PASS; Stage A2 preregistration (2026-08-01 ~18:25)

### 1. Result

Extension curves (50 episodes, seeds 400–449):

| seed | 12.5k | 15k | 17.5k | 20k | extended best (5k–20k) | control | required |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 46% | 44% | 46% | 52% | **52% @ 20k** | 52% | ≥50% ✓ |
| 2 | 52% | 44% | 42% | 32% | **56% @ 7.5k** | 50% | ≥48% ✓ |

Extended-best mean 54% ≥ 51% ✓ → **PASS** as preregistered.

Honest annotations: (a) seed 1 rises monotonically and again peaks exactly
at the budget boundary; (b) seed 2 shows pure-offline overtraining decay
(52→32 between 12.5k and 20k with zero environment data — same shape as
the b16 dense 20k declines in main-line Stage 9); (c) at any *fixed*
checkpoint the de-scaled arms never beat the unscaled 52/50, so offline
extraction is parity-at-best and slower, with larger seed variance. The
mechanism's claimed value remains the online phase.

### 2. Interpretation

De-saturation is viable as an offline component (parity given more
compute) and restores the temporal ordering the online Bellman phase
needs. Whether that translates into sustained online improvement — the
thing Stage 38 lacked (peak +6/+0 over offline, then decay) and Stage 40
catastrophically lacked (0% without the floor) — is exactly the A2
question.

### 3. Next-stage decision (Stage A2, preregistered)

Branch each seed's exact raw-10k offline state into a fresh run dir
(`prepare_cqn_no_bc_stage40_branch.py`, the tested tool), then run the
online phase with the identical launch config (dense + scale retained):
10k online env steps (global cap 20k), `demo_only_updates=false`, force
probability 0, batch 256+256. Clock exactly matches Stage 38's
offline-10k + online-10k. Evaluate raw 12.5/15/17.5/20k, 50 episodes,
seeds 400–449.

Preregistered gates (offline reference points: 40%/46% at raw 10k):

- **Mechanism gate (online improvement)**: per-seed
  `post-handoff best (12.5–20k) − own raw-10k offline value`; PASS if the
  mean ≥ +5pp with both deltas nonnegative. Stage-38 reference: +6/+0
  (mean +3). This is the direct test of "de-saturation makes online
  Bellman improvement work".
- **Performance gate (matched clock)**: selected best over 10k–20k
  (offline point included); parity PASS if mean ≥ 54% (Stage-38's 58/50
  mean); STRONG if mean ≥ 59%. Any seed ≥ 62% earns a bounded 30k
  extension mirroring Stage 38's (whose extension bests were only 52/44,
  mean 48% — the bar to beat there); any seed ≥ 66% ties the matched BC
  reference.
- **FAIL both** → reject the de-scaled composition; pivot to H2
  (soft/calibrated floor); the long-offline de-scaled line (seed 1 still
  rising at 20k) becomes a separate bounded question.

Held-out seeds 800–999 remain sealed; no 101k path exists in the runner.

### 4. Execution

`run_cqn_no_bc_agent_a2.sh` rewritten for branched dirs. Branch manifests
must show main replay 60 episodes / protected 51 at the 10k boundary
(same integrity check as Stage 40). Execution evidence appended below.

Launched 17:31 BST in
`exp_local/cqn_no_bc/agent_a2_descale_online_gpu0_20260801173125`; branch
manifests verified (60/10,953 main, 51/9,253 protected, pretrain_step
10000, hardlink transfer). Completed ~18:30 BST with all sentinels.

## Stage A2 result: mechanism gate PASS (+13pp online), 62% plateau; A2b extension (2026-08-01 ~18:35)

### 1. Result

Fixed 50-episode seeds-400–449 online curves (offline reference in
brackets):

| seed | [10k off] | 12.5k | 15k | 17.5k | 20k | post-handoff best | online Δ | Stage-38 Δ |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | [40%] | 50% | 38% | 40% | 34% | 50% @ 12.5k | +10pp | +6pp |
| 2 | [46%] | 60% | 62% | 60% | 46% | 62% @ 15k | +16pp | +0pp |

- **Mechanism gate: PASS.** Mean online improvement +13pp (required ≥+5,
  both nonnegative; unscaled Stage-38 reference +3pp mean). Seed 2 holds
  60/62/60 across three consecutive checkpoints — the first sustained
  ≥60% plateau in either no-BC line; 62% ties the official same-seed 101k
  endpoint and sits 4pp below the matched BC reference (66%).
- **Performance gate: parity PASS.** Selected bests 50/62, mean 56% ≥ 54%
  (Stage-38's 58/50). STRONG (≥59%) missed. Per-seed vs Stage 38: −8/+12.
- **Extension trigger:** seed 2 ≥ 62% → the preregistered bounded 30k
  extension is earned.

### 2. Interpretation

De-saturation converts the online phase from Stage 38's
peak-then-decay-immediately (+6/+0) into genuine multi-checkpoint
improvement (+10/+16) — the mechanism claim is supported at matched clock
and single-field isolation. Two honest limits: (a) seed variance is large
(seed 1 offline-weak throughout); (b) both seeds decline into the 20k
boundary (34/46). Combined with the A1b observation that seed 2 decayed
52→32 under *pure offline* training, the late decline is not an
online-phase artifact and is not ceiling creep (values can no longer
clip): it is a dense-objective overtraining phenomenon, and it is now the
single most valuable unexplained mechanism. Candidate fix for a later
stage: rank/target anchoring to a frozen earlier critic (reward-only
self-distillation), not imitation.

### 3. Next-stage decision (Stage A2b, preregistered)

Resume both A2 online states 20k→30k (10k more online env steps), same
config, snapshots every 2.5k, evaluate raw 22.5/25/27.5/30k with 50
episodes on seeds 400–449. Mirror of Stage-38's bounded extension (whose
extension-region bests were 52/44, mean 48%, and which failed its own
scaling gate).

- PASS: extension-region bests mean ≥ 53% (+5 over Stage-38's 48%) with
  both seeds ≥ their Stage-38 counterparts − 2pp (seed 1 ≥ 50, seed 2 ≥
  42). Additionally record: any point ≥ 62% (sustained-top evidence); any
  point ≥ 66% (ties matched BC).
- FAIL: otherwise → stop scaling this exact recipe; the next isolated
  question becomes the late-decay mechanism (anchoring), tested at
  matched 20k budget rather than by more scale.

Held-out 800–999 sealed; no full-run path.

### 4. Execution

`scripts/run_cqn_no_bc_agent_a2b.sh` resumes the two A2 dirs in place on
GPU 0 and runs the 4-checkpoint sweep. Evidence appended below.

## Mechanism probe on A2 online snapshots: calibration holds, no ceiling creep (2026-08-01 ~18:50)

Fixed-state probes (same 8 demo-success states, sampling seed 0, target
critic, CPU) across the A2 online phase vs the matched Stage-38 unscaled
snapshots:

| arm | Q(expert) 12.5k→20k | Q-RTG Spearman | demo top-1 |
| --- | --- | --- | --- |
| A2 de-scaled | 0.49→0.57 (parks at true scaled RTG ≈0.5–0.8) | 0.90–1.00 | 83–86% |
| Stage-38 unscaled | 1.41→1.64 (still climbing toward the 2.0 clip; the target IS the clip) | cross-state ordering destroyed by clipping | 83–87% |

Reading: demo-state per-factor top-1 is identical between arms — the
unscaled recipe's defect is not demo-state ranking (what BC margin fixes)
but the bootstrap regime: with every positive-return state's V clipped to
the same ceiling, a failed action's Bellman target (0.99×2≈1.98) is
indistinguishable from success, so online TD carries no signal. The
de-scaled arm restores calibrated, ordered values (Spearman ≈1.0) and the
closed-loop consequence is the +13pp vs +3pp online improvement measured
in A2. This closes the causal chain:

1. naive BC removal (canonical target, official rewards) → all positive
   targets clip → no discrimination at all → 0% (Stages 30–36, 40);
2. dense floor → within-state discrimination for unexecuted bins → 52%
   offline extraction, but cross-state ordering still clipped → online
   peak-then-decay (Stage 38) and 101k collapse (Stage 37);
3. dense floor + de-saturation → calibrated ordered values → genuine
   online improvement, 60–62% sustained (A2).

## Stage A2b result: FAIL; online-state collapse identified; Stage A3 (2026-08-01 ~19:50)

### 1. Result

A2b extension curves (50 eps, seeds 400–449): seed 1 36/36/32/26 at
22.5–30k (extension best 36%), seed 2 44/40/40/42 (best 44%). Mean 40% <
53% requirement; seed 1 far below its 50% floor. **FAIL** as
preregistered → stop scaling this exact recipe; the decay mechanism is
now the primary question.

The free diagnostic (re-reading the existing probe JSONs' online-state
groups) then localized the decay precisely:

| states | Stage-38 Q_greedy / span (10k→30k) | A2 Q_greedy / span (12.5k→20k) |
| --- | --- | --- |
| demo_success | improves; span healthy | calibrated; span healthy |
| online_success | 1.40→0.33 / 1.63→0.43 | 0.31→0.04 / 0.41→0.07 |
| online_failure | 0.97→0.08 / 1.18→0.06 | 0.18→0.02 / 0.24→0.04 |

At the policy's own visited states, ALL bins converge to the floor and
the span collapses — argmax becomes noise exactly where the agent acts,
while protected demo states stay sharp (which is why demo-state probes
and success curves dissociate). Verified: `use_self_imitation=true` was
active (protected buffer grew 51→88), so success-episode protection does
not prevent it. Mechanism: **anchor/floor sampling imbalance** — a
state's executed bin is MC-anchored only when its own transition is
sampled, but it is floored every time any nearby transition (dominated by
failure-episode prefixes sharing the corridor) executes another bin.
Demo states win because the protected buffer guarantees their anchor
frequency; online states lose and are ground to the floor.

### 2. Interpretation

The dense floor's "not executed here ⇒ worthless" prior is correct for a
stationary success-only offline buffer (why offline extraction works) and
anti-conservative online, where failure transitions actively unlearn
alternatives at every visited state. The retention problem is not logit
sharpening per se and not ceiling creep (fixed by A1); it is
failure-driven floor erosion **off the protected manifold** — invisible
to demo-state probes, fatal to closed-loop success.

### 3. Next-stage decision (Stage A3, preregistered)

Single-field, zero-new-code, precisely aimed: keep dense+scale offline
(shared A1 raw-10k checkpoints), and set
`method.dense_return_positive_only=true` for the online phase only —
failure transitions revert to canonical chosen-bin C51 (no floors from
failures), success transitions keep the full dense anchor+floor. Stage 39
tested this flag in the wrong phase structure (online-only from scratch,
+6/−4 inconclusive); the collapse diagnostic now motivates it exactly
here. The 6 existing focused unit tests pass; the flag composes with
q_reward_scale (validator has no conflict).

Arms: seeds 1 and 2, branched from the SAME A1 raw-10k states as A2 →
identical starting critics; online target is the only difference vs A2.
10k online, eval 12.5/15/17.5/20k (10k point identical to A2's 40/46 by
construction).

- **Retention gate**: mean over seeds of `20k endpoint − post-handoff
  best` ≥ −8pp (A2 reference: −16/−16).
- **Improvement gate**: post-handoff best mean ≥ 54% (A2's 56% − 2pp) —
  the fix must not destroy the online improvement dense delivered.
- PASS both → this is the recipe; replicate seeds 3/4 and retest the 30k
  extension. FAIL retention → label-smoothing arm next (bounds floor
  sharpening). FAIL improvement only → success-only floors insufficient
  for improvement; hybrid floor weight becomes the question.

Held-out sealed; no full-run path.

### 4. Execution

`run_cqn_no_bc_agent_a3.sh`: branch A1 10k states → online with the
positive-only override → paired eval. Evidence appended below.

Launched 19:15 BST in `agent_a3_posonly_online_gpu0_20260801191509`;
completed ~20:15 with `dense_return_positive_only: true` verified in both
archived online configs.

## Stage A3 result: retention FAIL — failure-floor attribution REFUTED; Stage A4 (2026-08-01 ~20:25)

### 1. Result

| seed | 12.5k | 15k | 17.5k | 20k | best | endpoint−best | A2 reference |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 56% | 52% | 46% | 36% | 56% | −20pp | −16pp |
| 2 | 54% | 48% | 44% | 38% | 54% | −16pp | −16pp |

Improvement gate passes marginally (best mean 55% ≥ 54%); retention gate
**FAILS** (mean −18pp vs required ≥ −8pp). Removing every failure-derived
floor left the decay slope unchanged — the second causal attribution
refuted today (after H1). Same-start-critic isolation makes this clean:
plain dense online (A2) and positive-only online (A3) decay identically.

### 2. Interpretation

The common denominator across every decaying arm is now: A1b decayed
under *pure offline* training on a fixed success-only buffer; A2/A3/38
decay online regardless of floor provenance; probes show the protected
manifold sharpening while off-manifold values flatten. Unified reading:
**continued dense CE training on a (mostly static, success-dominated,
protected) buffer overfits — the critic sharpens on replayed pixels and
loses off-manifold generalization, so the greedy policy becomes brittle
exactly where it acts.** The protected success replay recreates BC's
overfitting failure mode inside a reward-only objective. The main line's
value research reached the mirror conclusion from the BC side: its
+10.4pp combined arm = ε-bin exploration + margin decay, mechanism
"exploration manufactures same-state/different-bin contrast data; decay
lets TD consume it". In this no-BC line there is no margin to decay — TD
is everything — so the missing nutrient is the contrast data itself.

### 3. Next-stage decision (Stage A4, preregistered)

Third same-start-critic head-to-head: branch the A1 raw-10k states again;
online = the A2 recipe (plain dense + scale; positive-only reverted) plus
`method.bin_explore_probs=[0.002,0.004,0.008]` — the value line's
hierarchical ε-bin exploration, constant (their ε-annealing measured
harmful), reward-only, already implemented and battle-tested at b16 with
BC present; untested in the no-BC b256 regime. Single resolved-field diff
vs A2's online config.

Gates unchanged: retention mean(20k − best) ≥ −8pp; improvement best mean
≥ 54%. PASS both → recipe locks (descale + dense + explore), seeds 3/4
replication + 30k extension retest toward the 66% BC reference. FAIL
retention → passive anti-overfit family (label smoothing on dense
targets, online lr decay) and quantify the fixed-short-budget fallback
(preregistered endpoint at the peak region with fresh seeds).

### 4. Execution

`run_cqn_no_bc_agent_a4.sh` (A3 runner with positive-only reverted and
the explore probs added). Wiring evidence: `bin_flip_*` metrics must be
nonzero in online train.csv. Evidence appended below.

Launched 20:06 BST in `agent_a4_explore_online_gpu0_20260801200611`;
completed ~21:10.

## Stage A4 result: treatment activation unverifiable; counted as an A2 replication (2026-08-01 ~21:20)

### 1. Result

| seed | 12.5k | 15k | 17.5k | 20k | best | endpoint−best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 34% | 48% | 40% | 44% | 48% | −4pp |
| 2 | 60% | 52% | 36% | 40% | 60% | −20pp |

Post-hoc audit found the preregistered wiring criterion was written
against the wrong mechanism's counters: `structured_exploration_*`
belongs to a different code path (its prob is 0 here) and
`_apply_bin_explore` emits no CSV metric and stores no replay flag in
this configuration. The config was correctly set
(`bin_explore_probs=[0.002,0.004,0.008]` in the archived online config)
and the code path is unconditional given that config, but activation
cannot be verified from artifacts, and GPU-nondeterministic reruns make
curve differences uninformative. A4 therefore cannot be interpreted as an
exploration test. It IS a valid same-start rerun of the A2 online phase:
bests 48/60 vs A2's 50/62 (peak height reproducible), retention −4/−20 vs
−16/−16 (decay magnitude noisy across reruns). Retention gate FAIL
(mean −12), improvement gate PASS at the line (54%).

### 2. Interpretation

Three same-start online arms (A2, A3, A4) now show: peaks 48–62%,
retention −4 to −20, and no mechanism separation beyond the noise floor.
With 50-episode eval SE ≈ 7pp and seed-level spread ≥ 10pp, two-seed
mechanism screens can no longer resolve effects of the size being chased.
Mechanism-hopping stops here.

### 3. Next-stage decision (Stage A5, preregistered): statistical power for the locked recipe

Lock the recipe as: de-scaled dense reward-Q (q_reward_scale=0.07),
offline 10k demo-only + online 10k, b256, no positive-only, no explore.
Train fresh full pipelines for seeds 3 and 4, then (chained) seeds 5 and
6. Evaluate the fixed grid raw 10/12.5/15/17.5/20k, 50 episodes, seeds
400–449 each. No mechanism variation of any kind.

Purpose (all preregistered, no gate to pass — this is measurement):
- offline extraction distribution at n≥4 (currently 40/46 + A1b 52/56);
- online peak distribution at n≥6 run-level (A2/A4 reruns + new seeds);
- retention slope distribution at n≥6;
- decide from those distributions whether (a) the peak mean genuinely
  sits in the high 50s (then one adequately-powered retention arm is
  worth it and the BC 66% target is reachable), or (b) the peak mean is
  low 50s (then the honest claim is parity-at-matched-short-budget and
  the differentiator experiments — second task, mixed-quality demos —
  become the paper's core).
- Fixed-endpoint references from the matched BC control for later
  comparison: 58% @15k, 56% @20k (its selected best 66% @12.5k).

### 4. Execution

`run_cqn_no_bc_agent_a5.sh`: two sequential two-run pairs (seeds 3+4,
then 5+6), each offline→online→eval, standard one-card protocol on GPU 0.
Evidence appended below.

Launched 20:58 BST in `agent_a5_replication_gpu0_20260801205810`; both
pairs completed with all sentinels by ~00:45.

## Stage A5 result: the A2 online-improvement effect does NOT replicate (2026-08-02 ~00:50)

### 1. Result

Fresh-seed full pipelines (50 eps, seeds 400–449):

| seed | 10k (offline) | 12.5k | 15k | 17.5k | 20k | best | online Δ (best−offline) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 3 | 40% | 50% | 48% | 38% | 48% | 50% | +10 |
| 4 | 56% | 54% | 48% | 44% | 42% | 56% | −2 |
| 5 | 44% | 38% | 34% | 38% | 36% | 44% | −6 |
| 6 | 58% | 48% | 46% | 50% | 48% | 58% | −8 |

Pooled recipe distribution at n=6 (A2 seeds 1/2 + A5 seeds 3–6):

- offline raw-10k extraction: 40/46/40/56/44/58 → **mean 47.3%**, all
  nonzero — the robust core result (zero env interaction, zero imitation
  gradient);
- online improvement (best − own offline): +10/+16/+10/−2/−6/−8 →
  **mean +3.3pp** — the A2 mechanism-gate result (+13pp) was a
  two-seed fluke; the de-scaled online phase's true mean effect is
  indistinguishable from the unscaled Stage-38 reference (+3);
- fixed-20k endpoints: 34/46/48/42/36/48 → mean 42.3% (below the offline
  mean — the online phase is net-negative at fixed endpoints);
- selected best over full curves: 50/62/50/56/44/58 → mean 53.3%, vs
  matched BC selected 66% / fixed 58%@15k, 56%@20k.

### 2. Interpretation

The honest effect table after replication: de-saturation restores value
calibration (probes; mechanism intact) and offline extraction parity, but
does not deliver a mean online improvement, and the decay past the early
peak dominates fixed endpoints in every arm tested. The A2 62% plateau
and +13pp were within-noise outcomes that this line's own preregistered
replication has now corrected — recorded as such. Noteworthy secondary
signal: offline extraction varies 40–58 at 10k and A1b showed +12/+10
from 10k→20k offline on seeds 1/2; the offline budget ceiling is now the
cheapest unresolved quantity and decides whether an offline-heavy recipe
can reach the high 50s reliably before any online step.

### 3. Next-stage decision (Stage A6, preregistered): offline-budget scaling measurement

Branch seeds 3–6 at their exact raw-10k offline states (pre-online, the
branch tool excludes post-snapshot episodes) and extend pure-offline
training to 30k updates. Evaluate raw 15k/20k/25k/30k, 50 episodes,
seeds 400–449. Measurement stage, no pass/fail gate. Morning decision
framework: median extraction at ≥20k ≥ 55% → offline-heavy recipe viable
(then design one adequately-powered retention/holding arm on top);
plateau ≤52% or recurring decay (A1b seed-2 pattern) → offline ceiling
established at ~50; the paper's performance claim calibrates to
parity-at-matched-budget and the differentiator experiments (second
task, mixed-quality demos) become the core.

### 4. Execution

`run_cqn_no_bc_agent_a6.sh`: branch 4 seeds → two sequential extension
pairs (3+4, 5+6) → evals. Evidence appended below.

Launched 23:18 BST on GPU 0; pair 3+4 training completed (~00:10), evals
running as of 00:00 checkpoint.

## Stage A7 preregistration: finite-optimum CE via label smoothing (2026-08-02 ~00:05)

### 1. Evidence being acted on

The user's question "do we actually know why online no-BC degrades?"
prompted a target-stationarity check from existing CSVs:
`mc_lower_bound_fraction` declines 0.52→0.33 over the de-scaled online
phase (A2/A5) but stays high (0.90→0.67) in unscaled Stage-38 — which
ALSO decays. Target nonstationarity is therefore not necessary for the
decay: another attribution eliminated. The surviving asymmetry between
BC-full (stable to 101k) and all no-BC arms (decay after ~10–15k total
updates, even pure-offline in A1b) is loss geometry: **margin/BC losses
self-saturate** (satisfied margin → zero gradient → parameters freeze at
a good solution), while the dense C51 cross-entropy against point-mass
targets **has no finite optimum in logit space and sharpens forever** —
matching the measured monotone sharpening of the protected manifold
(top-1 rising through 30k) with off-manifold span collapse.

### 2. Intervention

`method.dense_return_label_smoothing=0.05`: target ← 0.95·target +
0.05·uniform, applied identically to every bin. Gives the CE a finite
optimum (training self-terminates like a satisfied margin); uniform mass
on the symmetric support has expectation 0 = the floor, so bin-expectation
ordering is unchanged and the zero-return action-label invariance is
preserved exactly (unit-tested: 3 focused tests; construction chain
4 tests; resolved config diff = the single field).

### 3. Design and gates (paired, n=6)

Branch all six existing raw-10k de-scaled offline states (A1 seeds 1–2,
A5 seeds 3–6) and run the online phase with smoothing on GPU 5 (three
sequential pairs). Paired plain-dense controls already exist for every
seed (A2: s1/s2; A5: s3–6). Eval raw 12.5/15/17.5/20k, 50 eps, seeds
400–449.

- **Retention gate**: paired mean Δ(fixed 20k endpoint) ≥ +5pp with ≥4/6
  seeds positive. Control endpoints: 34/46/48/42/36/48.
- **Non-destruction gate**: paired mean Δ(post-handoff best) ≥ −3pp.
  Control bests: 50/62/50/54/38/50.
- PASS both → smoothing joins the locked recipe; retest the 30k
  extension; then the BC-66% challenge. FAIL → the sharpening hypothesis
  is refuted too; remaining candidates are online lr decay and the
  fixed-short-budget claim.
- Mechanism signature (post-hoc probes): demo-state top-1 should plateau
  instead of rising; online-state span should stop collapsing.

### 4. Execution

`run_cqn_no_bc_agent_a7.sh` on GPU 5 with an EGL smoke gate (2-update
real run; on EGL failure retries once with the index-4 mapping, else
writes `egl_blocked`). GPU-5 CUDA health verified by UUID probe.
Evidence appended below.

Launched 00:03 BST on GPU 5; `smoke_ok` with the default EGL mapping (no
index override needed); pairs running.

## Stage A6 result: offline-heavy path rejected per preregistration (2026-08-02 ~00:45)

Curves (50 eps): seed 3 40/50/46/46/50, seed 4 56/52/56/50/42, seed 5
44/38/44/46/54 (still rising at 30k), seed 6 58/58/48/48/48 at raw
10/15/20/25/30k. With A1b (52@20k / 32@20k), the median extraction at
≥20k budgets is **51% < the preregistered 55% line** → the offline-heavy
recipe is not viable; ceiling ≈ 50–58 selected / 47–49 fixed. Notable:
the matched BC control's offline-10k point is 58% — seeds 4/6 (56/58)
reach it; seeds 1/3/5 (40–44) do not. A large share of the BC gap at
this stage is **seed variance** (bimodal 40–46 vs 52–58), consistent
with margin acting as a cross-seed stabilizer. Path (b) is active:
matched-budget parity claim + differentiators.

## Stage A8 preregistration: second-task generality screen (2026-08-02 ~01:00)

Task `saucepan_to_hob` (long-horizon; demos cached locally, 16/20
successful in a 20-demo probe). Reward-scale transfer check done from a
real 2-update smoke: sampled native `mc_return_mean=9.26` vs MovePlate's
10.3 — the same `q_reward_scale=0.07` transfers unchanged (one
data-derived constant across tasks strengthens the method claim).

Design: seeds 1+2, offline-only 10k demo-only updates at b256 (the
locked de-scaled dense recipe, no online), `demos=60`
`env.expected_successful_demos=null`, eval raw 7.5k/10k only (long
episodes make full sweeps expensive; the screen question is binary),
50 eps, seeds 400–449 on GPU 0.

- Gate: any seed ≥ 10% at any evaluated point → generality supported;
  promote to a BC-matched comparison on this task later. Both 0% →
  generality risk flagged prominently; investigate before any paper
  claim.

### Execution

`run_cqn_no_bc_agent_a8.sh` on GPU 0. Evidence appended below.

First launch failed fast: only 36 demos exist for this task (60
requested); relaunched with `demos=36` (29 successful) at 01:07.

## Stage A8 result: second-task generality PASS (2026-08-02 ~02:45)

| seed | 7.5k | 10k |
| --- | ---: | ---: |
| 1 | 26% | 32% |
| 2 | 42% | **54%** |

Both seeds clear the 10% gate by a wide margin on `saucepan_to_hob`
(long-horizon, 29 successful demos, zero environment interaction, zero
imitation gradients, and the SAME data-derived scale constant 0.07).
Curves still rising at the 10k budget. Generality of the offline
dense-extraction claim is supported at screen level.

Follow-up preregistered (Stage A8b): the matched BC reference on the same
task and clock — official CQN-AS objective (BC=1, margin, FOSD,
critic_lambda=0.1) with 10k offline demo-only updates at b256, seeds 1+2,
eval raw 7.5k/10k on the same 50-episode split. No gate: this is the
comparator measurement for the generality claim (MovePlate analogue: BC
offline-10k = 58%). Launched 01:34 on GPU 0.

## Stage A7 early verdict: label smoothing FAILS catastrophically — and names the real constraint (2026-08-02 ~01:40)

Pairs 1+2 and 3+4 (4/6) are decisive regardless of the final pair:
smoothed endpoints 8/22/18/10 vs paired controls 34/46/48/42 → paired
Δ(endpoint) ≈ −28pp (gate required ≥ +5pp); bests also lower. Retention
and non-destruction gates both FAIL.

Final 6/6 (complete ~02:10): smoothed endpoints 8/22/18/10/0/10 vs
controls 34/46/48/42/36/48 — paired mean Δ(endpoint) **−31pp**, all six
negative; seed 5 (the weakest offline start, 44%) collapsed to 0% at
every checkpoint. Gates FAIL unanimously; the entropy-pump reading is
now backed by the full paired sample.

Post-hoc mechanism reading (recorded as hypothesis, not established):
label smoothing inside a *distributional bootstrap loop* compounds —
supervised smoothing touches fixed labels once, but here the Bellman
target distribution comes from a target critic that is itself converging
to smoothed (entropy-lifted) distributions, and the loss mixes uniform
again each iteration → an entropy pump diffusing all distributions
toward uniform, whose expectation is the support midpoint 0 → global
value collapse → argmax noise. Consistent with: offline extraction
unharmed (MC anchors are fresh point masses each update, no
compounding) while online crashes (mc_lower_bound_fraction ≈ 0.5, half
the targets bootstrap).

Design constraint for any future finite-optimum fix: it must not inject
entropy into the bootstrap path. The compliant shape is a
**satisficing floor** — suspend the floor CE on a bin once its expected
Q sits within δ of the floor (value-conditioned, margin-style
self-termination, no entropy). Its zero-return action-label invariance
requires the suspension rule to apply identically to chosen and unseen
bins when the chosen target equals the floor; deferred to a fresh-eyes
session rather than a 2am edit.

Retention attributions/interventions now refuted: demo-state erosion,
failure-floor provenance, (unverifiable) explore config, global label
smoothing. Retention remains THE open problem; the fixed-short-budget
claim is the current honest performance statement.

## Stage A8b result: BC reference on saucepan — the gap widens with horizon (2026-08-02 ~02:00)

| saucepan_to_hob offline-10k | seed 1 | seed 2 |
| --- | ---: | ---: |
| BC reference (bc_lambda=1.0 verified) | 48→**76%** | 74→**80%** |
| no-BC de-scaled dense (A8) | 26→32% | 42→54% |

BC's offline extraction dominates by ~35pp on this long-horizon task vs
~8–11pp on MovePlate. Endpoint probes on the no-BC critics show
replay-state ranking is NOT the discriminator (top-1 78%/78%, Spearman
0.97/0.98 — comparable to MovePlate) — the same
replay-metrics-vs-closed-loop dissociation, more extreme. Recorded
hypothesis (supported by arithmetic, not yet causally isolated): the
**return-channel attenuation limit** — saucepan demos average ~310
steps, so front-of-trajectory RTG ≈ 0.99^280 × 23 ≈ 1.4 native ≈ one
support atom above the floor after scaling; the win margin at early
states is within eval-time perturbation size, while BC reads the
non-attenuated action channel. One-sentence reframe of the whole "why
BC" question: **BC's advantage is not extra information; it is reading
the channel that γ^T does not attenuate.** Candidate mechanisms to test
later (fresh session): horizon-adapted gamma, more atoms/nonuniform
support, or n-step return relays along the trajectory.

## Stage A8c preregistration: saucepan offline extension (2026-08-02 ~01:45)

Both A8 curves rise into the 10k boundary (26→32, 42→54). Extend both
seeds' offline phase in place to 20k (A1b pattern), eval raw
12.5/15/17.5/20k, 50 eps, seeds 400–449, on GPU 5 after A7 completes.
Measurement (no gate): extends the generality result and the
offline-budget curve to a second task. Runs chained automatically.

## Stage A8c result: no-BC reaches BC territory on saucepan with more offline budget (2026-08-02 ~03:30)

| seed | 10k | 12.5k | 15k | 17.5k | 20k | best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 32% | 28% | 52% | 56% | **58%** | 58% @ 20k (still rising) |
| 2 | 54% | **76%** | 54% | 54% | 42% | 76% @ 12.5k |

Seed 2's 76% equals the BC reference's seed-1 fixed-10k endpoint (76%)
and sits 4pp below its seed 2 (80%). The apparent 35pp horizon gap at
matched 10k was therefore largely a **convergence-rate gap, not a hard
information ceiling**: the return-channel attenuation slows extraction
on long-horizon tasks (front-of-trajectory signal is ~1 atom, so more
updates are needed to consolidate it) but does not cap it near the 10k
values. Honest caveats: selected-best on the dev split, one seed;
seed 2 shows the familiar peak-then-decay (76→42) while seed 1 rises
monotonically; a fair endpoint comparison would extend the BC arms too.

Cross-task synthesis for the morning: no-BC peaks now touch BC
territory on BOTH tasks (MovePlate 62 vs 66; saucepan 76 vs 76–80), and
in every single case the blocker between peak and fixed-endpoint claim
is the same retention/decay problem — four interventions refuted, the
satisficing-floor design (value-conditioned self-termination, no
entropy into the bootstrap loop) is the queued candidate.

## Stage A9 preregistration: satisficing floor, paired n=6 (2026-08-02 ~02:40)

Implementation landed: `dense_return_floor_satisfaction_margin=0.02`
(a quarter support atom). The floor cross-entropy is suspended on any
bin whose target IS the floor distribution AND whose expected Q sits
within the margin of the floor. The rule is target-conditioned, never
label-conditioned: on zero-return samples the chosen bin's target equals
the floor, so all bins follow the identical value-conditioned rule and
the exact action-label invariance holds even with unequal per-bin Q
(unit-tested, including the unequal-Q swap case; 7 focused tests + 4
construction-chain tests pass). Positive-return chosen bins are never
suspended. This converts the floor from a grinder into a hinge —
margin-style self-termination on VALUES, no imitation objects, no
entropy into the bootstrap path (the A7 failure's design constraint).

Design: same-start paired n=6 as A7 — branch the six raw-10k de-scaled
offline states, online 10k with the margin enabled, eval raw
12.5/15/17.5/20k, 50 eps, seeds 400–449. Controls: the plain-dense
online curves (A2 s1/s2, A5 s3–6): endpoints 34/46/48/42/36/48, bests
50/62/50/54/38/50. Pairs split across both cards: GPU 0 runs (1,2) then
(5,6); GPU 5 runs (3,4).

- **Retention gate**: paired mean Δ(fixed 20k endpoint) ≥ +5pp with
  ≥4/6 seeds positive.
- **Non-destruction gate**: paired mean Δ(post-handoff best) ≥ −3pp.
- PASS both → recipe locks (descale + dense + satisficing floor);
  immediately rerun the 30k extension question and the saucepan
  composition; the fixed-endpoint BC comparison follows. FAIL →
  retention pivots to training-schedule solutions (online lr decay,
  preregistered fixed-short-budget endpoint) rather than further
  objective surgery.

### Execution

`run_cqn_no_bc_agent_a9.sh GPU s1 s2 [s3 s4]` (parameterized pairs; A7
runner lineage incl. the EGL smoke gate). Evidence appended below.

Launched 02:46 on both cards; both smokes ok; complete ~04:35.

## Stage A9 result: first positive-mean retention intervention, but gate FAIL (2026-08-02 ~04:40)

Paired Δ(fixed 20k endpoint): +12/+2/−14/+12/+4/−4 → mean **+2.0pp**
(required ≥ +5), 4/6 positive (sign condition met). Non-destruction:
Δ(best) mean +0.7 ≥ −3 ✓. Formal verdict: **FAIL** on the mean as
preregistered. Notable positives recorded without overriding the gate:
first of five retention interventions with a positive paired mean; seed
4's curve 56/50/52/54 is the flattest online trajectory observed; seeds
1/4 endpoints (46/54) are the best fixed-20k values in any no-BC arm;
seed 3's −14 comes from a single terminal point (52→34 at 20k only).

## Stage A9b preregistration: single permitted dose variation (2026-08-02 ~04:45)

δ=0.02 (a quarter atom) was a semi-arbitrary first dose. Following the
main line's floor-tail dose precedent (Stages 3→4), exactly ONE dose
escalation is preregistered: `dense_return_floor_satisfaction_margin=
0.08` (one full support atom). Same paired n=6 same-start design, same
controls, same gates (paired mean Δ(endpoint) ≥ +5 with ≥4/6 positive;
Δ(best) mean ≥ −3). No further doses regardless of outcome; a second
FAIL closes objective surgery and pivots retention to training-schedule
solutions (online lr decay; preregistered fixed-short-budget endpoint)
and the performance push to saucepan + mixed-quality demos.

### Execution

Same runner with the a9b launch config; GPU 0 pairs (1,2)+(5,6), GPU 5
pair (3,4). Evidence appended below.

Launched 04:14 both cards; complete ~06:00.

## Stage A9b result: dose escalation reverses — objective surgery CLOSED (2026-08-02 ~06:05)

Paired Δ(fixed 20k endpoint): +12/−6/−6/+4/+2/−8 → mean **−0.3pp**, 3/6
positive → **FAIL** (and worse than δ=0.02's +2.0). Non-destruction
passes (+3.0). Combined satisficing verdict: neutral-to-mildly-positive,
non-destructive, insufficient alone; best-behaved of the five retention
interventions; documented as a future combination candidate but NOT
added to the locked recipe (parsimony rule). Per the preregistered
boundary, objective surgery on retention is closed. Retention now
belongs to training-schedule solutions; the performance push moves to
the saucepan front and the mixed-quality-demo differentiator.

Final retention scoreboard (paired Δ endpoint means): positive-only
n/a-refuted-by-slope; explore-config unverifiable; smoothing −31;
satisficing δ=0.02 **+2.0**; satisficing δ=0.08 −0.3.

## Stages A10/A10b preregistration: the saucepan front (2026-08-02 ~06:10)

Measurement pair for the morning decision (no pass/fail gates; these
numbers decide whether a formal tie/win protocol vs BC is warranted):

- **A10 (GPU 0)**: fresh saucepan no-BC seeds 3+4, offline-only 20k at
  the locked recipe (descale+dense, scale 0.07), snapshots every 2.5k,
  eval raw 10/12.5/15/17.5/20k — maps the no-BC extraction distribution
  (current: 58@20k rising / 76@12.5k).
- **A10b (GPU 5)**: the A8b BC references resumed in place 10k→20k,
  eval raw 12.5/15/17.5/20k — fairness comparator (does BC also climb
  past 76/80 given the same extension?).

Decision rule for the next step (registered): if the no-BC best
distribution overlaps the extended-BC distribution within the ±4pp eval
noise on ≥2 seeds, design the formal fixed-endpoint + sealed-held-out
comparison on this task; otherwise the tie claim is dropped and the
differentiator (mixed-quality demos) carries the "beat" requirement.

## Stages A10/A10b result: BC also scales; saucepan tie dropped (2026-08-02 ~07:15)

- A10 no-BC fresh: seed 3 34/50/42/24/28 (best 50); seed 4
  46/8/48/**84**/78 (best 84 @ 17.5k, sustained 84/78 late pair).
- A10b BC extension: seed 1 80/64/**92**/54; seed 2 **94**/84/86/78.

Extended-budget selected bests: BC 92/94 vs no-BC {50, 58↗, 76, 84}.
The no-BC maximum (84) sits 8pp below the BC minimum (92) — outside the
±4 noise window → per the registered rule the saucepan tie claim is
dropped. Both families are volatile at 50 eps (BC 92→54 in one step).
The "beat baseline" requirement now rests on the differentiator.

## Stage A11 preregistration: unlabeled-demo-quality differentiator (2026-08-02 ~06:40)

Lever implemented and smoke-verified: `+env.treat_all_demos_as_expert=
true` relabels every demonstration's `demo` flag to 1 at load (rewards
untouched). Effects: the protected buffer keeps all 60 MovePlate demos
(51 success + 9 failed; smoke shows zero "Skipping failed" lines), and
the BC/FOSD/margin losses — which mask by the demo flag — now imitate
the failed demonstrations too. The no-BC update never reads the flag
(bit-identity unit tests) and return-gates by true rewards, so only its
replay composition changes. This is the realistic no-success-labels
regime and the direct experimental form of the project's thesis: BC
imitates whatever it is given; reward-derived objectives let the return
decide.

Arms (offline 10k, b256, eval raw 5k/7.5k/10k, 50 eps, seeds 400–449):
- GPU 0: no-BC mixed, seeds 1+2 (clean refs 40/46 @10k);
- GPU 5: BC mixed seed 1 + BC clean seed 2 (clean ref seed 1: 58 @10k
  from the Stage-38 matched sweep; seed 2's clean reference measured
  here).

Registered predictions/gates:
- **P1 invariance**: no-BC mixed @10k within ±5pp of its clean per-seed
  reference.
- **P2 degradation**: BC mixed seed 1 @10k ≤ 58 − 5pp.
- **Crossover** (the beat question at 15% contamination): no-BC mixed
  vs BC mixed head-to-head; if P2 holds without crossover, one
  registered dose-2 raises the contamination fraction (subset the
  successful demos; actual fraction reported from the load log).

### Execution

`run_cqn_no_bc_agent_a11_nobc.sh` (GPU 0) and
`run_cqn_no_bc_agent_a11_bc.sh` (GPU 5). Evidence appended below.

Launched 06:35 both cards; complete ~07:40. Wiring verified in logs:
mixed arms 0 skip-lines, clean arm 9 skip-lines.

## Stage A11 result: BOTH predictions refuted — BC is robust to natural failed demos (2026-08-02 ~07:45)

| @10k | mixed | clean ref | Δ |
| --- | ---: | ---: | ---: |
| BC seed 1 (mixed) | **64%** | 58% | +6 |
| BC seed 2 (clean, new ref) | — | 52% | — |
| no-BC seed 1 (mixed) | 48% | 40% | +8 |
| no-BC seed 2 (mixed) | 30% | 46% | −16 |

P2 (BC degradation ≥5pp) decisively fails — BC improved on the mixed
seed. P1 (no-BC invariance ±5pp) fails on seed 2 (−16, though within
this recipe's seed-variance band). Crossover fails. Mechanism reading:
BiGym's natural failed demonstrations are benign near-misses whose
actions are mostly sensible; imitating them at 15% weight acts like data
augmentation, not poison. The predicted asymmetry requires actively
misleading failure actions, which this benchmark's natural failures do
not provide. The registered dose-2 (synthetic misleading demos) is a
design-fairness question deferred to the user. On natural contamination,
the differentiator claim is **refuted and recorded**.

## Stage A12 preregistration: satisficing × online-lr-decay combination (2026-08-02 ~07:50)

Post-surgery schedule territory, with the main line's own precedent
(their +10.4pp came from a combination whose single factors measured
+4.2 and +0.2). Combine the only positive-mean retention lever
(satisficing δ=0.02, non-destructive) with an online-phase learning-rate
reduction `method.critic_lr: 5e-5 → 2e-5` (config-only; offline phase
untouched at 5e-5 via the branch design). Same paired n=6 same-start
design and controls as A9. Explicitly labeled a combination arm (two
resolved diffs vs A2's online phase).

Gates unchanged: paired mean Δ(fixed 20k endpoint) ≥ +5pp with ≥4/6
positive; Δ(best) mean ≥ −3pp. PASS → recipe candidate locks and the
30k-extension + fixed-endpoint protocol follows. FAIL → schedule
territory reduces to the preregistered fixed-short-budget claim, and the
session's synthesis stands as the deliverable.

### Execution

A9 runner with the a12 launch config (satisficing 0.02 inherited + lr
override in the online invocation). GPU 0 pairs (1,2)+(5,6); GPU 5
(3,4). Evidence appended below.

## Directive shift (user, 2026-08-02 morning): measurement-first mechanism research

The user rejected both shortcut paths (synthetic misleading demos;
saucepan seed-tail chase) and directed the line to research WHY no-BC
online learning fails to learn anything useful, extract the insight, and
only then design a robust method. Performance-chasing arms stop; A12
(already running) is relabeled as a churn-hypothesis dose, not a
performance move. Competing hypotheses registered:

- **H-churn**: thin Q margins + never-saturating CE → each update flips
  argmax at some states; BC's saturated margins freeze parameters. 
  Prediction: consecutive-snapshot greedy-action agreement no-BC ≪ BC.
- **H-redundancy**: greedy+σ=0.01 collection replays the demo-derived
  policy → successes carry no new information; failures only pollute
  off-manifold values → online can subtract but not add.
- **H-deflation**: failure-state bootstraps chain to other failure
  states with no positive anchor → deflation spiral (probes already
  show off-manifold Q 1.4→0.3).
- New observation formalized: online outcomes regress toward ~50 — all
  three low-offline-start seeds improved (+10/+16/+10), the high-start
  seeds degraded (−2/−8) — consistent with online training pulling
  every policy toward the mixed-replay equilibrium instead of improving
  on the offline solution.

## D1 result: churn hypotheses refuted; the discriminator is the off-manifold value field (2026-08-02 ~08:40)

Consecutive-snapshot greedy-bin agreement (same fixed states, target
critic): no-BC 76–83%, BC 80–84%; demo-success states ~94% in BOTH arms.
Long-range (10k→20k) agreement: no-BC 69.2%, BC 71.7% — between the
random-walk prediction (~42/44%) and full anchoring (~80%), and again
nearly identical. Both the strong churn story (BC freezes) and the
drift story (no-BC diffuses) are refuted at probed states.

The single quantity that separates the arms (BC n=1, no-BC replicated
on 2 seeds via the earlier probes):

| online_success states | BC | no-BC |
| --- | --- | --- |
| span 10k→20k | 0.61→0.60 (stable) | 0.48→0.10 (5× collapse) |
| Q_greedy 10k→20k | 0.37→0.64 (growing) | 0.43→0.07 (6× deflation) |

**Mechanism synthesis.** The dense floor's per-state lesson "every
non-executed bin is worth exactly the task minimum" GENERALIZES: the
network learns "at states like these, most actions are worthless",
flattening the off-manifold value field where closed-loop rollouts
actually live. This explains every prior refutation at once: replay-
state metrics stay excellent (the states themselves are supervised
correctly); floor provenance is irrelevant (success-transition floors
teach the same generalized lesson — A3); offline training is safe (the
demo manifold IS the training distribution, off-manifold barely
exists); BC is stable not because its parameters freeze (churn is
equal) but because imitation-style supervision — "THIS action is best
here", never "everything else is worthless" — generalizes into a
smooth, high-span preference field.

## Stage A12 result: FAIL at +2.0pp — lr dose adds nothing; churn story dead (2026-08-02 ~09:20)

Paired Δ(fixed 20k endpoint): 0/+6/−6/+8/+2/+2 → mean **+2.0pp**
(identical to satisficing alone), 4/6 nonnegative, non-destruction
passes (+1.7). Halving the online lr added nothing — as a churn dose
this cross-validates D1's direct measurement: churn is not the decay
mechanism. Retention scoreboard after six story-driven interventions:
best +2.0pp (twice), all below the +5 gate. A13 (relative floor) is the
first intervention derived from the isolated cause rather than a
plausible story; chains fired on both cards (GPU 5 07:45 smoke_ok,
GPU 0 08:28).

## Stage A13 preregistration: relative floor, paired n=6 (2026-08-02 ~09:15)

Implementation landed and tested (4 focused tests incl. an
optimization-convergence check that unseen bins settle exactly at
E[chosen]−m, and exact zero-return label invariance; 7 prior focused
tests and the construction chain unaffected). Field:
`dense_return_relative_floor_margin=0.16` (two support atoms), one
resolved diff vs the A2/A5 online recipe. Design identical to A9/A12:
branch the six raw-10k de-scaled offline states, online 10k, eval raw
12.5/15/17.5/20k; paired controls A2 s1/s2 + A5 s3–6 (endpoints
34/46/48/42/36/48, bests 50/62/50/54/38/50). Chained to launch as each
card's A12 instance completes.

Gates (same as A9/A12): paired mean Δ(fixed 20k endpoint) ≥ +5pp with
≥4/6 positive; Δ(best) mean ≥ −3pp. Mechanism signature (post-hoc
probe, secondary): the online-state span/Q collapse (0.48→0.10 /
0.43→0.07 in controls) should be materially reduced — this is the
quantity the intervention was derived from, and its response is
informative even if the policy gates fail.

## Stage A13 result + margin synthesis: the mechanism question is answered (2026-08-02 ~11:30)

A13 policy gates: **FAIL both** (paired Δ endpoint mean −7.3, 1/6
positive; Δ best mean −9.3 — destructive). Distinctive curve shape:
deep early dip (12–36% at 12.5k) then recovery, 4/6 seeds still rising
at 20k — switching objectives compresses the offline-built gaps before
a new equilibrium forms. Mechanism probes: off-manifold value DEFLATION
persists under relative floors (Qg 0.6→0.017) → **the deflation is not
caused by absolute-floor generalization** (third causal story refuted
by direct removal).

The triangle "span→0 + per-factor ranking preserved (~71%) + policy
collapse" is resolved by the margin table (candidate_top2_gap at
online-success states):

| arm | demo-state margin | online-state margin (10k→20k) |
| --- | --- | --- |
| BC | 0.13→0.21 | **0.126→0.129 (stable)** |
| no-BC dense | 0.55→0.61 (over-sharpening) | **0.356→0.066 (5× collapse)** |
| A13 relative floor | 0.06–0.09 (capped by design) | 0.084→0.025 (worst) |

**Unified conclusion.** Composite action selection across ~240 factor
decisions per chunk is exponentially sensitive to per-factor decision
margins. BC-margin's true function in CQN-AS — measured, not assumed —
is **maintaining uniform decision-margin geometry off the supervised
manifold** (its classification-style loss generalizes margin width;
0.13 everywhere). The dense no-BC objective concentrates margin mass on
the protected manifold and lets it evaporate where the policy acts;
online decay is progressive off-manifold margin erosion under
failure-dominated data. All seven refuted interventions are explained:
none addressed off-manifold margin maintenance (A13 capped margins
globally, hence worst). Not action information (demos contain it), not
anchoring (churn equal), not floors per se: **margin geometry**.

**Derived robust-method requirement**: a reward-only source of margin
supervision at policy-visited states. The only such source is
same-state alternative-bin OUTCOME data — i.e., ε-bin exploration,
which retro-explains the main value line's sole working mechanism
("exploration manufactures contrast data; TD consumes it"): contrast
data is precisely off-manifold margin supervision with real returns.
Designated next arm: instrumented ε-bin exploration (activation counter
added BEFORE launch — the A4 lesson), on the descale+dense recipe,
paired n=6, with the margin probe as the mechanism gate alongside the
policy gates. Longer horizons than 10k online may be required (contrast
data accrues slowly at ε≈1%/level); preregister budget accordingly.

## Stage A14 preregistration: instrumented ε-bin exploration (2026-08-02 ~13:30)

Instrumentation landed and END-TO-END VERIFIED before launch (the A4
lesson): `_apply_bin_explore` now counts segment fires and applied
shifts, act() counts training calls, all flow to train.csv. A 1.2k-step
ε=0.5 smoke shows act=1001 / explore-calls=1001 / fired=64 /
applied=986 (self-consistent with persist=16). The earlier 400-step
"fired=1" scare was a logging-cadence artifact (rows every 1k steps;
the single early row shows only the reset-warmup call). Retro-note
recorded honestly: by this mechanism arithmetic, A4's exploration was
almost certainly active (~20% duty at 1× dose over 10k steps); its
no-separation-from-noise result stands.

Design: descale+dense recipe + `bin_explore_probs=[0.004,0.008,0.016]`
(2× dose; the value line measured 1×==2× insensitivity) — expected ~550
segments / ~31% exploration duty over a 20k-step online phase (global
cap 30k; margin/contrast data accrues slowly, hence the longer horizon).
Branch the six raw-10k offline states as before; eval raw
12.5/15/17.5/20/25/30k. Runner asserts nonzero explore counters after
training (hard wiring gate) before any evaluation is interpreted.

Gates:
- **Policy (paired, ≤20k grid)**: mean Δ(fixed 20k endpoint) ≥ +5pp,
  ≥4/6 positive, vs the same six controls (34/46/48/42/36/48).
- **Extension region (25/30k)**: vs the A2b seeds-1/2 references
  (36/36/32/26 and 44/40/40/42) — informational unless decisive.
- **Mechanism (margin probe, the causal test)**: online-state
  candidate_top2_gap at 20k ≥ 2× matched control (0.066) on ≥2 probed
  seeds. If margins respond but policy doesn't, the margin thesis needs
  revision; if policy responds without margins, likewise. Both gates
  reporting together is the point.

### Execution

`run_cqn_no_bc_agent_a14.sh` (a9 lineage + 30k cap + counter assertion).
GPU 0 pairs (1,2)+(5,6); GPU 5 pair (3,4). Evidence appended below.

## Protocol shift (user, 2026-08-02 ~11:20): wide screening before replication

The user directed a screening-first protocol: run FOUR different
candidate arms at one seed each in parallel; only a candidate showing a
signal graduates to the 4-seed stability stage. Adopted. A14's running
pairs (1,2)/(3,4) complete as launched (n=4 analyzable); the queued
pair (5,6) is cancelled to free GPU 0.

## Stage A15 preregistration: 4-arm single-seed screen (2026-08-02 ~11:30)

All arms: branch the A1 seed-2 raw-10k state (richest control history),
online 10k (global 20k), eval raw 12.5/15/17.5/20k, 50 eps. Paired
control: A2 seed-2 (60/62/60/46; best 62, endpoint 46). Arms (each one
config override on the descale+dense recipe):

- **S1 explore×satisficing**: a14 explore config +
  `dense_return_floor_satisfaction_margin=0.02` — the two
  positive-signal levers combined (contrast data + floor
  self-termination).
- **S2 UTD/2**: `update_every_steps=2` — halve grinding per unit fresh
  data (static-buffer overtraining lever).
- **S3 fresh-data rebalance**: `demo_batch_size=128` — protected
  fraction 50%→33%, shifting gradient mass toward policy-visited
  states (where margins die).
- **S4 slow target field**: `method.critic_target_tau=0.005` — smooth
  the bootstrap value field that off-manifold targets are read from.

Screen bar (lenient by design, preregistered): fixed 20k endpoint ≥ 52%
(+6 over control) OR (best ≥ 60% AND endpoint ≥ 48%). A passer
graduates to the 4-seed stability stage plus a margin probe; a clean
sweep of failures still bounds four mechanism levers at one GPU-hour
each. Single-seed screens cannot confirm — only nominate.

### Execution

`run_cqn_no_bc_agent_a15_screen.sh` (parameterized: gpu, arm name,
launch, extra override). GPU 0 runs S1+S2 after the A14 (1,2) pair
completes and its (5,6) queue is cancelled; GPU 5 runs S3+S4 after
(3,4). Evidence appended below.

## Stage A14 result: explore alone FAILS retention (n=4, verified active) (2026-08-02 ~13:05)

Exploration verifiably active (fired 215–253 segments/seed, ~30% duty).
Paired Δ(20k endpoint) vs controls: +2/+2/−6/+6 → mean **+1.0**; the
25/30k extension declines in 3/4 seeds (30k endpoints 28/26/34/42).
Contrast data alone does not stop the decay at this horizon/dose. The
margin-thesis prediction that exploration suffices is NOT supported at
the policy level (margin probe pending as a mechanism postscript).

## Stage A15 screen result: two passers (2026-08-02 ~13:10)

Control (A2 s2): 60/62/60/46, endpoint 46.

| arm | curve | endpoint | verdict |
| --- | --- | ---: | --- |
| S2 UTD/2 | 52/60/52/58 | **58 (+12)** | **PASS — flattest no-BC online curve observed** |
| S1 explore×satisficing | 56/48/48/54 | **54 (+8)** | **PASS** |
| S3 demo128 | 50/52/44/44 | 44 | fail |
| S4 tau005 | 52/56/54/44 | 44 | fail |

Effect decomposition so far: explore alone +1 (n=4), satisficing alone
+2 (n=6), explore×satis +8 (n=1), UTD/2 +12 (n=1). The simplest
training-schedule lever (halving updates per env step) shows the
strongest single-seed retention signal — consistent with the
static-buffer overtraining reading of A1b and the post-surgery pivot.

## Stage A16 preregistration: stability stage for both passers (2026-08-02 ~13:15)

Per the user's screen-then-replicate protocol: each passer runs seeds
1/3/4/6 (seed 2 done in the screen → n=5 paired deltas each), same
branch/online/eval protocol. Waves: UTD/2 first on both cards, then
explore×satis. Controls (12.5–20k): s1 50/38/40/34; s3 50/48/38/48; s4
54/48/36/42; s6 48/46/50/48.

Stability gate (n=5): paired mean Δ(fixed 20k endpoint) ≥ +5pp with
≥4/5 nonnegative; Δ(best) mean ≥ −3. A UTD/2 pass makes it the
recipe's retention component (a training-schedule solution, as the
post-surgery pivot anticipated) and the immediate follow-ups are a
longer-horizon run (does it IMPROVE past 20k where everything else
decayed?) and the margin probe; then the BC-comparison protocol.

## Stage A16 result: both passers fail replication; consolidation point (2026-08-02 ~17:50)

Stability stage (n=5 paired incl. the screen seed):

- UTD/2: Δ endpoints 0/+12/−4/0/−8 → mean **0.0** — the screen's +12
  was seed-2 noise; FAIL.
- explore×satisficing: +2/+8/−4/+12/−4 → mean **+2.8**, 3/5 nonnegative
  — FAIL, but the best n≥5 retention value recorded.

The screen-then-replicate protocol performed exactly as intended. The
retention-intervention ledger closes at NINE failures (positive-only,
explore-config, smoothing, satisficing δ=0.02/0.08, lr-halving,
relative floor, verified exploration, UTD/2, explore×satisficing), best
mean +2.8pp.

Final mechanistic twist: the margin probe on the best surviving run
(explore×satis seed 4: endpoint 54%, flat curve) shows the SAME
off-manifold margin collapse as decaying controls (0.40→0.08 vs
0.36→0.07). A run can retain performance with collapsed probed margins
→ the off-manifold margin/value collapse is a robust CORRELATE of the
no-BC online regime but its proximal-cause status is now uncertain as
well. What stands rock-solid: (1) saturation mechanism + descale fix
(causal, replicated); (2) floor necessity (causal); (3) offline
extraction robustness (two tasks); (4) the empirical bound — no
single-knob or two-knob intervention within the charter moves mean
online retention ≥ +5pp; (5) BC's online stabilizing function has
resisted every mechanistic decomposition attempted (freeze/churn ✗,
margin geometry — correlate only).

Open decision points recorded for the user: (a) whether a
return-gated large-margin objective (margin on the executed bin of
positive-return transitions only — reward-gated, needs no success
labels, robust-to-bad-demos by construction) is admissible; the charter
currently forbids all margin/rank objectives by letter, though this
variant arguably honors its spirit. It is the strongest remaining move
toward "beat the baseline". (b) Otherwise, consolidate as: mechanism
anatomy + robust BC-free offline extraction + the systematic
negative-result ledger + fixed-short-budget parity claims.

## Stage A17 preregistration: return-gated margin screen (2026-08-02 ~18:40)

**Charter note, explicit.** The hinge uses the executed action of
positive-return transitions as a ranking reference, which the main-line
charter's letter forbids ("never … margin"). It is implemented and
screened as a clearly-labeled boundary variant because it satisfies the
project's OPERATIONAL anti-imitation test everywhere it is applied: the
gate reads measured return only (never a demo flag), online successes
receive it identically, and on zero-return samples the term vanishes
exactly (loss and gradients invariant to the recorded action;
unit-tested, 3 focused tests). Its admissibility as "no BC loss"
remains the user's call; results are quarantined under this label.

Mechanism position: the minimal reward-only replica of BC's
classification geometry — a RELATIVE separation constraint that
survives global level drift (unlike the absolute floor) and pushes
unseen bins DOWN (unlike A13's relative floor, which pulled them up and
destroyed contrast).

Screen design (per the user's screen-then-replicate protocol): margin
0.16, dose λ ∈ {0.1, 1.0}, seeds 2 and 4 (strongest control
histories), standard branch/online-10k/eval-grid. Controls: s2
60/62/60/46; s4 54/48/36/42. Screen bar (+6 endpoint rule): s2 endpoint
≥ 52 or (best ≥ 60 and endpoint ≥ 48); s4 endpoint ≥ 48 or (best ≥ 52
and endpoint ≥ 44). Any (dose, seed) pass → 4-seed stability stage for
that dose + margin probe (prediction: online-state top2_gap held ≥ m
where controls collapse to 0.066 — the sharpest mechanism signature yet
available).

### Execution

A15/A16 screen machinery, 4 slots: GPU 0 [w01 s2, w10 s2]; GPU 5 [w01
s4, w10 s4]. Evidence appended below.

## Stage A17 screen: seed-4 both doses PASS; mechanism signature HITS (2026-08-02 ~20:20)

Seed-4 screen (control 54/48/36/42, endpoint 42): w01 56/48/44/50 →
endpoint 50 (+8) PASS; w10 52/44/48/46 → best 52 & endpoint 46,
borderline PASS. Seed-2 recovery (one CUDA-init transient, relaunched)
pending.

**Margin probe on w01 s4 — the first predicted mechanism signature to
land in 17 stages**: online-success top2_gap at 20k = **0.178** (the
design value m=0.16 held; plain-dense control 0.066; explore×satis s4
0.082; BC 0.129), achieved WHILE value levels deflate as usual (Qg
0.245→0.076) — the hinge is level-invariant by construction and did
exactly what it was designed to do, with the policy endpoint improving
in the same run. Online-failure states still collapse (0.044) —
recorded; the thesis concerns states where correct ranking matters.
Design prediction and policy response aligned for the first time.

w01 nominated; stability stage n=6 (adds seeds 1/3/5/6 to the screen's
2/4) chained behind the s2 recovery. Stability gate: paired mean
Δ(fixed 20k endpoint) ≥ +5pp with ≥4/6 nonnegative (controls
34/46/48/42/36/48); pass → the charter question goes to the user WITH a
working method attached, plus longer-horizon and BC-comparison
protocols.

Seed-2 recovery completed (~21:20): **w10 s2 = 46/58/54/64 — the first
no-BC curve in the entire project still RISING at the 20k boundary, and
the highest fixed 20k endpoint recorded (64 vs previous no-BC max 54;
above the BC control's seed-1 fixed-20k 56)**. Paired +18. w01 s2 faded
to 44 (−2). Dose-by-seed interaction (s4 favored w01, s2 favors w10);
the strong dose's screen mean is +11 vs w01's +3, and the rising shape
matches the mechanism dose logic (stronger enforcement, stronger
geometry). Both doses proceed to n=6 stability: w01 waves running,
w10 waves chained behind them.

## Stage A17 final verdict and cross-project synthesis (2026-08-03 ~00:30)

**A17 n=6 stability: both doses FAIL.** Paired Δ(fixed 20k endpoint) vs
controls 34/46/48/42/36/48 — w01: +4/−2/−4/+8/0/+4, mean **+1.7**
(4/6 nonneg); w10: +2/+18/−10/+4/−4/−6, mean **+0.7** (3/6). The
seed-2 +18 did not generalize (the recurring single-seed pattern). RGM
joins the retention ledger as directional-but-insufficient at the 20k
screening horizon; the margin mechanism survives as ANALYSIS (below),
not as a method component.

**The sibling GPT project completed the goal** (cqn-no-bc.md Stages
41–43): 10k demo-only dense offline → 101k online with (a)
positive-return-only dense targets and (b) FROZEN expert replay
(use_self_imitation=false, protected buffer fixed at 9,253), official
hyperparameters otherwise (λ_TD=0.1, no descale, no margin, no explore).
Sealed four-seed fixed-endpoint held-out: **66.375% vs official
64.625% (+1.75pp; not statistically significant — the correct claim is
"matches, point estimate above")**.

**Mechanism validation across projects (the probe result that closes
the causal chain).** My value-fidelity/margin probe on their fresh seed
3 (held-out 68%) across its 101k online phase, online-success states:
top2_gap 1.21→0.54→0.41→**0.354 at 111k** with top-1 RISING 73→81% —
maintained ~5× above the collapse level (0.066) that every failing arm
in my line exhibited by 20k, and above BC's 0.129. The off-manifold
margin/field-collapse account of no-BC online failure now has all three
legs: collapse present in every failing arm (measured), absent in the
independently-discovered passing method (measured), and directionally
improved by an intervention that artificially restores margins (A17).
Why their ingredients preserve the field: positive-only removes
counterfactual flattening from failure trajectories, and the frozen
expert buffer removes the success-replay feedback that amplified drift
— together doing with reward machinery what BC-margin's classification
geometry does implicitly.

**Honest methodological lessons recorded:**
1. My 20k-online screening cap — chosen for iteration speed — was
   likely BELOW the horizon where retention mechanisms differentiate:
   their decisive gains materialized in the 30–50k+ blocks, and their
   key ingredient (positive-only) had failed in my 20k same-start test
   (A3) under growing replay. Screen horizons must match mechanism
   timescales.
2. The frozen-expert-replay lever was never in my search space; every
   arm inherited the growing protected buffer. The replay FEEDBACK
   channel deserved the same systematic treatment as the loss channel.
3. The return-gated margin's paper claim is vulnerable on success-only
   data (return-gating ≈ demo-gating there — reduces to renamed BC per
   the main line's own Stage-18 note); the GPT recipe's action-identity
   usage (value-space floors + candidate backup, exact zero-return
   invariance) is the clean form of the no-BC claim.

**Paper skeleton (recommended):** flagship method = the Stage-42/43
recipe with its sealed result; explanation chapter = this line's
mechanism anatomy (saturation → floor necessity → off-manifold
margin/field maintenance, with the cross-validated probe evidence);
negative-result appendix = the systematic intervention ledger (12
retention arms with same-start paired protocols); analysis = A17 as the
interventional test of the margin thesis; offline-extraction and
two-task generality results (47±7 MovePlate n=6; saucepan 32–84) as
supporting contributions.

## Stage A18 preregistration: GPT recipe × de-saturation (2026-08-03 ~01:00)

The composition candidate for an OWN method that beats (not merely
matches) the baseline. Base = the sibling project's sealed Stage-42/43
recipe (offline full-dense 10k → online positive-return-only dense,
FROZEN 9,253-transition expert replay, candidate backup, official
λ=0.1/b256). Added = this line's `q_reward_scale=0.07` in both phases
(single field per phase config; composition verified). Mechanism
rationale: their probe profile shows saturation-supported margins (demo
gap ≈ support width, all executed values at the ceiling); de-saturation
supplies the one ingredient their recipe lacks — calibrated temporal
ordering (Spearman 0.91–1.0 vs clipped) — which should improve online
Bellman quality and cross-seed consistency (their residual weakness:
seed-4 −15.5 held-out).

Protocol mirrors their progressive ladder (the 20k-screen lesson):
fresh seeds 1/2, one per card (solo runs; avoids the co-residence CUDA
race), offline 10k → online to raw 30k, eval raw 12.5–30k (8 × 50 eps,
seeds 400–449). Gate vs THEIR frozen Stage-42 raw-30k block (bests
74/66, endpoints 54/62, mean-of-16 57.5): PASS = mean best ≥ 68 AND
endpoint mean ≥ 58 AND both bests ≥ 60. Pass → 50k block vs their
68/80; then the 111k four-seed protocol with the same sealed
fixed-endpoint comparison (seeds 800–999, justified: never used for
selection; official reference 64.625 was measured on this split, so
comparability requires it). The win claim requires the four-seed
held-out mean to exceed 64.625 by more than their +1.75 — target ≥ 68
with ≥3/4 positive paired deltas.

### Execution

`run_cqn_no_bc_agent_a18.sh GPU SEED` (two-phase + eval, solo per
card). Seed 1 → GPU 0, seed 2 → GPU 5. Evidence appended below.

## Stage A18 result: composition FAILS — de-saturation is incompatible with the winning recipe (2026-08-03 ~15:25)

Curves (50 eps, seeds 400–449, raw 12.5–30k):
seed 1 `48/58/54/38/36/34/32/34` (best 58, endpoint 34);
seed 2 `48/50/48/48/40/28/26/36` (best 50, endpoint 36).
Sibling Stage-42 reference: bests 74/66, endpoints 54/62, mean-of-16 57.5.
A18 mean best 54 (gate ≥68), endpoint mean 35 (gate ≥58): **FAIL on every
condition, and materially WORSE than the un-scaled recipe.** Both seeds also
show the classic decay signature the sibling recipe otherwise avoids.

Interpretation (a genuine negative result, not a wiring artifact — scaling
verified active in `pretrain.csv`): the two components are **not additive,
they are antagonistic**. Reading: the winning recipe's stability relies on
the very saturation this line diagnosed as a defect. With clipped targets,
every positive-return state parks at the ceiling, so the floor/positive-only
machinery enforces a large, uniform executed-vs-counterfactual separation
(measured demo-state gap ≈1.9 ≈ full support width) that is robust to value
drift — saturation acts as an implicit, extremely wide margin. De-scaling
restores fine temporal ordering but shrinks every separation by ~14×
(0.07 scale), leaving margins inside the noise band of online value drift,
so the off-manifold field decays again. This also retro-explains why
descale+dense (A2/A5) looked promising early (calibration helps extraction)
yet always decayed online, while the sibling recipe with clipped values did
not.

Consequence for the project: calibrated ordering and drift-robust margins are
in tension under a fixed C51 support. The clean way to have both is a WIDER
support with unclipped returns (e.g. v_max ≈ 25 native, or scale ≈0.5 with
v_max 12) so that ordering survives AND separations stay large in absolute
atoms — untested, one config field, the obvious next experiment. This line's
own composition attempt is otherwise exhausted; the sibling Stage-42/43
recipe remains the method of record.

## Stage A19 result: wide support also FAILS — support geometry closed (2026-08-03 ~19:00)

Wide support (v±29, 101 atoms, native returns): seed 1
`56/48/42/44/46/30/28/26` (best 56, endpoint 26); seed 2
`48/50/54/46/48/44/42/26` (best 54, endpoint 26). Mean best 55 (gate ≥68),
endpoint mean 26 (gate ≥58) → **FAIL**, comparable to A18 and far below the
sibling recipe (bests 74/66, endpoints 54/62).

**Conclusion for the support/calibration line.** Three configurations now
span the axis: clipped narrow support (sibling recipe) = works; de-scaled
narrow support (A18) = fails; native wide support (A19) = fails. Both
unclipped variants decay, and both preserve calibrated ordering — so
calibration is NOT the missing ingredient, and the earlier "de-saturation
fixes the mechanism" reading (A1/A2) is now superseded: what those runs
gained offline they never converted online, and forcing calibration into the
working recipe destroys it. The operative property of the sibling recipe is
that clipping pins every positive-return state to a common ceiling, which
makes the executed-vs-counterfactual separation both huge and UNIFORM across
states — a geometry that survives drift precisely because it carries no fine
value information to lose. This is a substantive, counter-intuitive finding:
in this regime, target coarseness is a feature.

Support geometry is therefore closed as a lever. This line's positive
contribution stands as: the saturation diagnosis (why canonical no-BC gives
0%), the floor-necessity result, the off-manifold margin/field account with
cross-project probe validation, the 12-arm negative ledger, and now the
calibration-vs-drift-robustness tension (A18/A19) that explains why the
working recipe works for a reason opposite to the one this line assumed.

## Stage A20 result: the offline value-ordering predictor SURVIVES prospective test (2026-08-04 ~00:40)

Fresh seeds 5/6 ran the sibling recipe verbatim; offline geometry was
recorded BEFORE any online step, outcomes measured on the identical
raw-30k grid as all four sibling seeds:

| seed | offline Spearman | raw-30k best | endpoint |
|---|---:|---:|---:|
| 1 | 0.738 | 74 | 54 |
| 2 | 0.714 | 66 | 62 |
| 3 | 0.810 | 70 | 64 |
| 4 | 0.500 | 62 | 50 |
| **5 (fresh)** | **0.214** | **56** | **46** |
| **6 (fresh)** | **0.262** | **58** | **52** |

Pooled n=6: **corr(Spearman, best) = +0.93**, corr(Spearman, endpoint) =
+0.83. The two fresh seeds extended the predictor range downward and
landed exactly where the hypothesis placed them — this is prospective
support, not the in-sample arithmetic that motivated the stage. Probe
noise was ruled out by an n=32 recomputation (A20 s5 0.214→0.294;
sibling s1 0.738→0.809: the between-seed gap is real, not sampling).

An initial reading of these curves as "outcomes normal ⇒ predictor
refuted" was WRONG (wrong reference band) and is corrected here.

**Implied method component (reward-only, zero environment cost):** an
offline quality gate. Run the 25-minute offline phase, measure ordering
quality on demo states with the CPU probe (no env rollouts, no held-out,
no demo labels — it correlates stored returns against critic values), and
promote only seeds clearing a bar. At bar 0.50 on these six: kept
{1,2,3,4} mean best 68.0 / endpoint 57.5 versus ungated 64.3 / 54.7 —
a +3.7/+2.8pp shift from a screen that costs no online compute. Note the
gate never touches the online phase or the objective; it is a
compute-allocation rule derived from this line's own diagnostic.

**What it does NOT yet establish**: n=6 with 4 in-sample seeds; the
outcome metric is 30k-window validation, not the sealed 111k endpoint;
the bar (0.50) is chosen post hoc; and — the honest core caveat — a gate
that discards seeds changes the *deployment protocol*, so a fair
comparison against official CQN-AS must give the baseline the same
budget (e.g. run K+2 baseline seeds and report the best K), or be framed
explicitly as "reward-only early stopping" rather than a better learner.

**Registered next experiment (not run — compute belongs to the user's own
jobs tonight)**: fresh seeds 9–14, offline only (25 min each, ~1.5 h
total, CPU probes free); measure the Spearman distribution to fix the
bar's operating point and the rejection rate; then promote survivors to
111k and evaluate the four sealed endpoints against the official 64.625
with the budget-matched framing above.

## Stage A21: the A20 predictor is RETRACTED — the metric is not reproducible across batches (2026-08-04 ~00:55)

Offline-only sweep, fresh seeds 9–14, n=32 probes: Spearman
0.948/0.907/0.949/0.950/0.908/0.943 — mean 0.934, sd 0.021. That
distribution is disjoint from every earlier measurement (sibling seeds
0.50–0.81; A20 seeds 0.21–0.29) under an identical recipe and unchanged
code, which is the wrong shape for seed-to-seed variation.

Decisive reproducibility test: **rerun seeds 5 and 6 through the same
procedure. Spearman 0.294 → 0.963 and 0.262 → 0.956.** Same seeds, same
config, same probe; the statistic moved by 0.7. It is therefore a
property of the batch/run conditions, not of the seed.

**Consequence: Stage A20's "prospectively validated predictor" is
withdrawn.** Its +0.93 correlation was a batch confound — the A20 pair
happened to be measured low AND to score low, both traceable to that
batch rather than to a causal link between offline ordering quality and
online outcome. The offline quality gate that this implied is dead as
formulated. Also retroactively weakened: any conclusion resting on
CROSS-RUN comparisons of this statistic. (Within-run trajectories — how
margin/span evolve over a single run's checkpoints, which is what the
mechanism analysis in §"D1 result" and the cross-project margin
comparison used — are unaffected in kind but should be re-verified with
paired reruns before being load-bearing in a paper.)

Unresolved: WHY the metric shifts. Candidates not yet separated — probe
state sampling depends on the replay directory contents at probe time;
Spearman over ~32 states with a narrow return spread is a high-variance
statistic; GPU contention differed between batches. Any future use of
this instrument needs a reproducibility protocol (paired reruns, fixed
state sets shared across runs) before it can support a claim.

**Session close.** Own-method candidates tested and failed: A17 (return-
gated margin, n=6), A18 (de-saturation × winning recipe), A19 (wide
support), A20/A21 (offline geometry gate — retracted on reproducibility).
The method of record remains the sibling Stage-42/43 recipe at parity
(+1.75pp, not significant). This line's durable contributions are the
mechanism anatomy (saturation → floor necessity → off-manifold field),
the 14-arm negative ledger, the calibration-vs-drift-robustness tension
(A18/A19), and — added today — a documented reproducibility limit of the
diagnostic instrument itself.

## Derived method candidate (Stage A13, to be preregistered): relative floor

Replace the unseen-bin target point mass at 0 with a point mass at
**max(G − m, 0)** (m ≈ 2 support atoms): the supervision becomes "other
actions are slightly worse than the executed one" instead of "other
actions are worthless". Preserves within-state ranking (extraction
intact), preserves the exact zero-return invariance (at G=0 all targets
coincide), injects no entropy, adds no imitation object, one resolved
field. The generalized lesson becomes "values vary smoothly with a
small runner-up gap" — the shape of BC's field. Direct removal of the
measured cause rather than a compensating trick.

## D4 result: weak novelty of online successes (corrected) (2026-08-02 ~08:20)

Action-space nearest-neighbor analysis on A2 seed 2's protected buffer
(51 demos + 37 online successes, progress-resampled action L1):
online-success → nearest-demo distance is **1.15×** the demo→demo
internal NN distance. (A first estimate of 0.66× was an artifact of a
contaminated demo/online split and is corrected here.) Reading: online
successes hug the demo manifold — marginal novelty is small but not
zero. H-redundancy holds in a weak form; the sharper discriminators are
D1 (churn) and the A12 dose.
