# CQNAS without imitation loss

## Goal and non-negotiable boundary

The only research goal is to remove BC/imitation from original CQNAS and learn
MovePlate from expert demonstrations through RL targets alone, while matching
or exceeding the original policy. There is no actor pretraining, action
cross-entropy, FOSD, large-margin imitation, AWR-weighted cloning, flow-policy
fitting, or frozen behavior policy. Demonstration actions may be used only as
the actions belonging to reward-based replay transitions or as data-supported
Bellman maximization candidates; they are never action labels for a policy,
margin, rank, reconstruction, or likelihood objective. The deployed policy is
the canonical CQNAS `argmax Q`.

The launch enables `method.strict_demo_rl_only=true`. Configuration composition
must fail if any forbidden path is enabled. Demonstrations may contribute only
ordinary RL transition information `(s, a, r, s')` and completed returns in a
stage that explicitly preregisters them.

## Corrected evaluation protocol (2026-07-31)

The earlier stage reports used the clean original CQNAS
`68%/72%/76%` (mean `72%`) as a hard development gate. That number is useful
for mechanism development but is not a full-run endpoint reference:

| result | training budget | evaluation | selection |
| --- | ---: | --- | --- |
| clean original CQNAS | 10.5k | 50 episodes/checkpoint, seeds 400--449 | best of 2.5k/5k/7.5k/10k |
| independent recheck of those selected checkpoints | 10.5k | 100 episodes/seed, seeds 43000--43099 | checkpoints already selected |
| official full CQNAS | 101k | 200 episodes/seed, seeds 800--999 | fixed final checkpoint |

The 10.5k selected values are `68%/72%/76%`, mean `72%`, in
`exp_local/cqn_value_fidelity_stage22/clean_multiseed_summary.json`. The
independent 100-episode recheck is `59%/56%/69%`, mean `61.3%`, in
`exp_local/cqn_value_fidelity_stage22/clean_multiseed_heldout_seed43000/summary.json`.
The official fixed-endpoint matrix is `62%/60.5%/62%/74%`, mean `64.6%`, as
recorded in `cqn-flow.md` section 39.9.

Consequently, all earlier 10k gates below are retained as the historical
development decisions that generated the next experiments, not as evidence
that a mechanism fails at full scale. From Stage 22 onward:

1. The 10.5k, 50-episode checkpoint sweep is only a relative screen against a
   matched no-BC control. It cannot establish official parity or full-budget
   failure.
2. A reproducible relative gain is confirmed with an additional training seed
   and an independent nonsealed development evaluation. A candidate whose
   best point is still at the 10k boundary is eligible for a 20k continuation
   rather than automatic rejection.
3. The final comparison trains the selected no-BC recipe with four training
   seeds to 101k. No checkpoint is reselected. Each fixed 101k endpoint is
   evaluated on the sealed 200 episodes, seeds 800--999.
4. Final success means the four-seed no-BC fixed-endpoint mean matches or
   exceeds the official `64.6%`. The checkpoint-selected `72%` remains
   descriptive context only.

## Stage 0: existing evidence and first decision (2026-07-30)

### 1. Previous-stage result

- Historical original-transition pure TD (`bc_lambda=0`) on MovePlate measured
  `0/0/0/0%` at 2.5k/5k/7.5k/10k; clean first-success pure TD also measured
  `0/0/0/0%`.
- The matched full CQNAS clean reproduction has validation-best success
  `68/72/76%` for training seeds 1/2/3, mean `72.0%`. Its exact run paths and
  curves are in
  `exp_local/cqn_value_fidelity_stage22/clean_multiseed_summary.json`.
- The historical no-BC launches retained `critic_lambda=0.1`; original CQNAS
  optimized `0.1 * TD + 1.0 * BC`. Thus removing BC also reduced the remaining
  objective's scale by tenfold.
- The official 100k held-out reference is `62.0/60.5/62.0/74.0%` on 200
  episodes from seeds 800--999, mean `64.6%`.

### 2. Interpretation

The 0% result establishes that simply deleting BC from the old configuration
does not work. It does not isolate whether failure is caused by missing action
identification or by the tenfold TD rescaling. Training loss is not policy
quality, so neither finite TD loss nor Q separation will count as success.

The first new hypothesis follows the conservative RL construction in
[Q-Transformer](https://arxiv.org/pdf/2309.10150): the replayed action receives
only Bellman supervision, while every replay-unseen bin is regressed to the
task's valid minimum return (zero for sparse-success MovePlate). Unlike CQL
softmax behavior regularization, this term contains no action classification or
likelihood. If all returns and Q values are at the floor, changing the recorded
action produces exactly zero loss and zero gradient.

### 3. Next-stage decision

Stage 1 asks one question only: after normalizing TD scale, does unseen-action
return-floor conservatism recover actionable demo information?

- Matched control: strict demo-RL-only CQNAS, `critic_lambda=1.0`,
  `unseen_return_floor_weight=0`.
- Treatment: identical configuration and seed, with
  `unseen_return_floor_weight=1`.
- Both use the clean first-success MovePlate MDP, seed 1, the same demonstrations,
  16 online + 16 demo samples/update, and checkpoints at
  2.5k/5k/7.5k/10k.
- Selection split: fixed environment seeds 400--449, 50 episodes per
  checkpoint. Select the highest success checkpoint; exact ties choose the
  earlier checkpoint.
- Held-out split: untouched in Stage 1. Seeds 800--999 are reserved for final
  confirmation after validation selection.
- Policy metric: MovePlate success rate. Mechanism diagnostics:
  `unseen_return_floor_loss`, chosen/unseen expected-Q gap, finite gradients,
  and training throughput.
- Stage-1 pass: treatment validation-best success is at least 20% and at least
  10 percentage points above normalized TD control. A pass triggers matched
  seed-2/3 confirmation. A fail rules out this coefficient/mechanism alone and
  triggers a separate return-propagation stage; it does not authorize adding
  imitation.
- Final goal gate: follow the corrected protocol above; only four 101k fixed
  endpoints on sealed seeds 800--999 are compared with the official `64.6%`.

### 4. Execution

Implemented:

- canonical C51 expected-Q return-floor regularization with the replayed bin
  masked out;
- strict configuration audit;
- unit checks that the replayed bin has zero conservative gradient, zero-return
  batches cannot reveal action identity, and flipping every `demo` flag leaves
  the update exactly unchanged;
- matched launch configs, one-GPU/two-run controller, fixed-seed snapshot sweep,
  and validation-best summarizer.

Stage 1 launched on GPU 0 at 2026-07-30 23:09 local time:

- controller:
  `exp_local/cqn_no_bc/stage1_seed1_gpu0_20260730230924`;
- normalized TD control:
  `exp_local/cqn_no_bc/stage1_seed1_gpu0_20260730230924/control_td`;
- return-floor treatment:
  `exp_local/cqn_no_bc/stage1_seed1_gpu0_20260730230924/treatment_floor`.

The runs started 120 seconds apart with `xla_mem_fraction=0.45` each. Measured
co-resident GPU use was 30.25 GiB with 1.85 GiB free. Both produced finite real
updates. At its first update the treatment logged floor loss `3.55e-15` and
chosen-minus-unseen expected-Q gap `1.64e-8`, as expected when all initial
values sit at the zero return floor. This verifies wiring only; policy quality
remains unresolved until the fixed validation sweep completes.

## Stage 1 result and Stage 2 decision (2026-07-30)

### 1. Previous-stage result

`stage1_summary.json` records the complete fixed-split result:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| normalized TD | 0% | 0% | 0% | 0% | 0% @ 2.5k |
| TD + return floor | 0% | 0% | 0% | 0% | 0% @ 2.5k |

Every cell is 50 MovePlate episodes on environment seeds 400--449. The
preregistered mechanism gate failed: treatment-minus-control was 0 percentage
points and treatment best was below 20%.

### 2. Interpretation

The result rules out TD coefficient rescaling as the explanation for the old
pure-TD failure, and rules out unseen-action floor regression alone at this
data/update budget. The treatment's chosen-minus-unseen Q gap grew from
approximately zero to `0.059` at 9k, so the conservative term did alter action
ranking, but no checkpoint produced a successful policy. That separates
action-support control from the unresolved propagation of one sparse terminal
reward through a long expert trajectory.

### 3. Next-stage decision

Stage 2 isolates return propagation by completing a 2-by-2 design:

- existing normalized TD: no MC lower bound, no floor;
- existing floor: no MC lower bound, floor weight 1;
- new MC-only: Q-Transformer-style
  `target = max(discounted Monte-Carlo return, Bellman target)`, no floor;
- new MC+floor: the same target plus floor weight 1.

The Monte-Carlo return is computed from rewards only after a complete episode.
It is not an action label, actor objective, or pretraining loss. The max is one
chosen-action Q-learning target rather than a second auxiliary loss.

Both new arms use training seed 1 and all Stage-1 settings. Selection remains
50 episodes/checkpoint on seeds 400--449 at 2.5k/5k/7.5k/10k; exact ties select
the earlier checkpoint and then the simpler MC-only method. Held-out seeds
800--999 remain untouched. The propagation gate passes if the best MC arm is
at least 20% and improves by at least 10 percentage points over its matched
no-MC arm. A pass advances the validation-selected variant to training seeds 2
and 3. A fail rules out this return target at the 10k budget and moves to a
separate representation/action-factorization stage.

### 4. Execution

Implemented `method.mc_lower_bound_target`. When enabled, replay stores
discounted reward-to-go for completed demo and online episodes. For each
replayed action head, the C51 Bellman distribution is retained unless its
expected value is below the reward-to-go; in that case a projected point mass
at reward-to-go replaces it. `mc_return_weight` remains zero, so there is no
second MC auxiliary loss.

Focused checks establish that:

- the replayed `demo` flag has no effect on the update;
- no policy/flow parameters exist;
- MC returns are produced by the episode reward sequence;
- the new target does not emit `mc_return_loss`;
- existing canonical MC and no-MC signatures still work.

Stage 2 launched on GPU 0 at 2026-07-30 23:37 local time:

- controller: `exp_local/cqn_no_bc/stage2_seed1_gpu0_20260730233733`;
- MC-only: `.../mc_only`;
- MC+floor: `.../mc_floor`.

The starts were staggered by 120 seconds and measured co-resident GPU use was
30.23 GiB. MC-only reached a real 1k update with finite loss; its initial
logged `mc_lower_bound_fraction` was `0.451` and reward-to-go mean was `0.488`.
This is target-use evidence only, not policy-quality evidence.

## Stage 2 result and ranking diagnostic (2026-07-31)

### 1. Previous-stage result

`stage2_summary.json` completed the fixed-split factorial:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| TD | 0% | 0% | 0% | 0% | 0% |
| floor | 0% | 0% | 0% | 0% | 0% |
| MC-only | 0% | 0% | 0% | 0% | 0% |
| MC+floor | 0% | 0% | 2% | 4% | 4% @ 10k |

Each cell is 50 episodes on seeds 400--449. MC+floor improved over its matched
floor-only arm by 4 percentage points but failed the preregistered 20% / +10pp
propagation gate.

### 2. Interpretation

The interaction produced the first successful no-imitation MovePlate
trajectories: reward-to-go alone and floor alone both remained at zero, while
their combination reached 2/50. This establishes that reward information can
reach action selection without BC, but does not establish a useful policy or
rule out sampling noise. The result remains far below the original 72% target.

Training showed a mean chosen-minus-unseen Q gap around 0.18--0.25, but a mean
gap does not establish that the replayed action beats the single largest
unseen bin. Greedy control is determined by that maximum.

### 3. Next-stage decision

Before changing the objective, run a read-only ranking diagnostic on the 10k
MC-only and MC+floor target critics. Use the same 32 stratified replay states
(8 each from demo-success, demo-failure, online-success, online-failure) from
the historical clean first-success run, sampling seed 0. This preserves an
existing full-CQNAS reference on the identical data.

Metrics are demo-success replay-bin top-1, current-action top-1,
max-minus-replay-Q, Q span, and return correlation. No validation or held-out
environment seeds are consumed.

- If MC+floor demo-success top-1 is below 60% or current-action top-1 below
  50%, Stage 3 will target the maximum unseen action rather than the mean.
- If both thresholds pass while task success remains 4%, ranking is adequate
  on replay states and Stage 3 instead targets action-sequence/prefix
  consistency or distribution shift.

### 4. Execution

The two probes run sequentially on GPU 0 after all Stage-2 evaluation processes
have exited. Exact outputs are stored under the Stage-2 controller's
`probes/` directory.

## Stage 2 ranking result and sequence-position audit (2026-07-31)

### 1. Previous-stage result

The fixed-state target-critic probe produced:

| method | demo-success top-1 | current-action top-1 | max - replay Q | Q span | Q / return Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC-only @ 10k | 27.99% | 34.17% | 0.00066 | 0.00195 | 0.774 |
| MC+floor @ 10k | 61.93% | 59.72% | 0.02816 | 0.37264 | 0.759 |
| original CQNAS @ 8k | 85.54% | 58.33% | 0.00726 | 0.72572 | -0.405 |

The MC+floor outputs are
`stage2_seed1_gpu0_20260730233733/probes/mc_floor_10000.json`; the original
reference is
`cqn_value_fidelity_stage2/probes/full_first_success_8000.json`. Both use the
same stratified replay source and sampling seed. The all-sequence and
current-action preregistered thresholds passed, even though environment
validation remained 4%.

### 2. Interpretation

This establishes two points. First, MC+floor contains nontrivial value
information: its demo-success Q/return Pearson correlation is positive and its
current-action top-1 rate is slightly above the original CQNAS reference.
This rules out the explanation that reward-to-go never reached the critic.
Second, the full action sequence remains 23.6 percentage points below the
original, and its largest competing bin exceeds the replayed bin by about four
times the original gap. Thus good replay-state current-action ranking is not
sufficient for closed-loop success. What remains unresolved is whether the
deficit is concentrated at particular future chunk positions, consistent with
temporal-ensemble/prefix inconsistency, or is uniform and therefore more
consistent with state-distribution shift or insufficient global ranking
margin.

### 3. Next-stage decision

Before implementing a causal prefix critic, rerun the MC+floor and original
CQNAS probes on the identical states while retaining per-sequence-position
statistics. Report, for each of the 16 action positions, replay-bin top-1,
max-minus-replay-Q, and candidate Q span, averaged over C2F levels, action
dimensions, and the eight demo-success states.

This diagnostic consumes no environment seeds. The temporal-prefix branch is
selected only if the future-position top-1 deficit relative to original CQNAS
is at least 15 percentage points larger than the position-0 deficit, or if the
MC+floor future max-minus-replay-Q is at least twice its position-0 value.
Otherwise Stage 3 will not claim a prefix-specific defect and will target the
replay-to-policy distribution gap with a separately matched pure-RL
experiment.

### 4. Execution

`scripts/analyze_cqn_value_fidelity.py` now records the three
sequence-position vectors without changing checkpoint state or action
selection. The original 8k snapshot used by the earlier aggregate reference
has since been pruned, so the position comparison uses the available clean
seed-1 reproduction at its validation-selected best checkpoint, 5k (68% on
seeds 400--449), rather than silently substituting its final checkpoint. The
two matched probes are run sequentially on otherwise idle GPU 0; no training
or environment evaluation shares that card.

## Stage 2 sequence-position result and Stage 3 protocol (2026-07-31)

### 1. Previous-stage result

On the eight matched demo-success states, MC+floor position-0 top-1 was 59.72%
and the mean over positions 1--15 was 62.07%. Its max-minus-replay-Q was
0.02891 at position 0 and 0.02811 over future positions. The clean original
CQNAS seed-1 validation-best checkpoint (5k, 68%) measured 73.89% and 80.17%
top-1 respectively. Thus the original-minus-MC+floor top-1 deficit increased
from 14.17 points at position 0 to 18.09 points in the future, only 3.92
points; the MC+floor future/position-0 Q-gap ratio was 0.97.

Exact artifacts:

- `stage2_seed1_gpu0_20260730233733/probes/mc_floor_10000_by_sequence.json`;
- `stage2_seed1_gpu0_20260730233733/probes/original_clean_seed1_best5000_by_sequence.json`.

### 2. Interpretation

Neither preregistered temporal-prefix condition passed: the future top-1
deficit did not grow by 15 points relative to position 0, and the future
max-Q gap did not double. This rules out claiming that a position-specific
causal-prefix defect is the measured bottleneck. MC+floor instead leaves a
roughly uniform small set of high unseen bins: its average unseen Q is low,
but greedy control is determined by the maximum unseen bin. Replay-state
ranking is still observational, so state-distribution shift remains
unresolved.

### 3. Next-stage decision

Stage 3 tests one hypothesis only: averaging the absolute reward-floor penalty
over all unseen bins dilutes the gradient on the largest OOD competitor. The
matched arms are:

- control: MC lower-bound target + mean unseen-bin return floor;
- treatment: the identical target and coefficient, but regress the maximum
  unseen bin in every C2F action head to the same absolute return floor.

The replayed bin is masked in both arms and remains trained only by the single
max(MC return, Bellman return) target. The floor is fixed at zero from the
task's reward support; it never uses a demo flag, action likelihood, chosen-Q
margin, actor, or pretrained policy. In particular, if every bin predicts the
floor, the conservative term has zero loss and zero gradient for every
possible recorded action.

Both arms use training seed 1, the same 60 demos, 10k updates, and all other
Stage-2 settings. Selection is again 50 episodes per checkpoint on fixed seeds
400--449 at 2.5k/5k/7.5k/10k, with earliest-checkpoint tie breaking. Held-out
seeds 800--999 remain untouched. The treatment passes if its
validation-selected best is at least 20% and at least 10 percentage points
above the matched mean-floor control. A pass advances to training seeds 2 and
3; a fail rejects worst-bin conservatism at this budget and requires a new
distribution-shift hypothesis.

### 4. Execution

Implemented `method.unseen_return_floor_reduction` with legacy `mean` and new
`max` modes. The max mode selects the largest unseen expected Q independently
for every level/action head, while masking the replayed bin before the max.
The paired launch configurations, selector, and one-GPU runner are:

- `cqn_as_pixel_bigym_nobc_mc_mean_floor_gate.yaml`;
- `cqn_as_pixel_bigym_nobc_mc_max_floor_gate.yaml`;
- `scripts/summarize_cqn_no_bc_stage3.py`;
- `scripts/run_cqn_no_bc_stage3.sh`.

The complete CQN-AS unit file passes (`88 passed`), including max-floor
replayed-bin gradient masking, zero-signal non-cloning, demo-label invariance,
and the matched launch contract. Stage 3 launched on GPU 0 at 2026-07-31 00:20
local time:

- controller: `exp_local/cqn_no_bc/stage3_seed1_gpu0_20260731002044`;
- control: `.../mc_mean_floor`;
- treatment: `.../mc_max_floor`.

Both processes reached a real 1k update with finite losses. The treatment's
first 1k row records `mc_lower_bound_fraction=0.700`,
`unseen_return_floor_loss=0.0327`, and no BC metric. Co-resident GPU use was
30.25 GiB with both processes alive, consistent with the one-card/two-run
protocol. These are launch and wiring checks only, not policy-quality
evidence.

## Stage 3 result and post-gate diagnostic (2026-07-31)

### 1. Previous-stage result

The fixed validation split completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC + mean floor | 0% | 0% | 0% | 0% | 0% @ 2.5k |
| MC + max floor | 4% | 0% | 0% | 16% | 16% @ 10k |

Every cell is 50 episodes on seeds 400--449. The treatment improved its
validation-selected best by 16 percentage points but missed the preregistered
absolute 20% threshold, so `stage3_summary.json` records
`worst_case_floor_gate: fail`. The max-floor arm produced 8/50 successful
episodes at 10k; the mean-floor control produced none at any checkpoint.

### 2. Interpretation

Worst-bin conservatism has a substantial causal effect under the matched
objective: replacing only the unseen-bin reduction changed validation from
0% to 16%. This establishes that a small number of high OOD bins was a real
control bottleneck and rules out treating the earlier 4% MC+mean result as the
best obtainable behavior from reward-only supervision. It does not establish
a useful policy: 16% failed the gate and remains 52 points below the matched
seed-1 original CQNAS validation-best result (68%). The non-monotone
4/0/0/16 curve also makes checkpoint selection essential.

### 3. Next-stage decision

Before selecting the next distribution-shift hypothesis, probe the target
critics of the matched mean-floor and max-floor 10k checkpoints on the same 32
stratified replay states and sampling seed used in Stage 2. Report
demo-success replay-bin top-1, current-action top-1, max-minus-replay-Q,
candidate span, and Q/return correlation, including per-sequence vectors.
This consumes no validation or held-out environment seeds.

- If max-floor reaches at least 80% demo-success top-1 and
  max-minus-replay-Q at most 0.01 while validation remains 16%, Stage 4 treats
  replay-state action ranking as adequate and targets replay-to-closed-loop
  state distribution.
- Otherwise Stage 4 targets remaining conservative value separation rather
  than adding a prefix model or actor.

The Stage-4 experiment must retain the strict no-imitation guard, use a
matched control, select checkpoints on seeds 400--449, and leave held-out
seeds 800--999 untouched until the validation gate passes.

### 4. Execution

Stage 3 completed cleanly at
`exp_local/cqn_no_bc/stage3_seed1_gpu0_20260731002044`; training and validation
completion sentinels, both CSVs, all snapshots, and the summary JSON are
present, and no Stage-3 process remains. The two 10k probes run sequentially
on now-idle GPU 0 and write under the controller's `probes/` directory.

## Stage 3 probe result and Stage 4 protocol (2026-07-31)

### 1. Previous-stage result

The matched 10k target-critic probes produced:

| arm | demo-success top-1 | current top-1 | max - replay Q | Q span | Q / return Pearson |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC + mean floor | 63.02% | 61.67% | 0.02704 | 0.37099 | 0.747 |
| MC + max floor | 62.83% | 60.56% | 0.00930 | 0.59355 | -0.039 |

The max-floor arm passed the preregistered Q-gap threshold but failed the 80%
top-1 threshold. Exact artifacts are
`probes/mc_mean_floor_10000.json` and
`probes/mc_max_floor_10000.json` under the Stage-3 controller.

### 2. Interpretation

Worst-bin conservatism reduced the magnitude by which the best unseen bin beat
the replayed bin by 65.6%, while leaving the fraction of losing action heads
essentially unchanged. Together with the 0% to 16% policy improvement, this
establishes that smaller OOD argmax errors are behaviorally useful. It rules
out declaring replay-state ranking solved: roughly 37% of demo-success heads
still select another bin, and demo-state return correlation is not retained
by max-floor. A plausible unresolved mechanism is rotating near-maximum
competitors: updating only one unseen bin per head may expose the next one,
whereas averaging all four unseen bins diluted the useful tail gradient.

### 3. Next-stage decision

Stage 4 tests that tail-coverage hypothesis with one matched change:

- control: MC lower-bound + maximum-unseen-bin floor (top-1 tail);
- treatment: MC lower-bound + top-2-unseen-bin floor, averaging the squared
  absolute-floor error over the two largest unseen Q bins in each head.

Both mask the replayed bin before tail selection. Neither term uses chosen Q,
an action margin, demo identity, likelihood, policy head, pretraining, or
self-imitation. When all bins equal the zero reward floor, both objectives
have exactly zero loss and gradient for every possible recorded action.

Both runs use training seed 1, 60 demos, 10k updates, and otherwise identical
Stage-3 settings. Checkpoint selection remains 50 episodes on seeds 400--449
at 2.5k/5k/7.5k/10k, earliest on ties. Held-out seeds 800--999 remain
untouched. The Stage-4 gate requires the top-2 treatment's validation-best to
be at least 30% and at least 10 percentage points above the matched top-1
control. A pass advances the validation-selected treatment to training seeds
2 and 3; a fail rejects the tail-coverage hypothesis at this budget.

### 4. Execution

Implementation and anti-cheat validation begin immediately after this
preregistration. The paired runner will again stagger two 0.45-memory jobs by
120 seconds on GPU 0 and will serialize validation after both training jobs
exit.

Implemented `unseen_return_floor_reduction=topk` with a statically validated
`unseen_return_floor_topk`. The replayed bin is replaced by negative infinity
before tail selection. Focused tests show gradients only on the requested
upper unseen tail, and the complete CQN-AS suite passes (`93 passed`).

Stage 4 launched at 2026-07-31 00:54 local time:

- controller: `exp_local/cqn_no_bc/stage4_seed1_gpu0_20260731005430`;
- control: `.../mc_top1_floor`;
- treatment: `.../mc_top2_floor`.

Both reached a real 1k update with finite losses. The top-2 row records
`mc_lower_bound_fraction=0.770` and
`unseen_return_floor_loss=0.0423`; these establish target use, not policy
quality. Co-resident GPU use was 30.25 GiB with both processes live and no
OOM.

## Stage 4 result and Stage 5 protocol (2026-07-31)

### 1. Previous-stage result

The fixed validation split completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC + top-1 floor | 0% | 0% | 6% | 6% | 6% @ 7.5k |
| MC + top-2 floor | 0% | 0% | 2% | 2% | 2% @ 7.5k |

Top-2 lost 4 percentage points to its matched control and failed both parts of
the preregistered 30% / +10pp gate. Exact results are in
`exp_local/cqn_no_bc/stage4_seed1_gpu0_20260731005430/stage4_summary.json`.

### 2. Interpretation

Widening the conservative tail did not improve policy quality and instead
weakened it. This rules out the proposed rotating-top-two explanation at this
budget: the single worst unseen bin is the more useful conservative target.
Together, Stages 3 and 4 support retaining max-floor but not spending another
stage on tail width.

The completed control artifact exposes a separate distribution issue. At 10k,
the ordinary replay contained 19,609 transitions while the dedicated
successful-demo replay contained 8,061. Each update concatenated 16 samples
from each buffer. Because the ordinary replay began with demos and later
accumulated online rollouts, its expected composition at 10k was about 70.6%
demo-origin samples and 29.4% online-origin samples, before ordinary replay
sampling variance. Most learned-policy episodes had zero reward. Thus the
critic was not actually optimized on a purely offline expert distribution.

### 3. Next-stage decision

Stage 5 tests one offline-RL hypothesis: zero-return online state/action
transitions interfere with reward-only identification of the expert action in
a shared critic. The matched arms are:

- control: the retained MC + top-1 floor objective with the current
  16 ordinary + 16 dedicated-demo batch;
- treatment: the identical objective and total batch size 32, sampled only
  from the dedicated successful-demo replay.

The treatment is the final policy training objective, not actor or policy
pretraining. Online environments still advance the matched 10k-step schedule,
but their transitions are excluded by replay-source routing before the update.
The critic update itself remains invariant to the `demo` label and contains no
BC, likelihood, chosen-action margin, actor, flow head, or self-imitation.

Both arms use seed 1, the same 60 source demonstrations, and all other Stage-4
settings. Selection remains 50 episodes on seeds 400--449 at
2.5k/5k/7.5k/10k with earliest-checkpoint tie breaking. Held-out seeds
800--999 remain untouched. The demo-only treatment passes if its
validation-selected best is at least 40% and at least 15 percentage points
above the matched mixed-replay control. A pass advances the selected
checkpoint rule to training seeds 2 and 3; a fail rejects demo-only replay as
the missing distribution correction.

### 4. Execution

Implementation begins immediately. The data-source switch will preserve a
32-sample update batch, retain both replay artifacts for audit, and be tested
to ignore ordinary-replay values independently of the stored demo flag.

Implemented `replay.demo_only_updates`. The iterator consumes only the
dedicated demo replay, keeps the ordinary replay artifact and closes both
iterators correctly. The treatment uses `demo_batch_size=32`; the control
keeps `batch_size=16` plus `demo_batch_size=16`. CQN-AS tests pass (93), and
the replay-source/fast-merge/config checks pass (11), including a check that
ordinary replay is never consumed and stored demo labels do not control
selection.

Stage 5 launched at 2026-07-31 01:27 local time:

- controller: `exp_local/cqn_no_bc/stage5_seed1_gpu0_20260731012718`;
- control: `.../mc_mixed_replay`;
- treatment: `.../mc_demo_only`.

Both reached a real 1k update with finite losses. The demo-only row records
`mc_lower_bound_fraction=0.842` and a 32-sample source batch in its Hydra
artifact. Co-resident GPU use was 30.72 GiB without OOM. This is source-routing
and target-use evidence only.

## Stage 5 result and Stage 6 protocol (2026-07-31)

### 1. Previous-stage result

The fixed validation split completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC + max floor, mixed replay | 0% | 0% | 2% | 4% | 4% @ 10k |
| MC + max floor, demo-only | 0% | 0% | 2% | 8% | 8% @ 10k |

Demo-only improved by 4 points but failed the preregistered 40% / +15pp gate.
The first demo-only eval attempt exposed an eval-only configuration bug: the
sweep cleared `demo_batch_size` but retained
`replay.demo_only_updates=true`. The sweep now disables that replay-only flag
during evaluation; existing snapshots were evaluated without retraining.
`stage5_summary.json`, both CSVs, and completion sentinels are present under
`exp_local/cqn_no_bc/stage5_seed1_gpu0_20260731012718`.

### 2. Interpretation

Removing online failure transitions is mildly beneficial but not the missing
solution. This rules out replay-source contamination as the dominant cause:
even the clean successful-expert distribution reached only 8%. Both pure-RL
curves began improving only at the final checkpoint, but their per-head critic
still optimizes every action dimension independently against the same return.
For a 15-dimensional robot action, that factorization discards conditional
information about which later joint/action component is valuable given the
earlier components. More updates cannot restore information that the Q
function's conditioning omits.

### 3. Next-stage decision

Stage 6 tests a Q-Transformer-style joint-action hypothesis while preserving
the exact reward-only targets. The current CQN-AS critic predicts all 15
action dimensions in parallel. The treatment instead factorizes each
time-step action autoregressively:

`Q(s, a_0), Q(s, a_0, a_1), ..., Q(s, a_0, ..., a_14)`.

During training, the replay prefix is an input to the corresponding
conditional Q head; the current replay bin is still trained only by
max(MC return, Bellman return), and unseen current bins are trained to the
absolute zero return floor. During greedy control, dimensions are selected in
the same causal order and the selected prefix is fed to later Q heads. There
is no action likelihood, BC classification, actor, chosen-action margin,
pretraining, or demo-identity branch.

The matched arms use mixed replay to isolate architecture:

- control: existing parallel-dimension MC + max-floor critic;
- treatment: autoregressive-dimension MC + max-floor critic.

Both use seed 1, 60 demos, 10k updates, and a 32-sample merged batch.
Selection remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k, earliest on ties. Held-out seeds 800--999 remain untouched.
The treatment passes if its validation-best is at least 40% and at least 15
points above its matched control. A pass advances to seeds 2 and 3; a fail
rules out action-dimension conditioning in this implementation.

### 4. Execution

Implementation starts immediately. The new critic must retain the legacy model
unchanged when disabled, perform one base critic evaluation plus a lightweight
causal recurrent correction, and pass zero-return non-cloning, replayed-bin
gradient masking, demo-label invariance, shape, greedy-prefix, and matched
launch tests before GPU training.

Implemented `method.autoregressive_action_dims`. The disabled branch still
constructs the original `C2FSequenceDistributionalCritic` with its unchanged
parameter tree. The treatment wraps that critic with a shared action-dimension
GRU: state features are projected once, dimension `d` consumes only the
selected center from `d-1`, and a zero-initialized correction is added to the
conditional C51 logits. Training teacher-forces only the strict replay prefix;
greedy evaluation builds the same prefix causally.

The full CQN-AS suite passed (`97 passed`). After replacing a wasteful repeated
raw-feature projection with one state projection per critic call, the three
treatment-specific checks passed again. They establish strict-prefix
causality, identical updates when only the stored demo identity bit changes,
absence of policy/flow parameters, and matched launch composition. The
existing reward-floor checks continue to show zero replayed-bin gradients and
zero action-dependent signal when all returns equal the absolute floor.

Stage 6 launched at 2026-07-31 02:09 local time:

- controller: `exp_local/cqn_no_bc/stage6_seed1_gpu0_20260731020957`;
- control: `.../mc_parallel_dims`;
- treatment: `.../mc_autoregressive_dims`.

Both Hydra artifacts confirm the strict no-BC contract, and the only
treatment switch is `autoregressive_action_dims=true`. Both processes are
live together on GPU 0 at 30.26 GiB. The control reached 9k; after a
103-second first JIT/update, the treatment reached a real 1k update with
`critic_loss=1.112`, `mc_lower_bound_fraction=0.821`,
`unseen_return_floor_loss=0.0351`, and steady-state backend update time
0.049 seconds. These establish execution and target use only; policy quality
awaits the fixed validation sweep.

## Stage 6 result and Stage 7 protocol (2026-07-31)

### 1. Previous-stage result

The fixed validation split completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| parallel-dimension MC + max floor | 0% | 2% | 0% | 0% | 2% @ 5k |
| autoregressive-dimension MC + max floor | 0% | 0% | 0% | 0% | 0% @ 2.5k |

The treatment lost 2 percentage points and failed both parts of the
preregistered 40% / +15pp gate. Exact curves and checkpoint selection are in
`exp_local/cqn_no_bc/stage6_seed1_gpu0_20260731020957/stage6_summary.json`.

Matched validation-best probes on eight common demo-success states found:

| arm | replay-bin top-1 | current-action top-1 | max minus replay Q | Q/return Pearson |
| --- | ---: | ---: | ---: | ---: |
| parallel @ 5k | 55.26% | 50.00% | 0.01074 | 0.361 |
| autoregressive @ 2.5k | 52.59% | 45.56% | 0.01348 | 0.782 |

Probe artifacts are `probes/parallel_best5000.json` and
`probes/autoregressive_best2500.json` under the same Stage-6 controller.

### 2. Interpretation

The autoregressive critic learned a stronger ordering of replay actions by
future return, but made the expert bin less likely to win and increased the
largest competing-bin error. This rules out missing action-dimension
conditioning as the bottleneck in this residual implementation. It also
sharpens the unresolved issue: return information reaches the critic, yet the
current objective does not convert it into sufficiently separated action
values for greedy control.

The earlier mean, max, and top-k floor studies all regress expected Q with a
separate conservative term. Mean spreads one unit of loss across four unseen
bins; max/top-k train only a changing upper tail. None supplies a complete
categorical return target to every candidate bin.

### 3. Next-stage decision

Stage 7 tests one objective hypothesis: use a single dense C51 return
regression over every action bin. For each state/head:

- the replayed bin receives `max(MC return, Bellman return)`;
- every replay-unseen bin receives the task's absolute minimum return, zero;
- the loss is the summed categorical return cross-entropy over all five bins.

This is one Q-return objective, not an auxiliary action likelihood, ranking
margin, or expert-labelled branch. Its anti-cheat invariant is exact: when
the replayed return also equals zero, all bins receive the same categorical
target, so changing the recorded action bin cannot change either the loss or
its logit gradient.

The matched arms return to the parallel critic to isolate the objective:

- control: Stage-6 parallel MC + max expected-Q floor;
- treatment: dense all-bin categorical return regression, with the separate
  expected-Q floor disabled.

Both use seed 1, 60 source demos, mixed 16+16 replay, 10k updates, and all
other settings unchanged. Selection remains 50 episodes/checkpoint on seeds
400--449 at 2.5k/5k/7.5k/10k, earliest on ties. Held-out seeds 800--999 remain
untouched. The treatment passes at validation-best at least 40% and at least
15 points over the matched control. A pass advances to seeds 2 and 3; a fail
rejects dense all-bin categorical return supervision at this budget.

### 4. Execution

Implementation begins immediately. Tests must prove action-label-invariant
loss and gradients at the zero-return floor, nonzero return-dependent
separation, strict rejection of simultaneous floor/BC paths, unchanged legacy
composition, and a real finite update before paired GPU execution.

Implemented `method.dense_return_q_target`. The treatment replaces the
chosen-bin TD cross-entropy plus expected-Q floor with one categorical
return-regression tensor: it preserves the full chosen-bin TD/MC gradient and
adds one zero-return C51 target for each of the four unseen bins. The separate
floor is configuration-invalid in this mode.

All five treatment-specific tests passed, including exact equality of loss
and every logit gradient after changing replay bins when the chosen target is
also zero. A positive return target is required before the replayed and
unseen-bin gradients differ. The complete CQN-AS suite passes (`102 passed`);
`git diff --check`, Python compilation, shell syntax, and launch composition
also pass.

Stage 7 launched at 2026-07-31 02:51 local time:

- controller: `exp_local/cqn_no_bc/stage7_seed1_gpu0_20260731025138`;
- control: `.../mc_max_floor`;
- treatment: `.../mc_dense_return`.

Both processes are live on GPU 0 at 30.25 GiB. The treatment's first update
reported `critic_loss=dense_return_q_loss=19.659` (the summed five-bin
categorical loss) and no expected-Q floor metric. At 1k it reported finite
loss 3.447, `mc_lower_bound_fraction=0.745`, chosen Q 0.244, unseen Q 0.064,
and a 0.179 chosen/unseen gap. This verifies the single return objective and
real optimization, not policy quality.

## Stage 7 result and Stage 8 confirmation protocol (2026-07-31)

### 1. Previous-stage result

The fixed validation split completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| MC + max expected-Q floor | 0% | 0% | 0% | 10% | 10% @ 10k |
| dense all-bin categorical return | 4% | 12% | 40% | 40% | 40% @ 7.5k |

The treatment improved validation-best by 30 points and passed both parts of
the preregistered 40% / +15pp gate. The exact artifact is
`exp_local/cqn_no_bc/stage7_seed1_gpu0_20260731025138/stage7_summary.json`.
At 10k its chosen/unseen Q gap was 0.345 versus the control's 0.255, but
checkpoint selection correctly uses task success and chooses the earlier 7.5k
snapshot.

### 2. Interpretation

This is the first no-BC stage to pass its mechanism gate. Dense categorical
return targets convert reward information into materially better closed-loop
behavior; the improvement is not inferred from loss. It rules out the claim
that action likelihood or an imitation margin is necessary for any useful
demo policy in this setup.

It does not yet establish the goal. Seed-1 dense validation-best is 40%,
whereas original CQNAS seed 1 is 68%, and the original three-seed
validation-best mean is 72%. A single seed is also insufficient to distinguish
a repeatable objective from seed variance.

### 3. Next-stage decision

Stage 8 is a locked replication, not a new method:

- train the exact Stage-7 dense objective with training seeds 2 and 3;
- select each checkpoint independently on the same fixed 50 validation
  episodes, seeds 400--449, at 2.5k/5k/7.5k/10k with earliest tie breaking;
- combine those results with the already selected seed-1 7.5k checkpoint;
- compare the three-seed validation-best mean against the locked original
  CQNAS values 68/72/76%, mean 72%.

Held-out seeds 800--999 remain untouched. The Stage-8 pass criterion is a
no-BC three-seed validation-best mean of at least 72%. A pass immediately
evaluates the three frozen selected checkpoints on 200 held-out episodes each
and requires mean success at least the locked official 64.6% reference. A
validation fail does not spend held-out seeds and instead motivates a new
pure-RL hypothesis using the multi-seed failure pattern.

### 4. Execution

Seeds 2 and 3 will run together on GPU 0 at 0.45 XLA memory fraction each,
staggered by 120 seconds. Training, sequential validation, checkpoint
selection, the three-seed aggregate, and completion sentinels are handled by
one durable controller.

Stage 8 launched at 2026-07-31 03:17 local time:

- controller: `exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717`;
- replication runs: `.../dense_seed2` and `.../dense_seed3`;
- locked seed-1 source:
  `stage7_seed1_gpu0_20260731025138/mc_dense_return`.

Both replication processes are live together at 30.25 GiB. Seed 2 reached a
real 4k update and seed 3 completed its first compiled update; both report
`critic_loss=dense_return_q_loss`, no floor loss, finite MC fractions, and
the exact strict configuration. This is execution evidence only.

## Stage 8 result and Stage 9 compute test (2026-07-31)

### 1. Previous-stage result

The complete dense-return validation curves are:

| training seed | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 4% | 12% | 40% | 40% | 40% @ 7.5k |
| 2 | 4% | 8% | 32% | 46% | 46% @ 10k |
| 3 | 6% | 30% | 36% | 46% | 46% @ 10k |

The dense three-seed mean is 44%, compared with original CQNAS
68/72/76%, mean 72%; the deficit is 28 points. The exact aggregate is
`exp_local/cqn_no_bc/stage8_dense_multiseed_gpu0_20260731031717/stage8_summary.json`.
The validation gate failed and held-out seeds 800--999 remain sealed.

### 2. Interpretation

Dense all-bin return regression is repeatable: all three seeds learn a
nontrivial policy and converge to a narrow 40--46% validation-best band. This
rules out the Stage-7 gain being a lucky seed. It also rules out claiming
parity: the selected best checkpoints remain materially below every original
CQNAS seed.

The failure pattern leaves one bounded question. Seeds 2 and 3 improved from
32/36% at 7.5k to 46/46% at 10k, and both selected their last checkpoint.
Unlike original CQNAS, whose best checkpoints occur by 5k, the dense objective
fits five categorical return targets per head and may simply converge more
slowly. Training loss or Q gap cannot decide that question.

### 3. Next-stage decision

Stage 9 tests compute limitation without changing the method. Resume the exact
seed-2 and seed-3 10.5k states, including optimizer and replay, and extend
training to 20k. Validate only the new 12.5k/15k/17.5k/20k checkpoints on the
same seeds 400--449, then select each seed's best over its combined 2.5k--20k
curve with earliest tie breaking.

The compute-limited hypothesis passes only if both seeds improve at least 15
points over their locked 46% best and their two-seed extended-best mean is at
least 68%. A pass justifies extending seed 1 before reconsidering the final
72% validation gate. A fail rules out more updates alone and moves to a new
pure-RL objective. Held-out seeds remain untouched in either case.

### 4. Execution

The resume controller will preserve a copy of each 10k Hydra configuration,
reuse the exact run directories so `train_fast.py` restores
`latest_snapshot.pkl`, set only `num_train_frames=20000`, and run both
extensions on GPU 0 at 0.45 memory fraction with a 120-second stagger.

Stage 9 launched at 2026-07-31 03:42 local time:

- controller: `exp_local/cqn_no_bc/stage9_dense_extend20k_gpu0_20260731034235`;
- exact resumed runs: the Stage-8 seed-2 and seed-3 directories;
- preserved pre-resume configs: `.../configs_10k/seed2.yaml` and
  `seed3.yaml`.

Both processes are live at 30.25 GiB. Resume is artifact-verified rather than
inferred: seed 2 appended 11k--16k rows while retaining its episode counter,
and seed 3 appended an 11k row from the 10.5k latest snapshot. Both keep
`critic_loss=dense_return_q_loss` and finite values.

## Stage 9 result and Stage 10 episodic success-Q protocol (2026-07-31)

### 1. Previous-stage result

The exact-resume 20k validation curves completed:

| training seed | 12.5k | 15k | 17.5k | 20k | extended validation-best | gain over old best |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 2 | 46% | 40% | 44% | 38% | 46% @ 10k | +0pp |
| 3 | 50% | 56% | 50% | 40% | 56% @ 15k | +10pp |

The two-seed extended-best mean is 51%. Neither seed improved by the required
15 points and the mean is below the preregistered 68% threshold, so the
compute gate failed. The exact merged selection artifact and frozen snapshot
paths are in
`exp_local/cqn_no_bc/stage9_dense_extend20k_gpu0_20260731034235/stage9_summary.json`.
Training, validation, summary, and completion sentinels are all present.
Held-out seeds 800--999 remain sealed.

### 2. Interpretation

More updates alone do not explain the Stage-8 gap at this tested budget.
Seed 2 never exceeded 46%; seed 3 briefly reached 56%, then both fell to
38--40% at 20k. This rules out extending the unchanged dense objective to
20k as the route to parity. It does not claim that arbitrary training lengths
can never help, but the declining curves make another blind extension
unjustified.

A read-only audit of the actual Stage-8 demo replay identifies a more specific
information bottleneck. Its 51 complete episode files contain 8,061 valid
transitions: 50 episodes have terminal return 1 and one has return 0.
Nevertheless, discounting reduces the stored reward-to-go to mean 0.487,
median 0.448, and tenth percentile 0.217. Thus the existing dense target
throws away much of the common terminal-success bit for early expert
transitions even though selection is undiscounted task success.

### 3. Next-stage decision

Stage 10 tests one hypothesis: the relevant RL value is episodic success
probability, so every transition in a completed successful trajectory should
receive Monte-Carlo target 1 and every transition in a failed trajectory
target 0. For every state/head, the replayed bin receives that binary episode
outcome and all replay-unseen bins receive zero, in the same single dense C51
Q cross-entropy.

This is reward-gated Monte-Carlo control, not action likelihood:

- the update never reads the `demo` flag;
- the one failed demonstration receives no positive action target;
- successful online trajectories receive the same target as successful demos;
- failed online trajectories and failed demos are treated identically;
- when outcome is zero, every bin has the same zero-return target, so changing
  which bin was recorded cannot create replay-versus-unseen gradient
  separation at the categorical loss.

There is no actor, policy head, behavior pretraining, FOSD, margin, AWR,
flow, self-imitation, or auxiliary loss. Greedy action selection comes only
from the learned Q distribution.

The matched seed-1 arms are:

- control: the Stage-7/8 discounted MC-lower-bound dense-return objective;
- treatment: binary episodic-success dense-Q regression, replacing the
  discounted/Bellman chosen-bin target.

Both train fresh for 10k with the same 60 requested source demos, mixed 16+16
replay, architecture, optimizer, exploration, and snapshots. Selection is
again 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k, earliest on ties. The mechanism gate requires treatment
validation-best at least 60% and at least 15 points above its fresh matched
control. A pass advances the exact treatment to training seeds 2 and 3. A
fail rejects binary episodic success-Q in this form and moves to a separate
pure-RL hypothesis. Held-out remains sealed.

### 4. Execution

Implementation begins immediately. Before GPU launch, focused tests must show
that the target is the completed reward outcome rather than demo identity,
that zero outcome provides no categorical action-label separation, that the
new mode is mutually exclusive with the discounted MC-lower-bound target,
that no policy or imitation metric/path exists, and that the matched configs
differ only in the chosen return target mode.

Implemented `method.episodic_success_q_target`. It makes episodic success the
chosen-bin C51 distribution and fully replaces the Bellman/discounted-MC
target; configuration rejects combining the two or adding a separate MC
loss. Replay stores the completed reward-to-go only to recover whether the
trajectory ever received positive reward. The update never reads demo
identity.

The focused Stage-10 tests pass (8/8): positive discounted returns recover
outcome 1, non-positive returns recover 0, swapping every `demo` flag leaves
all parameters and metrics bit-identical, no policy/flow parameters or
imitation metrics exist, the only reported loss is dense Q loss, and the two
target modes are mutually exclusive. The full CQN-AS suite passes
(`106 passed`). Python compilation, shell syntax, summary dry-run, and
`git diff --check` also pass.

Stage 10 launched at 2026-07-31 04:13 local time:

- controller: `exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356`;
- fresh control: `.../discounted_dense`;
- treatment: `.../episodic_success_q`.

The starts were staggered by 120 seconds. Both processes are live together on
GPU 0 at 30.25 GiB. The treatment completed a real 1k update with finite
`critic_loss=dense_return_q_loss=2.419`,
`episodic_success_fraction=1.0`, chosen Q 0.432, unseen Q 0.117, and no
MC-lower-bound, BC, policy, flow, or auxiliary-loss metric. This verifies
execution and reward-outcome target use only; checkpoint validation determines
policy quality.

## Stage 10 result and Stage 11 ordered success-return protocol (2026-07-31)

### 1. Previous-stage result

The fresh matched validation curves completed:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| discounted dense-return control | 12% | 24% | 34% | 48% | 48% @ 10k |
| binary episodic-success Q | 0% | 30% | 48% | 36% | 48% @ 7.5k |

The treatment improved by 0 points, stayed below 60%, and failed both parts of
the preregistered gate. Exact curves and selected snapshot paths are in
`exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/stage10_summary.json`.
Held-out seeds 800--999 remain sealed.

Two validation-selected, target-critic probes used exactly the same 32 replay
states and actions from the fresh control:

| target | demo-success replay-bin top-1 | current-action top-1 | max minus replay Q | Q/discounted-return Pearson |
| --- | ---: | ---: | ---: | ---: |
| discounted @ 10k | 77.92% | 69.17% | 0.0150 | 0.856 |
| binary success @ 7.5k | 71.49% | 63.89% | 0.0443 | 0.176 |

The probe artifacts are `probes/discounted_best10000.json` and
`probes/episodic_best7500.json` under the Stage-10 controller.

### 2. Interpretation

An undiscounted terminal-success bit is sufficient to recover a 48% policy,
but replacing all reward-to-go magnitudes by that bit does not improve the
best checkpoint and overtrains by 10k. The common-state probes explain the
failure mechanistically: binary targets discard useful temporal return
ordering, reduce replay-action top-1, and make the largest wrong-bin errors
about three times larger. This rules out discount attenuation alone as the
remaining bottleneck.

The retained discounted objective has strong value ordering and nearly the
original critic's replay-bin top-1, but its early successful transitions still
receive weak targets. The next bounded question is whether terminal success
can strengthen those targets without erasing their ordering.

### 3. Next-stage decision

Stage 11 tests one reward-target hypothesis. For completed return
`G_gamma`, form one ordered success return

`G_ordered = 0.5 * G_gamma + 0.5 * I[G_gamma > 0]`.

Among successful trajectories this strictly preserves the ordering of
`G_gamma` while lifting every positive target toward one; failures remain
zero. `G_ordered` replaces `G_gamma` inside the existing max of Monte-Carlo
and Bellman distributions. Dense unseen bins remain zero. This is still one
C51 Q target and one loss, not a second outcome loss or action margin.
Demo identity is never read, and successful online trajectories receive the
same transformation.

To use the two-run GPU budget without rerunning already locked controls, train
the exact treatment with seeds 1 and 2 together. Their matched dense controls
are locked before this hypothesis:

- seed 1: fresh Stage-10 control, 48% @ 10k;
- seed 2: Stage-8 dense control, 46% @ 10k.

Both treatments use the same architecture, 60 requested demos, mixed 16+16
replay, optimizer, 10k budget, and validation split. Each checkpoint is
selected independently on 50 episodes, seeds 400--449, at
2.5k/5k/7.5k/10k with earliest tie breaking. The gate requires both treatment
seeds at least 60%, their mean at least 64%, and a mean improvement of at
least 15 points over the locked 47% control mean. A pass trains seed 3 before
the final 72% three-seed gate. A fail rejects this ordered success-return
target. Held-out remains sealed.

### 4. Execution

Implementation begins immediately. Tests must establish the exact formula,
zero preservation, strict monotonic ordering among positive returns, demo-flag
invariance, mutual exclusion from the direct binary mode, one reported Q
loss, and matched configuration composition before launch.

Implemented `method.ordered_success_return_mix`; Stage 11 fixes it at 0.5.
The transformed scalar replaces the MC scalar before C51 projection and the
existing MC-versus-Bellman target choice, so no second loss is introduced.

All four focused Stage-11 checks pass: the exact targets for
`[-1, 0, .2, .6, 1]` are `[0, 0, .6, .8, 1]`, positive ordering is strict,
demo/online flag swaps give bit-identical parameters and metrics, and the
configuration rejects use outside the single dense MC-lower-bound target.
The full CQN-AS suite passes (`110 passed`). Python compilation, shell syntax,
summary dry-run, and `git diff --check` also pass.

Stage 11 launched at 2026-07-31 04:47 local time:

- controller: `exp_local/cqn_no_bc/stage11_ordered_success_gpu0_20260731044744`;
- treatments: `.../ordered_success_seed1` and
  `.../ordered_success_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

The 120-second stagger completed and both treatments are live together at
30.15 GiB. Seed 1 reached a real 1k update with raw MC mean 0.423, transformed
target mean 0.649, finite `critic_loss=dense_return_q_loss=2.982`, and no
second loss or imitation metric. Seed 2 is initialized on the same strict
configuration. These are execution and target-use checks only.

## Stage 11 result and Stage 12 chunk-horizon alignment protocol (2026-07-31)

### 1. Previous-stage result

The two treatment validation curves and locked matched controls are:

| seed | ordered success 2.5k/5k/7.5k/10k | treatment best | locked dense best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 6% / 36% / 36% / 46% | 46% @ 10k | 48% @ 10k | -2pp |
| 2 | 8% / 8% / 32% / 36% | 36% @ 10k | 46% @ 10k | -10pp |

Treatment mean is 41% versus the locked control mean 47%, a -6 point change.
Neither treatment seed reached 60%; the 64% / +15pp gate failed. Exact curves
and snapshots are in
`exp_local/cqn_no_bc/stage11_ordered_success_gpu0_20260731044744/stage11_summary.json`.
Held-out remains sealed.

### 2. Interpretation

Preserving discounted-return ordering while lifting long-horizon successes
does not fix the gap and is harmful on seed 2. Together with Stage 10, this
rules out simple target-magnitude attenuation as the main bottleneck: using
the success bit alone ties the control, while a monotone midpoint blend loses
six points across two seeds.

The remaining structural mismatch is temporal. Current CQN-AS predicts an
action sequence but replay supplies a one-step Bellman return and rollout
replans every step. BC can supervise every future action token despite that
mismatch; once BC is removed, the Q target should denote the same action
object that is actually executed.

### 3. Next-stage decision

Stage 12 tests one chunk-credit hypothesis with `K=8`:

- predict an eight-action sequence;
- execute that plan for eight primitive decisions before replanning, in both
  training and validation;
- for the treatment, use replay `nstep=8`, so the Bellman target contains the
  discounted rewards generated by exactly those eight actions and bootstraps
  at `s_(t+8)`;
- retain the same dense discounted MC/Bellman Q objective, greedy critic,
  mixed replay, and every anti-imitation guard.

The matched control also uses action-sequence 8 and replan interval 8 but keeps
`nstep=1`. Thus the only arm difference is whether the RL return horizon
matches the executed chunk. There is still no actor, action likelihood,
pretraining, self-imitation, margin, FOSD, flow, or auxiliary loss.

Both fresh arms use training seed 1 and run together on GPU 0. Selection
remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k, earliest tie breaking, with train and validation both using
the same replan-8 execution semantics. The treatment gate is validation-best
at least 64% and at least 15 points above the matched nstep-1 control. A pass
replicates seeds 2 and 3 before comparison with original CQNAS mean 72%. A
fail rejects chunk-aligned eight-step backup in this form. Held-out stays
sealed.

### 4. Execution

This stage is a configuration-only intervention. Composition tests must lock
`action_sequence=8`, `temporal_ensemble_replan_interval=8`, and identical
strict no-imitation settings while proving the sole arm difference is
`replay.nstep: 1 -> 8`. Shell, summary, and config checks precede launch.

The composition audit passes and the fully flattened resolved experiment
configs differ at exactly one key: `replay.nstep` (1 versus 8). Both lock
`action_sequence=8`, replan interval 8, dense discounted MC/Bellman Q,
mixed 16+16 replay, and every strict anti-imitation setting. The sparse-replan
execution tests pass (2/2); shell syntax, summary dry-run, and
`git diff --check` also pass.

Stage 12 launched at 2026-07-31 05:12 local time:

- controller: `exp_local/cqn_no_bc/stage12_chunk_horizon_gpu0_20260731051253`;
- matched control: `.../k8_nstep1`;
- chunk-aligned treatment: `.../k8_nstep8`.

Both processes are live together after a 120-second stagger at 30.24 GiB.
The treatment replay logs `nstep: 8` for both online and demo buffers and
completed a real 1k update with finite
`critic_loss=dense_return_q_loss=3.640`,
`mc_lower_bound_fraction=0.866`, and no imitation or second-loss metric.
This establishes correct execution and return plumbing only.

## Stage 12 result and Stage 13 local-return-kernel protocol (2026-07-31)

### 1. Previous-stage result

The complete matched validation curves are:

| arm | 2.5k | 5k | 7.5k | 10k | validation-best |
| --- | ---: | ---: | ---: | ---: | ---: |
| K=8, replan 8, nstep 1 | 0% | 2% | 18% | 20% | 20% @ 10k |
| K=8, replan 8, nstep 8 | 0% | 0% | 6% | 22% | 22% @ 10k |

The chunk-aligned treatment improves its matched control by only 2 points,
is 42 points below the preregistered 64% absolute gate, and therefore fails.
The exact curves and selected snapshot paths are in
`exp_local/cqn_no_bc/stage12_chunk_horizon_gpu0_20260731051253/stage12_summary.json`.
Training and validation completed for both arms; held-out seeds 800--999
remain sealed.

### 2. Interpretation

An eight-step Bellman return matching an eight-step executed plan is not
sufficient to replace BC in this form. It rules out the specific claim that
the original gap is primarily a one-step-versus-eight-step credit mismatch.
It does not rule out all chunk horizons: forcing eight open-loop actions also
reduces the matched nstep-1 control to 20%, far below the retained
short-replanning dense-Q controls. Thus most of the stage is dominated by an
execution-horizon penalty rather than a useful return-horizon contrast.

The retained dense objective instead has a continuous-action generalization
problem. It gives the exact replay bin the full return and every other bin
zero, even at the finest C2F level where an adjacent bin changes the decoded
continuous action by only one small cell. This creates an unnecessarily
discontinuous Q target around every successful action.

### 3. Next-stage decision

Stage 13 tests one local-smoothness hypothesis while restoring the retained
`action_sequence=16`, one-step replanning, and one-step replay setup. For each
finest-level action head, use the single categorical target

`T_b = kappa_b * T_G + (1 - kappa_b) * T_0`,

where the replayed bin has `kappa=1`, its immediate in-range neighbors have
`kappa=0.5`, and all other bins and all unseen coarse-level bins have
`kappa=0`. `T_G` is the existing max of discounted Monte-Carlo and Bellman
return distributions; `T_0` is the valid zero-return distribution.

This is a local continuous-action Q prior, not action likelihood. If the
chosen return is zero, `T_G=T_0` and every target, loss value, and logit
gradient is exactly invariant to the recorded action. Demo identity is never
read, and successful online experience receives the same kernel. There is
still one dense C51 Q cross-entropy and no actor, policy head, pretraining,
FOSD, margin, AWR, flow, self-imitation, or auxiliary loss.

Use the two-run GPU budget for treatment seeds 1 and 2. Their exact matched
dense-Q controls were locked before this hypothesis:

- seed 1: Stage-10 fresh dense control, 48% @ 10k;
- seed 2: Stage-8 dense control, 46% @ 10k.

Both treatments keep the same 60 requested demos, mixed 16+16 replay,
architecture, optimizer, and 10k budget. Select independently on 50 episodes
per checkpoint, seeds 400--449, at 2.5k/5k/7.5k/10k with earliest tie
breaking. The treatment passes only if both seeds reach at least 60%, their
mean reaches at least 64%, and their mean improves by at least 15 points over
the locked 47% control mean. A pass trains seed 3 before the final original
CQNAS 72% validation-mean gate. A fail rejects this finest-neighbor kernel.
Held-out remains sealed.

### 4. Execution

Implementation starts immediately. Before launch, tests must prove the exact
neighbor target, exact zero-return action-label invariance of both loss and
gradients, demo-flag invariance in a real update, mutual exclusion from any
second objective, unchanged zero-kernel behavior, and matched launch
composition.

Implemented `method.dense_return_finest_neighbor_weight`; Stage 13 fixes it
at 0.5. The convex target is constructed inside the existing all-bin
categorical target tensor before the same single cross-entropy reduction.
Only immediate neighbors at the last C2F level are changed.

All six focused checks pass. They include bit-exact loss and gradient
invariance after changing every replay bin at zero return with the kernel
enabled, explicit gradient checks that coarse and non-neighbor bins remain at
the floor, a real update whose parameters and metrics are bit-identical after
flipping every demo flag, one-loss assertions, and a flattened composition
audit showing the launch differs from its dense control only at the new
weight. The complete CQN-AS suite passes (`116 passed`); Python compilation,
shell syntax, summary dry-run, and `git diff --check` also pass.

Stage 13 launched at 2026-07-31 05:42 local time:

- controller:
  `exp_local/cqn_no_bc/stage13_finest_neighbor_gpu0_20260731054229`;
- treatments: `.../finest_neighbor_seed1` and
  `.../finest_neighbor_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

The 120-second stagger completed and both Python processes are live together
on GPU 0 at about 30.0 GiB. Seed 1 completed a real 1k update with finite
`critic_loss=dense_return_q_loss=4.582`,
`mc_lower_bound_fraction=0.792`, chosen/unseen Q gap 0.158, and no imitation
or auxiliary-loss metric. This verifies execution only.

## Stage 13 result and Stage 14 primitive-action Q protocol (2026-07-31)

### 1. Previous-stage result

The complete local-kernel curves and locked dense controls are:

| seed | local kernel 2.5k/5k/7.5k/10k | treatment best | locked dense best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 14% / 18% / 36% / 56% | 56% @ 10k | 48% @ 10k | +8pp |
| 2 | 4% / 22% / 34% / 28% | 34% @ 7.5k | 46% @ 10k | -12pp |

The treatment mean is 45% versus the locked control mean 47%, a -2 point
change. Neither the two-seed 64% mean nor both-seeds-at-least-60% condition
passes. Exact curves and selected snapshots are in
`exp_local/cqn_no_bc/stage13_finest_neighbor_gpu0_20260731054229/stage13_summary.json`.
Training, validation, summary, and completion sentinels are present; no
Stage-13 process remains. Held-out seeds 800--999 remain sealed.

### 2. Interpretation

A finest-level local return kernel can improve one seed by eight points, but
the effect reverses by twelve points on seed 2 and is negative on average.
This rules out a fixed 0.5 immediate-neighbor kernel as a repeatable route to
parity. It does not justify tuning the kernel against the same validation
split: the cross-seed sign reversal indicates objective instability rather
than a merely under-sized constant.

The remaining causal mismatch is visible in the retained K=16 update. The
same current-state return target trains all 16 planned action tokens, although
closed-loop rollout replans after one primitive decision. BC has a direct
future action label for every token; a transition-based Q target only
identifies the action that was actually executed from the current state.
Stage 12 made the opposite repair by executing a long open-loop plan, but its
20--22% results show that MovePlate pays a large eight-step open-loop penalty.

### 3. Next-stage decision

Stage 14 tests the causally aligned closed-loop endpoint:

- set `action_sequence=1`, so the critic's sole action token is exactly the
  primitive action executed by the transition;
- retain replan interval 1 and replay `nstep=1`;
- restore the unmodified dense discounted MC/Bellman target with neighbor
  weight zero;
- keep every other optimizer, replay, demo, exploration, and evaluation
  setting fixed.

This is standard greedy primitive-action Q control inside the CQN-AS critic,
not a policy or action-likelihood model. Every action-specific target remains
return-gated; zero-return samples have the exact all-bin invariance. There is
no actor, pretraining, BC, FOSD, margin, AWR, flow, self-imitation, or
auxiliary loss.

Train primitive-Q seeds 1 and 2 together and compare them with the same
pre-hypothesis K=16 dense controls, 48% and 46%. Selection remains 50
episodes/checkpoint on seeds 400--449 at 2.5k/5k/7.5k/10k with earliest tie
breaking. The gate requires both primitive-Q seeds at least 60%, mean at least
64%, and mean improvement at least 15 points over the 47% control mean. A
pass trains seed 3 before the final original-CQNAS 72% validation-mean gate.
A fail rejects the K=1 causal action object at this budget. Held-out remains
sealed.

### 4. Execution

The legacy constructor rejects `action_sequence=1` even though the GRU,
greedy action, replay, and temporal-ensemble code are shape-valid at K=1.
Execution therefore includes one minimal compatibility change: relax that
lower-bound check from 2 to 1. A flattened composition audit must still show
the experiment config differs from the retained dense configuration only in
`action_sequence: 16 -> 1`; focused tests must verify K=1 action/critic and
replan shapes plus the existing strict no-imitation invariants. Shell,
summary, Python, full regression, and diff checks precede execution.

The lower-bound compatibility change and launch configuration are
implemented. Seven focused checks pass, including a real finite K=1 dense-Q
update, action/critic shape checks, per-step replan behavior, strict one-loss
assertions, and a flattened audit whose only resolved experiment difference
is `action_sequence`. The complete CQN-AS suite passes (`118 passed`);
shell syntax, summary dry-run, Python compilation, and `git diff --check`
also pass.

Stage 14 launched at 2026-07-31 06:12 local time:

- controller: `exp_local/cqn_no_bc/stage14_primitive_q_gpu0_20260731061215`;
- treatments: `.../primitive_q_seed1` and `.../primitive_q_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Seed 1 is live on GPU 0 and completed a real 1k update with
`critic_loss=dense_return_q_loss=3.181`,
`mc_lower_bound_fraction=0.742`, chosen/unseen Q gap 0.171, and no imitation
or auxiliary-loss metric. Its resolved config records K=1, nstep 1, neighbor
weight zero, and every strict guard. Seed 2 starts after the planned
120-second stagger. This is execution evidence only.

## Stage 14 result and Stage 15 replay-SARSA Q protocol (2026-07-31)

### 1. Previous-stage result

The complete K=1 curves are:

| seed | primitive-Q 2.5k/5k/7.5k/10k | treatment best | locked K=16 dense best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 0% / 0% / 0% / 0% | 0% @ 2.5k | 48% @ 10k | -48pp |
| 2 | 0% / 0% / 2% / 0% | 2% @ 7.5k | 46% @ 10k | -44pp |

The primitive-Q mean is 1% versus the locked control mean 47%, a -46 point
change. Exact curves and selected snapshots are in
`exp_local/cqn_no_bc/stage14_primitive_q_gpu0_20260731061215/stage14_summary.json`.
Training, validation, summary, and completion sentinels are present; no
Stage-14 process remains. Held-out seeds 800--999 remain sealed.

### 2. Interpretation

Training only the primitive action that causally generated each transition is
not enough to recover a MovePlate policy at 10k. The consistent two-seed
collapse rules out K=1 as the route to parity and establishes that the
long-horizon action-sequence representation supplies essential information.
Together with Stage 12, the result exposes the constraint: the learner must
retain future plan tokens without forcing a long open-loop execution horizon.

The retained dense K=16 method bootstraps from the critic's own greedy next
plan. Early in offline/demo-driven learning that maximization can select
unsupported next actions and inject OOD target error. Replay already contains
the actual consecutive action sequence and therefore supports a standard
SARSA Bellman target along both expert and online trajectories.

### 3. Next-stage decision

Stage 15 keeps K=16, one-step replanning, full-sequence dense discounted
MC/Bellman Q, and every retained setting. It changes only the Bellman
bootstrap action:

- control: current Double-CQN greedy next plan;
- treatment: shift the replayed consecutive action sequence by one token, so
  the next-state target evaluates
  `[a_(t+1), ..., a_(t+15), a_(t+15)]`.

This is replay SARSA, a standard RL target. It does not optimize the
likelihood, margin, rank, or reconstruction of a replay action; it only
evaluates the action actually observed at the bootstrap state. Demo identity
is never read, successful and failed online data follow the same rule, greedy
Q alone acts at evaluation, and there is no actor, policy head, pretraining,
BC, FOSD, AWR, flow, self-imitation, or auxiliary loss. The existing
Monte-Carlo lower bound and dense all-bin target remain one categorical Q
target.

Train treatment seeds 1 and 2 together against the locked greedy-target dense
controls, 48% and 46%. Select on the unchanged 50 episodes/checkpoint,
seeds 400--449, at 2.5k/5k/7.5k/10k with earliest tie breaking. The gate
requires both SARSA seeds at least 60%, mean at least 64%, and mean improvement
at least 15 points over the 47% control mean. A pass trains seed 3 before the
original-CQNAS 72% validation-mean gate. A fail rejects replay-next SARSA in
this form. Held-out remains sealed.

### 4. Execution

Implementation minimally permits `td_target_action_source=replay_next` while
`separate_bc_policy=false` and under the strict guard; behavior-policy and
policy-value target sources remain forbidden. Tests must prove that no policy
parameters or policy metrics exist, flipping every demo flag is bit-identical,
the replay sequence is shifted exactly, only one Q loss is reported, and the
resolved treatment differs from control only in target-action source. Full
regression, shell, summary, compilation, and diff checks precede launch.

The strict no-imitation validator and constructor now permit only
`replay_next` in addition to the existing critic source when no policy head
exists. The shift operation is factored into a tested helper; policy and
policy-value sources remain forbidden.

All eleven focused checks pass: exact sequence shift and boundary repeat,
bit-identical parameters and metrics after flipping every demo flag, absence
of all policy parameters and metrics, a real finite one-loss update, strict
rejection of BC-policy target selection, and a flattened one-key composition
diff. The complete CQN-AS suite passes (`122 passed`); shell syntax, summary
dry-run, Python compilation, and `git diff --check` also pass.

Stage 15 launched at 2026-07-31 06:39 local time:

- controller:
  `exp_local/cqn_no_bc/stage15_replay_sarsa_gpu0_20260731063921`;
- treatments: `.../replay_sarsa_seed1` and `.../replay_sarsa_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Seed 1 is live on GPU 0 with the exact strict K=16 replay-next configuration
and completed a real 1k update with
`critic_loss=dense_return_q_loss=3.670`,
`mc_lower_bound_fraction=0.895`, chosen/unseen Q gap 0.187, and no policy or
auxiliary-loss metric. Seed 2 starts after the planned 120-second stagger.
This verifies execution only.

## Stage 15 result and Stage 16 sequence-aligned return protocol (2026-07-31)

### 1. Previous-stage result

The replay-SARSA curves and locked greedy-target controls are:

| seed | replay SARSA 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 6% / 12% / 56% / 50% | 56% @ 7.5k | 48% @ 10k | +8pp |
| 2 | 4% / 14% / 36% / 30% | 36% @ 7.5k | 46% @ 10k | -10pp |

Treatment mean is 46% versus control mean 47%, a -1 point change. The large
seed-1 7.5k gain does not replicate and both preregistered per-seed/mean gates
fail. Exact curves and snapshots are in
`exp_local/cqn_no_bc/stage15_replay_sarsa_gpu0_20260731063921/stage15_summary.json`.
All completion sentinels are present; no Stage-15 process remains. Held-out
seeds 800--999 remain sealed.

### 2. Interpretation

Following replay actions in the Bellman bootstrap can improve one seed but is
not a stable replacement for BC. This rules out greedy OOD next-action
selection as the main remaining bottleneck. Together with the similarly
signed Stage-13 result, it also warns against combining two seed-1-only gains
without a separate mechanism.

There is a direct target error still shared by all retained K=16 variants.
The replay sequence contains actions at times `t ... t+15`, but the same
current-state discounted return `G_t` is broadcast to every token. In the
first-success sparse-reward MDP, the exact return belonging to token `k`
before success is

`G_(t+k) = min(G_t / gamma^k, 1)`.

Thus the current target systematically undervalues later expert actions and
does not use the temporal information encoded in the scalar return.

### 3. Next-stage decision

Stage 16 tests one sequence-credit hypothesis. Keep K=16, one-step replanning,
greedy Double-CQN bootstrap, dense all-bin categorical Q, mixed replay, and
all other retained settings. Change only the Monte-Carlo lower bound:

- control: broadcast scalar `G_t` to all 16 action tokens;
- treatment: use `G_(t+k)=min(G_t/0.99^k,1)` for token `k`, repeated only over
  that token's primitive action dimensions and C2F levels.

Zero and failed returns remain zero at every token. The transformed values
replace the scalar inside the same MC-versus-Bellman maximum and the same
single dense C51 loss; they are not an auxiliary temporal or policy loss.
The update never reads demo identity, and successful online trajectories use
the identical formula. With zero return, every action bin still has the exact
same target and gradient.

Train treatment seeds 1 and 2 against locked scalar-return dense controls,
48% and 46%. Selection remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k with earliest tie breaking. The gate requires both aligned
seeds at least 60%, mean at least 64%, and mean improvement at least 15 points
over the 47% control mean. A pass trains seed 3 before the original-CQNAS 72%
validation-mean gate. A fail rejects exact sparse-return token alignment at
this budget. Held-out remains sealed.

### 4. Execution

Implementation adds one optional sequence-alignment discount, null by
default and fixed to the replay gamma 0.99 here. Tests must prove the exact
formula and C51 projection, zero-return/action-label invariance, strict
single-loss and demo-flag invariance, full-sequence requirement, unchanged
legacy default, and a one-key resolved config diff. Full regression, shell,
summary, compilation, and diff checks precede launch.

Implemented `method.sequence_aligned_mc_discount`. The per-token point masses
replace the scalar MC distribution only inside the existing MC-versus-Bellman
selection; the dense target and its single cross-entropy are unchanged.

All six focused checks pass: exact powers-of-gamma returns, exact off-grid C51
projection, combined zero-return loss/gradient action-label invariance,
bit-identical demo-flag updates with one reported loss, rejection outside
full-sequence Q, and a flattened one-key config diff. The complete CQN-AS
suite passes (`128 passed`); shell syntax, summary dry-run, Python
compilation, and `git diff --check` also pass.

Stage 16 launched at 2026-07-31 07:15 local time:

- controller:
  `exp_local/cqn_no_bc/stage16_sequence_return_gpu0_20260731071516`;
- treatments: `.../sequence_return_seed1` and
  `.../sequence_return_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Seed 1 is live on GPU 0 and completed a real 1k update with
`critic_loss=dense_return_q_loss=3.610`,
`mc_lower_bound_fraction=0.745`, chosen/unseen Q gap 0.173, raw MC mean
0.454, and aligned-token mean 0.485. There is no imitation or auxiliary-loss
metric. Seed 2 starts after the planned 120-second stagger. This verifies
execution only.

## Stage 16 result and Stage 17 effective replay-SARSA protocol (2026-07-31)

### 1. Previous-stage result

The sequence-aligned return curves and locked scalar-return controls are:

| seed | aligned return 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 6% / 28% / 32% / 40% | 40% @ 10k | 48% @ 10k | -8pp |
| 2 | 2% / 20% / 46% / 44% | 46% @ 7.5k | 46% @ 10k | 0pp |

Treatment mean is 43% versus the locked control mean 47%, a -4 point change.
Neither seed reaches 60% and every preregistered gate fails. Exact curves and
selected snapshots are in
`exp_local/cqn_no_bc/stage16_sequence_return_gpu0_20260731071516/stage16_summary.json`.
Training, validation, summary, and completion sentinels are present; no
Stage-16 process remains. Held-out seeds 800--999 remain sealed.

### 2. Interpretation

Correcting the sparse-return value attached to each future action token does
not improve MovePlate policy quality at this budget. The result rules out the
specific scalar-return broadcast bias as the main parity bottleneck; it does
not rule out action-sequence Q learning itself.

A post-result code-path audit found that the Stage-15 replay-SARSA
intervention was not operative. Under the required
`separate_bc_policy=false` setting, `CQNAs._build_update_fn()` delegated to
the parent CQN update, and that parent always selected the greedy critic
bootstrap action. The replay-shift branch existed only in the separate-policy
update path, which strict no-imitation mode forbids. Therefore the measured
Stage-15 curves remain real policy results but constitute a fresh stochastic
replication of the greedy-target control, not a replay-SARSA experiment.
They cannot rule replay-SARSA in or out.

### 3. Next-stage decision

Stage 17 repeats the originally intended single-variable causal test after
making the target source effective:

- retain K=16, replan interval 1, replay nstep 1, dense discounted
  MC/Bellman categorical Q, mixed expert/online replay, and every optimizer,
  exploration, and evaluation setting;
- control uses the online critic argmax at `s_(t+1)`;
- treatment evaluates the actual consecutive replay sequence
  `[a_(t+1), ..., a_(t+15), a_(t+15)]` at `s_(t+1)`.

This is one standard one-step SARSA Q target, not an action-likelihood,
ranking, reconstruction, or auxiliary objective. Demo identity is not read;
failed demos and online samples use the identical update; no actor, policy
head, pretraining, BC, FOSD, margin, AWR, flow, or self-imitation path exists.
Greedy Q alone acts at evaluation.

Train corrected treatment seeds 1 and 2 concurrently against the locked
greedy-target controls, 48% and 46%. Select on 50 episodes/checkpoint with
seeds 400--449 at 2.5k/5k/7.5k/10k and earliest tie breaking. The gate
requires both treatment seeds at least 60%, mean at least 64%, and mean
improvement at least 15 points over the 47% control mean. A pass trains seed
3 before the original-CQNAS 72% validation-mean gate. A fail rejects
one-step replay-SARSA only after the intervention is genuinely active.
Held-out remains sealed.

### 4. Execution

The parent single-objective CQN update now calls a target-action hook.
CQN-AS overrides that hook only for `replay_next`, returning the exact shifted
replay sequence; ordinary CQN and critic-target CQN-AS retain greedy
Double-CQN behavior. The separate-policy implementation is unchanged.
Treatment logging emits `td_target_replay_next=1` as execution evidence, not
as a loss.

Focused tests prove the shifted sequence exactly, directly inspect the
selected bootstrap action, instrument a real no-policy update to prove the
parent update calls the new hook, flip every demo flag with bit-identical
parameters and metrics, require a single reported Q loss, and audit the
resolved treatment as the same strict one-key configuration contrast.

All five focused checks pass. The complete CQN-AS suite passes
(`129 passed`); summary dry-run, shell syntax, Python compilation, and
`git diff --check` also pass.

Stage 17 launched at 2026-07-31 07:49 local time:

- controller:
  `exp_local/cqn_no_bc/stage17_effective_replay_sarsa_gpu0_20260731074940`;
- treatments: `.../effective_replay_sarsa_seed1` and
  `.../effective_replay_sarsa_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Both treatments are live concurrently on GPU 0 with 0.45 memory fractions.
At a real 1k-or-later update, seed 1 reports
`critic_loss=dense_return_q_loss=2.753`,
`td_target_replay_next=1`, and no policy/BC/auxiliary metric; seed 2 reports
3.620, 1, and the same strict absence. Total GPU allocation is approximately
30.2/32.6 GB. This establishes that the corrected SARSA intervention is
actually executing, not policy quality.

## Stage 17 result and Stage 18 dense expected-Q protocol (2026-07-31)

### 1. Previous-stage result

The corrected replay-SARSA curves and locked greedy-target controls are:

| seed | corrected SARSA 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 4% / 10% / 48% / 44% | 48% @ 7.5k | 48% @ 10k | 0pp |
| 2 | 4% / 10% / 34% / 44% | 44% @ 10k | 46% @ 10k | -2pp |

Treatment mean is 46% versus control mean 47%, a -1 point change. Neither
seed reaches 60% and all preregistered gates fail. Exact curves and selected
snapshots are in
`exp_local/cqn_no_bc/stage17_effective_replay_sarsa_gpu0_20260731074940/stage17_summary.json`.
All completion sentinels are present, no Stage-17 process remains, GPU 0 is
free, and held-out seeds 800--999 remain sealed.

### 2. Interpretation

After correcting the dormant code path, one-step replay-SARSA is reproducibly
no better than greedy Double-CQN at validation-selected best checkpoints. It
can reach the same seed-1 best earlier, but the replicated mean does not
improve. This now validly rules out next-action OOD maximization as the main
parity bottleneck.

The retained dense target uses C51 cross-entropy for all five bins of every
action head. Even when every bin should have return zero, it spends gradient
and model capacity making each full 51-atom distribution a sharp point mass
at zero, although greedy action selection reads only its expectation. That
distribution-shape work is action-label invariant and not imitation, but it
may dilute the return contrast that must replace BC.

### 3. Next-stage decision

Stage 18 tests only the Q statistic matched to control:

- retain K=16, replan interval 1, greedy Double-CQN, replay nstep 1,
  discounted MC lower bounds, mixed replay, and every optimizer/exploration
  setting;
- keep the exact return target: the replayed bin receives the expectation of
  its existing Bellman/MC C51 target and every counterfactual bin receives the
  valid task floor zero;
- replace only per-bin C51 cross-entropy with one half-squared expected-Q
  Bellman/return regression over all bins.

The categorical critic architecture and greedy expected-Q action selection
remain unchanged. There is one critic objective and no auxiliary
distribution loss. A zero-return sample gives every bin the same zero target;
its loss and gradient are exactly invariant to the replay action label. Demo
identity is never read, and successful online trajectories use the same
return-gated target. No actor, policy head, pretraining, BC, margin, FOSD,
AWR, flow, self-imitation, or return-independent action supervision exists.

Train treatment seeds 1 and 2 together against the locked categorical dense-Q
controls, 48% and 46%. Selection remains 50 episodes/checkpoint on seeds
400--449 at 2.5k/5k/7.5k/10k with earliest tie breaking. The gate requires
both treatment seeds at least 60%, mean at least 64%, and mean improvement at
least 15 points over the 47% control mean. A pass trains seed 3 before the
original-CQNAS 72% validation-mean gate. A fail rejects mean-only dense
return regression at this budget. Held-out remains sealed.

### 4. Execution

Implementation adds one default-off dense expected-Q loss switch. Tests must
prove exact targets and loss, zero-return loss/gradient action-label
invariance, bit-identical demo-flag updates, one reported Q loss, rejection
of incompatible modes, legacy default identity, and a one-key resolved config
diff. Full regression, shell, summary, Python compilation, and diff checks
precede launch.

Implemented `method.dense_return_expected_q_loss`. Five focused checks pass,
including exact hand-computed targets, exact zero-return gradient invariance,
a real demo-agnostic single-loss update, invalid-mode rejection, and the
one-key config audit. The complete CQN-AS suite passes (`134 passed`);
summary dry-run, shell syntax, Python compilation, and `git diff --check`
also pass.

Stage 18 launched at 2026-07-31 08:22 local time:

- controller:
  `exp_local/cqn_no_bc/stage18_expected_q_gpu0_20260731082233`;
- treatments: `.../expected_q_seed1` and `.../expected_q_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Both treatments are live concurrently on GPU 0 with 0.45 memory fractions.
At real 1k-or-later updates, seed 1 reports
`critic_loss=dense_return_q_loss=0.030`,
`dense_return_expected_q_target=1`, and no policy/BC/auxiliary metric; seed 2
reports 0.071, 1, and the same strict absence. Total GPU allocation is
approximately 30.2/32.6 GB. This verifies execution only.

## Stage 18 result and Stage 19 reward-scaled dense-C51 protocol (2026-07-31)

### 1. Previous-stage result

The expected-Q curves and locked categorical dense-C51 controls are:

| seed | expected-Q 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 18% / 28% / 32% / 24% | 32% @ 7.5k | 48% @ 10k | -16pp |
| 2 | 24% / 34% / 44% / 42% | 44% @ 7.5k | 46% @ 10k | -2pp |

Treatment mean is 38% versus control mean 47%, a -9 point change. Both seeds
learn faster at early checkpoints, but neither reaches its matched
validation-best control and every preregistered gate fails. Exact curves and
snapshots are in
`exp_local/cqn_no_bc/stage18_expected_q_gpu0_20260731082233/stage18_summary.json`.
All completion sentinels are present, no Stage-18 process remains, GPU 0 is
free, and held-out seeds 800--999 remain sealed.

### 2. Interpretation

Mean-only Q regression removes unnecessary C51-shape work and accelerates
early policy learning, but loses the categorical target's useful
regularization and overtrains. This rules it out as the final objective.

Existing common-state target-critic probes expose a more precise gap. On
successful demonstration states:

| objective | candidate Q span | replay-bin top-1 | Q/return Pearson |
| --- | ---: | ---: | ---: |
| original CQNAS with BC margin | 0.726 | 85.5% | -0.405 |
| dense-C51 pure RL | 0.305 | 77.9% | +0.856 |

The original has stronger action separation but value ordering that is
anti-correlated with return; the pure-RL critic contains more authentic return
information but at less than half the Q dynamic range. The probe artifacts
are
`exp_local/cqn_value_fidelity_stage2/probes/full_first_success_8000.json`
and
`exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/probes/discounted_best10000.json`.
This does not justify adding a return-weighted action cross-entropy: on the
success-only expert buffer that would reduce to renamed BC and is excluded.

### 3. Next-stage decision

Stage 19 tests a policy-invariant RL target transformation. For any positive
constant `c`, replacing rewards and all Q values by `c` times their originals
preserves every action argmax and therefore the optimal policy. Set `c=2`,
the largest scale whose terminal success target still lies exactly on the
existing C51 support maximum `v_max=2`.

Keep K=16, replan interval 1, categorical dense all-bin Q, greedy
Double-CQN, nstep 1, mixed replay, and every optimizer/exploration/evaluation
setting. Multiply both immediate Bellman rewards and completed-trajectory MC
returns by two inside the same single categorical Q target; bootstrap Q is
already represented on the scaled support. Unseen floor zero remains zero.
No architecture, sampling, loss weight, actor, policy, pretraining, action
likelihood, margin, FOSD, AWR, flow, self-imitation, or demo-identity path is
added. Zero-reward samples retain exact action-label invariance.

Train scaled seeds 1 and 2 against locked unit-scale controls, 48% and 46%.
Selection remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k with earliest tie breaking. The gate requires both scaled
seeds at least 60%, mean at least 64%, and mean improvement at least 15
points over the 47% control mean. A pass trains seed 3 before the
original-CQNAS 72% validation-mean gate. A fail rejects 2x policy-invariant
reward scaling. Held-out remains sealed.

### 4. Execution

Implementation adds a default-one Q reward scale and applies it consistently
to immediate rewards and MC lower bounds. Tests must prove Bellman and MC
scaling, terminal support fit, zero-return invariance, demo-flag invariance,
one loss, invalid-mode rejection, default identity, and a one-key resolved
config diff. Full regression and launch audits precede execution.

Implemented `method.q_reward_scale`. Focused checks prove separate immediate
Bellman and MC-return scaling paths, terminal-support fit, demo-flag
bit identity, one Q loss with no actor/policy parameters, invalid-mode
rejection, and the one-key resolved-config contrast. The complete CQN-AS
suite passes (`138 passed`); shell syntax, summary dry-run, Python
compilation, resolved launch audit, and `git diff --check` also pass.

Immediately before launch, GPU 0 has only Xorg/desktop allocations
(111/32607 MiB, 0% utilization) and no training or evaluation process.
The resolved treatment has `strict_demo_rl_only=true`,
`q_reward_scale=2`, `bc_lambda=bc_margin=0`,
`separate_bc_policy=flow_policy=false`, `num_pretrain_steps=0`,
`use_self_imitation=false`, and 16+16 mixed expert/online batch samples.

Stage 19 launched at 2026-07-31 08:58 local time:

- controller:
  `exp_local/cqn_no_bc/stage19_reward_scale_gpu0_20260731085818`;
- treatments: `.../reward_scale_seed1` and
  `.../reward_scale_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Both treatments are live concurrently on GPU 0 with 0.45 memory fractions
and approximately 30.25/32.61 GB total allocation. At real 1k updates, seed
1 reports `critic_loss=dense_return_q_loss=4.302`,
`q_reward_scale=2`, and `scaled_mc_return_mean=0.918`; seed 2 reports
3.982, 2, and 0.846. Neither CSV exposes a policy, BC, or auxiliary-loss
metric. This establishes that the intended single Q objective is executing,
not policy quality.

## Stage 19 result and Stage 20 gap-increasing Bellman protocol (2026-07-31)

### 1. Previous-stage result

The 2x reward/Q-scale curves and locked unit-scale controls are:

| seed | 2x scale 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 6% / 18% / 40% / 46% | 46% @ 10k | 48% @ 10k | -2pp |
| 2 | 2% / 16% / 36% / 46% | 46% @ 10k | 46% @ 10k | 0pp |

Treatment mean is 46% versus control mean 47%, a -1 point change. Neither
seed reaches 60%; the mean, improvement, and per-seed gates all fail. Exact
curves and selected snapshots are in
`exp_local/cqn_no_bc/stage19_reward_scale_gpu0_20260731085818/stage19_summary.json`.
The training, validation, summary, and completion sentinels are present, no
Stage-19 process remains, GPU 0 is free, and held-out seeds 800--999 remain
sealed.

### 2. Interpretation

Positive reward scaling preserves the task's optimal policy and produces a
modestly steeper middle curve, but it does not increase validation-best
performance or reproduce the original CQNAS action separation. This rules
out insufficient absolute reward/Q units as the bottleneck. It leaves the
relative action gap under an approximate critic unresolved: multiplying all
values does not explicitly make the greedy action robust to estimation error.

### 3. Next-stage decision

Stage 20 tests only an optimality-preserving, gap-increasing RL operator.
Bellemare et al.'s advantage-learning operator is
`T_AL Q(s,a) = T Q(s,a) - alpha [V(s) - Q(s,a)]`, with
`V(s)=max_b Q(s,b)` and `alpha in [0,1)`. The cited theorem establishes
optimality preservation and action-gap increase in the exact setting:
<https://arxiv.org/abs/1512.04860>.

Set `alpha=0.5`. For every bin in the existing dense-C51 target, translate
its return distribution by half its stop-gradient current disadvantage.
The replayed bin retains its Bellman/MC base distribution and all
counterfactual bins retain their zero-return base distribution; the
transformation is applied uniformly, inside the same categorical Q target.
This approximately doubles fixed-point action gaps while avoiding the
support-saturating aggressiveness of alpha near one.

Keep reward scale 1, K=16, replan interval 1, categorical dense all-bin Q,
greedy Double-CQN, nstep 1, mixed replay, and every optimizer, architecture,
exploration, and evaluation setting. There is no second loss. The operator
never reads demo identity, never compares the replay action to another bin,
and uses no action-likelihood, margin, FOSD, actor, policy head, pretraining,
AWR, flow, or self-imitation. When the Bellman/MC target is the zero floor,
every bin has the same base target and the transformation remains exactly
independent of the replay action label.

Train treatment seeds 1 and 2 against locked ordinary-Bellman controls, 48%
and 46%. Selection remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k with earliest tie breaking. Record validation-best success,
curve, selected checkpoint, per-seed delta, mean delta, and learned
chosen-vs-unseen Q gap. The gate requires both treatments at least 60%, mean
at least 64%, and mean improvement at least 15 points over the 47% control
mean. A pass trains seed 3 before the original-CQNAS 72% validation-mean
gate. A fail rejects alpha=0.5 gap-increasing dense C51. Held-out remains
sealed.

### 4. Execution

Implementation will add a default-zero advantage-learning coefficient inside
the existing dense categorical target. Tests must prove the exact
distribution translation, `alpha=0` bit identity, zero-return loss/gradient
action-label invariance, demo-flag bit identity, one reported Q loss,
invalid-mode rejection, and a one-key resolved config diff. Full regression,
shell, summary, compilation, resolved-config, process, and GPU audits precede
launch.

Implemented `method.dense_return_advantage_alpha`. Five focused checks pass,
including exact C51 translation/clipping, exact `alpha=0` identity,
zero-return loss and gradient action-label invariance, a real demo-flag
invariant single-Q update, invalid-mode rejection, and the one-key
resolved-config audit. The complete CQN-AS suite passes (`143 passed`);
summary dry-run with selected Q-gap extraction, shell syntax, Python
compilation, and `git diff --check` also pass.

Immediately before launch, no training/evaluation process exists and GPU 0
has only 111/32607 MiB desktop allocation at 0% utilization. The resolved
treatment has `dense_return_advantage_alpha=0.5`,
`strict_demo_rl_only=true`, `q_reward_scale=1`,
`bc_lambda=bc_margin=0`, no policy/flow head, no pretraining or
self-imitation, and the same 16+16 mixed replay batch as control.

Stage 20 launched at 2026-07-31 09:31 local time:

- controller:
  `exp_local/cqn_no_bc/stage20_advantage_gap_gpu0_20260731093142`;
- treatments: `.../advantage_gap_seed1` and
  `.../advantage_gap_seed2`;
- locked baselines are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Both treatments are live concurrently on GPU 0 with 0.45 memory fractions
and approximately 30.25/32.61 GB total allocation. At real 1k updates, seed
1 reports `critic_loss=dense_return_q_loss=6.349`,
`dense_return_advantage_alpha=0.5`, and chosen-vs-unseen Q gap 0.200;
seed 2 reports 6.313, 0.5, and 0.226. The corresponding locked-control 1k
gaps are 0.187 and 0.162, so the intervention is already measurably active.
Neither treatment exposes a policy, BC, or auxiliary-loss metric. This is
execution evidence, not policy-quality evidence.

## Stage 20 result and Stage 21 clipped-advantage protocol (2026-07-31)

### 1. Previous-stage result

The constant advantage-learning curves and locked ordinary-Bellman controls
are:

| seed | constant AL 2.5k/5k/7.5k/10k | treatment best | locked control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 0% / 0% / 0% / 0% | 0% @ 2.5k | 48% @ 10k | -48pp |
| 2 | 0% / 0% / 0% / 4% | 4% @ 10k | 46% @ 10k | -42pp |

Treatment mean is 2% versus control mean 47%, a -45 point change. Every gate
fails. Exact curves, selected snapshots, and nearest logged Q metrics are in
`exp_local/cqn_no_bc/stage20_advantage_gap_gpu0_20260731093142/stage20_summary.json`.
Seed 1's earliest-tie selected snapshot is 2.5k, so its Q diagnostic explicitly
records the nearest earlier metric step, 2k, rather than pretending an exact
2.5k train row exists. All completion sentinels are present, no Stage-20
process remains, GPU 0 is free, and held-out remains sealed.

### 2. Interpretation

Constant advantage learning does enlarge the measured chosen-vs-unseen Q
gap: at 10k it reaches 0.357 and 0.374, versus locked-control 0.278 and
0.267. Yet policy success collapses. This establishes that gap magnitude was
not the missing sufficient condition and rules out unconditional alpha=0.5.
The paired mechanism/result specifically supports early wrong-greedy
lock-in: a genuinely good action that the approximate critic initially ranks
below another bin has its target repeatedly reduced before its demonstration
return can correct the ranking.

### 3. Next-stage decision

Stage 21 tests the direct causal repair, clipped advantage learning, while
keeping the same single Q objective. The primary definition and motivation
are from Zhang et al.:
<https://arxiv.org/abs/2203.11677>. Keep `alpha=0.5`, set clipping ratio
`c=0.9`, and use the existing C51 lower support `Q_l=v_min=-2`.
Apply the disadvantage shift only when
`(Q(s,a)-Q_l)/(V(s)-Q_l) >= c`; otherwise use the ordinary dense
Bellman/MC target unchanged.

Thus only bins already within the top 10% of the current support-relative
value range are separated. A mistakenly undervalued expert action receives
the full return-driven Q correction instead of a self-confirming negative
shift. The intervention differs from Stage 20 by exactly the clipping ratio.
It never reads demo identity or replay-action identity in its mask, and it
adds no likelihood, margin, policy, pretraining, auxiliary, flow, AWR, FOSD,
or self-imitation term. Zero-return targets remain action-label invariant.

Train clipped-AL seeds 1 and 2. Compare causally against constant-AL 0%/4%
and for the goal gate against locked ordinary controls 48%/46%. Selection
remains 50 episodes/checkpoint on seeds 400--449 at
2.5k/5k/7.5k/10k with earliest tie breaking. Record validation-best,
selected checkpoint, curves, Q gaps, per-seed and mean deltas. Pass requires
both clipped seeds at least 60%, mean at least 64%, and at least +15 points
over the ordinary-control 47% mean. A pass trains seed 3 before the original
CQNAS 72% validation-mean gate. A fail rejects `alpha=0.5,c=0.9` clipped AL.
Held-out seeds 800--999 remain sealed.

### 4. Execution

Implementation will add a default-null clipping ratio to the same
advantage-learning target. Tests must prove the exact clipped mask and shift,
null identity with Stage 20, zero-return loss/gradient action-label
invariance, demo-flag bit identity, one Q loss, invalid-mode rejection, and a
one-key Stage-20-to-21 config diff. Full regression and launch audits precede
execution.

Implemented `method.dense_return_advantage_clip_ratio`. Five focused checks
pass, including the exact near-greedy mask/shift, null constant-AL identity,
clipped zero-return loss/gradient action-label invariance, a real
demo-flag-invariant single-Q update, invalid-mode rejection, and the one-key
Stage-20-to-21 config audit. The complete CQN-AS suite passes
(`146 passed`); dual-control summary dry-run, shell syntax, Python
compilation, resolved config, and `git diff --check` also pass.

Immediately before launch, GPU 0 has only 111/32607 MiB desktop allocation
at 0% utilization and no Stage-21 process. Other users' jobs on GPUs 2 and 5
are untouched. The resolved treatment has `alpha=0.5`, `clip_ratio=0.9`,
strict no-imitation settings, and the same replay/training configuration as
Stage 20.

Stage 21 launched at 2026-07-31 10:05 local time:

- controller:
  `exp_local/cqn_no_bc/stage21_clipped_advantage_gpu0_20260731100539`;
- treatments: `.../clipped_advantage_seed1` and
  `.../clipped_advantage_seed2`;
- ordinary and constant-AL controls are materialized in four path files.

Both treatments are live concurrently on GPU 0 with 0.45 memory fractions
and approximately 30.25/32.61 GB total allocation. At real 1k updates, seed
1 reports `critic_loss=dense_return_q_loss=5.384`, alpha 0.5, clip ratio
0.9, and Q gap 0.156; seed 2 reports 5.321, 0.5, 0.9, and 0.149. These gaps
are below constant-AL's 0.200/0.226, directly confirming that clipping is
active. Neither CSV contains a policy, BC, or auxiliary-loss metric. This is
execution evidence only.

## Stage 21 result and Stage 22 demo-candidate Bellman protocol (2026-07-31)

### 1. Previous-stage result

The clipped-AL curves, constant-AL parent, and ordinary dense-Q controls are:

| seed | clipped AL 2.5k/5k/7.5k/10k | clipped best | constant AL best | ordinary best |
| --- | --- | ---: | ---: | ---: |
| 1 | 0% / 0% / 10% / 12% | 12% @ 10k | 0% | 48% |
| 2 | 0% / 4% / 0% / 10% | 10% @ 10k | 4% | 46% |

Clipped mean is 11%, rescuing 9 points over constant-AL's 2% but remaining
36 points below ordinary dense-Q's 47%. Every gate fails. Exact curves,
selected snapshots, dual controls, and selected Q diagnostics are in
`exp_local/cqn_no_bc/stage21_clipped_advantage_gpu0_20260731100539/stage21_summary.json`.
All sentinels are present, no Stage-21 process remains, GPU 0 is free, and
held-out remains sealed.

### 2. Interpretation

Clipping correctly limits final Q gaps to about 0.201 on both seeds, versus
0.357/0.374 under constant AL, and partially restores learning. It still
destroys most policy quality. This rules out the tested gap-operator family:
both unconditional and support-ratio-clipped self-referential gap expansion
interfere with learning the expert-return ordering under function
approximation. The next stage must strengthen information arriving from
actual expert returns rather than amplify the critic's current ranking.

### 3. Next-stage decision

Stage 22 tests demo-augmented Bellman maximization with the retained best
dense-C51 reward target. Keep mixed 16 ordinary + 16 expert replay, the
discounted MC lower bound, architecture, optimizer, action sequence 16,
replay `nstep=1`, one-step replanning, exploration, and evaluation unchanged.
Change only the bootstrap maximization set.

For a transition sampled at replay index `t`, replay additionally returns the
true action chunk beginning at the bootstrap state:

`u_B = [a_(t+n), ..., a_(t+n+K-1)]`.

It uses the replay buffer's existing edge/zero padding rule. The critic also
constructs its ordinary coarse-to-fine greedy chunk `u_Q`. Both complete
chunks are scored by the online critic along their own coarse-to-fine paths,
using only the deepest level and averaging expected C51 value over the
`K * action_dim` heads. The higher-scoring chunk is evaluated by the target
critic inside the unchanged C51 Bellman/MC target:

`u_* = argmax_(u in {u_Q, u_B}) S_online(s_(t+n), u)`.

This candidate rule is applied to both expert and online replay transitions;
the agent never reads the demo flag. On an expert transition, `u_B` is the
expert continuation. It is not forced to win and receives no likelihood,
margin, rank, reconstruction, or policy gradient. The only optimized tensor
remains the reward-based categorical Q target. This distinguishes Stage 22
from Stage 17: corrected Stage 17 always used an approximate shifted chunk
`[a_(t+1), ..., a_(t+15), a_(t+15)]`; it neither returned the true final
action nor compared replay and greedy candidates.

The current sequence critic already initializes both value and advantage
heads with zero weights and biases, making every initial action-bin expected
value exactly zero. Therefore Stage 22 uses `u_B` on exact score ties and then
lets the learned Q scores decide; it does not add a time schedule or an
expert-only forcing branch. Pessimistic distribution initialization, twin
critics, and randomized-head exploration remain separate later hypotheses so
this stage has one interpretation.

Train candidate-backup seeds 1 and 2 against the locked ordinary dense-Q
controls, 48% and 46%. Selection remains the development-only 50
episodes/checkpoint on seeds 400--449 at 2.5k/5k/7.5k/10k with earliest tie
breaking. Record the curves, best checkpoints, per-seed deltas, deepest-level
behavior-candidate selection fraction, candidate Q gap, demo value
calibration, and expert-bin top-1/top-2 diagnostics.

The corrected development decision is relative:

- immediate replication if mean improvement is at least 5 points and neither
  seed is worse than its matched no-BC control;
- add seed 3 if the mean improvement is positive but the two seed signs
  disagree;
- continue to 20k if the mean is within 5 points of control and a treatment
  curve is still nondecreasing into its 10k boundary;
- otherwise stop this development candidate without claiming full-budget
  failure.

Promotion to the 101k experiment requires at least three training seeds and
an independent, nonsealed 100-episode comparison against the matched no-BC
controls, with mean improvement at least 5 points and at least two of three
seed deltas nonnegative. A scale-continuation result may also promote if that
criterion is first reached at 20k. Seeds 800--999 remain sealed until the
four-seed fixed-endpoint final evaluation.

### 4. Execution

Replay gains an opt-in `action_tp1` sequence whose start index is exactly
`t+nstep`. A deterministic replay test must prove, for `nstep=1`, that current
observation/action, next observation, and next behavior action are indexed
`obs[t]`, `action[t]`, `obs[t+1]`, and `action[t+1]`, including terminal
padding. Agent tests must prove deepest-level-only scoring, exact candidate
selection, target-critic evaluation of the selected chunk, one Q loss,
demo-flag invariance, and unchanged legacy behavior when the option is off.

The Stage-22 launch config differs from the retained control only by enabling
the next-action replay element and `td_target_action_source=critic_replay_max`.
Full CQN-AS/replay regression, summary dry-run, shell syntax, compilation,
resolved-config diff, process, log, and GPU audits precede launch.

Implemented the opt-in `action_tp1` path in all three nonsequential replay
samplers: scalar, explicit-index batch, and vectorized batch. Tests establish
the exact `t+nstep` start, terminal edge padding, and byte equality between
scalar and vectorized batches. The single-objective CQN update now passes the
true next chunk to a CQN-AS hook. That hook scores greedy and replay chunks
along their own zoom paths, uses only the deepest level, selects replay on
ties, and returns only diagnostics; the target critic and C51/MC loss are
unchanged.

The full CQN-AS suite passes (`151 passed`), and the replay plus Stage-22
decision tests pass (`13 passed`). Shell syntax, Python compilation,
resolved-config audit, and `git diff --check` also pass. The resolved treatment
has 16 ordinary + 16 expert samples, `critic_lambda=1`,
`dense_return_q_target=true`, `mc_lower_bound_target=true`,
`include_next_action=true`, `td_target_action_source=critic_replay_max`, and
all BC/FOSD/margin/pretraining/self-imitation paths disabled.

Stage 22 launched on GPU 0 at 2026-07-31 10:57 local time:

- controller:
  `exp_local/cqn_no_bc/stage22_demo_candidate_gpu0_20260731105705`;
- treatments: `.../demo_candidate_seed1` and
  `.../demo_candidate_seed2`;
- locked no-BC controls are materialized in `baseline_seed1.txt` and
  `baseline_seed2.txt`.

Both treatments are concurrently resident at approximately 30.25/32.61 GiB.
The exact initialization behavior is visible in seed 2's compiled step-zero
row: behavior-candidate fraction `1.0` and score gap `0.0`. By seed 1's real
1k update, the same fraction is `0.0` and behavior-minus-greedy score is
`-0.0168`, while `critic_loss=dense_return_q_loss=3.447`, MC-target use is
`0.685`, and chosen-minus-unseen Q is `0.144`. This is execution and mechanism
evidence only; policy quality awaits the fixed validation sweep.

## Stage 22 result and Stage 23 replication/scale protocol (2026-07-31)

### 1. Previous-stage result

Stage 22 completed training, the fixed 50-episode validation sweep, and
summary generation:

| seed | demo-candidate 2.5k/5k/7.5k/10k | treatment best | matched no-BC best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 4% / 18% / 38% / 56% | 56% @ 10k | 48% @ 10k | +8pp |
| 2 | 4% / 20% / 32% / 44% | 44% @ 10k | 46% @ 10k | -2pp |

The candidate-backup mean is 50% versus the locked-control mean of 47%, a
+3 point change. Both treatment curves select their 10k boundary. At that
checkpoint, the replay continuation is selected for 9.375% of each logged
batch. Its mean score trails the critic-greedy candidate by 0.0235 on seed 1
and 0.0252 on seed 2. The initial exact tie behavior was separately observed:
the replay candidate fraction is 1.0 and score gap is 0.0 at compiled step
zero.

The exact curves, selected snapshots, candidate scores, and decision flags
are in
`exp_local/cqn_no_bc/stage22_demo_candidate_gpu0_20260731105705/stage22_summary.json`.
The controller has `training_complete`, `validation_complete`, and `complete`
sentinels; no Stage-22 train/eval process remains. GPU 0 is at 111/32607 MiB
and 0% utilization. Seeds 800--999 were not evaluated.

### 2. Interpretation

The result establishes that exact next-behavior chunks can participate in
the Bellman maximization without an imitation gradient and can improve one
matched seed. It does not yet establish a stable improvement: seed signs
disagree and the two-seed mean gain is only 3 points on a noisy 50-episode
selection split. It therefore neither establishes parity with official
CQN-AS nor rules out the method at full budget.

The mechanism diagnostics localize the remaining issue. Ties initially route
reward through the replay continuation as designed, but after learning begins
the critic-greedy chunk wins about 91% of logged comparisons and has a higher
mean predicted value. Candidate augmentation is active but only weakly
controls extrapolative greedy maximization. Because both treatment curves
are still rising at 10k, a short-budget rejection would also confound method
quality with scale. Stage 22 satisfies both preregistered continuation rules:
mixed-sign seed-3 replication and a matched 20k scale check.

### 3. Next-stage decision

Stage 23 runs two distinct experiments concurrently on one GPU:

1. **Replication arm:** train a fresh demo-candidate seed 3 to 10.5k and
   validate 2.5k/5k/7.5k/10k with 50 episodes/checkpoint, seeds 400--449.
   Compare its validation-selected best checkpoint against the locked
   ordinary no-BC seed-3 curve, 6%/30%/36%/46%.
2. **Scale arm:** exactly resume the Stage-22 seed-2 candidate run from its
   10.5k snapshot to a 20k budget. Validate only 12.5k/15k/17.5k/20k with the same
   50 episodes/checkpoint and seeds 400--449, then select over the union of
   its old and new checkpoints. Compare against the already completed
   matched ordinary no-BC seed-2 extension, whose 12.5k/15k/17.5k/20k curve
   is 46%/40%/44%/38% and whose union-selected best is 46% @ 10k.

The replication and scale questions remain separate in the summary. For the
three-seed 10k screen, pass to an independent nonsealed 100-episode
confirmation requires mean improvement of at least 5 points over matched
controls and at least two of three per-seed deltas nonnegative. For the
scale arm, a candidate union-selected best at least 5 points above the
matched extended control, occurring after 10k, is evidence that the method
benefits from scale and triggers matched candidate extensions for the other
seeds before independent confirmation. If neither criterion passes, this
exact candidate-only variant stops at development stage; that still is not
a claim of full-budget impossibility.

Metrics are success curves and selected checkpoints, per-seed/mean deltas,
candidate-use fraction and Q gap at the selected treatment checkpoints, and
chosen-vs-unseen Q gap. The objective, replay mixture, architecture,
optimizer, exploration, and all no-imitation constraints remain unchanged.
The final goal gate is still four 101k training seeds, fixed final
checkpoints, and sealed 200-episode seeds 800--999 against the official
64.6% mean.

### 4. Execution

Stage 23 will reuse the tested Stage-22 launch config for both arms. The scale
arm resumes in the existing seed-2 run directory so `train_fast.py` loads
`snapshots/latest_snapshot.pkl`; `num_train_frames=20000` is the only runtime
budget override. New validation uses a separate
`val50_ext20k_seeds400.csv` and skips all old and endpoint-only snapshots, so
the original Stage-22 evidence remains intact. A dedicated summary must
validate every expected step, preserve earliest-checkpoint tie breaking, and
emit the two preregistered decisions independently before launch.

Implemented `scripts/summarize_cqn_no_bc_stage23.py`,
`scripts/run_cqn_no_bc_stage23.sh`, and the Stage-23 protocol tests. The
summary validates every required short and extended checkpoint, selects the
earliest checkpoint on ties, keeps replication and scale gates independent,
and labels a failed development gate without making a full-budget claim. All
six Stage-22/23 decision tests pass; shell syntax, Python compilation, and
`git diff --check` pass.

The resolved fresh-seed config has `num_train_frames=10500`,
`action_sequence=16`, replay `nstep=1`, 16 ordinary plus 16 demonstration
samples, `include_next_action=true`,
`td_target_action_source=critic_replay_max`, `critic_lambda=1`,
`dense_return_q_target=true`, `mc_lower_bound_target=true`, and zero
BC/margin/separate-policy paths. The seed-2 10.5k snapshot and the locked
control's complete extended-validation CSV both exist. Immediately before
launch, GPU 0 was at 111/32607 MiB and 0% utilization.

Stage 23 launched at 2026-07-31 11:29 local time:

- controller:
  `exp_local/cqn_no_bc/stage23_candidate_repl_scale_gpu0_20260731112918`;
- replication treatment: `.../demo_candidate_seed3`;
- scale treatment: the existing Stage-22 `.../demo_candidate_seed2`;
- all three matched baselines and all treatment paths are materialized in
  the controller directory.

The controller and seed-3 process are live. The seed-3 log reports its exact
workspace and demonstration loading, and GPU 0 has allocated 15087 MiB. The
controller intentionally staggers the seed-2 resume by 120 seconds before
both jobs run concurrently at 0.45 memory fractions. This verifies the first
arm started; the second-arm process and snapshot-resume message are checked
next.

At 11:33, both Python processes are concurrently live. GPU 0 is at
30256/32607 MiB and 73% utilization. Seed 3 has logged through 3k. The
resumed seed-2 `train.csv` has grown from the original 10k row to 11k while
retaining the old rows, which is direct artifact evidence that the 10.5k
state was restored and continued rather than restarted. Both process CPU
times are advancing. Current throughput gives an approximate training ETA of
11:42, followed by six to eight minutes for the two fixed validation sweeps
and summary.

Both training arms completed normally. The flushed seed-2 log explicitly
contains `resuming: .../snapshots/latest_snapshot.pkl`; that symlink pointed
to `10500_snapshot.pkl` at launch. Its new numbered snapshots are 12.5k,
15k, 17.5k, and 20k, with
`latest_snapshot.pkl -> 20000_snapshot.pkl`. Seed 3 has its 10.5k endpoint
snapshot. The controller wrote `training_complete` and started the read-only
seed-3 validation sweep.

## Stage 23 result and Stage 24 multi-seed scale protocol (2026-07-31)

### 1. Previous-stage result

The fresh seed-3 replication curve is:

| seed | candidate 2.5k/5k/7.5k/10k | candidate best | matched no-BC best | delta |
| --- | --- | ---: | ---: | ---: |
| 3 | 0% / 32% / 32% / 50% | 50% @ 10k | 46% @ 10k | +4pp |

Combining Stage 22 and Stage 23, the three short-budget deltas are
+8/-2/+4 points for seeds 1/2/3. Candidate mean is 50.0% versus 46.67% for
the matched ordinary no-BC controls, a +3.33 point change. Two of three
deltas are nonnegative, but the preregistered mean-improvement requirement
is +5 points, so the independent-confirmation gate does not pass at 10k.

The seed-2 scale curve and its matched control are:

| arm | 12.5k | 15k | 17.5k | 20k | union-selected best |
| --- | ---: | ---: | ---: | ---: | ---: |
| demo-candidate | 56% | 38% | 46% | 44% | 56% @ 12.5k |
| ordinary no-BC | 46% | 40% | 44% | 38% | 46% @ 10k |

Thus the scale-selected treatment gain is +10 points and occurs after 10k,
which passes the independent scale gate. At the nearest logged treatment
step, 12k, replay-candidate use is 0%, replay-minus-greedy score is -0.0231,
and chosen-minus-unseen Q is 0.366. These are local batch diagnostics, not
success estimates.

The complete artifact is
`exp_local/cqn_no_bc/stage23_candidate_repl_scale_gpu0_20260731112918/stage23_summary.json`.
The controller has training, validation, and final completion sentinels; no
Stage-23 process remains. Seeds 800--999 remain sealed.

### 2. Interpretation

The 10k result rules out a stable three-seed improvement of the
preregistered magnitude at that budget. It does not rule out the method:
the matched seed-2 extension gives a concrete counterexample to using the
10k boundary as the sole compute gate. With identical objective and data,
the candidate method first reaches its selected 56% at 12.5k while the
ordinary no-BC control never exceeds 46% through 20k.

The late curve is not monotonic, so this is evidence for a
validation-selected scale benefit, not for a superior 20k endpoint: the
candidate endpoint itself is 44%. It is also only one training seed.
Therefore it establishes that the candidate backup can matter after the
short screen and justifies multi-seed extension, but it does not yet
establish reproducible improvement, official parity, or final-checkpoint
quality. The low replay-candidate fraction at 12k suggests the useful effect
may arise from earlier propagation rather than continued behavior forcing;
the single logged batch is insufficient to settle that mechanism.

### 3. Next-stage decision

Stage 24 follows the emitted decision
`extend_candidate_seeds1_3_before_confirmation`. Exactly resume candidate
seeds 1 and 3 from their 10.5k snapshots to a 20k budget, concurrently on
one GPU. Evaluate 12.5k/15k/17.5k/20k with 50 episodes/checkpoint and seeds
400--449, and select over each full 2.5k--20k union with earliest tie
breaking.

Seed 3 has a complete, locked, same-seed ordinary no-BC control through 20k:
its union-selected best is 56% @ 15k. A reproduced scale improvement
requires candidate seed 3 to exceed that by at least 5 points at a
post-10k checkpoint; a nonnegative delta is recorded as partial replication.
Seed 1's current locked control only runs to 10.5k. Its extended candidate
curve will be locked in Stage 24, but no post-10k method delta will be
claimed until Stage 25 blindly extends the seed-1 ordinary control to the
same 20k checkpoints. That control extension is mandatory regardless of the
Stage-24 seed-1 curve.

After the seed-1 control is complete, the final development gate uses all
three same-budget, validation-selected pairs. Independent nonsealed
100-episode confirmation requires mean treatment-minus-control improvement
of at least 5 points and at least two of three deltas nonnegative. Only a
confirmed result can be promoted to the four-seed 101k fixed-endpoint run.
Success curves/checkpoints, per-seed deltas where matched, candidate-use and
candidate-score gaps, and chosen-vs-unseen Q remain the metrics. The
objective and all no-imitation constraints remain unchanged.

### 4. Execution

GPU 0 became occupied during the end of Stage-23 evaluation by two unrelated
201k BC-weaning jobs, each using about 15 GiB. They are not modified. GPU 1
is currently free at 313/32607 MiB; an isolated JAX CUDA device query and
jitted reduction completed correctly on it. Stage 24 therefore targets GPU
1 rather than interfering with other work.

The Stage-24 runner must preserve each 10k Hydra config, load the existing
`latest_snapshot.pkl`, append the four numbered snapshots, write extension
validation to a separate CSV, and independently summarize seed 1 and the
matched seed-3 scale replication. Protocol tests, shell syntax, compilation,
and a resolved no-imitation config audit precede launch.

Implemented the Stage-24 runner, summary, and protocol tests. The summary
keeps the matched seed-3 scale claim separate from seed 1's pending-control
status and cannot label a 10k-selected seed-3 result as strong scale
replication. Eight Stage-22--24 decision tests pass; shell syntax, Python
compilation, and `git diff --check` pass. Both candidate runs still point
their latest snapshots to 10.5k and have no 20k snapshot before resume. GPU 1
is healthy and idle at 158/32607 MiB.

Stage 24 launched on GPU 1 at 2026-07-31 11:59 local time:

- controller:
  `exp_local/cqn_no_bc/stage24_candidate_extend20k_gpu1_20260731115908`;
- resumed treatments: the Stage-22 seed-1 candidate and Stage-23 seed-3
  candidate directories;
- locked seed-1 short and seed-3 extended baseline paths are materialized in
  the controller directory.

The controller and seed-1 Python process are live, its log reports the exact
existing treatment workspace and demonstration loading, and GPU 1 has
allocated 15138 MiB. Seed 3 is intentionally staggered by 120 seconds; the
second process, explicit resume line, and appended train row are checked
next.

Both resume processes are now concurrently live on GPU 1 at
30307/32607 MiB and 75% utilization. Their existing `train.csv` files have
advanced to 14k for seed 1 and 11k for seed 3, proving both restored 10.5k
states are updating. No Stage-24 failure sentinel is present.

## Stage 24 result and Stage 25 matched-control protocol (2026-07-31)

### 1. Previous-stage result

Both candidate extensions completed:

| seed | candidate 12.5k/15k/17.5k/20k | union-selected candidate best | matched control best | delta |
| --- | --- | ---: | ---: | ---: |
| 1 | 34% / 44% / 38% / 46% | 56% @ 10k | pending 20k extension; at least 48% @ 10k | at most +8pp |
| 3 | 44% / 38% / 40% / 42% | 50% @ 10k | 56% @ 15k | -6pp |

Seed 3 fails both strong and partial scale-replication criteria. Neither
candidate improves its own 10k best after extension. Exact curves, selected
snapshots, and candidate diagnostics are in
`exp_local/cqn_no_bc/stage24_candidate_extend20k_gpu1_20260731115908/stage24_summary.json`.
All training/validation/final sentinels are present, no Stage-24 process
remains, and GPU 1 is free at 158/32607 MiB. Held-out remains sealed.

### 2. Interpretation

The matched seed-3 result rules out a reproducible 20k scale improvement for
exact candidate maximization alone: the +10 point seed-2 benefit reverses to
-6 points on seed 3. Together with both extended candidate curves declining
from their 10k selections, this establishes that merely adding the replay
continuation to `{u_Q, u_B}` is not sufficient for stable no-BC learning.
It does not rule out demo-augmented reward backups with a better early
selection rule.

The three-seed +5 point development gate is already mathematically
unreachable. Seed 2 contributes +10 points and seed 3 contributes -6.
Seed-1 candidate best is 56%, while its control union already contains 48%
at 10k, so its eventual delta is bounded above by +8. The total is therefore
at most +12 points, or +4 points on average. An independent 100-episode
confirmation would be unjustified regardless of the new seed-1 control
checkpoints.

### 3. Next-stage decision

Stage 25 still executes the preregistered mandatory matched-control
extension, so the seed-1 treatment curve is not left compared against a
shorter-budget baseline. Resume the ordinary no-BC seed-1 control from 10.5k
to 20k with its original objective/config. Evaluate
12.5k/15k/17.5k/20k using 50 episodes/checkpoint, seeds 400--449, and select
over the full union. Then compute all three exact same-budget deltas and the
mean.

The gate remains mean improvement at least 5 points and at least two of
three nonnegative deltas. Its known upper bound is below threshold, so Stage
25 is a fairness/completeness measurement rather than another opportunity
to reselect the treatment. On failure, the exact candidate-only variant is
rejected at development budget without making a full-101k impossibility
claim. Seeds 800--999 remain untouched.

### 4. Execution

The Stage-25 runner and three-seed summary are implemented. They resume only
the locked seed-1 ordinary control, use a separate extension CSV, and read
all other treatment/control curves without mutation. The decision function
has focused tests for the mean/sign gate and explicitly names a failed
result as development-only. Shell syntax, compilation, and diff checks pass.
Before launch, the seed-1 control must still have a 10.5k latest snapshot and
no 20k snapshot; GPU 1 must remain free.

The prelaunch audit confirms `latest_snapshot.pkl -> 10500_snapshot.pkl`,
no 20k snapshot, `critic_lambda=1`, zero BC/margin, dense C51/MC target, and
ordinary critic bootstrap. Ten Stage-22--25 protocol tests pass. Stage 25
launched on GPU 1 at 2026-07-31 12:32:

- controller:
  `exp_local/cqn_no_bc/stage25_seed1_control20k_gpu1_20260731123228`;
- resumed run:
  `exp_local/cqn_no_bc/stage10_seed1_gpu0_20260731041356/discounted_dense`.

The process is live, the log reports the exact existing workspace and
demonstration loading, and GPU 1 has allocated 15138 MiB.

## Stage 25 result and Stage 26 exact-trajectory protocol (2026-07-31)

### 1. Previous-stage result

The completed ordinary seed-1 extension is 44%/60%/54%/46% at
12.5k/15k/17.5k/20k, selecting 60% @ 15k over the full union. The final
three matched pairs are:

| seed | candidate-only best | ordinary no-BC best | delta |
| --- | ---: | ---: | ---: |
| 1 | 56% @ 10k | 60% @ 15k | -4pp |
| 2 | 56% @ 12.5k | 46% @ 10k | +10pp |
| 3 | 50% @ 10k | 56% @ 15k | -6pp |

Both means are exactly 54%. Mean improvement is 0 points and only one of
three deltas is nonnegative, so the development gate fails. The exact
artifact is
`exp_local/cqn_no_bc/stage25_seed1_control20k_gpu1_20260731123228/stage25_summary.json`.
Training, validation, and final completion sentinels are present, no
Stage-25 process remains, GPU 1 is free, and held-out seeds remain sealed.

### 2. Interpretation

Across matched 20k budgets, exact replay-candidate maximization redistributes
which seed succeeds but adds no mean policy quality. This rules out
`max{u_Q,u_B}` alone as a stable replacement for BC at this development
budget. It also confirms why the seed-2 +10 point result could not be promoted
in isolation: the ordinary seed-1 and seed-3 controls themselves improve
with scale.

The mechanism logs show why the next intervention is distinct. Replay
continuations are selected in only 0--9% of representative late batches and
usually score below the greedy candidate. The replay action is therefore
available but is not reliably used long enough to propagate the expert
trajectory before critic extrapolation takes over. Stage 17 cannot answer
this exact question: it used the approximate shifted current chunk, forced
that target for demo and online samples alike, and had no true final
`action_tp1` token or later candidate-max phase.

### 3. Next-stage decision

Stage 26 tests exact demo-trajectory backup followed by candidate
maximization, with no new optimized loss:

- **Phase A, 0--10.5k:** for demonstration samples, force the Bellman
  bootstrap action to the true replay chunk beginning at `s_(t+1)`. For
  online samples, retain the ordinary candidate maximum. This is
  trajectory-SARSA reward propagation, not action imitation.
- **Phase B, 10.5k--20k:** set the force probability to zero and resume the
  same optimizer/replay state. All samples then use the Stage-22
  `{critic-greedy, replay-continuation}` candidate maximum, allowing the
  learned policy to depart from the expert.

This is a deliberately sharp `eta: 1 -> 0` schedule: it isolates the value of
an early guaranteed propagation phase before testing smoother schedules.
The only target remains the dense reward-based C51 Bellman/MC distribution,
and the only loss is its cross-entropy. There is no actor, BC, likelihood,
margin, FOSD, MSE action reconstruction, pretraining, self-imitation, or
auxiliary objective. Failed demonstrations receive the same reward-derived
target rule; demo identity only chooses the supported bootstrap action in
Phase A.

Run treatment seeds 1 and 2 directly through 20k because the previous stage
proved the mechanism is scale-sensitive. Evaluate all eight
2.5k--20k checkpoints with 50 episodes/checkpoint, seeds 400--449, and
earliest tie breaking. Compare against both locked ordinary no-BC controls
and candidate-only controls at the same union-selected budget. Record
success curves, force/candidate use, candidate score gap, and
chosen-vs-unseen Q.

A strong pass requires mean improvement of at least 5 points over ordinary
no-BC and both seed deltas nonnegative. It then trains seed 3 before any
independent confirmation. A positive mean with mixed signs also trains seed
3 solely to resolve replication.

Do not use 20k as a hard rejection boundary when the treatment is still
scaling. Independently of the relative gate, continue seeds 1 and 2 to 50k
when at least one treatment has a 20k endpoint of at least 50%, that endpoint
is no lower than 17.5k, and the two-seed validation-selected mean trails the
matched ordinary controls by no more than 5 points. At 50k, continue to the
101k fixed-endpoint experiment if the curve is still rising or the mean has
entered the 60--65% official-performance range. Otherwise this exact force
schedule stops at development budget without a full-run failure claim.
Seeds 800--999 remain sealed.

### 4. Execution

Implemented `method.demo_behavior_force_probability`. At probability one,
the target-action hook forces the exact `action_tp1` only where `demo=1`,
even when the online critic scores greedy higher; online samples retain the
candidate rule. At zero, the hook is exactly the Stage-22 candidate path.
The RNG used for the force mask is folded from, rather than replacing, the
existing greedy-action key.

Tests prove exact demo-only selection, a real finite update with
`critic_loss == dense_return_q_loss`, force fraction/probability diagnostics,
absence of policy/BC metrics and parameters, invalid-mode rejection, and a
one-key Stage-22-to-Phase-A config diff. The complete CQN-AS suite passes
(`155 passed`); the Stage-26 decision tests pass (`3 passed`). Shell syntax,
Python compilation, and `git diff --check` pass.

The runner creates fresh seed-1/2 treatments, executes Phase A concurrently,
preserves both Phase-A configs, resumes both directories for Phase B with
force probability zero, then performs the fixed eight-checkpoint sweep and
matched dual-control summary. A resolved Phase-A/Phase-B objective audit and
GPU/process check precede launch.

The resolved audit confirms Phase A is 10.5k with force probability 1.0 and
Phase B is 20k with probability 0.0. Both use 16 ordinary plus 16 demo
samples, true next-action replay, `critic_lambda=1`, dense C51/MC targets,
zero BC/margin, no policy, and no pretraining. Thirteen Stage-22--26 protocol
tests pass.

Stage 26 launched on GPU 1 at 2026-07-31 12:50:

- controller:
  `exp_local/cqn_no_bc/stage26_demo_trajectory20k_gpu1_20260731125030`;
- fresh treatments: `.../trajectory_seed1` and `.../trajectory_seed2`;
- both ordinary and candidate-only locked controls are materialized in the
  controller directory.

Seed 1 Phase A is live, its log reports the exact fresh workspace and demo
loading, and GPU 1 has allocated 15138 MiB. Seed 2 starts after the fixed
120-second stagger.

Both Phase-A seeds are now concurrently resident at 30291/32607 MiB. Seed
1's real 1k row reports force probability 1.0, forced fraction 0.96875,
behavior-candidate fraction 0.96875, and behavior-minus-greedy score
-0.0163. Thus the replay continuation is being used despite the critic
preferring greedy, exactly as intended. The same row has
`critic_loss=dense_return_q_loss=3.556`; no second loss or policy metric is
present.

Phase A completed for both seeds and the controller wrote
`phase_a_complete`. The boundary is visible in seed 1's appended metrics:
10k has force probability 1.0 and forced fraction 0.71875; after exact resume,
11k has probability 0.0, forced fraction 0.0, and behavior-candidate fraction
0.0. This proves Phase B continues the same state while removing the
trajectory override.

Both Phase-B runs completed and the controller wrote `training_complete`.
The fixed seed-1 checkpoint sweep is live. The Stage-26 summarizer now
computes the pre-registered 20k boundary and relative-quality conditions
directly from the eight-point curves and emits an explicit 50k-continuation
decision; its four focused decision tests pass. This prevents a discretionary
post-hoc extension based on a visually selected peak.

Following the scale-budget correction, all earlier two-seed no-BC curves were
also re-audited rather than treating their 10k rejection as final. The
clearest not-yet-scaled candidate is Stage 19's policy-invariant 2x reward/Q
scale: seeds 1/2 rise
`6/18/40/46%` and `2/16/36/46%`, respectively, and their selected mean is
46%, only 1 point below the matched ordinary no-BC mean. Both endpoints are
their selected maxima, unlike the mixed Stage-13 finest-neighbor
`14/18/36/56%` versus `4/22/34/28%` or Stage-15 replay-SARSA
`6/12/56/50%` versus `4/14/36/30%`. Stage 22 has already received the
matched 20k extension and therefore is not being reconsidered from its 10k
peak.

Stage 19 is consequently queued as a separate 10k-to-20k scale question:
resume both saved optimizer/replay states, evaluate 12.5/15/17.5/20k with
the same 50 episodes and seeds 400--449, and compare union-selected best
checkpoints against the already-extended ordinary no-BC seeds 1/2. It is not
mixed into Stage 26. Its mechanism pass requires a mean gain of at least 5
points and both seed deltas nonnegative. Independently, it may continue to
50k under the same scale rule as Stage 26: at least one nondecreasing 20k
endpoint at or above 50%, with selected mean no more than 5 points below
ordinary. The sole execution blocker is that GPU 1 is currently performing
Stage 26's locked sweep; the other visible GPU workloads are unrelated and
will not be interrupted.

The queued Stage-27 runner and summarizer are implemented. They resume the
two exact Stage-19 directories to 20k with one card/two jobs, validate the
four new checkpoints concurrently, and compare them with the already
extended ordinary controls. The summarizer distinguishes the mechanism gate
from the scale-continuation gate. Its four focused tests (including an
end-to-end fixed-curve summary) plus the four Stage-26 decision tests pass;
shell syntax, Python compilation, and
`git diff --check` pass. No Stage-27 process has been launched while the
Stage-26 sweep owns the research GPU.

The resolved Stage-27 job audit is `num_train_frames=20000`,
`demo_batch_size=16`, `critic_lambda=1`, `bc_lambda=bc_margin=0`,
`dense_return_q_target=true`, `q_reward_scale=2`,
`td_target_action_source=critic`, `use_self_imitation=false`, and one update
step. It therefore extends the original single reward-derived C51/MC
objective rather than introducing an imitation or policy objective.

Stage 26's seed-1 fixed sweep is complete:
`6/28/28/46/48/46/46/48%` at
2.5/5/7.5/10/12.5/15/17.5/20k. Earliest-tie selection chooses
48% @ 12.5k. This is -12 points from the matched ordinary no-BC seed-1
60% @ 15k and -8 points from candidate-only 56% @ 10k. Its 20k endpoint is
48%, below the pre-registered 50k-continuation boundary. The runner has
started the seed-2 fixed sweep; no cross-seed decision is made before that
curve completes.

## Stage 26 result and Stage 27 historical-scale execution (2026-07-31)

### 1. Previous-stage result

The fixed Stage-26 seed-2 curve is
`2/18/22/24/34/48/44/44%`, selecting 48% @ 15k. Together with seed 1,
the matched union-selected comparison is:

| seed | trajectory then candidate | ordinary no-BC | delta | candidate-only | delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| 1 | 48% @ 12.5k | 60% @ 15k | -12pp | 56% @ 10k | -8pp |
| 2 | 48% @ 15k | 46% @ 10k | +2pp | 56% @ 12.5k | -8pp |

The treatment mean is 48%, versus 53% ordinary and 56% candidate-only:
-5 and -8 points, respectively. Its 20k endpoints are 48%/44%; neither is
at least 50%. Consequently `strong_pass=false`,
`scale_continuation=false`, and both per-seed 20k-boundary flags are false.
The exact artifact is
`exp_local/cqn_no_bc/stage26_demo_trajectory20k_gpu1_20260731125030/stage26_summary.json`.
Training, validation, summary, and completion sentinels are present; the
controller and evaluators have exited and GPU 1 is free.

### 2. Interpretation

Forcing the exact next demonstrated action chunk through the first 10.5k
steps does not provide a stable policy-quality gain. Seed 2 recovers from
24% @ 10k to 48% @ 15k after the force is removed, but then falls to 44%;
seed 1 plateaus at 46--48%. This rules out the sharp
`eta=1 until 10.5k, then eta=0` schedule at the matched 20k development
budget. It does not rule out a smoother schedule or no-BC learning at the
official 101k budget, and it is not reported as a full-budget failure.
Training-loss equality established objective purity, not policy quality.

### 3. Next-stage decision

Run the already pre-registered Stage-27 historical scale check. Its
hypothesis is that Stage 19's policy-invariant 2x reward/Q representation,
whose two seeds both reach their 10k maxima at 46%, needs more Bellman
updates rather than a new objective. Resume the exact optimizer/replay states
to 20k, keep the original matched ordinary controls, select over the full
2.5k--20k curve on seeds 400--449, and keep seeds 800--999 sealed.

The mechanism gate is a two-seed mean gain of at least 5 points with both
deltas nonnegative. Independently, the 50k scale gate requires at least one
20k endpoint at or above 50% and no lower than 17.5k, while selected mean
trails ordinary by no more than 5 points.

### 4. Execution

The tested Stage-27 runner is ready to resume both treatments concurrently
on GPU 1 and then validate the four new checkpoints. Launch follows
immediately below.

Stage 27 launched at 14:01 with controller
`exp_local/cqn_no_bc/stage27_reward_scale20k_gpu1_20260731140155`.
Seed 1 resumed the exact Stage-19 directory and has written new 11k/12k
training rows beyond its saved 10.5k state. Seed 2 started after the fixed
120-second stagger. Both processes are live on GPU 1 with about 31.1 GiB
allocated, confirming the intended one-card/two-job execution.

Both Stage-27 treatments reached 20k and wrote their numbered snapshots plus
the controller `training_complete` sentinel. The runner then launched the
two fixed 12.5/15/17.5/20k validation sweeps concurrently on GPU 1. No
policy-quality decision is inferred from the training rows.

The first fixed extension point is 58%/50% for seeds 1/2 at 12.5k, versus
46%/46% at 10k. Thus the old rising curve shows a replicated +12/+4 point
scale gain at the first new checkpoint. The current treatment mean of 54%
is 1 point above the ordinary controls' union-selected mean of 53%, but the
mechanism and boundary gates remain unresolved until all four new points
finish. The unusually long rollouts are explained by the configured
3000-step episode limit: unsuccessful policies often avoid early failure and
run to timeout.

Seed 1 then reaches 66% @ 15k. Since seed 2 already has 50% @ 12.5k, the
union-selected treatment deltas are now at least +6 points versus ordinary
seed 1's 60% and +4 points versus ordinary seed 2's 46%. The pre-registered
mechanism gate (mean +5 points, both nonnegative) is therefore already
irreversibly satisfied, and a fresh seed-3 replication is required. The
separate 50k decision still waits for the 20k endpoints. These remain
50-episode development values, not sealed official-budget evidence.

Because this union-selected gate can no longer be reversed by later
checkpoints, Stage 28 was pre-registered and launched without waiting for
the separate boundary gate. It trains a fresh reward-scale seed 3 to 20k and
compares it with the existing ordinary no-BC seed-3 20k curve. Its
three-seed pass requires mean gain at least 5 points and at least two
nonnegative deltas; a pass triggers independent 100-episode confirmation.
The resolved seed-3 config again has `critic_lambda=1`,
`bc_lambda=bc_margin=0`, dense reward-scaled C51/MC targets, no
self-imitation, and no policy objective.

Stage 28 launched at 14:53 with controller
`exp_local/cqn_no_bc/stage28_reward_scale_seed3_gpu1_20260731145354`.
The fresh training process is live; GPU 1 remains within memory capacity.
After training it waits for Stage 27's completion sentinel before starting
its own fixed eight-checkpoint sweep, so no third evaluator is introduced.

## Stage 27 result and Stage 28 replication protocol (2026-07-31)

### 1. Previous-stage result

The completed reward-scale extension curves are:

| seed | 12.5k | 15k | 17.5k | 20k | union best | ordinary best | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 58% | 66% | 58% | 54% | 66% @ 15k | 60% @ 15k | +6pp |
| 2 | 50% | 50% | 46% | 46% | 50% @ 12.5k | 46% @ 10k | +4pp |

The treatment mean is 58% versus 53% ordinary, a +5 point gain with both
seed deltas nonnegative. The mechanism gate passes. Neither endpoint passes
the independent boundary gate: seed 1 has 54% < 58% at 17.5k, while seed 2
has 46% < 50%. Thus `mechanism_pass=true` and
`scale_continuation=false`. The exact artifact is
`exp_local/cqn_no_bc/stage27_reward_scale20k_gpu1_20260731140155/stage27_summary.json`;
all completion sentinels are present.

### 2. Interpretation

The 2x reward/Q representation has a real scale-dependent effect at this
development budget: the same saved runs that were only 46%/46% at 10k
become 66%/50% and beat their matched ordinary controls by +6/+4 points.
This directly confirms that the old 10k gate would have discarded a useful
no-BC mechanism. It does not yet establish three-seed replication,
independent-checkpoint reliability, or official 101k parity. The decline
after the selected peaks also means two seeds alone do not justify a 50k
claim from endpoint trend.

### 3. Next-stage decision

Use fresh reward-scale seed 3 against the already-extended ordinary seed 3,
with identical 2.5k--20k selection seeds and checkpoints. A three-seed mean
gain of at least 5 points with at least two nonnegative deltas passes
replication.

Before seeing any seed-3 policy result, the longer-budget rule is
pre-registered as follows: a replication pass triggers both independent
100-episode evaluation of all three selected checkpoints and matched
extension of reward-scale seeds 1/2/3 to 50k. This replication-based compute
gate is stronger than the failed two-seed endpoint gate and implements the
instruction to continue genuinely good curves without making a post-hoc
decision from seed 3.

### 4. Execution

Stage 28 seed 3 has completed 20k training and written
`training_complete`. It observed Stage 27's completion sentinel and has
started its fixed eight-checkpoint validation sweep on GPU 1.

To avoid a worst-case serial sweep exceeding two hours under the 3000-step
timeout, the fixed checkpoint set was sharded without changing any
evaluation sample. The original evaluator handles 2.5/5/7.5/10k; while it
runs, a second evaluator handles 12.5/15k. Once the lower shard frees its
slot, 17.5/20k use that slot. All retain 50 episodes and seeds 400--449. A
recovery controller requires exactly 4+2+2 rows before summary. The first
two seed-3 points are 0%/34% at 2.5k/5k.

## Stage 28 result and Stage 29 early-force protocol (2026-07-31)

### 1. Previous-stage result

The fresh reward-scale seed-3 curve is
`0/34/46/36/40/44/38/40%`, selecting 46% @ 7.5k. Its matched ordinary
no-BC control selects 56% @ 15k, a -10 point delta. Across all three seeds,
the reward-scale deltas are +6/+4/-10 points, so mean improvement is exactly
0 points. Two of three deltas are nonnegative, but the pre-registered +5
point mean requirement fails. `replication_pass=false` and the selected
decision is `stop_reward_scale_variant_without_full_budget_claim`.

The exact artifact is
`exp_local/cqn_no_bc/stage28_reward_scale_seed3_gpu1_20260731145354/stage28_summary.json`.
Training, all three validation shards, merged summary, and completion
sentinels are present. The sharding retained exactly eight checkpoints with
50 episodes each and seeds 400--449.

### 2. Interpretation

Policy-invariant 2x reward/Q scaling can produce a large scale-dependent
gain on seeds 1/2, but that gain does not replicate on seed 3. This rules out
reward scaling alone as a stable BC replacement at 20k and prevents both
independent confirmation and the proposed 50k extension. It does not imply
failure at an unseen 101k budget or invalidate the more general
demo-augmented Bellman hypothesis.

Stage 26 provides a separate actionable signal: exact demo-continuation
forcing improves some 5k points, but by 7.5--10k both seeds lag and seed 2
only recovers after the force is removed. The unresolved question is whether
the guaranteed reward-propagation phase was useful but simply too long.

### 3. Next-stage decision

Stage 29 tests a shorter schedule as a separate mechanism:

- fresh seeds 1/2, 0--5k: force exact `action_tp1` only for demo Bellman
  backups;
- same optimizer/replay state, 5k--20k: force probability zero and retain
  exact `{greedy, replay continuation}` candidate-max.

The only optimized objective remains the single reward-derived dense
C51/MC cross-entropy. Compare all eight checkpoints with ordinary no-BC,
candidate-only, and the completed long-force Stage 26. A stable mechanism
pass requires mean gain at least 5 points over ordinary with both deltas
nonnegative. A scale continuation remains a separate endpoint rule. Held-out
seeds 800--999 stay sealed.

### 4. Execution

Implementation and launch follow immediately.

## Stage 29: scale audit and ordered-success 20k continuation (2026-07-31)

### 1. Previous-stage result

The failed-arm audit used the actual fixed-seed validation curves and saved
checkpoints, not training loss. Candidate-only has already been matched to
20k and fails at validation-selected best: `56/56/50%` versus ordinary
no-BC `60/46/56%`, deltas `-4/+10/-6pp`, equal 54% means. Exact demo
trajectory then candidate-max also has a complete 20k result: `48/48%`
versus ordinary `60/46%`, a `-5pp` mean delta. Neither treatment is rising
at the 20k boundary.

Among the earlier failures that stopped at 10k, most peak before the budget
edge and then fall (replay-SARSA, corrected replay-SARSA, expected-Q), are
far below the viable control (primitive-Q, advantage variants), or have only
one rising seed (finest-neighbor and sequence-return). Ordered-success is the
only unextended two-seed treatment whose selected checkpoint is the 10k
boundary for both seeds: seed 1 `6/36/36/46%`, seed 2 `8/8/32/36%` at
2.5/5/7.5/10k. Reward-scale had the same boundary pattern and its ongoing
20k extension has already improved to provisional best `66/50%`, validating
the scale-audit premise.

### 2. Interpretation

The old 10k rejection is insufficient specifically for ordered-success:
both curves are still learning at the cutoff, so slower learning after
removing BC remains a live explanation. This does not reopen variants whose
curves already roll over, nor does it justify a 101k claim from development
data. Loss finiteness is only a wiring check.

### 3. Next-stage decision

Resume the exact ordered-success seed-1/2 optimizer and replay states from
10.5k to 20k as one isolated mechanism. Compare against the already extended
ordinary no-BC controls, select the earliest best checkpoint over
2.5k--20k on seeds 400--449 (50 episodes/checkpoint), and keep seeds
800--999 sealed. The mechanism gate is mean delta at least `+5pp` with both
seed deltas nonnegative. The separate scale gate requires at least one 20k
endpoint at least 50% and no lower than 17.5k, while selected mean trails
ordinary by no more than 5pp. A mechanism pass triggers fresh seed 3 plus an
independent 100-episode confirmation; scale-only pass permits a pre-registered
50k continuation without a quality claim.

### 4. Execution

The Stage-29 runner and dedicated summarizer are implemented as
`scripts/run_cqn_no_bc_stage29.sh` and
`scripts/summarize_cqn_no_bc_stage29.py`; shell syntax, Python compilation,
resolved no-BC config (`bc_lambda=bc_margin=0`, ordered-success mix 0.5), and
`git diff --check` pass. Numeric CUDA ordinal probing currently falls back to
CPU, so the runner pins physical GPU 3 by UUID; an isolated UUID JAX probe
confirmed a real CUDA device before launch.

Controller `1714528` launched at 15:30 in
`exp_local/cqn_no_bc/stage29_ordered_success20k_gpu3_20260731153042`.
Seed 1 restored successfully and wrote a real 11k row with finite critic loss
1.323, then reached 12k; seed 2 started after the required 120-second stagger.
Both processes are resident on physical GPU 3 at about 15.1 GiB each, matching
the measured two-run allocation. Training is expected to finish in roughly
10--15 minutes; fixed 12.5/15/17.5/20k evaluation then runs sequentially so
no evaluator shares the card with training or another evaluator.

Stage-27 completion update: the final reward-scale 20k endpoints are 54%/46%.
Union-selected best remains 66% @ 15k and 50% @ 12.5k versus ordinary
60%/46%, hence deltas `+6/+4pp`, mean `+5pp`, and `mechanism_pass=true`.
Both curves decline after their selected point, so both 20k boundary flags
and `scale_continuation` are false: this supports extending the original 10k
screen, but rules out an automatic 50k continuation under the registered
scale criterion. Stage 28 training is complete and its fresh seed-3 fixed
validation sweep has started; that result decides replication.

## Stage 29 result and offline-phase correction (2026-07-31)

### 1. Previous-stage result

The ordered-success continuation completed before the offline-phase audit.
Its fixed 50-episode development curves on seeds 400--449 are:

| seed | 2.5k | 5k | 7.5k | 10k | 12.5k | 15k | 17.5k | 20k | selected best | ordinary best | delta |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 6% | 36% | 36% | 46% | 40% | 44% | 44% | 48% | 48% @ 20k | 60% @ 15k | -12pp |
| 2 | 8% | 8% | 32% | 36% | 32% | 42% | 38% | 34% | 42% @ 15k | 46% @ 10k | -4pp |

The selected mean is 45% versus 53% for the matched ordinary no-BC
controls, a `-8pp` delta. Neither 20k endpoint satisfies the registered
`>=50%` nondecreasing boundary rule. Thus both the mechanism and scale gates
fail. The recovered summary is
`exp_local/cqn_no_bc/stage29_ordered_success20k_gpu3_20260731153042/stage29_summary.json`;
training, validation, summary, and completion artifacts are present.

### 2. Interpretation

This rules out ordered-return mixing as a stable improvement at the matched
20k development budget. It is not evidence against the user's proposed
offline-then-online demo-driven RL algorithm. The resolved original CQN-AS
config and every modern no-BC stage through Stage 29 have
`num_pretrain_steps=0`: a random critic begins environment interaction and
demonstrations are only mixed into online updates.

The original BC/FOSD/margin terms supply direct expert-action ranking as soon
as online updates begin. Removing those gradients while retaining the same
phase schedule was therefore not a neutral ablation: the no-BC policy acts
before demonstration returns have propagated through the critic, and its
early online replay is generated by that uncalibrated policy. Stage 22
candidate-max and Stage 26 continuation forcing changed the Bellman target
during online learning, but neither constituted an offline RL phase.

The strict audit guard also encoded this mistaken assumption by rejecting
every nonzero `num_pretrain_steps`. That guard conflated forbidden actor/BC
pretraining with allowed reward-only offline critic learning. It has now been
corrected: it still rejects all action-imitation paths, while permitting
offline C51/MC Q updates when the agent is critic-only.

### 3. Next-stage decision

Stage 30 asks one isolated question: does reward-only Q-learning on
demonstrations *before the first environment action* improve an otherwise
identical no-BC online Q learner?

Use fresh training seeds 1/2 and two matched arms:

- `online_only`: zero offline updates, then 20k environment interactions;
- `offline_then_online`: 10k gradient updates from 100% protected successful
  demonstration replay, then the same 20k environment interactions.

Both arms use the same canonical replayed-action C51 cross-entropy,
`critic_lambda=1`, discounted MC return as a reward lower-bound target,
`nstep=1`, and exact next-action candidate support. Dense unseen-bin targets,
BC/FOSD/margin, policy/flow heads, self-imitation, AWR, and any actor are all
disabled. During offline learning only, demonstrated transitions force
`action_tp1` as the Bellman continuation. Online learning uses the maximum
over the critic-greedy and exact replay-continuation candidates, with batches
of 16 main-replay plus 16 protected-demo samples.

Evaluate the offline endpoint separately, then select both arms over the same
eight online checkpoints from 2.5k through 20k, using 50 episodes and seeds
400--449 with earliest-checkpoint tie breaking. Seeds 800--999 remain sealed.
Primary policy metrics are the full success curves and validation-selected
best checkpoints. Mechanism diagnostics are offline force/candidate fraction,
MC-lower-bound fraction, candidate Q gap, and the absence of imitation-loss
metrics. Finite training loss is only a wiring check.

The two-seed mechanism gate requires mean selected improvement of at least
`+5pp` and both per-seed deltas nonnegative. A separate scale gate allows a
50k continuation when selected mean trails by at most `5pp` and at least one
20k endpoint is `>=50%` and nondecreasing from 17.5k. The treatment
intentionally has 10k additional offline Q updates; this first stage tests
the complete offline phase, not compute-matched ordering. A mechanism pass
therefore requires fresh seed 3 followed by a separate update-count-matched
control before any method claim, plus independent 100-episode confirmation.

RoboBase currently includes pretrain updates in its displayed global step.
The treatment raw snapshots are therefore 10k for the offline endpoint and
12.5/15/17.5/20/22.5/25/27.5/30k for 2.5--20k online steps. The summarizer
performs this fixed offset mapping; the environment interaction budget is
still exactly 20k.

### 4. Execution

The new launch, controller, summarizer, strict-guard regression, and phase
contract tests are implemented as:

- `robobase/cfgs/launch/cqn_as_pixel_bigym_nobc_stage30_offline_then_online_gate.yaml`;
- `scripts/run_cqn_no_bc_stage30.sh`;
- `scripts/summarize_cqn_no_bc_stage30.py`.

Focused strict/config/summarizer tests pass (`15 passed`), shell syntax and
Python compilation pass, all three resolved phase contracts match the
registered values, and `git diff --check` is clean for the touched files.
The broader Stage-30 regression set subsequently completed with `174 passed`
(`tests/unit/test_cqn_as.py`, replay next-action/equivalence tests, snapshot
sweep, Stage-30 summarizer, and value-fidelity diagnostics; one warning,
334.27 seconds). This is implementation evidence only, not policy-quality
evidence.

An end-to-end smoke at
`exp_local/cqn_no_bc/stage30_offline_smoke_gpu3_20260731160657` performed two
offline updates before any environment interaction. Its `pretrain.csv`
records `env_episodes=0`, no `train.csv`,
`behavior_candidate_fraction=1.0`,
`demo_behavior_force_fraction=1.0`,
`demo_behavior_force_probability=1.0`,
`mc_lower_bound_fraction=0.96875`, and finite critic loss 3.93, with no
imitation metric. Resuming the same snapshot with the online config did not
add another pretrain row, switched to 16+16 mixed replay and force probability
zero, executed two real environment steps, and saved the next checkpoint.

Physical GPU 3 is currently empty with 32.1 GiB free. The controller runs
the matched control and treatment concurrently per seed on that one card,
with a 120-second compilation stagger, then evaluates the two arms
concurrently. Launch follows immediately; expected training plus fixed
validation wall time is approximately 75--100 minutes.

Stage 30 launched at 16:12 with controller PID 1785027 in
`exp_local/cqn_no_bc/stage30_offline_then_online_gpu3_20260731161256`.
The seed-1 online-only control and offline treatment are both live on physical
GPU 3 after the registered 120-second stagger. The control has reached 3k
real environment steps with finite critic loss. The treatment has written
offline rows through 300 updates with `env_episodes=0`,
`behavior_candidate_fraction=1.0`,
`demo_behavior_force_fraction=1.0`, and finite critic loss. The two Python
processes use about 2.8 GiB each under dynamic allocation, confirming the
intended one-card/two-run execution with substantial memory headroom.

The seed-1 offline phase then completed all 10k updates and wrote
`offline_seed1_complete`, `10000_snapshot.pkl`, and its archived phase-A
config. The controller resumed the same run with
`num_pretrain_steps=10000`, global limit 30k (10k offline plus 20k real
online steps), 16+16 replay, and force probability zero. Its first phase-B
row is at raw step 10k/online step zero with finite critic loss 1.765 and
zero force fraction; `pretrain.csv` remains fixed at 101 lines. This verifies
that phase B neither reran offline updates nor reset the optimizer/critic.

The read-only value-fidelity probe was also extended for the selected
checkpoints to report expert-bin top-2 rates, RTG calibration MAE in fixed
distance-to-terminal buckets, greedy-chunk value, and
`expert_Q - greedy_Q`, in addition to the existing top-1/rank/correlation
metrics. A real historical CQN-AS checkpoint smoke completed on CPU with
`status=ok` in 43.9 seconds at
`exp_local/cqn_no_bc/stage30_diagnostic_smoke/real_checkpoint.json`. These
mechanism diagnostics remain separate from the environment success gate.

Seed-1 matched training subsequently completed. The control has its fixed
20k final snapshot, the treatment has raw 30k (10k offline + 20k online),
`training_seed1_complete` is present, and control/offline/online resolved
configs are archived. The controller then launched fresh seed 2 without
inspecting or selecting any seed-1 policy checkpoint.

The pre-registered read-only mechanism probe was run on both 10k-offline
endpoints using the same eight held-in successful-demo transitions per seed.
Reward-value propagation is reproducible: success-demo RTG MAE is
0.0392/0.0320 and Pearson correlation is 0.779/0.787 for seeds 1/2. Action
extraction remains imperfect: current-action expert-bin top-1 is 31.9%/38.6%
(random reference 20%), top-2 is 55.3%/60.0% (random reference 40%), and mean
`Q(expert)-Q(greedy)` is -0.0233/-0.0245. Thus the offline phase has learned
the demonstrated returns, but unseen/greedy action overestimation can still
defeat the expert candidate. These diagnostics do not select a checkpoint or
establish policy quality; the fixed environment sweep remains decisive. The
full artifacts are `offline_endpoint_seed{1,2}_value_fidelity.json` in the
Stage-30 directory.

Seed-1 validation then completed all fixed checkpoints. The online-only
control is 0/50 at every 2.5k--20k online checkpoint. The treatment offline
endpoint is also 0/50, and every treatment checkpoint from 2.5k through 20k
online is 0/50. This is rollout evidence that calibrated demonstration
returns did not become a usable factored greedy policy in seed 1; it does not
yet distinguish action-ranking extrapolation from rollout covariate shift.
The controller immediately started the identical seed-2 sweep without using
the seed-1 result for selection.

## Stage 30 complete: offline first is necessary but not sufficient

### 1. Previous-stage result

Stage 30 completed with all training, validation, phase-contract, summary,
and completion artifacts present at
`exp_local/cqn_no_bc/stage30_offline_then_online_gpu3_20260731161256`.
For both seeds, the online-only curve at
`2.5/5/7.5/10/12.5/15/17.5/20k` is
`0/0/0/0/0/0/0/0%`. The reward-only offline treatment is 0% at its 10k
offline endpoint, and its same eight online checkpoints are also
`0/0/0/0/0/0/0/0%` for both seeds. Earliest-checkpoint tie breaking selects
2.5k online for every arm: control mean 0%, treatment mean 0%, delta 0pp.
Both the registered mechanism gate and the rising-20k scale gate are false.

This policy result coexists with successful value fitting. On held-in
successful-demo transitions, the two offline endpoints have RTG MAE
0.0392/0.0320 and Pearson 0.779/0.787. Current-action expert top-1 is only
31.9%/38.6%, top-2 is 55.3%/60.0%, and mean
`Q(expert)-Q(greedy)` is -0.0233/-0.0245. The archived final offline metric
rows likewise report behavior-minus-greedy gaps of -0.0494/-0.0340 while the
demonstrated continuation is forced on every offline sample.

### 2. Interpretation

This establishes that the corrected offline-before-online phase really does
propagate reward return into the critic, but calibrated value on replayed
expert actions is not sufficient to extract a usable high-dimensional greedy
policy. With the dueling head, its shared state-value stream can lift all
bins while only weakly separating the replayed bin; unseen/greedy chunks then
remain above the demonstrated chunk. Zero offline rollout and zero online
curves are policy evidence, not an inference from training loss.

The result rules out merely prepending 10k reward-only offline updates to
this canonical replay-action, dueling-C51 candidate-backup recipe at 20k
online budget. It does not rule out offline-first demo-driven RL, a critic
whose unseen actions remain pessimistic, or no-BC learning at full scale.
The user's permission to extend promising curves does not apply to this arm:
neither seed has a nonzero point or an improving 17.5k-to-20k boundary, so
the pre-registered 50k scale criterion is not met.

### 3. Next-stage decision

Stage 31 isolates the measured action-ranking failure. Replace only the
dueling `V + A - mean(A)` categorical head with a direct per-bin C51 head
(`method.use_dueling=false`). The direct head leaves unobserved bins at their
failure-return initialization instead of raising them through a shared
positive value stream. It adds no loss, actor, policy likelihood, behavior
constraint, or action label objective.

Use fresh direct-head runs with training seeds 1/2 and compare them against
the immutable Stage-30 offline-then-online arms with the same seeds. All
other phase settings remain exact: 10k protected-success-demo offline
updates, forced demonstrated Bellman continuation only offline, then 20k
online interactions with 16+16 replay and candidate-max. Select over the same
eight online checkpoints with 50 episodes and seeds 400--449; report the
offline endpoint separately and keep seeds 800--999 sealed. Primary metrics
are the full success curves and validation-selected best checkpoints;
mechanism metrics are RTG MAE, expert top-1/top-2, and
`Q(expert)-Q(greedy)`.

The direct-head mechanism gate requires mean selected gain of at least 10pp,
both per-seed deltas nonnegative, and at least one seed at 20% or higher. A
separate scale gate permits 50k only when mean gain is at least 5pp and a 20k
endpoint is at least 20% and nondecreasing from 17.5k. A mechanism pass
requires seed 3 and an update-matched confirmation before any method claim.

### 4. Execution

The isolated launch, two-seed one-card controller, summary contract, and
tests are implemented in
`cqn_as_pixel_bigym_nobc_stage31_offline_direct_head_gate.yaml`,
`run_cqn_no_bc_stage31.sh`,
`summarize_cqn_no_bc_stage31.py`, and
`test_summarize_cqn_no_bc_stage31.py`. Config/decision tests pass (3 tests),
the direct head performs a finite reward-only update with no BC metric (2
tests), shell syntax and Python compilation pass, and `git diff --check` is
clean. The full Stage-31 controller is launched immediately after this
Stage-30 closure.

Stage 31 launched at 17:28 under controller PID 1914482 in
`exp_local/cqn_no_bc/stage31_offline_direct_head_gpu3_20260731172806`.
Both seed pipelines are live on physical GPU 3 after the registered
120-second stagger. At the first two-run verification, seed 1 had reached
9.4k/10k offline updates and seed 2 1.3k/10k. Both record zero environment
episodes, behavior/force fractions 1.0, force probability 1.0, finite C51
loss, and no imitation metric. Resolved seed-1 config confirms
`use_dueling=false`, `demo_only_updates=true`, `strict_demo_rl_only=true`,
BC/margin zero, FOSD false, and the reward-only MC/candidate target. The two
processes use 5.6 GiB total on GPU 3, verifying the intended one-card/two-run
execution. The controller will resume each exact offline checkpoint into its
20k online phase, run the fixed validation sweep, and write the Stage-31
summary automatically.

Both Stage-31 offline endpoints and both 20k online training phases then
completed. Fixed offline diagnostics show that direct heads improve the
success-demo mean `Q(expert)-Q(greedy)` from the Stage-30 dueling values
-0.0233/-0.0245 to -0.0130/-0.0114 and improve Pearson RTG correlation to
0.894/0.867. They do not improve exact action extraction consistently:
current-action top-1 is 32.2%/30.0% and top-2 is 49.4%/49.2%. Thus direct
heads reduce shared-value leakage but leave an unsupported-action ranking
gap. The controller has started the registered 50-episode validation sweep
for both seeds; no policy conclusion is taken from these diagnostics.

## Stage 31 complete: direct C51 does not solve policy extraction

### 1. Previous-stage result

Stage 31 completed with the phase contracts, all nine fixed validation
points per seed, summary, and completion marker present at
`exp_local/cqn_no_bc/stage31_offline_direct_head_gpu3_20260731172806`.
The 10k-offline endpoint is 0/50 for both training seeds. After resuming the
same critic and optimizer online, each seed's success curve at
`2.5/5/7.5/10/12.5/15/17.5/20k` is
`0/0/0/0/0/0/0/0%`. Earliest-checkpoint tie breaking therefore selects
2.5k online at 0% for both the Stage-30 dueling baseline and the direct-head
treatment: selected means are 0% versus 0%, for a 0pp gain. The mechanism,
seed-3, and 50k scale gates are all false.

The fixed offline diagnostic remains mechanism evidence only. Relative to
the Stage-30 dueling endpoints, direct heads improve success-demo RTG
Pearson correlation from 0.779/0.787 to 0.894/0.867 and reduce the diagnostic
mean `Q(expert)-Q(greedy)` deficit from -0.0233/-0.0245 to
-0.0130/-0.0114. Exact current-action top-1 remains only 32.2%/30.0% and
top-2 49.4%/49.2%. In the training batches, mean
`behavior-minus-greedy-Q` worsens rather than improves from the first 2k to
the last 2k offline updates: -0.0228 to -0.0423 for seed 1 and -0.0228 to
-0.0370 for seed 2, even while C51 loss falls.

### 2. Interpretation

This rules out the isolated claim that removing the shared dueling value
stream is enough to turn reward-only offline value fitting into a working
greedy MovePlate policy. It also gives no curve-based justification for
merely extending this arm to 50k: both 17.5k and 20k endpoints are zero, and
the offline action gap is not improving with more updates. Falling training
loss is explicitly not policy-quality evidence.

The result does not rule out the corrected offline-before-online research
protocol. Both Stage 30 and Stage 31 show that the critic can encode expert
returns while unsupported compound actions still outrank the demonstrated
action. What remains unresolved is whether pessimistic action selection can
convert the already learned reward information into a supported greedy
policy without adding any behavior or imitation constraint.

### 3. Next-stage decision

Stage 32 tests exactly that unresolved mechanism. Keep the direct per-bin
C51 architecture and every Stage-31 offline/online setting fixed, but use
two independently initialized critics. Coarse-to-fine action and candidate
selection use the minimum of their expected returns; the Bellman target uses
the full categorical distribution from the lower-valued target critic. Both
critics minimize the same reward TD/MC C51 cross-entropy and their two losses
are averaged. There is still no actor, BC, margin, likelihood, conservative
penalty, or extra action objective.

The matched baseline is the immutable Stage-31 direct-head arm, with fresh
Stage-32 training seeds 1/2. Each run receives 10k protected-demo offline
updates followed by 20k online interactions with the same 16+16 replay and
candidate-max setup. The offline endpoint is reported separately. Selection
uses the same eight online checkpoints, 50 episodes per checkpoint, and
seeds 400--449; seeds 800--999 remain sealed. Primary metrics are full
success curves and validation-selected best checkpoints. Mechanism metrics
add critic disagreement and lower-target usage to RTG calibration, action
rank, and candidate usage.

The pre-registered mechanism pass remains mean selected gain >=10pp, both
per-seed deltas nonnegative, and at least one seed >=20%. A 50k extension is
allowed only for mean gain >=5pp plus at least one 20k endpoint >=20% and
nondecreasing from 17.5k. A mechanism pass requires seed 3 and an
update-matched confirmation before any quality claim; otherwise this twin
mechanism is rejected and the next question becomes supported exploration.

### 4. Execution

The isolated implementation and protocol are in
`cqn_as_pixel_bigym_nobc_stage32_offline_pessimistic_twin_gate.yaml`,
`run_cqn_no_bc_stage32.sh`, and
`summarize_cqn_no_bc_stage32.py`. Unit coverage checks independent critic
parameters, pessimistic categorical selection, reward-only updates under
JIT and non-JIT, action inference, exact Stage-31/32 config isolation, phase
contracts, summary decisions, and full parameter/target/optimizer checkpoint
round trips. The complete `test_cqn_as.py` suite passes 163 tests, the final
checkpoint-enhanced twin test passes both parameterizations, the related
summary/diagnostic tests pass, and shell syntax, Python compilation, and
`git diff --check` are clean.

A real pixel offline-to-online smoke completed at
`exp_local/cqn_no_bc/stage32_twin_resume_smoke_gpu3_20260731175915`.
Before environment interaction its offline row records `env_episodes=0`,
finite critic-1/critic-2/C51 loss 3.9318, demo force and behavior candidate
fractions 1.0, and no imitation metric. Resuming snapshot 2 with the online
config leaves `pretrain.csv` unchanged, executes two environment steps,
writes snapshot 4, switches force probability/fraction to zero, and records
finite separate critic losses 3.91309/3.91323 with nonzero critic
disagreement 0.000189. This verifies the twin parameter and optimizer state
survive the actual workspace resume boundary. The full matched Stage-32
controller is launched immediately after this smoke.

Stage 32 launched at 18:03 under durable controller PID 1984728 in
`exp_local/cqn_no_bc/stage32_offline_pessimistic_twin_gpu3_20260731180304`.
After the registered 120-second stagger, both seed pipelines are performing
real offline updates on physical GPU 3: seed 1 has passed 3.8k/10k and seed
2 has written its first post-compilation update. Their rows record zero
environment episodes, behavior and demo-force fractions 1.0, finite and
separately valued critic-1/critic-2 losses, nonzero twin disagreement, and no
imitation metric. The two processes occupy about 2.8 GiB each; together with
an unrelated 5.2-GiB process already on the card, 21.3 GiB remains free. The
controller will resume each endpoint online, evaluate all fixed checkpoints,
and write the Stage-32 summary without manual checkpoint inspection. Based
on current update throughput and the registered sweep cost, the next complete
policy result is expected in roughly 45--60 minutes.

### Pre-registered seed-3 matched confirmation on the second card

Before any Stage-32 policy checkpoint has been evaluated, the user made
physical GPU 5 available for two concurrent runs. It is used only to
accelerate the same Stage-32 research question: training seed 3 runs a fresh
Stage-31 direct-head baseline and a fresh Stage-32 pessimistic-twin treatment,
both with the identical 10k reward-only offline plus 20k online protocol. No
new loss, exploration mechanism, or research question is mixed into this
pair.

The seed-1/2 primary gate remains immutable. The seed-3 controller is required
to wait for `stage32_summary.json`, copy that decision as a frozen artifact,
and only then launch its two fixed validation sweeps. Consequently seed 3
cannot retroactively rescue or modify a failed primary gate. If the primary
mechanism gate passes, confirmation additionally requires the seed-3 twin
validation-best success to be at least 20% and noninferior to its matched
direct-head baseline. Only that conjunction can recommend the full 101k
protocol. If the primary gate fails, seed 3 is reported as supplemental
robustness evidence only.

This contract is implemented in
`run_cqn_no_bc_stage32_seed3_confirmation.sh` and
`summarize_cqn_no_bc_stage32_seed3.py`; the three decision tests pass, shell
syntax and Python compilation pass, and `git diff --check` is clean. The
durable controller launched on GPU 5 at 18:14 in
`exp_local/cqn_no_bc/stage32_seed3_matched_gpu5_20260731181459`. Its direct
baseline has written real offline updates with `env_episodes=0`, force and
behavior fractions 1.0, finite reward-only C51 loss, and no imitation metric.
The matched twin arm starts after the same 120-second compilation stagger
used by the primary experiment.

Both GPU-5 arms are now verified beyond process existence. The direct arm has
passed 8.5k/10k offline updates; the twin arm has written its first compiled
reward-only update. Both show `env_episodes=0`, behavior/demo-force fractions
1.0, finite C51 loss, and no BC or policy-loss column. With both resident,
GPU 5 uses about 7.1 GiB and retains about 25.0 GiB free.

The fixed read-only value probe has completed on both primary 10k-offline
twin endpoints using exactly the same eight held-in successful-demo
transitions as Stage 31. Twin seed 1/2 RTG MAE is 0.00583/0.00806, Pearson is
0.988/0.986, and current-action top-1 is 33.6%/34.2% (top-2 50.0%/50.8%).
For comparison, the matched Stage-31 direct endpoints had MAE 0.0311/0.0369,
Pearson 0.894/0.867, top-1 32.2%/30.0%, and top-2 49.4%/49.2%.
`Q(expert)-Q(greedy)` changes from -0.0130/-0.0114 to
-0.00593/-0.01280. Thus twin pessimism reproducibly improves return
calibration and modestly improves bin rank, but it does not yet establish a
usable policy; fixed environment success remains the decisive gate. The
artifacts are `offline_endpoint_seed{1,2}_value_fidelity.json` in the primary
Stage-32 directory and consume no held-out environment seed.

The GPU-5 twin arm subsequently completed all 10k offline updates, archived
its phase-A config and 10k snapshot, and resumed the exact checkpoint online.
Its `pretrain.csv` remains fixed at 101 lines while the first online row has
force probability/fraction zero, finite separate twin losses, and nonzero
disagreement. Thus both seed-3 matched arms have crossed the real
offline-to-online workspace boundary without resetting or adding an
imitation objective.

Both primary seeds have now completed the full 10k-offline plus 20k-online
budget with final raw-30k snapshots and phase completion markers. The
controller immediately launched the two pre-registered fixed validation
sweeps (processes 2028922/2028923): offline endpoint plus eight online
checkpoints, 50 episodes each, seeds 400--449. No checkpoint result was
inspected before the full training phase closed. On GPU 5 the matched direct
arm is complete and sealed, while its twin arm has reached 17k/20k online.

The seed-3 matched fixed-state probe completed after both offline endpoints
were already frozen. Direct versus twin RTG MAE is 0.0285 versus 0.00572,
Pearson is 0.915 versus 0.992, and `Q(expert)-Q(greedy)` is -0.0146 versus
-0.00684. Current-action top-1 moves from 28.9% to 31.9% and top-2 from
45.6% to 54.2%. Together with primary seeds 1/2, this reproduces the twin
calibration improvement across three training seeds, while also confirming
that exact expert-bin extraction remains modest. This still does not select
a checkpoint or establish task success.

## Stage 32 complete: calibrated twin values still yield zero task success

### 1. Previous-stage result

The primary Stage-32 experiment is complete at
`exp_local/cqn_no_bc/stage32_offline_pessimistic_twin_gpu3_20260731180304`.
Both training seeds completed 10k demo-only reward-Q updates followed by 20k
online interactions, and all phase configs, snapshots, fixed validation CSVs,
the summary, and completion markers are present. For both seeds, the 10k
offline endpoint and every online checkpoint at
`2.5/5/7.5/10/12.5/15/17.5/20k` are 0/50 on validation seeds 400--449.
Earliest-checkpoint tie breaking selects 2.5k online at 0% for both the
immutable Stage-31 direct-head baseline and Stage-32 twin treatment. Selected
means are therefore 0% versus 0%, with per-seed and mean gains of 0pp. The
mechanism gate and the separate 50k scale-continuation gate are both false.

The fixed-state mechanism result is different from the policy result. Across
primary seeds 1/2, twin critics reduce offline success-demo RTG MAE from
0.0311/0.0369 to 0.00583/0.00806 and raise Pearson correlation from
0.894/0.867 to 0.988/0.986. Current-action top-1 changes from 32.2%/30.0% to
33.6%/34.2%, top-2 from 49.4%/49.2% to 50.0%/50.8%, and
`Q(expert)-Q(greedy)` from -0.0130/-0.0114 to -0.00593/-0.01280. The
pre-registered seed-3 matched pair on GPU 5 independently reproduces the
calibration improvement (MAE 0.0285 to 0.00572; Pearson 0.915 to 0.992), but
its fixed validation sweep is supplemental because the primary decision was
frozen first. At this report it has completed the offline endpoint and the
first five online checkpoints through 12.5k; all six observed points are 0/50
for both direct and twin, and the remaining checkpoints are still running.

### 2. Interpretation

Stage 32 establishes that clipped twin pessimism materially improves the
critic's reward-return calibration across three training seeds. It rules out
the narrower hypothesis that this calibration improvement, by itself, is
enough to extract a successful greedy MovePlate policy within 20k online
steps. Falling C51 loss and low RTG error are mechanism evidence, not policy
quality; the complete primary success curves are the rejection evidence.

This does not rule out the user's corrected offline-before-online protocol or
no-BC learning at full budget. It identifies the unresolved boundary more
precisely: reward information is present, but pessimistic evaluation from a
single deterministic greedy trajectory does not collect useful recovery or
task-completing online data. Since both 17.5k and 20k endpoints are zero, this
specific deterministic behavior policy has no pre-registered evidence for a
50k extension.

### 3. Next-stage decision

Stage 33 isolates supported exploration. It keeps the Stage-32 twin direct-C51
architecture, offline phase, replay mixture, Bellman/MC targets, optimizer,
training seeds, and pessimistic min-Q evaluation unchanged. The only method
difference is online data collection: at each training-environment episode
reset, sample critic head 0 or 1 and greedily follow that same head for the
entire episode. This is episode-persistent randomized value-function
exploration. It adds no actor, reward, action label, BC/margin/FOSD term,
likelihood, conservative penalty, or other auxiliary objective.

The immutable matched baseline is Stage-32 seed 1/2. Fresh Stage-33 seed 1/2
receive 10k protected-success-demo offline updates and 20k online interactions.
The offline endpoint is reported separately; validation selection uses the
same eight online checkpoints, 50 episodes per checkpoint, seeds 400--449,
and earliest tie breaking. Seeds 800--999 remain sealed. Primary metrics are
the complete success curves and validation-selected best checkpoints.
Mechanism metrics additionally require the number of episodic head
assignments and head-0/head-1 usage rates to prove that randomized exploration
actually occurred.

The mechanism gate remains mean selected gain >=10pp, both seed deltas
nonnegative, and at least one seed >=20%. The independent scale gate permits a
50k continuation only for mean gain >=5pp plus at least one 20k endpoint >=20%
and nondecreasing from 17.5k. A mechanism pass requires a matched seed-3
confirmation before any full-run claim. A primary failure rejects only this
two-head episodic exploration mechanism and moves to a separately isolated
supported action-search experiment.

### 4. Execution

Stage 33 is implemented in
`cqn_as_pixel_bigym_nobc_stage33_episodic_twin_explore_gate.yaml`,
`run_cqn_no_bc_stage33.sh`, and `summarize_cqn_no_bc_stage33.py`. The action
path stores one sampled head per training environment until that environment
resets, while evaluation passes a sentinel that preserves Stage-32 min-Q
selection. Checkpoints persist the sampler RNG but deliberately clear active
heads on resume because environment state is not checkpointed. The summary
verifies the strict reward-only phase contract, the single exploration-field
difference from archived Stage 32, full curves, gates, and final head-use
metrics. Focused algorithm/config/JIT and summary-contract tests pass 15
checks in the two latest groups, including rejection of exploration without
twin critics.

A real MovePlate pixel offline-to-online smoke completed at
`exp_local/cqn_no_bc/stage33_episodic_twin_smoke_gpu3_20260731191557`.
Before interaction, the offline row has `env_episodes=0`, force and behavior
fractions 1.0, finite critic-1/critic-2 loss 3.93183, and no imitation metric;
snapshot 2 was written. Resuming the same critic and optimizer leaves that
offline table unchanged, switches force probability to zero, writes later
snapshots, and records finite independent critic losses. The smoke initially
revealed that rollout diagnostics were logged only for evaluation. Training
CSV wiring was added to the shared workspace and covered by a regression test.
The repeated real online smoke now records one head assignment, head-0/head-1
rates 0/1, and the same assignment at consecutive steps, directly verifying
episode persistence. This instrumentation does not affect the action or loss.

The formal two-seed controller is launched next on physical GPU 3. It uses the
registered 120-second compile stagger and then keeps both runs concurrent on
the same card. GPU 5 continues the already frozen Stage-32 seed-3 direct/twin
validation pair concurrently; no result from that supplemental pair can alter
the Stage-33 launch or its primary gate.

The durable Stage-33 controller launched at 19:26 BST under PID 2076763 in
`exp_local/cqn_no_bc/stage33_episodic_twin_explore_gpu3_20260731192643`.
After the registered two-minute stagger, both seed pipelines are performing
real offline updates on physical GPU 3. At the two-run acceptance check,
seed 1 is at 4.3k/10k and seed 2 at 0.6k/10k. Both have zero environment
episodes, behavior-candidate and demo-force fractions 1.0, force probability
1.0, finite and separately valued critic-1/critic-2 C51 losses, and nonzero
twin disagreement after initialization. Their resolved configs verify
demo-only replay, `strict_demo_rl_only=true`, BC/margin zero, FOSD false,
twin critics enabled, and no actor or policy loss. The two processes use about
5.6 GiB in addition to an unrelated 5.2-GiB process on the card, leaving
about 21.3 GiB free. This is a genuine two-run launch, not process-only
evidence; both CSVs contain completed reward-Q updates.

The pre-registered GPU-5 seed-3 matched confirmation subsequently completed
and wrote `stage32_seed3_summary.json`. Direct and twin are both 0/50 at the
offline endpoint and every online checkpoint through 20k; earliest selection
therefore gives 0% versus 0%, a 0pp seed-3 difference. The frozen primary gate
was already false, twin is below the required 20%, and the registered outcome
is `supplemental_only_primary_gate_failed`. This third training seed
strengthens the Stage-32 interpretation: twin critics repeatedly improve
fixed-state value calibration, but that mechanism does not produce any
MovePlate success at this development budget. Held-out seeds 800--999 remain
unopened. It does not alter Stage 33 or justify a Stage-32 full-budget run.

After formal launch, the expanded regression suite completed with 177 tests
passing (`test_cqn_as.py`, fast-workspace logging/merge, and Stage-33 summary
contracts; one warning, 373.47 seconds). This is implementation and protocol
evidence only; Stage-33 checkpoint selection still depends on its independent
fixed environment rollouts.

Both Stage-33 offline endpoints then completed at exactly 10k updates and
resumed the same critic/target/optimizer snapshots into the 20k online phase.
Each `pretrain.csv` is fixed at 100 data rows with zero environment episodes,
and each online resolved config switches from 32-demo-only/force-1 to the
registered 16-online+16-demo/force-0 contract. The first real online CSV rows
record finite twin reward-C51 losses and one episodic head assignment: seed 1
initially samples head 1 while seed 2 samples head 0. By raw step 11k, seed 1
has completed three episodes and its assignment count rises from one to four,
with cumulative head-0/head-1 rates 25%/75%. This directly verifies that the
randomized critic choice is used for online collection, remains active across
steps, and is resampled at episode boundaries. It is wiring evidence only;
success will be judged by the pre-registered fixed validation sweep.

Two concurrent read-only offline-endpoint probes then completed on GPU 5
without consuming an environment evaluation seed. On the same eight fixed
successful-demo transitions, Stage-33 seed 1/2 have RTG MAE
0.00605/0.00676, Pearson 0.992/0.970, current-action top-1
34.7%/31.4%, top-2 51.1%/49.4%, and
`Q(expert)-Q(greedy)` -0.0143/-0.0129. These values remain in the same
high-calibration/negative-action-gap regime as the Stage-32 offline endpoints.
This supports the intended isolation: before online behavior collection,
Stage 33 has not introduced a qualitatively different value target or an
imitation path. It also reiterates why the online success curve is essential:
accurate reward returns alone have not made the expert chunk greedy.

### Pre-registered Stage-33 full-scale sentinel on GPU 5

No Stage-33 policy checkpoint has yet been evaluated. Before observing any
primary validation result, a separate matched scale sentinel is registered to
address the known risk that coherent exploration can have a late onset. It is
not a post-hoc rescue and cannot change the seed-1/2 20k primary gate.

The sentinel uses training seed 4 for both arms on GPU 5. Its baseline is the
Stage-32 deterministic pessimistic-twin behavior and its treatment is the
Stage-33 episode-persistent sampled-twin behavior. Both receive the same 10k
demo-only reward-Q offline updates, 101k online environment interactions,
16+16 online/demo replay, Bellman/MC targets, twin architecture, optimizer,
Gaussian action noise, and reward-only C51 loss. The exploration flag is the
only method difference. Thus this pair asks only whether episodic randomized
value-function exploration requires the official-scale online budget.

Snapshots are registered at online 10/20/30/40/50/60/70/80/90/100k and the
exact 101k endpoint, plus the offline endpoint. Each is evaluated for 50
episodes on seeds 400--449 with earliest-checkpoint tie breaking; seeds
800--999 remain sealed. Training may run concurrently with the primary, but
the controller must copy and freeze `stage33_summary.json` before starting
sentinel validation. A strong scale signal requires treatment
validation-best >=40% and paired gain >=10pp, which launches the full
multi-seed 101k protocol. A weak signal requires treatment >=20% and
noninferiority, which requires a second matched scale seed. Otherwise the
sentinel does not support this exploration mechanism. None of these outcomes
is itself a final comparison with official CQNAS.

The durable runner, evaluator contract, exact phase verifier, summarizer, and
decision tests are implemented in
`run_cqn_no_bc_stage33_scale_sentinel.sh`,
`summarize_cqn_no_bc_stage33_scale_sentinel.py`, and
`test_summarize_cqn_no_bc_stage33_scale_sentinel.py`. Shell syntax, Python
compilation, exact strict-no-BC phase checks, decision cases, and the existing
JIT/config isolation tests pass nine focused tests. The matched pair is
launched next, still before any Stage-33 fixed validation observation.

The scale-sentinel controller launched durably at 19:48 BST under PID
2105966 in
`exp_local/cqn_no_bc/stage33_scale_sentinel_seed4_gpu5_20260731194804`.
After the registered 120-second stagger, both seed-4 arms are performing real
offline updates on physical GPU 5. Their logs skip the exact same nine failed
demonstrations (4/10/13/14/15/23/29/35/54), and both CSVs record zero
environment episodes, behavior/demo-force fractions 1.0, finite separate
twin losses, and no imitation metric. This verifies matched prior data as
well as process health. Together the two processes use about 6.8 GiB, leaving
about 25.3 GiB free. The controller will resume both to the exact 101k online
endpoint, freeze the primary decision, evaluate the registered checkpoints,
and write the scale summary automatically.

Head assignment alone does not prove behavioral diversity, so the read-only
value-fidelity probe was extended to evaluate each twin critic's independently
greedy coarse-to-fine action on the same fixed states. The implementation and
summary tests pass, and a CPU probe of the Stage-33 seed-1 offline endpoint
completed with `status=ok`. On the eight successful-demo states, the two heads
differ on 38.3% of their complete coarse-to-fine bin path, every sampled chunk
and every current action has at least one differing path factor, normalized
full-chunk action L1 is 0.165, and normalized current-action L1 is 0.193.
Across all 32 fixed states, path disagreement is 35.6% and current-action
disagreement remains 100%. Thus episode-level head sampling is not a no-op:
it produces materially different actions. This is still mechanism evidence,
not evidence that those actions improve task success.

Both scale-sentinel arms subsequently completed their exact 10k offline
phase and crossed the same snapshot/optimizer resume boundary into global
step 111k (101k online interactions). The deterministic baseline's online
rows keep episodic assignments at zero; the treatment's first online row has
one sampled assignment to head 1. Both switch force probability to zero and
retain finite separate twin C51 losses. This confirms that the full-scale pair
implements the registered behavior contrast rather than merely differing in
configuration files.

The Stage-33 primary training phase subsequently completed for both seeds.
Each run wrote all registered snapshots through raw step 30k, corresponding
to the 10k offline endpoint plus 20k online interactions, and the controller
wrote per-seed and joint training-complete markers at 20:05/20:07 BST. The
final logged training rows contain 64 and 66 episodic head assignments, with
head-0/head-1 fractions 48.4%/51.6% and 59.1%/40.9%, respectively. All
training-environment episodes have zero reward, but this is not substituted
for fixed-policy evaluation. At 20:07 the controller launched two concurrent,
fixed validation sweeps on GPU 3 over raw steps 10k, 12.5k, ..., 30k, each
using exactly 50 episodes from seeds 400--449. Process inspection confirms
both evaluators are live and each holds about 1.7 GiB. Based on the matched
Stage-32 sweep duration, the expected completion time is about 20:24 BST; no
validation success observation had been emitted at this registration point.

## Stage 33 complete: diverse twin behavior is not useful coverage at 20k

### 1. Previous-stage result

The Stage-33 primary experiment completed at
`exp_local/cqn_no_bc/stage33_episodic_twin_explore_gpu3_20260731192643`.
Both seeds contain the exact 10k offline endpoint, all eight online
checkpoints through 20k, phase contracts, fixed validation CSVs, summary, and
completion markers. For each seed, the offline endpoint and the online
`2.5/5/7.5/10/12.5/15/17.5/20k` curve are all 0/50 on seeds 400--449.
Earliest tie breaking therefore selects 0% at 2.5k for Stage 33 and for the
immutable Stage-32 matched baseline. Per-seed and mean gains are 0pp;
`mechanism_pass=false` and `scale_continuation=false`. This is 18 fixed
checkpoint evaluations and 900 validation episodes, not a conclusion from
training reward or critic loss. Held-out seeds 800--999 remain sealed.

The treatment was active. Its final training rows record 64/66 episode-level
head assignments, with head-0/head-1 rates 48.4%/51.6% and 59.1%/40.9%.
On fixed successful-demo states, the two critics' independently greedy paths
differ on 38.3% of coarse-to-fine bins, every current action differs in at
least one factor, and normalized current-action L1 is 0.193. The offline
reward diagnostic nevertheless remains well calibrated (RTG MAE
0.00605/0.00676 and Pearson 0.992/0.970) while
`Q(expert)-Q(greedy)` remains negative (-0.0143/-0.0129).

### 2. Interpretation

Stage 33 rules out the narrow 20k hypothesis that merely sampling one of two
meaningfully different reward-trained critic heads per episode turns the
calibrated offline critic into useful MovePlate coverage. The all-zero fixed
curves, rather than finite losses or zero training episodes, are the policy
evidence. The action-diversity probe rules out an implementation no-op: the
heads explore different actions, but those actions do not reach useful task
states within this budget.

This result does not establish full-budget failure. Before any Stage-33
validation observation, the matched seed-4 deterministic/episodic pair was
registered and launched for the exact official-scale 101k online budget on
GPU 5. That late-onset sentinel continues independently and cannot rewrite
the failed primary gate. It also does not resolve whether independent
per-factor maximization is composing a poor joint chunk even when useful
runner-up bins exist.

### 3. Next-stage decision

Stage 34 isolates that remaining policy-extraction question. Starting from
Stage 33, retain the twin direct-C51 architecture, 10k reward-only offline
phase, exact demo-continuation backup, MC lower bound, replay mixture,
optimizer, episodic sampled-head identity, Gaussian noise, and every loss
unchanged. Change only the coarse-to-fine rollout maximizer. At each level,
retain the two highest reward-Q bins for every factor and run a width-8 beam
over complete factor assignments; each beam is extended and scored by one
fixed sampled critic during online collection. Evaluation uses the same
joint beam construction under clipped twin Q and finally ranks complete
chunks by their deepest-level pessimistic expected return. No independently
random factor choices are combined, and no demo action, likelihood, actor,
BC, margin, FOSD, conservative loss, or auxiliary objective is introduced.
Bellman training remains exactly Stage 33 so this stage isolates rollout
search rather than silently changing both search and the learned target.

Fresh seeds 1/2 receive 10k demo-only offline Q updates then 20k online
interactions. The immutable matched baseline is Stage 33 seed 1/2. Report the
offline endpoint separately and select over the same eight online checkpoints
using 50 episodes, seeds 400--449, and earliest ties; 800--999 stay sealed.
Primary metrics are the complete curves and selected best. The mechanism gate
is mean gain >=10pp, both deltas nonnegative, and at least one seed >=20%.
The independent 50k gate is mean gain >=5pp plus at least one 20k endpoint
>=20% and nondecreasing from 17.5k. A pass requires a fresh seed-3 matched
confirmation. If the primary gate passes, a pre-registered read-only
standard-argmax evaluation of the same snapshots will distinguish better
checkpoint learning from beam-only extraction; it cannot change selection.

### 4. Execution

The Stage-33 controller emitted `stage33_summary.json` at 20:20 BST and
exited successfully. Its summary independently verifies every strict
reward-only phase field, the full all-zero curves, balanced head use, failed
gates, and the next decision
`stop_episodic_twin_and_test_supported_beam_exploration`. GPU 3 is now free
for the isolated Stage-34 implementation and two-seed run. In parallel, the
pre-registered GPU-5 scale pair remains live: at the latest completed check,
baseline/treatment had reached roughly 14k/11k online steps and both had
written their 10k-online snapshots with finite twin C51 losses. Stage-34
implementation, exact config-diff tests, a jitted beam test, and a real pixel
smoke start immediately below; formal launch is contingent on those safety
checks, not on the GPU-5 outcome.

Stage 34 is implemented as a default-disabled pure-JAX rollout operator in
`robobase/method/cqn_as.py`, with launch, controller, summary, and protocol
artifacts in
`cqn_as_pixel_bigym_nobc_stage34_joint_beam_gate.yaml`,
`run_cqn_no_bc_stage34.sh`, and
`summarize_cqn_no_bc_stage34.py`. A dynamic program exactly retains the best
width-8 complete assignments from each factor's top-two bins without
independently sampling factors. Sampled-head online rollout uses one critic
for the entire beam; fixed evaluation uses clipped twin bin and complete-chunk
scores. Width 1 remains the legacy path, and guards reject width >1 without
the parallel, pessimistic, episodic-twin platform. Update-time Bellman search
does not call the beam. Resolved Stage-33/34 configs differ only at
`method.twin_rollout_beam_width` (1 versus 8). Twenty related action, JIT,
config, strict-platform, Stage-33, and Stage-34 tests pass; shell syntax,
Python compilation, and focused `git diff --check` pass.

The real MovePlate pixel smoke at
`exp_local/cqn_no_bc/stage34_joint_beam_smoke_gpu3_20260731203448` completed
both phases. Its first two updates occur before any environment action with
demo candidate/force fractions 1.0, MC-lower-bound fraction 1.0, finite twin
loss 3.9318, and no imitation metric. Resuming the same critic, target,
optimizer, and replay to online switches force to zero, records one episodic
head assignment, finite independent losses 3.9126/3.9137, and writes raw
steps 3/4. A separate timing continuation reaches raw 24 and exits zero. Its
roughly 70-second first rollout compile dominates the two-step rate, while
the final 20 real action steps and two 453-MiB snapshots complete in about
one to two seconds after compilation. Peak memory remains safe, so the
width-8 protocol is retained rather than changed for convenience.

The formal Stage-34 controller launched durably at 20:43 BST under PID
2165104 in
`exp_local/cqn_no_bc/stage34_joint_beam_gpu3_20260731204358`. After the
registered 120-second stagger, both seed pipelines are concurrently resident
on physical GPU 3 and have completed real reward-Q updates. At acceptance,
seed 1/2 are at 4.0k/0.7k of 10k offline updates. Their resolved configs both
verify beam width 8, strict demo-RL-only, direct pessimistic twin critics,
episodic head sampling, BC/margin zero, FOSD false, and no policy/flow head.
Both CSVs have zero environment steps, behavior candidate and force fractions
1.0, active MC lower bounds, finite separately valued critic losses, and no
imitation field. Including unrelated processes, GPU 3 uses 12.0/32.6 GiB.
The controller will resume both exact offline states for 20k online steps,
run the fixed 9-checkpoint validation sweeps, and write the protocol-checked
summary automatically. Expected completion is approximately 21:45--22:00
BST, with the next meaningful milestone being both offline endpoints crossing
into online collection.

The independent GPU-5 scale sentinel remains healthy at the same acceptance
time. Its deterministic baseline and episodic treatment have reached about
41k/39k online steps, respectively; baseline has written its 40k-online
snapshot and treatment its 30k-online snapshot. Both processes have finite
twin losses and together use about 6.9 GiB. At the observed throughput, the
101k training endpoints are expected around 22:00 BST and their concurrent
fixed validation sweeps around 22:20--22:30. These estimates do not replace
artifact-backed completion checks.

At the 20:59 BST offline-to-online milestone, both Stage-34 seeds have written
their exact raw-step-10k offline snapshots and `offline_seed*_complete`
markers. Seed 1 has resumed the same state through raw step 12k (2k online),
with seven episode-persistent head assignments, head fractions 42.9%/57.1%,
force probability zero, and finite separate twin losses 1.4786/1.4790. Seed 2
is resident in the corresponding online process after its offline marker and
is still inside the first width-8 rollout compilation; it has not yet emitted
an online CSV row, so no collection metric is inferred for it. Process and GPU
inspection show both jobs live, no failure marker or matched error signature,
and 12.2/32.6 GiB used on GPU 3 including unrelated users. This establishes
the intended offline-first phase boundary and stateful resume, not policy
quality. The broader CPU regression also completed with 180 tests passing;
the sole warning is GLFW reporting the deliberately absent X11 display.

Immediately after that snapshot, seed 2 emitted its first online row at raw
step 11k (1k online): three environment episodes, four persistent head
assignments with 75%/25% head use, force probability zero, and finite twin
losses 1.5955/1.5916. Both seeds have therefore crossed the offline-first
boundary with measured online collection, rather than merely starting a
resume process.

The GPU-5 matched scale pair is likewise still live. Its deterministic
baseline is at raw step 60k (50k online) with a raw-60k snapshot, while the
episodic treatment is at raw step 57k (47k online) with its latest completed
raw-50k snapshot. Their latest separate twin C51 losses are finite
(1.1118/1.1095 and 1.1378/1.1405), the baseline records zero episodic head
assignments, and the treatment records 160 with 46.9%/53.1% head use. The two
training processes each hold about 2.67 GiB on physical GPU 5. The registered
101k endpoint and fixed validation protocol remain unchanged.

A post-launch contract audit verified that the resolved Stage-33/34 configs
differ only at `method.twin_rollout_beam_width`, and that `_joint_beam_action`
is reachable only from rollout/evaluation action construction. The pessimistic
twin update still uses the Stage-33 greedy candidate rather than the beam, so
the Bellman/MC objective is unchanged. Replay does not approximate the
bootstrap behavior chunk by shifting the current chunk: with
`include_next_action=true`, scalar and vectorized samplers independently read
the complete window beginning at `idx + nstep`, apply the registered edge
padding, expose it as `action_tp1`, and the update rejects a missing field.
The explicit nstep-1 test checks `[11,12,13] -> [12,13,14]`; the focused
scalar plus vectorized replay suite passes 8/8 tests. This rules out the main
off-by-one and silent-objective-change threats to interpreting the eventual
fixed-policy curve.

## Literature boundary: no-BC RLfD exists, but is not a CQN-AS drop-in

An audit of the primary DDPGfD, R2D3, RLPD, Hy-Q, and CQN-AS papers resolves
the training-order terminology. Demo-driven RL is not definitionally
offline-to-online. DDPGfD loads demonstrations into replay before training;
R2D3 continuously mixes a demo replay and agent replay; RLPD explicitly starts
online RL from random parameters with no offline pretraining or imitation
constraint; and Hy-Q alternates online collection with fitted Q regression on
aggregated offline and online data. Original CQN-AS likewise trains its RL
agents from scratch with a demo-initialized replay and auxiliary BC. By
contrast, the method pursued here deliberately uses a reward-only offline-Q
phase before online interaction because removing BC leaves no useful initial
behavior; this is a project design choice and an experimentally testable
hypothesis, not the definition of demo-driven RL.

The papers do establish valid no-BC precedents. DDPGfD uses demonstration
replay, prioritized sampling, one- plus n-step critic returns, high UTD, and
deterministic policy gradients, but also has an actor and L2 regularizers.
R2D3 is structurally closest to critic-only CQN-AS: its learner uses n=5
double Q-learning while stochastically mixing two replay buffers; BC appears
as a separate baseline, not its optimized objective. RLPD explicitly removes
both offline pretraining and a BC constraint and adds symmetric 50:50
sampling, critic LayerNorm, large ensembles, and high update ratios. Hy-Q is
pure fitted Q on hybrid data, but its finite-horizon fitted-regression loop and
action maximization assumptions are not a direct implementation for CQN-AS's
factorized continuous action chunks.

Several transferable ingredients are already present or already rejected as
sufficient in isolation. Original CQN-AS and the present runs already use
separate 50:50 demo/online sampling; the current method adds 10k reward-only
offline updates, exact next-demo continuation, MC return lower bounds, direct
C51 twin critics, and episode-persistent value exploration. The matched
Stage-12 long-horizon experiment changed nstep 1 to 8 under an executed K=8
plan and gained only 2pp (20% to 22%); it does not exactly test a simultaneous
one-plus-four-step loss, but rules out treating longer credit assignment as an
automatic solution. Exact replay-candidate maximization was 54% versus 54%
over three matched 20k seeds. Most importantly, Stage 30 fit demo returns
offline yet produced all-zero fixed-policy curves because
`Q(expert)-Q(greedy)` remained negative. Twin calibration and genuinely
different episodic heads then remained all-zero through 20k. Thus the generic
RLfD recipes are useful strong baselines, but do not remove the measured
CQN-AS-specific action-ranking bottleneck by themselves.

The running Stage-34 and pre-registered 101k scale sentinel are not changed by
this audit. If both fail their frozen gates, a literature-complete transfer
baseline may test the still-unresolved mechanisms, but each research question
must remain isolated: first a matched one-plus-four-step distributional target
against one-step at the same update count, then sequence-level prioritized
demo replay, and only then an update-ratio ablation. These may use demo identity
for sampling but every gradient must still derive solely from reward TD/return
targets. Actor gradients, actor pretraining, BC, margin, FOSD, likelihood,
demo-action regression, and implicit policy constraints remain forbidden.

At 21:33 BST, both Stage-34 training seeds completed the exact registered
10k reward-only offline phase plus 20k online interactions. Each run contains
the full raw-step checkpoint set through `30000_snapshot.pkl`, the latest
snapshot link points to raw 30k, both offline and online phase configs are
archived, and the controller wrote both per-seed markers and joint
`training_complete`. The periodic training CSV ends at raw 29k by logging
cadence, so the raw-30k snapshot and completion marker, rather than an
invented final row, establish the endpoint. Final logged head assignments are
66/64 with 48.5%/51.5% and 59.4%/40.6% use; losses remain finite. These are
mechanism and execution facts, not success evidence.

The controller immediately launched two concurrent fixed validation sweeps
on GPU 3. Process arguments independently confirm nine raw checkpoints
`10k,12.5k,...,30k`, 50 episodes per checkpoint, seeds 400--449, and 25
parallel evaluation environments. Both evaluators are live, hold about
711 MiB each, and have not emitted a first checkpoint CSV row yet. The next
result is therefore the complete validation-selected Stage-34 comparison;
no training reward or partial checkpoint is used to change its frozen gate.
In parallel, the GPU-5 101k scale pair remains live at 76k/73k online for the
deterministic/episodic arms.

## Stage 34 complete: joint beam rollout does not recover the policy at 20k

### 1. Previous-stage result

Stage 34 completed at
`exp_local/cqn_no_bc/stage34_joint_beam_gpu3_20260731204358`. Both seeds have
the registered 10k reward-only offline endpoint and all eight online
checkpoints. On the fixed 50-episode selection split, seeds 400--449, the
offline endpoint and online `2.5/5/7.5/10/12.5/15/17.5/20k` curves are all
0/50 for both seeds. Earliest-tie selection therefore chooses online 2.5k
(raw snapshot 12.5k) at 0% for each seed. The immutable Stage-33 matched
baseline has the same two all-zero curves and the same selected values, so
the per-seed gains and mean gain are 0pp. This result comprises 18 fixed
checkpoint evaluations and 900 Stage-34 validation episodes. Seeds 800--999
remain sealed.

The controller wrote `validation_complete`, a protocol-checked
`stage34_summary.json`, and `complete`, then exited normally. The summary
independently verifies the two offline/online phase contracts and the sole
method difference: beam width 1 in Stage 33 versus width 8 in Stage 34. It
also verifies direct pessimistic twin critics, episode-persistent sampled-head
exploration, exact replay continuation, unit-scale reward C51 TD/MC losses,
BC/margin zero, FOSD false, and no actor, flow head, conservative loss, or
auxiliary objective. The mechanism and scale gates are both false.

### 2. Interpretation

This rules out the isolated hypothesis that width-8 joint top-two beam
maximization at rollout converts the current reward-trained factored Q values
into successful MovePlate behavior within 20k online interactions. It is not
an implementation-no-op result: the beam path was active, the episodic heads
were assigned 66/64 times with approximately balanced use, and the complete
policy curves rather than finite training losses determine the rejection.
Together with Stage 33, it says that neither genuinely diverse twin-head
actions nor joint recomposition of their top-two bins is sufficient at the
development budget.

It does not establish failure at the official 101k scale, does not show that
reward-only offline-to-online learning is impossible, and does not authorize
opening held-out seeds. A late-onset effect remains unresolved because the
pre-registered matched scale sentinel was launched before the Stage-33
primary validation result and is still running. The Stage-34 failure cannot
rewrite or cancel that independent test.

### 3. Next-stage decision

Stop the Stage-34 beam branch and complete the already registered GPU-5
scale sentinel. Its deterministic Stage-32 twin baseline and episodic
Stage-33 twin treatment share training seed 4, demonstrations, reward-only
objective, 10k offline updates, 101k online interactions, replay, optimizer,
and evaluation; the sole difference is the episode-persistent sampled critic
used for treatment behavior. Selection evaluates the offline endpoint and
online `10/20/.../100/101k` checkpoints with 50 episodes on seeds 400--449,
chooses validation-best with earliest ties, and leaves seeds 800--999 sealed.
The strong gate is treatment best >=40% with paired gain >=10pp and triggers a
multi-seed 101k protocol. The weak gate is treatment best >=20% and
noninferior, which requires a second matched scale seed. Otherwise the scale
sentinel supplies no support for episodic twin exploration. No Stage-34
checkpoint can enter this selection.

### 4. Execution

At 21:58 BST, immediately after the Stage-34 summary froze this decision, the
GPU-5 controller remains live under PID 2105966 and both training children are
resident and error-free. The deterministic baseline has reached raw step
104k (94k online), and the episodic treatment raw step 101k (91k online);
both have written their raw-100k snapshots. Their latest twin losses are
finite (0.8481 and 0.9105), which establishes health only. The two jobs use
about 5.3 GiB of GPU memory, and physical GPU 5 is at 7.1/32.6 GiB including
other allocations. No failure marker or matched error signature is present.
The controller will finish the exact raw-111k endpoints, freeze the primary
Stage-33 decision, run both registered 12-checkpoint validation sweeps in
parallel, and emit its scale summary automatically. At current measured
throughput, training should end around 22:05--22:10 BST; validation is the
next policy-quality milestone, so this estimate is not reported as a result.

## Stage 33 full-scale sentinel complete: no late-onset twin-exploration effect

### 1. Previous-stage result

The pre-registered scale sentinel completed at
`exp_local/cqn_no_bc/stage33_scale_sentinel_seed4_gpu5_20260731194804`.
Both matched seed-4 arms contain exact raw-111k snapshots, representing 10k
demo-only reward-Q updates followed by 101k online interactions. The offline
endpoint and every online checkpoint at
`10/20/30/40/50/60/70/80/90/100/101k` are 0/50 on fixed selection seeds
400--449 for both the deterministic pessimistic-twin baseline and the
episodic sampled-head treatment. Earliest-tie selection chooses 10k online
(raw 20k) at 0% for each arm; paired validation-best gain is 0pp, and both
raw-111k final endpoints are also 0%. This is 24 checkpoint evaluations and
1,200 development episodes. Held-out seeds 800--999 remain sealed.

The treatment was active throughout the full budget: its final logged state
contains 338 episode-head assignments, split 51.18%/48.82% between the two
critics. The protocol summary verifies that both phases and arms use only the
averaged twin reward-based C51 TD/MC cross-entropy, exact replay candidates,
direct heads, unit critic scale, BC/margin zero, FOSD false, and no actor,
policy, conservative, or auxiliary loss. The sole arm difference is
episode-persistent sampled-head behavior. The controller wrote training and
validation completion markers, a frozen primary decision, the checked scale
summary, and `complete`, then exited normally. Both the >=40%/+10pp strong
gate and the >=20%/noninferior weak gate are false.

### 2. Interpretation

This directly resolves the budget ambiguity for this mechanism on the
registered training seed. The 20k all-zero Stage-33 result did not merely
miss a late-onset episodic-twin effect: the same matched treatment remains
zero at every point through the complete 101k online budget. Balanced head
use rules out a disabled or collapsed exploration switch. Together with the
deterministic baseline's identical curve, it rules out two-head pessimism or
episode-persistent selection alone as sufficient MovePlate exploration in
this reward-only offline-to-online construction.

It does not prove that no-BC CQN-AS is impossible. This is one matched
training seed on the development selection split, not four fixed endpoints
on the sealed split, and therefore it is not compared as a final number with
the official 64.6%. It also does not test the DDPGfD/R2D3 credit-assignment
recipe of optimizing one-step and longer-horizon Bellman returns together.
The earlier Stage-12 nstep-1 versus nstep-8 result changed the sole replay
horizon under a K=8 dense-return setup; it is not a simultaneous 1+4 loss in
the current canonical offline-first twin critic.

### 3. Next-stage decision

Stage 35 isolates that remaining literature-complete reward-learning
mechanism. Use a fresh deterministic Stage-32-style direct twin critic in
both arms; episodic head exploration and joint beam search are disabled
because their matched gates failed. Both replay buffers return exact
one-step and four-step reward sums, bootstrap observations, terminal masks,
discounts, and behavior-action continuations from the same sampled
transition. The control optimizes the existing one-step C51 target. The
treatment optimizes the scale-normalized average
`0.5 * (L_C51_1step + L_C51_4step)`. Each horizon independently uses clipped
double-Q action selection, the data-supported replay candidate, and the
same reward-derived MC lower bound. There is no demo-conditioned branch,
actor, BC, margin, likelihood, ranking, conservative target, or auxiliary
representation loss.

To avoid repeating the small-budget ambiguity, run two full-scale matched
pairs from the outset: seed-1 control/treatment on GPU 3 and seed-2
control/treatment on GPU 5, two runs per card. Every run receives 10k
demo-only offline Q updates and 101k online interactions. Report the offline
endpoint separately and select validation-best over online
`10/20/.../100/101k` using 50 episodes on seeds 400--449 with earliest ties;
also report each fixed raw-111k endpoint. Seeds 800--999 remain sealed.

The strong gate requires treatment selected mean >=40%, mean paired gain
>=10pp, and both seed deltas nonnegative; it promotes the frozen recipe to
two additional 101k training seeds and then the four fixed-endpoint sealed
comparison. The weak gate requires treatment selected mean >=20%,
nonnegative mean gain, and at least one strictly positive seed delta; it
requires a third matched 101k seed before deciding. Otherwise the
simultaneous 1+4 mechanism is rejected. Primary evidence is the complete
fixed success curve; separate one-step/four-step cross-entropies, target
values, candidate use, RTG calibration, and expert-action gap are mechanism
diagnostics only.

### 4. Execution

Implementation starts immediately after the scale summary. Replay will gain
a default-disabled auxiliary horizon that preserves every one-step sampling
start and truncates the four-step window exactly at episode termination;
tests must cover terminal-adjacent reward sums, effective discounts,
bootstrap observations, exact `t+4` behavior chunks, scalar/vectorized byte
equality, and the existing `t+1` off-by-one invariant. The critic change will
be isolated behind a nonnegative auxiliary C51 weight, normalized to preserve
the one-step loss scale, and strict-platform tests must prove demo-flag
invariance and absence of any imitation/policy loss. A real pixel
offline-to-online smoke, snapshot round trip, resolved control/treatment
config diff, and full focused regression precede formal launch.

At 23:01 BST, no Stage-33/34 process remains. Physical GPU 3 has 26.3 GiB
free after an unrelated allocation, and GPU 5 has 31.2 GiB free. These cards
are reserved for the registered two-pairs protocol after implementation and
smoke validation; no new policy-quality claim is made from availability.

Stage 35 is now implemented behind default-disabled replay and method fields.
`replay.auxiliary_nstep=4` preserves every valid one-step start and emits the
terminal-truncated four-step reward, effective discount, terminal flag,
bootstrap observation, and exact behavior-action chunk at that bootstrap
state. Scalar, explicit-index, and vectorized sampling share this contract;
the latter remains byte-identical to the scalar reference. The treatment's
isolated direct twin-C51 update computes each horizon's clipped double-Q
candidate target and MC lower bound independently, then optimizes
`(L_1 + w L_4) / (1 + w)`. The control sets `w=0`; no policy, demo-action,
margin, likelihood, conservative, or representation loss was added.

The implementation artifacts are
`cqn_as_pixel_bigym_nobc_stage35_one_step_control.yaml`,
`cqn_as_pixel_bigym_nobc_stage35_one_plus_four_gate.yaml`,
`run_cqn_no_bc_stage35.sh`, and
`summarize_cqn_no_bc_stage35.py`. Resolved MovePlate configs differ only at
`method.auxiliary_td_loss_weight` (`0.0` versus `1.0`); both retain nstep 1,
auxiliary nstep 4, deterministic width-1 pessimistic twins, exact replay
candidates, BC/margin zero, FOSD false, and the strict reward-only guard.
The pre-registered summarizer checks every archived offline/online phase,
selects only the fixed online 10k--101k curve with earliest ties, reports the
offline and raw-111k endpoints separately, and leaves seeds 800--999 sealed.

Validation completed before formal launch. The pre-existing CQN-AS suite
passes 172 tests after the critic change; the final replay/config/objective/
summary group passes 21 tests, including scalar/vector byte equality,
terminal-adjacent four-step truncation, JIT and non-JIT finite updates, the
exact normalized loss identity, strict contract rejection, and frozen gate
logic. Shell syntax, Python compilation, and the resolved single-field arm
diff also pass.

A real MovePlate pixel offline-to-online smoke completed at
`exp_local/cqn_no_bc/stage35_one_plus_four_smoke_gpu5_20260731232558`.
Its two-update offline phase has zero environment episodes, writes snapshot
2, and records finite twin losses with `critic_loss=3.931826`, equal one- and
four-step losses at initialization, unit auxiliary weight, both candidate and
forced-demo fractions 1.0, and both MC-lower-bound fractions 1.0. Resuming
the exact critic, target critics, optimizer, and replay leaves `pretrain.csv`
unchanged, executes two real environment steps, and writes snapshot 4. The
online row records `critic_loss=3.913243`, one-step loss 3.913251, four-step
loss 3.913235, their normalized mean 3.913243, separate critic losses,
nonzero twin disagreement 0.000181, force fractions zero, and finite
candidate/MC diagnostics for both horizons. Neither CSV contains a BC, FOSD,
margin, imitation, policy-loss, or MC-auxiliary-loss column. This establishes
wiring, real replay semantics, and resume health only; it is not policy
quality evidence.

The registered full-scale experiment launched durably at 23:30:59 BST under
controller PID 2334671 in
`exp_local/cqn_no_bc/stage35_one_plus_four_fullscale_20260731233059`.
GPU 3 carries seed-1 control and treatment; GPU 5 carries seed-2 control and
treatment, exactly two runs per card. The controls began together and the
treatments began after the registered 120-second compilation stagger. All
four processes have loaded the same task demonstrations, created main and
protected-demo replay with both horizons, and emitted finite reward-Q update
rows. The archived seed-1 control/treatment configs still differ only at
`method.auxiliary_td_loss_weight`; no failure marker exists.

At the formal acceptance check, seed-1/2 controls had reached 4.8k/5.9k of
10k offline updates. Their weights and auxiliary-loss contributions are zero,
and each logged `critic_loss` exactly equals its one-step loss. Seed-1/2
treatments had reached 0.6k/0.7k; their latest normalized losses are
1.90627 from one-step 1.91733 and four-step 1.89520, and 2.18508 from
one-step 2.21343 and four-step 2.15673. Both horizons use the forced replay
candidate on all offline samples, both critics remain distinct, and all
reported values are finite. These losses establish activation and matched
scale only, not policy quality. Physical GPUs 3/5 retain approximately
20.1/25.0 GiB free while all four jobs are resident.

Based on the measured 10.9--14.4 batched offline updates/s under four-way
contention and the prior 101k twin-run throughput, the coarse completion
window is approximately 02:30--04:00 BST, including the four fixed validation
sweeps. This estimate will be replaced by the measured online throughput once
both pairs cross the exact raw-step-10k offline boundary. The controller will
then evaluate raw steps 10k, 20k, ..., 110k, 111k on seeds 400--449 and emit
`stage35_summary.json`; it cannot inspect or open seeds 800--999.

The offline-to-online transition was accepted for all four formal runs at
23:49 BST. Every arm has an exact 10k snapshot of approximately 474 MB and an
`offline_complete` marker, and every resumed process has now written a real
online row beyond raw step 10k with `env_episodes > 0`. At that check, seed-1
control/treatment were at raw 16k/11k with 21/3 completed environment
episodes, and seed-2 control/treatment were at raw 16k/12k with 20/7 episodes.
The controls continue to record zero auxiliary weight and loss with total
critic loss equal to the one-step loss. The treatments continue to record
unit auxiliary weight and the exact normalized one-plus-four-step loss; for
example, the latest seed-1 treatment row is 1.324387 from one-step 1.330611
and four-step 1.318163. Forced-demo use is zero in every online row. The
controller remains live, physical GPUs 3/5 have 20.3/25.1 GiB free, and no
failure, CUDA/OOM, traceback, or non-finite marker exists.

This establishes only that the four matched reward-only agents completed the
registered offline Q phase, restored the full training state, and entered
genuine online interaction with the intended objectives. It neither measures
policy quality nor favors the treatment: success is not inferred from finite
or decreasing critic losses. The next quality decision remains frozen until
the complete 101k-online checkpoint curves and all 50-episode validation
evaluations exist. The summarizer will then apply the registered strong/weak
gates to validation-selected checkpoints; no sealed seed in 800--999 is
opened before a strong-gate promotion.

The first repeated online intervals provide the promised ETA refinement.
Recent per-run rates are approximately 12.7--13.8 raw steps/s for the
controls and 11.2--11.9 raw steps/s for the one-plus-four-step treatments;
these are computed from successive `(env_steps, total_time)` CSV rows rather
than instantaneous device utilization. At the slow treatment rate, training
should reach raw 111k near 02:10--02:20 BST. The directly comparable Stage-33
12-checkpoint, 50-episode sweep required about 47--50 minutes while two arms
shared one card. Allowing compilation and serialization margin gives a
revised Stage-35 summary ETA of approximately 03:00--03:20 BST on 1 August.

The first matched online-checkpoint milestone completed at 00:01:32 BST on
1 August: all four arms crossed raw step 20k, which is exactly 10k online
interactions after the offline phase. All four `20000_snapshot.pkl` files
exist at approximately 474 MB. At raw 20k, seed-1 control/treatment had 36/35
completed episodes and seed-2 control/treatment had 34/34. Objective checks
remain exact: control losses are 1.845279/1.727886 with zero auxiliary
contribution, while treatment losses are 1.617130 and 1.427335, exactly the
means of `(1.626129, 1.608130)` and `(1.420192, 1.434479)` one-/four-step
losses. Core metrics are finite, all four processes and the controller remain
live, no failure/error/non-finite marker exists, and GPU memory remains safe.

This checkpoint is recoverable and the matched objectives remain wired as
registered, but it is deliberately not a policy-quality result. The online
training rows' most recent episode success flags are zero in all arms, which
is neither a fixed-episode evaluation nor a checkpoint selection split. The
controller continues uninterrupted toward all raw 10k--111k snapshots; only
the subsequent seeds-400--449 50-episode sweep will supply the registered
best-checkpoint comparison and gate decision.

## Stage 36: batch-256 replication of the retained dense No-BC baseline (2026-08-01)

The user requested a return to the earlier ordinary dense No-BC runs that
reached validation-selected best success of 60%/46%/56% for training seeds
1/2/3, but at the original CQN-AS replay-batch scale. This is a separate
research question from Stage 35. Stage 35 remains a matched one-step versus
one-plus-four-step test on the canonical sparse offline-to-online branch;
Stage 36 tests whether the retained online-only dense-return objective
reproduces when `batch_size=256` and `demo_batch_size=256` rather than the
historical No-BC development setting of 16+16.

All three fresh runs use MovePlate, frame stack 4, three 84x84 cameras,
action sequence 16, execution length 1, zero pretraining, 20k online
environment steps, dueling C51, `dense_return_q_target=true`, the MC lower
bound, and exactly zero BC/margin/imitation loss. Snapshots are fixed at
2.5k intervals. Selection uses only seeds 400--449 with 50 episodes per
checkpoint and earliest-best tie breaking; seeds 800--999 remain sealed.
The immutable historical b16 best values are 60%/46%/56%. The registered
replication gate is a fresh three-seed validation-best mean of at least 49%
with at least two seeds reaching 40%. A pass promotes this exact b256 dense
No-BC configuration to a matched 101k No-BC-versus-BC comparison; a failure
does not license selection on held-out seeds.

GPU 4 is the only currently free card. Seeds 1 and 2 will use the measured
two-runs-per-card protocol with `xla_mem_fraction=0.45` and a 120-second JIT
stagger. Their fixed validation sweeps run only after both trainings finish;
seed 3 then trains and evaluates after the first wave has fully released the
card, so evaluation never shares a GPU with training. The durable controller,
run paths, live acceptance evidence, and ETA are recorded below after launch.

### Stage-35 recovery and GPU-3 diagnosis

At 01:53 BST the Stage-35 seed-2 one-plus-four treatment exited with status
143 after reaching raw step 99k; its latest durable snapshot is raw 90k. The
log ends with a resource-tracker shutdown warning and contains no traceback,
CUDA OOM, EGL failure, or non-finite metric. Kernel logs contain no NVIDIA
Xid/NVRM event. Physical GPU 3 is healthy and idle at 45 C, P8, 3 MiB, while
`nvidia-smi` reports its full 32.6 GiB capacity. The evidence therefore
identifies an external SIGTERM, not a GPU-3 hardware or driver failure.

Recovery is registered to resume the exact Stage-35 seed-2 treatment state
from its latest snapshot on GPU 3, preserving the existing run directory and
CSV history. After raw 111k exists, the recovery controller runs the original
four seeds-400--449, 50-episode checkpoint sweeps on GPU 3 in two no-training
waves and invokes the unchanged frozen Stage-35 summarizer. The historical
`training_failed` marker is retained as interruption provenance; recovery
uses separate completion markers.

## Stage 35 compute audit and corrective early-scaling gate (2026-08-01)

### 1. Previous-stage result

An audit after the user's compute-efficiency objection finds only two no-BC
candidate jobs that requested the official 101k online budget: the Stage-33
seed-4 scale sentinel and Stage 35. Stage 34 correctly stopped at 20k after
both complete eight-point validation curves were zero. The Stage-33 sentinel
was deliberately launched before observing its primary 20k curve to answer a
single late-onset question; it later measured 0/50 at every online checkpoint
from 10k through 101k in both arms. It should remain a one-off negative scale
control, not a template for later candidates.

Stage 35 did not inspect an early policy curve before allocating 101k. Its
runner saves every raw-10k checkpoint but defers all evaluation until all four
runs finish. At 01:37 BST, seed-1 control is complete at raw 111k, seed-1
treatment is at raw 100k, seed-2 control is at raw 101k, and seed-2 treatment
is at raw 86k; no validation CSV exists. The actual scalar training task-success
metric is zero in every row: 0 successes across 340/306/308/255 completed
environment episodes respectively. The similarly named double-underscore
field is static environment metadata and is not counted as success.

### 2. Interpretation

The Stage-35 allocation was too aggressive. Correcting the final comparison
from a 10k selected peak to a 101k fixed endpoint does not imply that every
untested mechanism deserves a blind full run. Early fixed-policy curves are
valid futility and scaling evidence even though they cannot establish final
parity with official CQNAS. Deferring every validation point until raw 111k
turns the intended development gate into a post-hoc report and wastes compute
when a curve is flat zero. Training success remains weaker than fixed-policy
evaluation, but four all-zero rollout streams make immediate validation more,
not less, important.

### 3. Next-stage decision

Insert a corrective coarse scaling gate before the registered 50-episode
sweep. On the existing selection split only, evaluate treatment checkpoints
at raw 20k/40k/60k/80k/100k for 20 fixed episodes, seeds 400--419. Run seed 1
as soon as GPU 3's completed control frees a slot; run seed 2 when one GPU-5
slot frees, using the same available checkpoints. This is a futility screen,
not the reported checkpoint selection and never opens seeds 800--999.

Pass immediately if any treatment point reaches at least 4/20 (20%), matching
the registered weak absolute gate; then resume the full matched 50-episode
curve. A 1--3/20 point is ambiguous and triggers a 50-episode evaluation only
for that point, its neighboring checkpoints, and the matched control. If both
treatment seeds are 0/20 at every tested point, reject the mechanism and skip
the four-arm 12-checkpoint sweep. Future stages must use the same progressive
budget rule: 20k first, extend to 50k only on a nonzero/rising curve, and
allocate 101k only after a reproducible development signal. The prior
Stage-33 sentinel is sufficient evidence that an all-zero curve need not be
re-run to 101k merely to rule out generic late onset.

### 4. Execution

Pause only the top-level Stage-35 coordinator before it can launch the large
post-training validation sweep; leave its current training children running
and snapshots intact. Launch the seed-1 treatment coarse sweep on GPU 3 as the
second run on that card, verify real evaluator progress, then apply the frozen
gate above. This reversible coordinator pause prevents new evaluation compute
while retaining the option to resume the original controller if the coarse
curve passes.

The coordinator was verified as stopped in reversible `T` state under PID
2334671 while all training children continued advancing. The seed-1 treatment
coarse sweep then completed normally at
`seed1/treatment/offline_twin_seed1/early_scale20_seeds400.csv`: raw
20k/40k/60k/80k/100k are respectively 0/20, 0/20, 0/20, 0/20, and 0/20.
This is a flat 0/100 fixed-policy scaling curve through 90k online steps, not
an inference from training loss. It fails the per-seed coarse pass but does
not yet trigger the two-seed futility decision. The second seed's available
raw 20k/40k/60k/80k checkpoints are evaluated next on GPU 3 after the first
evaluator exits, while the remaining training processes continue.

The second-seed sweep also completed flat zero: raw 20k/40k/60k/80k are
0/20 at every point. The frozen corrective gate therefore observes 0/180
fixed validation episodes across two treatment training seeds and rejects the
simultaneous one-plus-four-step mechanism. Seed-2 treatment was intentionally
stopped at raw 99k with its latest complete raw-90k snapshot; the other three
arms had already reached raw 111k. Resuming the paused coordinator reaped the
expected SIGTERM and exited before launching any large evaluator. There are
zero live Stage-35 processes and zero `val50_seeds400_full_raw_steps.csv`
files. `stage35_early_scaling_summary.json` records the decision and the
expected `training_failed` marker as a futility stop, not a crash. The coarse
gate used 180 episodes and avoided the planned 2,400-episode sweep, saving
2,220 validation episodes.

### Stage 36: official-batch, baseline-matched offline-to-online gate

#### 1. Previous-stage result

The Stage-35 result above establishes policy failure for the tested 1+4
variant and also exposes a comparison flaw: all prior no-BC development arms
used `batch_size=16` plus `demo_batch_size=16`. The verified official
new-infrastructure CQN-AS reference at
`exp_local/cqn_official_repro_newinfra/move_plate_official_seed1_gpu2_20260730234443`
uses 256 for each replay stream, receives an effective online update batch of
512 after concatenation, and achieves 67.5% on the sealed 200-episode raw-101k
endpoint. Its resolved config uses the canonical single dueling critic,
K=16, nstep 1, learning rate 5e-5, target tau 0.02, critic coefficient 0.1,
one update per environment step, and Gaussian scale 0.01.

#### 2. Interpretation

The batch-32 no-BC studies remain mechanism-development evidence but cannot
serve as a fair performance comparison to the official batch-256-per-stream
baseline. In this workspace the two batch fields are concatenated, so
"official batch 256" means 256 protected-replay samples offline and 256
online plus 256 protected samples (512 total) online. This is now a hard
protocol requirement. It does not rescue Stage 35 post hoc; that mechanism is
rejected by its measured flat curves.

#### 3. Next-stage decision

Stage 36 retests the strongest prior reward-only construction under the exact
official architecture and optimization setup. Both matched seed-1 arms use
the same 10k demo-only offline phase followed initially by only 10k online
interactions on physical GPU 3, with two runs on the card. The control keeps
canonical CQN-AS BC/FOSD/margin. The treatment removes every action-imitation
gradient and changes only the reward target: Double-CQN maximization includes
the exact replay continuation candidate and the completed reward return is a
lower bound on that same C51 target. No twin critic, direct head, beam,
auxiliary horizon, actor, or representation loss remains.

Both arms retain the official successful-online-trajectory protected replay.
For the treatment this is reward-only replay: `bc_lambda=bc_margin=0` and
FOSD false, so duplicated successful transitions receive only TD/MC gradients.
The strict guard requires an explicit success-replay permission and still
rejects every policy/imitation loss.

The progressive compute gate is mandatory. First evaluate only treatment at
the offline raw-10k point and raw 12.5k/15k/17.5k/20k (2.5k--10k online) for
20 episodes on seeds 400--419. A completely zero curve stops immediately. Any
nonzero point triggers the matched 50-episode control/treatment sweep on
seeds 400--449. Extension to 20k online additionally requires a treatment
late point (7.5k or 10k online) of at least 20%. Further extension to 50k and
then 101k must be separately earned by nonzero, non-collapsing scaling curves;
no runner may jump directly to full scale. Seeds 800--999 remain sealed.

#### 4. Execution

Added `cqn_as_pixel_bigym_stage36_offline_bc256_control.yaml` and
`cqn_as_pixel_bigym_stage36_offline_nobc_candidate256_gate.yaml`. Both resolve
to batch 256/256, offline 10k, canonical single dueling CQN-AS, and identical
baseline model/replay/optimizer settings. Their complete resolved difference
is restricted to BC/FOSD removal, strict reward-only guards, replay next-action
availability, candidate-max target source, and MC-lower-bound activation.
The strict validator now has an explicit reward-only success-replay permit;
its old default still rejects `use_self_imitation=true`. Nine focused tests
pass, including all legacy rejection cases, explicit reward-only replay, exact
official batch semantics, baseline hyperparameters, and the frozen resolved
config diff. A two-process GPU-3 offline/online memory smoke and progressive
runner validation are executed before the formal 10k-online launch.

## Stage 0--35 retrospective audit and Stage 37 dense replication (2026-08-01)

### 1. Previous-stage result

The complete summary-JSON audit starts at Stage 0 rather than treating the
later offline branch as the original no-BC baseline. Stage 0's pure TD arms
were 0% and also exposed a tenfold remaining-loss-scale mismatch. Stages 1--6
tested normalized TD, return floors, MC lower bounds, floor-tail reductions,
demo-only replay, and autoregressive dimensions; their best fixed success was
at most 16% and none produced a replicated rising curve.

Stage 7 introduced `dense_return_q_target=true` and changed seed-1 from a
0/0/0/10% max-floor curve to 4/12/40/40%. Stage 8 replicated it at b16+d16:
three-seed 10k best values were 40/46/46%. The completed 20k ordinary curves
are seed 1 `12/24/34/48/44/60/54/46%`, seed 2
`4/8/32/46/46/40/44/38%`, and seed 3
`6/30/36/46/50/56/50/40%`; validation-selected best values are 60/46/56%,
mean 54%.

Stages 11--21 did not improve that baseline reproducibly. Stage 22's replay
candidate reached 56/44/50% at 10k and Stage 25's complete 20k comparison was
56/56/50% versus ordinary 60/46/56%, equal 54% means; the added mechanism is
therefore unnecessary. Stage 26 was 48/48%. Reward scaling reached 66/50% in
two seeds but fresh seed 3 reached only 46% versus its 56% ordinary control,
leaving the three-seed mean delta exactly zero. Ordered success was lower.

Stages 30--35 are a separate configuration branch: the dense-return target
was disabled, the bootstrap source changed, and later stages added offline
warmup/direct heads/twins/exploration/beam/auxiliary horizon. Stage 30's
online-only controls were already all-zero, proving offline warmup alone did
not cause the discontinuity. Stages 30--34 remained zero at every fixed point;
the Stage-33 seed-4 sentinel remained zero at all checkpoints through 101k.
Stage 35's treatment coarse screen was 0/180 across two seeds through raw
100k/80k. This entire branch is rejected and receives no further recovery or
full evaluation.

### 2. Interpretation

The batch audit finds that all historical no-BC development runs used
16 ordinary plus 16 demo samples, whereas the official full CQN-AS runs use
256+256. Frame stack (4), three 84x84 cameras, K=16, execution length 1, and
the 512x512 critic width were not reduced. Thus batch scale is a real unmatched
factor for official comparison, but it does not explain the Stage-29-to-30
collapse: both the successful dense branch and Stage-30 online-only control
used b16+d16. The measured collapse coincides with disabling the dense target
and changing bootstrap semantics.

The simplest retained dense online-only objective is the only candidate with
three nonzero, broadly rising curves and no unnecessary mechanism. Candidate
backup and reward scaling are not promoted because their three-seed means do
not beat it. All-zero branches are explicitly abandoned.

### 3. Next-stage decision

Stage 37 reruns the retained dense online-only No-BC objective with
`batch_size=demo_batch_size=256`, three training seeds, 20k online steps, and
the fixed 50-episode seeds-400--449 curve every 2.5k. Promotion requires mean
validation-best at least 49%, at least two seeds with best at least 40%, and
at least two seeds reaching 40% at 17.5k or 20k. The late-point condition
prevents a transient early spike from earning a full run.

On a pass, pre-registered training seed 1 resumes to 101k. Its validation
curve is descriptive; the decisive metric is the fixed raw-101k checkpoint on
200 held-out episodes, seeds 800--999, against the official same-seed 62%
reference. It is not selected from the short-run winners. A seed-1 endpoint at
least 62% triggers the remaining training seeds; otherwise no parity claim is
made.

### 4. Execution

The first b256 dense attempt was externally terminated with status 143 at
seed-1/2 raw 6k/4k, without CUDA/OOM evidence. Durable 5k and 2.5k snapshots
remain. `run_cqn_no_bc_stage37_dense_b256.sh` resumes these exact runs after
the distinct GPU-4 Stage-36 offline gate releases the card, while fresh seed 3
starts on GPU 3 under the measured two-training-runs-per-card allocation. All
evaluation waits for training to leave GPU 4. The controller automatically
stops at the registered short gate or, only on a pass, runs the seed-1 101k
extension and fixed held-out endpoint.

The durable Stage-37 controller launched at 02:12:35 BST in
`exp_local/cqn_no_bc/stage37_dense_b256_20260801021235`, reusing the interrupted
run base `stage36_dense_b256_gpu4_20260801014727`. Fresh seed 3 is live on GPU
3 and has emitted its first post-JIT 1k row with finite critic loss 3.21745.
Its archived config verifies b256+d256, frame stack 4, zero pretraining,
`dense_return_q_target=true`, and `bc_lambda=0`. GPU 3 sustains 95--97% SM at
about 28.3 GiB total for the two co-resident training processes, with no OOM
or CUDA error. The preserved seed-1/2 snapshots are raw 5k/2.5k. Their resume
is queued behind the distinct Stage-36 GPU-4 gate, whose current offline
control/treatment progress is 5.5k/2.2k of 10k updates. No success claim is
made before the fixed Stage-37 validation curves exist.

A resolved-config diff against the historical seed-3 dense run confirms that
the only active research-scale change is `batch_size` and `demo_batch_size`,
both 16 to 256. Newly explicit fields are neutral defaults: no auxiliary TD,
advantage, ordered-success, episodic-head, twin, beam, forced-demo, or
reward-rescaling effect is active. The retained contract is dueling C51,
dense-return plus MC-lower-bound target, critic bootstrap, `critic_lambda=1`,
BC/margin zero, FOSD false, no separate BC/flow policy, K=16, nstep 1, and
frame stack 4. Relative to official CQN-AS, batch, vision/action geometry,
dueling architecture, bins/levels, and hidden width now match; the intended
research differences are removal of BC/margin and the reward-only dense/MC
target with critic coefficient normalized from 0.1 to 1.0.

The subsequently launched `stage36_dense_b256_gpu4_20260801014727` job is not
this matched experiment. It inherited the older dense-return construction,
used zero offline updates, and changed several non-baseline mechanisms. After
the user made exact baseline matching a hard requirement, both processes were
terminated at raw 5k/4k before any fixed-policy evaluation; their existing
2.5k checkpoints are retained only as interruption provenance. A training-row
success in those files is not a validation curve and is not used for method
selection. The accidentally restarted, already-rejected Stage-35 seed-2
treatment was also stopped again at its raw-93k CSV row. Its frozen coarse
result remains 0/180, and no deferred 50-episode sweep was launched.

The official-batch two-process smoke completed at
`exp_local/cqn_no_bc/stage36_batch256_pair_smoke_gpu4_20260801020620` with no
OOM or failure marker. Both arms used 256 replay samples offline and restored
for a real online update with 256 online plus 256 protected samples. The
control retained BC=1, margin=0.1, and FOSD; the treatment resolved to BC=0,
margin=0, FOSD=false, replay-candidate maximization, and the MC lower-bound.
Its offline row records behavior-force fraction 1.0 and MC-lower-bound
fraction 0.996; after restore the force fraction is zero and two real online
steps are present in each arm's `train.csv`. This is memory, objective, and
snapshot health evidence only, not policy-quality evidence. The new frozen
gate summarizer has four passing decision tests, and the Stage-36 strict and
resolved-config tests pass 2/2.

`run_cqn_no_bc_stage36_offline_gate.sh` now enforces the progressive budget in
the executable control flow. It cannot train beyond 10k demo-only updates plus
10k online interactions. It first opens only a treatment 20-episode curve at
raw 10k/12.5k/15k/17.5k/20k on seeds 400--419. An all-zero curve writes
`early_futility_stop` and exits without control evaluation. Any nonzero point
earns the matched 50-episode seeds-400--449 curve; even a pass only emits an
eligibility decision and never launches a longer training phase. Four
summarizer tests cover all-zero stopping, nonzero promotion, earliest-best
selection, the late 20% extension gate, and evaluation-protocol rejection.

The formal matched initial gate launched durably on physical GPU 4 at 02:10:21
BST under controller PID 2463686 in
`exp_local/cqn_no_bc/stage36_offline_gate_gpu4_20260801021021`. The control is
loading the MovePlate demonstrations and the treatment is scheduled after the
required 120-second compilation stagger; both use `batch_size=256`,
`demo_batch_size=256`, and `xla_mem_fraction=0.45`. The first ETA, based on the
measured smoke compilation and approximately 5--7 updates/s once compiled, is
03:20--03:40 BST for the treatment coarse scaling decision. It will be refined
from advancing `pretrain.csv` and `train.csv` artifacts after both processes
reach steady state.

The two-process acceptance check completed at 02:14 BST. Control PID 2463694
and treatment PID 2466461 are simultaneously resident on GPU 4, using 18.3 GiB
in total at 96% utilization with no failure/OOM marker. Their latest offline
rows are at 3.3k and 0.2k updates respectively. Under contention the measured
rates are 8.86 and 7.99 batched updates/s; both critic losses are finite. The
treatment records behavior-force fraction 1.0, MC-lower-bound fraction
0.95--0.98, and no environment episodes, confirming that this is the intended
reward-Q-only demo phase rather than online interaction or policy pretraining.
The measured rates retain the 03:20--03:40 BST coarse-decision ETA.

### Stage 36 completed result and Stage 37 handoff (2026-08-01 03:25 BST)

#### 1. Previous-stage result

The formal Stage-36 offline-to-online gate completed in
`exp_local/cqn_no_bc/stage36_offline_gate_gpu4_20260801021021`. Both arms
finished 10k demo-only updates and resumed from their raw-10k snapshots for
10k real online environment steps; the first online CSV rows were exactly at
global step 10k, confirming that the transition did not reset training. The
No-BC treatment's fixed 20-episode seeds-400--419 coarse curve was 0/20 at
every registered raw checkpoint: 10k, 12.5k, 15k, 17.5k, and 20k. The sweep
durations were 248.9, 213.5, 245.3, 243.4, and 218.6 seconds. The generated
`stage36_coarse_summary.json` records `coarse_nonzero=false`, and the runner
wrote both `early_futility_stop` and `complete`; no control evaluation or
larger budget was spent.

#### 2. Interpretation

Official batch scale and a matched offline warmup do not rescue this tested
reward-only replay-candidate/MC-lower-bound construction within 10k online
interactions. The result establishes policy failure for this Stage-36 branch,
not a general impossibility of learning without BC. In particular it does not
overturn the earlier online-only dense-return curves, which used different
target semantics and remain the only replicated nonzero candidate. Training
losses and finite updates are treated only as wiring evidence.

#### 3. Next-stage decision

Stage 36 is rejected under its pre-registered all-zero futility rule. The next
question remains the Stage-37 three-seed batch-256 replication of the retained
online-only dense-return objective. Selection uses fixed 50-episode seeds
400--449 at every 2.5k checkpoint and the already registered mean/breadth/late
promotion gate; held-out seeds 800--999 remain sealed.

#### 4. Execution

Stage-37 seed 3 completed raw 20k on GPU 3 and wrote
`dense_b256_seed3/snapshots/20000_snapshot.pkl` at 03:10:21 BST. When Stage 36
released GPU 4, the durable controller immediately wrote `gpu4_released` and
resumed seed 1 from its 5k snapshot at 03:23:15; its first new CSV row is at
step 5k, proving checkpoint rather than zero-step restart. Seed 2 launched on
the same card after the required 120-second stagger and is restoring from its
2.5k snapshot. Both use batch 256/256 and a 0.45 device slice. Evaluation will
start only after both training processes release GPU 4.

### Stage 37 first two fixed validation curves (2026-08-01 06:20 BST)

#### 1. Previous-stage result

All three batch-256 dense No-BC seeds completed raw 20k and have durable
snapshots at every registered 2.5k selection point. Fixed 50-episode
seeds-400--449 evaluation is complete for the first two seeds. Seed 1's curve
at raw 2.5/5/7.5/10/12.5/15/17.5/20k is
`38/52/50/40/46/44/32/42%`; seed 2's curve is
`30/50/44/50/50/48/46/42%`. Their validation-selected best values are 52%
at 5k and 50% at 5k (earliest-checkpoint tie break). Both raw-20k late
points are 42%.

#### 2. Interpretation

The retained dense objective is not an all-zero batch-256 failure: both seeds
rise to at least 50% by 5k and remain nonzero throughout. Seed 1 does show a
17.5k drop to 32%, so the result does not establish monotonic scaling; its
recovery to 42% at 20k and seed 2's 46/42% late pair nevertheless satisfy the
registered late criterion for these two seeds. This is policy-quality evidence
from fixed episodes, not an inference from training loss.

#### 3. Next-stage decision

Seed 3 must now complete the identical eight-point fixed sweep. Given the
first-two-seed best values of 52% and 50%, the registered three-seed mean-best
threshold of 49% requires seed 3 to reach at least 45%. The breadth condition
and the requirement for two seeds with a late point at least 40% are already
satisfied by seeds 1 and 2, but promotion remains forbidden until the complete
seed-3 curve and frozen summarizer confirm the full gate.

#### 4. Execution

The seed-3 sweep launched alone on GPU 4 at 06:19:35 BST after both seed-1/2
sweeps exited. It evaluates the same raw 2.5k--20k checkpoints for 50 episodes
on seeds 400--449. No training shares the card. On gate pass the durable
controller will immediately write `full_run_promoted` and resume the
pre-registered seed 1 to raw 101k; on failure it will write `complete` without
a full run.

### Stage 37 short gate passed and full run launched (2026-08-01 07:39 BST)

#### 1. Previous-stage result

Seed 3 completed the fixed raw 2.5/5/7.5/10/12.5/15/17.5/20k curve at
`38/48/36/42/44/36/36/36%`, selecting the earliest 5k checkpoint at 48%.
Together with seed 1's 52% and seed 2's 50%, the three-seed
validation-selected best mean is 50%. All three seeds have a best of at least
40%; seeds 1 and 2 each have a late best of at least 40% (42% and 46%). The
frozen `stage37_short_summary.json` therefore records `promotion_pass=true`.

#### 2. Interpretation

Official batch 256+256 reproduces a substantial, nonzero dense No-BC policy
curve and passes the pre-registered full-run gate. It does not produce
monotonic scaling: seed 1 dips at 17.5k and seed 3 peaks at 5k before settling
at 36%. The result establishes that the retained objective can learn without
BC at official batch scale, but it does not yet establish 101k endpoint parity
or superiority over official CQN-AS.

#### 3. Next-stage decision

The pre-registered seed 1 now continues from its raw-20k snapshot to raw 101k.
After training, fixed seeds 400--449 provide the descriptive 20k--101k curve;
the decisive game-performance metric remains the sealed 200-episode raw-101k
endpoint on seeds 800--999 versus the official same-seed 62% reference. A
held-out endpoint of at least 62% earns the remaining full training seeds.

#### 4. Execution

The controller wrote `short_validation_complete` and `full_run_promoted` at
07:37:20 BST, then launched PID 2751529 on GPU 4 with batch 256/256 and
`xla_mem_fraction=0.45`. Its first new CSV row is global step 20k, proving
resume from the durable 20k snapshot rather than a zero-step restart. GPU 4
also hosts one pre-existing external training process; the two processes use
about 30.2 GiB total, while the Stage-37 process is hard-sliced and currently
uses about 12.9 GiB. Both are computing without OOM or CUDA errors. ETA will be
derived from the first steady-state post-resume 1k interval.

#### 25k execution milestone (2026-08-01 07:54 BST)

The resumed full run wrote `25000_snapshot.pkl` (448,855,407 bytes) at
07:53:48 BST and `train.csv` reached global step 25k while PID 2751529
remained active. The raw 20k--25k interval took about 901 seconds, or 5.55
environment steps/s, with GPU 4 at 94% utilization and no fatal, CUDA, or OOM
entry in the full-run log. At that measured rate the remaining 76k training
steps have an ETA of about 3 hours 48 minutes (approximately 11:42 BST), after
which the controller will run validation and held-out evaluation without
sharing the GPU with training. This checkpoint is an execution-health result,
not policy-quality evidence; the registered evaluation protocol above remains
unchanged.

At 07:55 BST a live process audit found that GPU 4's second, unrelated
training process could outlive Stage 37 training. Because the original runner
would start evaluation immediately after its own child exited, controller PID
2467103 was stopped while training PID 2751529 was left running, and a detached
guard was installed to resume the controller only after the 101k child exits
and GPU 4 reports no compute processes. The runner was also amended with the
same idle-GPU gate for future launches. This changes only evaluation resource
isolation; training, checkpoints, seeds, and the registered metrics are
unchanged.

The first registered long-curve checkpoint, raw 30k, was written at 08:08:49
BST (448,855,406 bytes) with a matching `train.csv` row. Raw 20k--30k took
1,801.9 seconds (5.55 steps/s), confirming the earlier ETA rather than a
transient interval estimate. Training PID 2751529 remained active at 132% CPU,
GPU 4 was 98% utilized, and the fatal-log scan remained empty. The raw-30k
policy result stays unresolved until the post-training fixed validation sweep.

### Stage 37 full-run interruption recovery (2026-08-01 08:25 BST)

#### 1. Previous-stage result

The first full-run process stopped after writing its raw-33k CSV row and before
the registered raw-35k checkpoint. Its last durable artifact is the complete
`32500_snapshot.pkl` (448,855,406 bytes). The process log contains no Python
traceback, CUDA error, OOM, or numerical failure; its final entry is only the
multiprocessing resource tracker's interpreter-shutdown warning. The stopped
controller disappeared at the same time, and neither a completion nor a
training-failure marker was written.

#### 2. Interpretation

This is an infrastructure/process-lifecycle interruption, not evidence that
the No-BC policy failed or that training diverged. The available artifacts do
not identify who delivered the terminating signal, so the cause is not stated
more narrowly. Raw 32.5k is recoverable; the uncheckpointed interval after it
must not be treated as durable progress or policy evidence.

#### 3. Next-stage decision

Resume the same pre-registered seed 1, objective, batch 256/256, replay,
0.45-device slice, and raw-101k budget from the latest durable checkpoint.
Do not change the validation or held-out splits, metrics, or 62% pass line.
Avoid stopping the recovery controller; after training it must select an
otherwise idle GPU with at least 2 GiB free and three consecutive idle checks
before evaluation.

#### 4. Execution

`scripts/resume_cqn_no_bc_stage37_full.sh` launched controller PID 2801789 and
training PID 2801797 on GPU 4 at 08:21:57 BST. The first post-launch CSV write
at 08:25 reached raw 33k with updated total time, proving continuation from the
32.5k snapshot rather than a zero-step restart. GPU utilization was 98%, the
training process used about 14.4 GiB within its 0.45 slice, and the recovery
fatal-log scan was empty. The recovery controller will prefer idle GPU 5 for
evaluation and search the remaining cards if it is occupied at training
completion.

The recovery then crossed both raw 35k and 37.5k and wrote the registered
`40000_snapshot.pkl` (448,855,407 bytes) at 08:45:11 BST with a matching 40k
CSV row. Raw 35k--40k took 867.4 seconds (5.76 steps/s). Both controller and
training child remained live, GPU utilization was 99%, and the fatal-log scan
remained empty. This closes the recovery-health check; policy quality at 40k
still awaits the fixed post-training validation sweep.

The next registered long-curve checkpoint, `50000_snapshot.pkl`, was written
at 09:20:46 BST (448,855,408 bytes) with a matching 50k CSV row. Raw
40k--50k took 2,135.5 seconds (4.68 steps/s), slower than the earlier interval
because GPU 4 continued to share compute with the unrelated training process,
but both controller and training child remained live and no fatal/CUDA/OOM
entry appeared. At this measured recent rate, the remaining 51k training steps
have an ETA near 12:23 BST. Policy quality at raw 50k remains sealed until the
fixed post-training sweep.

The registered `60000_snapshot.pkl` was written at 09:50:34 BST
(448,855,412 bytes) with a matching 60k CSV row. Raw 50k--60k took 1,787.8
seconds (5.59 steps/s), showing that the earlier shared-card slowdown did not
persist across the whole interval. The controller and training child remained
live, GPU utilization was 98%, and the fatal-log scan was empty. At this rate
the remaining 41k training steps have an ETA near 11:53 BST; the raw-60k policy
result remains sealed for the post-training fixed validation sweep.

The registered `70000_snapshot.pkl` was written at 10:21:25 BST
(448,855,417 bytes) with a matching 70k CSV row. Raw 60k--70k took 1,851.2
seconds (5.40 steps/s). Controller and training child remained live, GPU
utilization was 98%, and the fatal-log scan remained empty. The remaining 31k
training steps therefore have an ETA near 11:57 BST; policy quality at raw 70k
continues to be reserved for the fixed post-training validation sweep.

The registered `80000_snapshot.pkl` was written at 10:54:51 BST
(448,855,417 bytes) with a matching 80k CSV row. Raw 70k--80k took 2,005.7
seconds (4.99 steps/s) while the second GPU-4 process again consumed most of
the remaining card capacity. Controller and training child stayed live, GPU
utilization was 99%, and no fatal/CUDA/OOM entry appeared. The remaining 21k
training steps have a conservative ETA near 12:05 BST; the raw-80k policy
result remains sealed for the post-training fixed validation sweep.

The registered `90000_snapshot.pkl` was written at 11:33:54 BST
(448,855,417 bytes) with a matching 90k CSV row. Raw 80k--90k took 2,343.2
seconds (4.27 steps/s) under continued GPU-4 contention. Controller and
training child remained live, GPU utilization was 98%, and no fatal/CUDA/OOM
entry appeared. The final 11k training steps have an ETA near 12:17 BST; the
raw-90k policy result remains sealed for the fixed post-training sweep.

### Stage 37 full training complete and isolated evaluation recovery (2026-08-01 12:21 BST)

#### 1. Previous-stage result

The promoted seed-1 run completed the registered raw-101k budget and wrote
`101000_snapshot.pkl` (448,855,416 bytes) at 12:16:53 BST. The recovery
controller wrote `full_training_complete` at 12:16:56; no training-failure,
fatal, CUDA, or OOM artifact was produced. The descriptive validation evidence
available before the final sweep remains 5k 52%, 20k 42%, 30k 44%, and 40k
42% on 50 episodes with seeds 400--449.

#### 2. Interpretation

The full training budget is now complete and the policy has not collapsed to
an all-zero run through the measured 40k point, but neither training health nor
the interim plateau establishes raw-101k game performance. The first final
evaluation attempt selected GPU 5 after three idle checks, but a distinct
Stage-38 training process started on that card about five seconds later. The
Stage-37 evaluator was terminated before it wrote any result row, so no
shared-GPU measurement is admitted.

#### 3. Next-stage decision

Keep the registered evaluation protocol unchanged: first evaluate raw
20/30/40/50/60/70/80/90/100/101k on validation seeds 400--449 for 50 episodes,
then evaluate only raw 101k on sealed held-out seeds 800--999 for 200 episodes.
The decisive pass line remains the official same-seed 62% endpoint. Every
evaluation attempt must start on an idle GPU and be aborted/retried if any
foreign GPU process appears; completed CSV rows may be reused by the sweep
script on retry.

#### 4. Execution

`scripts/resume_cqn_no_bc_stage37_eval.sh` launched isolated-eval supervisor
PID 3148343. Its first full-validation attempt selected GPU 2 after three idle
checks and launched evaluator PID/session 3149106 at 12:20:43 BST. Live NVML
and process-session inspection showed that PID 3149106 was GPU 2's only compute
process. The supervisor checks every five seconds and kills the evaluator's
entire process group on any foreign PID before retrying. Results append to
`val50_seeds400_full.csv`; the held-out sweep and frozen full summary follow
only after validation completes.

### Stage 37 interim 30k/40k validation launch (2026-08-01 08:48 BST)

#### 1. Previous-stage result

The registered short-run result remains the three-seed validation-selected
52%/50%/48% (mean 50%) result on seeds 400--449. The full-run training process
has reached raw 40k and written a complete 448,855,407-byte checkpoint, but its
training CSV is not policy-quality evidence. No post-20k fixed evaluation was
available at launch time.

#### 2. Interpretation

The durable 30k and 40k checkpoints permit a non-held-out scale check while
training continues on GPU 4. Evaluating them on the existing validation split
does not change checkpoint selection and does not open held-out seeds
800--999. It can establish whether the early nonzero policy survives to 40k,
but cannot establish final endpoint parity.

#### 3. Next-stage decision

Evaluate raw 30k and raw 40k with the frozen 50-episode seeds-400--449 split.
Write a separate interim CSV so the registered post-training full sweep remains
unchanged. Continue training to raw 101k regardless of these descriptive
interim values; final acceptance remains the fixed 200-episode raw-101k
held-out endpoint on seeds 800--999 versus the 62% same-seed reference.

#### 4. Execution

GPU 0 was verified idle with no compute PID before launch. Detached evaluator
PID 2852713 is running on physical GPU 0 with 25 eval environments,
`XLA_PYTHON_CLIENT_PREALLOCATE=false`, and only raw steps 30000 and 40000.
Results append to `val50_seeds400_interim_30k40k.csv`; launch diagnostics are
in `stage37_dense_b256_20260801021235/seed1_interim_val50_30k40k_gpu0.log`.
The process has allocated GPU-0 memory and emitted environment initialization
warnings, proving startup; no success row existed yet at this milestone.

### Stage 37 interim result and reward-gated dense-Q decision (2026-08-01 09:21 BST)

#### 1. Previous-stage result

The GPU-0 evaluator completed both registered descriptive points on the same
50 validation episodes, seeds 400--449: raw 30k is 44% (22/50) and raw 40k is
42% (21/50). Together with the existing seed-1 curve, the scale trajectory is
5k 52% selected-best, 20k 42%, 30k 44%, and 40k 42%. The full-run training
child remains live and has written a complete raw-50k checkpoint.

#### 2. Interpretation

This rules out the strong long-run-collapse hypothesis through raw 40k. The
method has reached a roughly 40--50% validation plateau with an early selected
peak, not a monotonic improvement curve. Falling critic loss and a persistent
chosen-minus-unseen Q gap therefore do not explain policy quality. Historical
matched evidence also rejects fixed neighbor smoothing (-2pp mean), expected-Q
regression (-9pp), reward scaling (-1pp), and advantage-gap expansion (-45pp)
as repeatable improvements.

The remaining untested objective mismatch is that every zero-return transition
currently sends all counterfactual bins to zero. This is action-label invariant
but supplies many unsupported negative counterfactual targets. A narrower
reward-only hypothesis is to retain the dense all-bin target only when the
completed trajectory has positive return, and otherwise use canonical
chosen-action C51 TD. The gate depends only on reward-to-go and applies equally
to demo and online success; it does not inspect demo identity or add BC.

#### 3. Next-stage decision

Do not perturb the running Stage-37 full run or conflate this hypothesis with
Stage 38's already registered exact-baseline offline-to-online question. First
complete Stage 38's fixed coarse curve. If it remains below parity, test the
reward-gated dense target as one isolated field against the corresponding
dense control with training seeds 1/2, batch 256+256, identical clocks and
seeds-400--449 50-episode selection. A mechanism pass requires both seed
deltas nonnegative and mean validation-best gain at least 5pp; only then add
seed 3. Held-out seeds 800--999 remain sealed.

#### 4. Execution

Stage 37 continues unchanged toward raw 101k and Stage 38 has recovered its
online boundary on GPU 5. An attempted raw-50k validation launch on otherwise
idle GPU 0 was stopped after a read-only JAX probe reproduced
`cuInit: CUDA_ERROR_NOT_INITIALIZED`; it had fallen back to CPU and produced no
policy row. No training process or checkpoint was affected. The raw-50k point
will be retried only after a fresh GPU-backend probe succeeds. A new
reward-gated training wave is intentionally not launched while Stage 38 is
answering the distinct matched question and no isolated training card is free.

## Stage 38: baseline-matched offline dense reward-Q gate (2026-08-01)

### 1. Previous-stage result

Stage 36 completed its exact official-batch offline-to-online gate with a
fixed treatment curve of 0/20 at raw 10k, 12.5k, 15k, 17.5k, and 20k. The
runner correctly wrote `early_futility_stop`; no matched 50-episode sweep or
long extension was launched. This is 0/100 fixed-policy evidence, not a loss
curve.

The distinct Stage-37 online-only dense construction has positive scaling
evidence and must not be conflated with that rejection. Its three complete
50-episode curves select 52%, 50%, and 48% (mean 50%), and two training seeds
remain at least 40% at a late 17.5k/20k point. It therefore legitimately
earned its registered seed-1 scale extension under the user's rule that a
supported curve may run longer. However, it retains the historical
`critic_lambda=1` and has no offline phase, so its eventual endpoint is scale
evidence for the dense mechanism rather than the final exact-baseline claim.

### 2. Interpretation

Stage 36 rules out official batch size, replay-continuation maximization, an
MC lower bound, and 10k demo-only Q warmup as a sufficient combination. Prior
fixed-state probes on the same construction already found the characteristic
failure mode: demo return calibration becomes good while
`Q(expert)-Q(greedy)` remains negative and expert-bin top-1 stays modest. The
missing operation is action coverage, not another return horizon.

`dense_return_q_target` addresses that boundary inside one categorical Q
objective. The replayed action bin receives its reward/Bellman/MC target and
unseen bins receive the task failure return. When the observed return is zero,
changing the replay action leaves the complete loss and all gradients
identical; positive reward is required before action-dependent separation
appears. Thus the signal is reward credit assignment, not demo identity,
margin, likelihood, or action regression. It is nevertheless reported
explicitly as a pessimistic dense Q target rather than claiming that demo and
online transitions are treated identically.

### 3. Next-stage decision

Stage 38 adds exactly one resolved field to the failed Stage-36 No-BC
treatment: `method.dense_return_q_target=true`. Everything else remains the
official baseline setting, including batch 256+256, single dueling critic,
K=16, nstep 1, critic learning rate 5e-5, target tau 0.02,
`critic_lambda=0.1`, one update per step, and noise 0.01. Both fresh training
seeds receive 10k demo-only reward-Q updates followed initially by only 10k
online interactions.

The coarse selection split is seeds 400--419 with 20 episodes at raw
10k/12.5k/15k/17.5k/20k. Both training seeds must have a nonzero point and at
least one must reach 20% at 17.5k or 20k before any 50-episode sweep is
allowed. A pass opens seeds 400--449 for the two dense seeds, the already
trained exact Stage-36 No-BC control, and its BC reference. Earliest-best
selection is fixed. A separately launched 20k-online extension additionally
requires dense mean best at least 40%, both dense seeds late at least 20%, and
a positive seed-1 gain over the exact Stage-36 No-BC control. The runner has
no 50k/101k training path, and held-out seeds 800--999 remain sealed.

### 4. Execution

Added
`cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate.yaml`,
`run_cqn_no_bc_stage38.sh`, and
`summarize_cqn_no_bc_stage38.py`. A flattened config test proves the Stage-38
launch differs from the Stage-36 No-BC treatment only at
`method.dense_return_q_target`; all official batch/model/optimizer fields and
strict no-imitation guards are asserted. Seven focused tests pass, covering
the single finite dense reward-Q update, exact config isolation, two-seed
coarse futility logic, earliest-best matched selection, late scaling, and
evaluation protocol rejection. Shell syntax, Python compilation, and scoped
`git diff --check` pass. A read-only common-state value/rank diagnostic is
also registered for the Stage-36 raw-10k and raw-20k treatment snapshots and
the raw-10k BC control; it uses CPU and cannot consume an environment
selection or held-out seed.

The Stage-36 diagnostic initially used the historical common replay anchors
for state/action comparability, then exposed a reward-semantics mismatch that
must not be hidden. The historical replay has one positive reward per success;
the exact current/official replay has approximately 23--27 consecutive
positive rewards after success. On the historical labels, the Stage-36
treatment is already saturated at the C51 upper atom: raw-10k/20k demo Q
means are 1.99997/1.999995, RTG MAE is 1.527/1.527, Pearson is
-0.315/-0.049, and candidate spans are 0.00558 and 0.0000032. The matched BC
control at raw 10k has Q mean 0.267, span 0.578, and current-action expert-bin
top-1 88.3%. These cross-reward calibration numbers are diagnostic of
saturation but are not reported as native-return calibration. The probe now
supports an explicit demo-only group selection and is rerunning raw 10k/20k
on the exact current replay/reward convention.

Stage 38 launched durably on physical GPU 5 at 08:30:24 BST under controller
PID 2815642 in
`exp_local/cqn_no_bc/stage38_offline_dense_b256_gpu5_20260801083024`.
Seed-1/2 PIDs 2815659/2821671 are simultaneously executing the 10k demo-only
phase with batch 256 and a 120-second stagger. At the first two-process check
they are at 2.2k/0.4k updates, both have force fraction 1.0, MC-lower-bound
fraction 0.95--0.98, and finite `critic_loss=dense_return_q_loss` with no BC,
policy, or auxiliary-loss column. Crucially, chosen-minus-unseen Q gaps are
already 1.17/0.60 instead of the all-bin saturation seen in Stage 36. The two
processes use 22.7 GiB at 96% GPU utilization with no OOM/failure marker.
Measured offline rates are about 7 updates/s each; the first coarse fixed
scaling decision is estimated for 09:55--10:15 BST and will supersede this
estimate when the online phase begins.

The exact-current-replay probes then completed and resolve the apparent
calibration contradiction. Successful-demo sampled RTG is 11.65 because the
official reward remains one for many post-success steps, while C51 support is
[-2, 2]. Stage-36 raw-10k/20k Q means are 1.999981/1.999997 and their MAE to
the support-clipped target is only 1.94e-5/3.28e-6: the critic has fitted the
target it is actually given. The failure is action collapse. Candidate span
shrinks from 0.00554 to 2.49e-6, top-two gap from 0.000620 to 9.39e-7, and
`Q(expert)-Q(greedy)` from -5.95e-6 to -1.12e-6. Raw-20k current-action
expert top-1 rises to 74.7%, but with all five bins numerically tied at the
upper support this is not a usable ranking signal. Stage 38 is therefore a
direct test of whether reward-derived unseen-bin failure targets preserve a
non-degenerate action gap and convert it into fixed rollout success.

### Stage 38 matched offline midpoint (2026-08-01 08:46 BST)

#### 1. Previous-stage result

Both fresh Stage-38 seeds have now written complete raw-5k demo-only
checkpoints. At the exact 5k rows, seed 1/2 have chosen-minus-unseen Q gaps
of 1.5103/1.4726 and the only optimization loss,
`critic_loss=dense_return_q_loss`, is 0.05643/0.06177. Over the matched
4.5k--5k windows the gaps are 1.4807/1.4847, chosen Q is 1.5737/1.5774, and
unseen Q is 0.0930/0.0927. Behavior-candidate and force fractions are 1.0
for both seeds, while MC-lower-bound fractions are 0.9670/0.9689. The
checkpoints are 448,855,387 and 448,855,398 bytes, respectively. There is no
BC, policy, FOSD, margin, or auxiliary-loss column. This is optimizer/value
evidence only; no Stage-38 policy evaluation has yet been run.

Separately, the already scaling-qualified legacy Stage-37 seed-1 extension
has written its complete raw-40k checkpoint. Its training process and the
Stage-38 controller/two workers remain live, GPU 4/5 utilization is 98%/97%,
and the fatal-log scan is empty.

#### 2. Interpretation

The two-seed midpoint establishes that the dense reward-Q target reproducibly
prevents the Stage-36 all-bin saturation during offline reward learning. It
rules out the earlier seed-1 gap as a one-seed initialization accident. It
does not establish that the greedy C2F policy can execute MovePlate, that the
gap survives online replay, or that performance scales; finite loss and Q
separation are not policy quality.

#### 3. Next-stage decision

Keep the frozen Stage-38 hypothesis and all matched settings unchanged:
batch 256/256, official single-dueling CQN-AS, 10k demo-only updates followed
by only 10k online interactions, and dense reward-derived action coverage as
the sole difference from the exact Stage-36 No-BC treatment. The matched
references remain the Stage-36 No-BC seed 1 and BC control seed 1. The
selection split remains seeds 400--419 for the five-point 20-episode coarse
curve and then seeds 400--449 only on a coarse pass; held-out seeds 800--999
remain sealed. Full-run promotion is prohibited at this stage. The pass and
fail criteria remain exactly those preregistered above, including immediate
rejection for an all-zero/weak curve.

#### 4. Execution

Both workers continued beyond raw 5k without restart. The controller is
still responsible for completing raw 10k offline, restoring each exact
checkpoint into the online phase with force probability zero, and producing
the fixed five-checkpoint curves. Based on the observed 6.3--7.0 updates/s,
offline completion remains expected near 09:00 BST and the first policy
scaling decision near 09:55--10:15 BST. The next report will verify the
phase transition from process arguments, `train.csv`, snapshots, and markers
before interpreting any rollout result.

### Stage 38 offline completion and EGL-safe online recovery (2026-08-01 09:01 BST)

#### 1. Previous-stage result

Both baseline-matched Stage-38 seeds completed all 10k demo-only updates and
wrote intact `10000_snapshot.pkl` files of 448,855,387 and 448,855,397 bytes.
Over their final logged 9.5k--9.9k windows, seed 1/2 have almost identical
chosen-minus-unseen Q gaps of 1.67813/1.67804, chosen Q of 1.73032/1.73064,
unseen Q of 0.05219/0.05261, and sole dense C51 losses of
0.03489/0.03405. Force and behavior-candidate fractions remain exactly 1.0.
This confirms completion of the registered offline phase but is still not a
policy-quality result.

The first automatic online restart failed for both seeds before workspace or
checkpoint loading with `Cannot initialize a EGL device display`. The
failure is isolated to fresh MuJoCo/EGL context creation: both checkpoint
files and copied offline configs predate and survive it. GPU 5 reports
`GPU Recovery Action: None`, no Xid was logged, and a standalone renderer
probe reproduces the EGL error. The original controller therefore correctly
wrote `training_failed` and exited without consuming any online training
steps.

The independent Stage-37 legacy extension has meanwhile written raw-42.5k
and logged raw 44k. It remains descriptive scaling work only; online episode
rewards are not substituted for its future fixed evaluation.

#### 2. Interpretation

Stage 38 establishes reproducible offline reward-Q action separation through
the complete 10k budget, ruling out the Stage-36 all-bin saturation in both
fresh seeds. What remains wholly unresolved is whether that separation
survives mixed online replay and produces a nonzero MovePlate scaling curve.
The failed transition is infrastructure evidence, not evidence for or against
the method, because execution stopped during import before loading either
policy.

#### 3. Next-stage decision

Resume both seeds from the exact 10k snapshots only after a same-card GPU-5
EGL renderer probe succeeds. Keep the registered online phase unchanged:
batch 256/256, `demo_only_updates=false`, force probability zero, raw-20k
hard cap, 120-second two-run stagger, and no extra objective. Then execute the
same seeds-400--419 five-checkpoint coarse curves. The all-zero/weak futility
rule, matched references, validation split, held-out seal, and prohibition on
full-run promotion remain unchanged.

#### 4. Execution

Added `scripts/resume_cqn_no_bc_stage38_after_egl.sh`. It validates both 10k
snapshots and the offline-complete marker, probes EGL on the assigned GPU,
retries only the specific pre-import EGL failure, resumes to exactly raw 20k,
and reconnects to the original coarse and conditional matched-validation
gates. Static validation passes: `bash -n`, scoped `git diff --check`, and
command inspection confirm batch 256/256, force zero, the five registered
10k--20k checkpoints, and absence of 50k/101k or held-out paths.

The durable recovery controller is PID 2875330 and began at 09:00:21 BST. Its
first GPU-5 renderer probe failed and wrote `egl_waiting`; it now probes every
60 seconds without allocating training memory or consuming steps. There is
no reliable wall-clock ETA while EGL is externally unavailable. Once the
probe succeeds, the prior throughput implies roughly 60--80 minutes to the
first coarse policy decision, including paired online training and fixed
evaluation.

### Stage 38 verified online boundary crossing (2026-08-01 09:13 BST)

#### 1. Previous-stage result

GPU-5 EGL became available on the recovery controller's ninth probe at
09:08:22 BST. Seed 1 then loaded the existing run at global step 10k and
wrote its first post-boundary row at raw 11k. Its effective online config is
the registered one: `num_train_frames=20000`, batch 256/256,
`demo_only_updates=false`, force probability zero, `critic_lambda=0.1`, one
update per step, dense reward-Q enabled, and BC/FOSD/margin absent. The raw
11k row reports force fraction zero, behavior-candidate fraction 0.2207,
chosen-minus-unseen Q gap 1.60375, and
`critic_loss=dense_return_q_loss=0.05315`. The process uses about 15.1 GiB and
GPU 5 is at 94% utilization.

Seed 2's first launch and three renderer retries have not yet obtained a
second EGL display while seed 1 and external graphics contexts are active.
It remains exactly at the intact raw-10k checkpoint and has consumed zero
online steps.

#### 2. Interpretation

Seed 1 proves that the offline snapshot restores without resetting either
the global clock or objective and that the intended offline-to-online switch
is operational. Its positive action gap survives the first 1k online steps,
but neither this minibatch metric nor the single successful training episode
is a fixed-policy scaling result. Seed 2's delay is still pre-import
infrastructure contention and provides no method evidence.

#### 3. Next-stage decision

Continue seed 1 to the frozen raw-20k cap and let seed 2's recovery worker
retry only same-card EGL initialization until it can resume the identical
phase. Do not move rendering to another GPU, change the 120-second stagger,
rerun offline updates, or open evaluation early. Coarse evaluation still
waits for both complete raw-20k curves, and every promotion criterion remains
unchanged.

#### 4. Execution

Seed-1 Python PID 2888424 and recovery controller PID 2875330 are live.
Seed 2's retry worker is also live and records each unavailable probe in
`egl_probe_history.log`. At seed 1's observed initial rate, its raw-20k
checkpoint is expected around 09:33--09:38 BST. The two-seed training and
coarse-evaluation ETA remains conditional on when the second EGL context
becomes available; no unsupported full run has been launched.

### Stage 38 capped training and coarse scaling result (2026-08-01 10:01 BST)

#### 1. Previous-stage result

Both exact Stage-38 arms completed the registered 10k offline plus 10k online
budget and wrote intact raw-20k snapshots of 448,855,393 and 448,855,403
bytes. Their copied online configs confirm batch 256/256,
`demo_only_updates=false`, force probability zero, `critic_lambda=0.1`, one
update per step, dense reward-Q enabled, and no BC/FOSD/margin. At the final
logged raw-19k rows the chosen-minus-unseen Q gaps remain positive at
1.3159/1.2112; these diagnostics are not used for policy selection.

The frozen 20-episode seeds-400--419 coarse curves are:

- seed 1: 55/40/50/60/55% at raw 10/12.5/15/17.5/20k;
- seed 2: 50/35/35/40/45% at the same checkpoints.

The coarse summarizer selects seed 1 raw 17.5k at 60% and seed 2 raw 10k at
50%. Both seeds are nonzero and their late best values are 60%/45%, so
`coarse_qualification_pass=true`. The raw-10k checkpoints are the outputs of
the pure demo-only reward-Q phase and already achieve 55%/50% without any
online interaction or imitation gradient.

#### 2. Interpretation

This is the first exact-baseline, batch-256 offline-to-online construction in
this study that decisively rules out the all-zero No-BC failure on two fresh
training seeds. It establishes that reward-derived dense action coverage can
turn expert trajectories into an executable greedy CQN-AS policy without BC.
It does not yet establish parity with the BC baseline: 20 episodes are noisy,
checkpoint selection is optimistic, and the online portion does not improve
monotonically over the strong offline checkpoint.

#### 3. Next-stage decision

Advance exactly as preregistered to 50-episode seeds-400--449 selection for
both dense seeds, followed by the already-trained exact Stage-36 No-BC and BC
references on the identical five checkpoints. Earliest-best tie breaking
stays fixed. A bounded 20k-online extension is eligible only if the dense
mean selected best is at least 40%, both seeds retain a late point of at least
20%, and seed 1 improves over the exact Stage-36 No-BC control. Held-out
seeds 800--999 remain sealed and `automatic_full_run=false`; even a matched
selection pass does not authorize raw 101k directly.

#### 4. Execution

The recovery controller wrote `coarse_validation_complete` and the frozen
`stage38_coarse_summary.json`, then launched candidate-selection PIDs
2976031/2976032 on GPU 5 at 10:00:42 BST. Each evaluates only the five
registered checkpoints for 50 episodes with seeds 400--449. Based on the
measured coarse evaluator time, candidate rows are expected by roughly
10:14 BST and the subsequent matched baseline wave by roughly 10:27 BST. No
full or held-out evaluation process exists.

## Stage 39: positive-return-gated dense reward-Q (2026-08-01 09:37 BST)

### 1. Previous-stage result

The completed Stage-37 interim validation gives raw 30k 44% (22/50) and raw
40k 42% (21/50) on seeds 400--449. Its validation-selected seed-1 and seed-2
short-curve best checkpoints remain raw 5k at 52% and 50%, respectively, for
a locked two-seed baseline mean of 51%. Thus the dense online-only objective
has a stable approximately 40--50% policy plateau rather than a long-run
collapse; decreasing training loss is not used as policy-quality evidence.

The attempted raw-50k evaluation failed before policy rollout when JAX logged
`cuInit(0): CUDA_ERROR_NOT_INITIALIZED` and fell back to CPU. GPU 0 had no
compute process, required no recovery, and logged no kernel Xid. A subsequent
fresh process on the same card reported the GPU backend, enumerated
`CudaDevice(id=0)`, and completed a device-side sum. The failure was therefore
a transient CUDA-context initialization failure during simultaneous system
JAX/EGL startup pressure, not a Stage-37 policy result or evidence of a GPU-0
hardware reset condition.

### 2. Interpretation

Stage 37 rules out catastrophic scaling collapse through raw 40k but leaves
an early peak and no improvement beyond the 40--50% band. The remaining
objective mismatch tested here is that the original dense target labels every
counterfactual action bin even on zero-return trajectories. Those negative
counterfactual labels are not supported by the executed trajectory. Stage 39
uses the same dense all-bin C51 target only when sampled reward-to-go is
positive; when it is zero, it falls back to canonical chosen-action C51. The
gate reads reward only, applies equally to demo and online data, and adds no
BC, margin, actor, imitation, or demo-identity signal. Finite initial losses
establish wiring only and do not establish better policy quality.

### 3. Next-stage decision

The isolated hypothesis is `dense_return_positive_only=true`; all other
method and training settings match the Stage-37 dense control. Train seeds 1
and 2 for raw 20k with batch 256+256, selecting independently over raw
2.5k, 5k, ..., 20k using 50 validation episodes and seeds 400--449. Compare
each treatment seed's selected best against the corresponding Stage-37
selected best (52% and 50%). Pass requires both seed deltas nonnegative and a
treatment mean of at least 56%, i.e. at least +5 percentage points over the
locked 51% mean. Any negative seed delta or mean below 56% fails. Only a pass
adds seed 3; held-out seeds 800--999 remain sealed.

### 4. Execution

Implemented the reward-only gate in the C51 update, configuration plumbing,
launch config, fixed sweep runner, and result summarizer. Focused unit tests
verify that a zero-return batch is parameter-identical to canonical C51, a
positive-return batch is parameter-identical to the original dense update,
invalid flag combinations are rejected, and the Stage-39 launch differs from
the control only by the registered gate; all eight focused checks pass across
the two invocations, and scoped compile, shell syntax, and diff checks pass.

The first launch exposed an independent runner error: without `MUJOCO_GL=egl`,
headless MuJoCo selected GLFW and exited with `gladLoadGL error` before
training. The runner now explicitly uses EGL, matching Stage 37, and the clean
durable restart is
`exp_local/cqn_no_bc/stage39_positive_return_dense_gpu0_20260801093310`.
Controller PID 2929895 launched seed-1 PID 2929899 and, after the registered
120-second stagger, seed-2 PID 2934283 on GPU 0. Both have finite first update
rows; seed 1 has crossed raw 1k with critic loss 3.1537 and positive-return
fraction 0.8809, while seed 2's initialization row has critic loss 18.4304 and
positive-return fraction 0.9219. The two processes use approximately
12.9 GiB each (26.1 GiB total), GPU utilization is 79%, and the fatal-log scan
is empty. At the measured early paired rate, training should finish in roughly
55--75 minutes; the sequential 16-point fixed validation sweep will then take
approximately four additional hours before the Stage-39 pass/fail decision.

### Stage 39 training completion and seed-1 validation progress (2026-08-01 11:15 BST)

#### 1. Previous-stage result

Both Stage-39 training seeds completed the registered raw-20k budget and each
wrote all eight 2.5k-spaced checkpoints through `20000_snapshot.pkl` (about
448.9 MB). Their last logged raw-19k rows remain finite: seed 1/2 critic loss
is 1.1929/1.1980, positive-return dense fraction is 0.7578/0.7754, and
chosen-minus-unseen Q gap is 0.3766/0.3714. These are training-health metrics,
not policy quality.

The fixed seed-1 validation has completed its first two checkpoints on seeds
400--449: raw 2.5k is 46% (23/50) and raw 5k is 58% (29/50). The current
validation-selected interim best is therefore raw 5k at 58%, which is +6pp
against the corresponding Stage-37 seed-1 selected best of 52%.

#### 2. Interpretation

The completed training artifacts establish that the reward gate runs stably
and reproducibly through the full budget. Seed 1's first two policy points are
encouraging and already clear its per-seed nonnegative-delta requirement, but
they do not establish the Stage-39 pass: six seed-1 checkpoints and all eight
seed-2 checkpoints remain unevaluated, and the gate requires both seed deltas
nonnegative plus treatment mean at least 56%. No held-out result has been
opened.

#### 3. Next-stage decision

Keep the registered selection sweep unchanged. Finish all eight seed-1 points,
then all eight seed-2 points sequentially on GPU 0; select each seed at its own
best validation checkpoint and apply the frozen comparison against 52%/50%.
Do not add seed 3 or run seeds 800--999 unless the complete two-seed gate
passes.

#### 4. Execution

The durable controller remains live and has written `training_complete`.
Seed-1 evaluator PID 3022170 is active on GPU 0 with about 1.6 GiB allocated;
the fatal/CUDA/OOM scan is empty. Each completed point took about 13.7 minutes.
At that measured rate, seed 1 should finish around 12:37 BST and the complete
two-seed selection plus summary around 14:25--14:35 BST.

### Stage 39 seed-1 near-complete selection curve (2026-08-01 12:20 BST)

#### 1. Previous-stage result

Seven of seed 1's eight fixed validation points are complete. At raw
2.5k/5k/7.5k/10k/12.5k/15k/17.5k the success rates are respectively
46/58/46/46/48/52/44% over the same 50 episodes, seeds 400--449. Its interim
validation-selected best remains raw 5k at 58%, +6pp over the matched
Stage-37 seed-1 selected best of 52%.

#### 2. Interpretation

The improvement is localized to the early raw-5k checkpoint rather than a
uniform upward shift of the whole curve. Seed 1 therefore satisfies its
nonnegative-delta arm so far, but the method-level gate remains unresolved
until raw 20k and all seed-2 points complete. Training losses are not used in
this policy comparison.

#### 3. Next-stage decision

Finish the already-running raw-20k seed-1 point and then run the unchanged
eight-point seed-2 sweep. Select seed 2 independently and apply the frozen
two-seed mean-at-least-56% rule; keep seed 3 and held-out seeds sealed.

#### 4. Execution

Seed-1 evaluator PID 3022170 is actively evaluating raw 20k on GPU 0 with
about 1.6 GiB allocated. Seven points each took 13.7--13.8 minutes and the
error scan is empty. Seed 1 is expected to finish near 12:30 BST; the full
two-seed summary is expected near 14:20 BST.

## Stage 38 matched selection and bounded scale extension (2026-08-01 10:38 BST)

### 1. Previous-stage result

The complete 50-episode seeds-400--449 curves at raw
10/12.5/15/17.5/20k are now:

- dense reward-Q seed 1: 52/52/56/58/50%, earliest selected best 58% at
  raw 17.5k;
- dense reward-Q seed 2: 50/40/50/38/42%, earliest selected best 50% at
  raw 10k;
- exact Stage-36 No-BC seed 1: 0/0/0/0/0%, selected best 0%;
- matched BC seed 1: 58/66/58/60/56%, selected best 66% at raw 12.5k.

The dense two-seed selected-best mean is 54%. Both dense late best values
remain nonzero at 58%/42%, and the matched seed-1 improvement over the exact
No-BC control is +58 percentage points. The frozen summarizer therefore
wrote `eligible_for_20k_online_extension=true`. All five fixed No-BC control
points are zero, so that failed construction remains stopped and receives no
larger budget.

### 2. Interpretation

This establishes with the matched 50-episode selection split that the one
reward-Q change, rather than loss-scale or batch differences, converts the
all-zero No-BC CQN-AS into a useful policy. It also rules out the 20-episode
coarse result as a pure small-sample accident. It does not establish parity:
the directly matched seed-1 candidate is 8 points below the BC selected best,
and the two-seed candidate mean cannot be treated as a matched two-seed BC
comparison. The strongest checkpoints are early, with the pure-offline raw
10k endpoints already at 52%/50%, so more online updates need an explicit
scaling test rather than an assumed benefit.

### 3. Next-stage decision

Test only whether the same exact offline-to-online recipe remains useful or
improves over a second 10k online block. Resume the existing two seeds from
their raw-20k snapshots to raw 30k, preserving batch 256+256, all baseline
architecture/optimizer/exploration settings, and the same single reward-Q
objective. Evaluate only raw 22.5/25/27.5/30k with 50 episodes and the same
selection seeds 400--449, then combine them with the frozen initial curve
using earliest-best tie breaking. The matched references remain the exact
Stage-36 No-BC and BC curves above; seeds 800--999 stay sealed.

The bounded extension supports a separately designed 50k scaling sentinel
only if both seeds have an extension-region best of at least 40%, their mean
extension-region best is at least 50%, and their combined-curve mean best is
not below the frozen initial 54%. Failure stops Stage 38 at raw 30k. Passing
does not authorize 101k: the extension summary hard-codes
`eligible_for_full_run=false`, and a new fixed scaling gate is required before
any full run.

### 4. Execution

Added `run_cqn_no_bc_stage38_bounded_extension.sh` and
`summarize_cqn_no_bc_stage38_extension.py`. The runner first verifies the
completed matched gate, resumes both existing snapshots on GPU 5 with a
120-second one-card/two-run stagger, caps them at raw 30k, and then executes
the paired fixed four-checkpoint sweep. It contains no 101k or held-out eval
command. The summarizer rejects an unqualified initial gate and can authorize
only the separate 50k sentinel. Shell syntax, Python compilation, scoped diff
checks, and all eight Stage-38 initial/extension summarizer tests pass. The
durable bounded controller is launched immediately after this protocol is
recorded; its PID and first artifact-backed boundary check are appended below.

The durable extension controller is PID 3029160 on physical GPU 5. Both
`latest_snapshot.pkl` links still point to the exact raw-20k snapshots. Its
first two renderer probes at 10:42:09 and 10:43:09 BST encountered the same
transient `Cannot initialize a EGL device display` condition seen at the
initial offline-to-online boundary, so no training process has been admitted
and no additional step has been consumed. The controller is live and retries
the same card every 60 seconds; genuine execution will be recorded only after
a worker loads raw 20k and writes a post-20k training artifact.

The wait exposed two infrastructure details before any update was consumed.
On this host GPU 1 is the boot-VGA card and is absent from NVIDIA's five-entry
EGL list, so physical GPU 5 maps to `MUJOCO_EGL_DEVICE_ID=4`, not 5. An index-4
renderer probe succeeds. The first corrected-EGL worker then found physical
GPU 5 temporarily unable to admit a new CUDA context and JAX attempted a CPU
fallback. The entire process group was terminated before checkpoint load;
both latest links and raw-20k file mtimes/sizes remain unchanged. The runner
now requires independent EGL and `JAX_PLATFORMS=cuda` probes, forces CUDA-only
training/evaluation, and retries transient CUDA initialization errors rather
than ever accepting CPU execution. Revalidation still gives eight passing
tests and clean shell/diff checks. Corrected controller PID 3049967 is live;
its first check reports `egl=available, cuda=unavailable`, so it is safely
waiting on the four external GPU-5 graphics contexts and has consumed zero
new steps.

By 11:06 BST all external GPU-5 graphics contexts had exited, but thirteen
consecutive CUDA probes through 11:08:42 still failed to create a new JAX
context (`CUDA_ERROR_NO_DEVICE`/`CUDA_ERROR_NOT_INITIALIZED`). GPU 5 is
otherwise quiescent at 3 MiB/P8 with 0% SM, no compute or graphics PID,
`GPU Recovery Action: None`, no Xid, default compute mode, and no pending
repair. Thus the remaining blocker is a stale/unavailable CUDA device state,
not memory contention or an experiment failure. Controller PID 3049967 stays
live behind the hard CUDA gate. Resetting physical GPU 5 or moving the exact
pair to another assigned card requires user authority; neither action is
silently taken.

CUDA enumeration also confirms why numeric pinning is unsafe here: the
boot-VGA GPU is currently absent from CUDA's compute list, so numeric ordinal
5 is `NO_DEVICE` while physical GPU 5 resolves by UUID to
`GPU-2f044e6a-9150-0e30-7d97-009bdd425b11` and returns the more precise
`NOT_INITIALIZED`. The bounded runner now resolves the assigned physical card
through `nvidia-smi`, exports that UUID for CUDA/JAX, passes `gpu_id=null` so
RoboBase cannot overwrite it with a drifting ordinal, and evaluates with
`--gpu-id -1` under the same UUID. Hydra composition confirms the unchanged
method contract (`256/256`, raw 30k cap, dense reward-Q true, BC zero), and all
eight tests still pass. Corrected UUID-pinned controller PID 3070966 is live;
its first dual probe remains `egl=available, cuda=unavailable`, with both run
artifacts exactly at raw 20k.

### Stage 38 CUDA blocked audit (2026-08-01 11:14 BST)

#### 1. Previous-stage result

The matched Stage-38 result remains 58%/50% selected best across two training
seeds versus 0% for exact No-BC and 66% for the matched seed-1 BC reference.
No new method result has been produced: both Stage-38 latest links still point
to raw 20k and no extension marker or post-20k snapshot exists. Physical GPU
5 is empty at 3 MiB, 0% SM, P8, default compute mode, no process, no Xid, and
`GPU Recovery Action: None`, but both the durable controller and an independent
UUID-pinned JAX probe still return `CUDA_ERROR_NOT_INITIALIZED`.

The legacy Stage-37 run has reached raw 82.5k but has no new fixed evaluation
beyond 44%/42% at raw 30k/40k. Stage 39 has so far evaluated only seed-1 raw
2.5k/5k at 46%/58%; its incomplete online-only curve is not substituted for
the exact offline-to-online question.

#### 2. Interpretation

This is an infrastructure block, not evidence against dense reward-Q: no
extension update has run. Numeric CUDA index drift and CPU fallback have been
ruled out by UUID pinning and a CUDA-only hard gate. The full research goal is
still unresolved because the bounded scaling curve, matched multi-seed BC
comparison, full-budget promotion, and sealed held-out test remain missing.

#### 3. Next-stage decision

After explicit authority to reset physical GPU 5, or assignment of another
card capable of creating a fresh CUDA context, resume the same two raw-20k
snapshots to raw 30k. Do not change batch 256+256, method settings, validation
seeds, checkpoint set, or promotion thresholds. Do not launch 101k or open
held-out seeds. If the raw-30k gate passes, separately design the registered
50k sentinel; otherwise stop this method.

#### 4. Execution

UUID-pinned controller PID 3070966 remains live behind independent EGL/CUDA
probes and cannot consume CPU updates. This identical blocker has now recurred
for three consecutive goal turns. The persistent goal is therefore marked
blocked pending reset/reassignment rather than left misleadingly active or
declared complete.

### Stage 38 resumed on assigned GPU 3 (2026-08-01 12:32 BST)

#### 1. Previous-stage result

The method result remains the frozen matched selection: dense reward-Q seeds
1/2 select 58%/50%, exact No-BC selects 0%, and matched BC seed 1 selects 66%.
While the goal was blocked, GPU 5 eventually admitted CUDA at 12:17:31 and the
waiting controller resumed both runs. Before migration it wrote a complete
raw-22.5k seed-1 snapshot; seed 2 had only its prior raw-20k durable snapshot.
Training rows beyond those durable boundaries are not policy evaluations and
are discarded on restore.

#### 2. Interpretation

The automatic GPU-5 start confirms that the UUID/CUDA hard gate works and that
the block was infrastructure-only. Moving the same serialized critic,
optimizer, RNG, replay, and global step to another GPU does not change the
research treatment. The durable raw-22.5k/raw-20k boundary preserves every
registered evaluation checkpoint and prevents partially executed online
segments from entering the comparison.

#### 3. Next-stage decision

Honor the newly assigned physical GPU 3 and continue the same two runs to raw
30k with the frozen batch 256+256 and all baseline-matched settings. Then run
the already registered 50-episode seeds-400--449 sweep at raw
22.5/25/27.5/30k. The pass/fail criteria, Stage-36 No-BC/BC references,
earliest-best selection, sealed held-out split, and prohibition on 101k remain
unchanged.

#### 4. Execution

GPU 3 is empty and passes both a UUID-pinned JAX CUDA probe and its physical
EGL index-2 renderer probe. The GPU-5 process group was terminated cleanly,
leaving seed-1/2 latest links at raw 22.5k/20k. The runner now records physical
GPU, UUID, and EGL index and uses GPU-specific log names so migration evidence
cannot overwrite prior logs. GPU-3 controller PID 3161894 launched seed-1 PID
3162179 and seed-2 PID 3167683 with the registered 120-second stagger.

Both workers are live simultaneously on the single card, using 12.9/15.1 GiB
(28.0 GiB total) at 96% utilization. Resolved configs show raw-30k cap,
batch/demo batch 256/256, `critic_lambda=0.1`, dense reward-Q enabled,
BC zero, online mixed replay, and force probability zero. Seed 1 has already
written a fresh post-restore raw-23k training row; seed 2 has completed GPU
compilation and written its restored-boundary raw-20k row. Fatal scans are
clean. At the observed Stage-38 throughput, paired training plus the fixed
four-checkpoint validation should reach its result around 13:15--13:35 BST;
artifact progress supersedes this estimate.

### Stage 37 full-budget final result and stop decision (2026-08-01 12:46 BST)

#### 1. Previous-stage result

The isolated formal validation sweep completed all ten registered checkpoints
on 50 episodes with seeds 400--449: raw 20/30/40/50/60/70/80/90/100/101k
scored 44/42/44/36/42/36/32/34/26/34%. Its post-20k validation best is 44%
at the earliest tie, raw 20k. Including the frozen short sweep, this seed's
overall validation-selected best remains 52% at raw 5k, versus 68% for the
matched original CQN-AS seed-1 validation best.

The pre-registered fixed raw-101k endpoint then scored 28.5% (57/200) on the
sealed held-out episodes with seeds 800--999. This is 33.5 percentage points
below the official same-seed 62% full-run endpoint. The exact artifacts are
`dense_b256_seed1/val50_seeds400_full.csv`,
`dense_b256_seed1/heldout200_seeds800_endpoint.csv`, and
`stage37_dense_b256_20260801021235/stage37_full_summary.json`.

#### 2. Interpretation

Batch 256+256 does not explain away the useful early No-BC behavior: the
replication learned a nonzero policy and reached 52% by raw 5k. However, the
full curve rules out the hypothesis that this retained dense No-BC objective
merely learns more slowly and catches up with a 101k budget. Performance is a
plateau followed by degradation, not positive scaling, and the held-out game
result is decisively below the matched BC reference. Training loss or healthy
updates are therefore not used as evidence of policy quality.

The earlier 30k/40k interim rows (44%/42%) are retained only as historical
monitoring. The formal isolated sweep (42%/44%) is authoritative: it used the
frozen run config and exact registered episode seeds after training completed.
Concurrent CQN-AS additions are config-gated; this run explicitly has BC zero,
dense return-Q true, and all newer positive-only/twin/advantage/auxiliary
treatments absent or false/zero. The small swapped two-point discrepancy does
not affect selection or the stop decision, and the independent 200-episode
held-out failure is much larger than that sampling variation.

#### 3. Next-stage decision

Reject Stage 37 for full-budget parity and do not train its remaining seeds to
101k. Also keep the previously audited all-zero Stages 0--6 and 30--36 stopped.
No additional full run is justified from a curve whose overall best is early
and whose endpoint loses 33.5 points. Any reward-gated or positive-return
dense-Q work remains a distinct research question under its own matched
validation gate; its result must not be pooled with Stage 37.

#### 4. Execution

The isolated evaluator completed held-out attempt 1 on GPU 5 without a foreign
GPU process at 12:45:14 BST and wrote `heldout_evaluation_complete`, the frozen
JSON summary, and `complete`. The earlier contaminated-card validation attempt
was killed before an admitted result and the full validation was retried to
completion in isolation. No Stage-37 seed-2/3 full-budget jobs were launched.
The Stage-37 question is closed as failed; ongoing Stage-38/39 jobs, if any,
continue only under their separately registered protocols.

### Stage 39 positive-return dense-Q result (2026-08-01 12:46 BST)

#### 1. Previous-stage result

On the fixed 50-episode validation seeds 400--449, positive-return-only
dense-Q selected 58% at raw 5k for seed 1 and 46% at raw 5k for seed 2. The
locked Stage-37 controls selected 52% and 50%, respectively. The paired deltas
are therefore +6pp/-4pp and the treatment mean is 52% versus the 51% control
mean, a +1pp gain.

#### 2. Interpretation

The treatment fails its registered requirement that both paired deltas be
nonnegative and mean gain be at least 5pp. It improves one seed but harms the
other, so it does not establish a reproducible improvement and its nonzero
losses are not treated as policy evidence.

#### 3. Next-stage decision

Reject positive-return-only dense-Q as a distinct mechanism. Do not run a
third seed, extend it, or open held-out seeds. Continue only the already
registered Stage-38 bounded offline-to-online extension, whose question and
baseline are separate.

#### 4. Execution

Both eight-checkpoint sweeps and
`stage39_positive_return_dense_gpu0_20260801093310/stage39_summary.json` are
complete; its `next_decision` is `reject_mechanism` and its `complete` marker
was written at 12:44:01 BST. No Stage-39 follow-on job was launched.

### Stage 38 bounded scaling result and stop decision (2026-08-01 13:11 BST)

#### 1. Previous-stage result

The fixed extension curves at raw 22.5/25/27.5/30k are 50/50/52/46% for
seed 1 and 44/38/36/38% for seed 2 on 50 episodes, seeds 400--449. Extension
validation-best is therefore 52% at raw 27.5k and 44% at raw 22.5k, mean 48%.
Across the complete raw-10k--30k curve, the validation-selected best remains
the earlier 58% at raw 17.5k and 50% at raw 10k, mean 54%. The raw-30k
endpoints are 46% and 38%. For context, the matched Stage-36 BC seed-1
validation best is 66%.

#### 2. Interpretation

The exact offline-to-online dense reward-Q treatment remains nonzero, but it
does not improve with the second 10k online block. Its extension-best mean
drops from the initial selected mean of 54% to 48%, seed 2 trends down, and
neither combined selected checkpoint moves later. This rules out the
registered positive-scaling hypothesis; it does not justify interpreting the
training losses as improved game behavior. Held-out seeds remain unopened.

#### 3. Next-stage decision

The registered 50k-sentinel gate required both extension bests at least 40%,
extension mean at least 50%, and combined mean no lower than the initial mean.
Although the first and third conditions pass, the 48% extension mean fails the
second. Stop Stage 38 at raw 30k. Do not design or launch the 50k sentinel and
do not promote this treatment to 101k or held-out evaluation.

#### 4. Execution

Both raw-30k snapshots were written, all eight fixed extension evaluations
completed after both training processes had exited, and
`stage38_extension_summary.json`, `extension_validation_complete`, and
`extension_complete` were written at 13:10:39 BST. The summary records
`eligible_for_separately_designed_50k_scaling_sentinel=false` and
`next_decision=stop_stage38_scaling_after_raw30k`. The controller exited and
no follow-on Stage-38 process was launched.

## Stage 40: dense-offline to canonical-online handoff (2026-08-01)

### 1. Previous-stage result

Stage 38 reached 52%/50% at the shared raw-10k offline boundary. Continuing
the dense reward-Q target online selected 58%/50% over raw 10--20k, but its
second online block selected only 52%/44% and ended at 46%/38% at raw 30k.
The registered extension gate therefore failed at a 48% mean. Separately,
Stage 37's legacy online-only dense run fell to 34% validation and 28.5%
held-out at raw 101k, while Stage 39's positive-return-only variant produced
paired deltas of +6pp/-4pp and failed its reproducibility gate.

### 2. Interpretation

Dense reward-Q is sufficient to create a nonzero policy during the pure
reward-based offline phase, but the available curves do not support continued
dense labeling online or a full-budget run. A specific unresolved mechanism
is whether counterfactual failure labels that are useful for initial offline
action ranking subsequently erase useful values as online state coverage
changes. This is distinct from changing the network, update ratio,
exploration, or replay recipe, and training loss is not used as policy-quality
evidence.

### 3. Next-stage decision

Stage 40 tests only a phase handoff. For each training seed 1/2, treatment and
the completed Stage-38 control share the exact same raw-10k dense reward-Q
offline checkpoint, optimizer, replay contents, and RNG state. Treatment then
sets `dense_return_q_target=false` and uses the canonical reward-derived C51
Bellman/MC target online; control kept the dense target online. Both use
batch/demo-batch 256/256, K=16, nstep=1, critic lambda 0.1, learning rate
5e-5, target tau 0.02, one update per environment step, noise 0.01, the same
single dueling critic, and zero BC, margin, FOSD, actor, or action-regression
loss.

The initial cap is raw 20k: 10k offline updates plus 10k online environment
steps. Selection uses exactly 50 episodes on seeds 400--449 at raw
10/12.5/15/17.5/20k, with earliest-checkpoint tie breaking. The raw-10k
evaluation must match the corresponding shared Stage-38 checkpoint. Over only
post-handoff checkpoints 12.5--20k, both per-seed treatment bests must be
non-worse than their matched controls, mean best gain must be at least 5pp,
and treatment raw-20k endpoint mean must be at least 50%. Passing authorizes
only a separately designed raw-30k scaling extension. Failure stops Stage 40.
Full 101k and held-out seeds 800--999 remain sealed in either case.

### 4. Execution

Added
`cqn_as_pixel_bigym_stage40_online_canonical_handoff_gate.yaml`,
`prepare_cqn_no_bc_stage40_branch.py`,
`run_cqn_no_bc_stage40.sh`, and
`summarize_cqn_no_bc_stage40.py`. Seven focused tests pass. The config test
proves Stage 40 differs from Stage 38 only in
`method.dense_return_q_target`; snapshot inspection proves that flag is not
stored in agent or checkpoint state. The branch preparation test rejects
non-contiguous replay and excludes episodes appended after the snapshot.

The durable controller registered this protocol at 15:20:03 BST in
`exp_local/cqn_no_bc/stage40_offline_dense_online_canonical_gpu3_20260801152002`.
Both branch manifests contain exactly the raw-10k state: main replay
60 episodes/10,953 transitions and protected demo replay 51 episodes/9,253
transitions for each seed. CUDA and EGL hard probes passed on physical GPU 3
(UUID `GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`, EGL device 2). Seed 1
loaded the raw-10k snapshot and entered the online process at 15:20:33 BST;
seed 2 loaded its paired raw-10k snapshot at 15:22:35 BST. Both runtime
configs record batch/demo-batch 256/256 and
`dense_return_q_target=false`; both wrote finite raw-10k boundary rows to
`train.csv`. At 15:23 BST the two JAX processes occupied 27,998 MiB together
on GPU 3 at 87% utilization, leaving about 4.6 GiB headroom. The controller is
responsible for the preregistered validation, summary, and stop decision, with
no full-run code path.

### Stage 40 final result and stop (2026-08-01 16:04 BST)

#### 1. Previous-stage result

The complete fixed 50-episode curves at raw 10/12.5/15/17.5/20k are
54/0/0/0/0% for seed 1 and 46/0/0/0/0% for seed 2. The corresponding
Stage-38 full-dense-online curves are 52/52/56/58/50% and
50/40/50/38/42%. Restricting selection to post-handoff checkpoints gives
0%/0% versus 58%/50%, paired deltas -58pp/-50pp and mean delta -54pp.

#### 2. Interpretation

Switching the same offline-trained critic directly to canonical chosen-action
C51 destroys the usable action ranking within 2.5k online steps in both
seeds, with no recovery by raw 20k. This establishes that the offline policy
requires a continuing reward-derived action-coverage anchor; it rules out a
hard dense-to-canonical handoff. The raw-10k re-evaluations differed by
+2pp/-4pp from the older evaluations even though each snapshot is the same
inode, demonstrating finite-rollout evaluation noise rather than a branch
state mismatch. Policy quality is determined by the complete curve, not
training losses.

#### 3. Next-stage decision

Stop Stage 40. It is ineligible for raw 30k, full 101k, or held-out seeds.
The next isolated question is Stage 41: keep the full dense reward-Q offline
phase, but online apply the dense target only when reward-to-go is positive;
zero-return trajectories use canonical C51. The condition reads completed
reward only, never demo identity. This tests whether reward-supported ranking
can be retained without continuing unsupported counterfactual failure labels
on failed online trajectories.

#### 4. Execution

`stage40_summary.json` was independently recomputed byte-identically. It
records `mechanism_pass=false`,
`eligible_for_bounded_raw30_extension=false`, and
`next_decision=stop_stage40_after_raw20_gate`. Training and eval processes
exited, the controller wrote `complete` at 16:03:12 BST, and GPU 3 returned to
3 MiB. No Stage-40 follow-on was launched.

## Stage 41: positive-return dense online handoff (2026-08-01)

### 1. Previous-stage result

Stage 38 shows that full dense reward-Q online remains nonzero but does not
scale: post-handoff validation best is 58%/50%, raw-20k endpoint 50%/42%, and
the raw-22.5--30k extension best falls to 52%/44%. Stage 40 shows the opposite
extreme, no dense target online, collapses both seeds to zero immediately.
Stage 39's random-initialization online-only positive-return gate produced
+6pp/-4pp and was rejected, but did not test preservation after a successful
offline reward-Q phase.

### 2. Interpretation

The two extremes isolate the unresolved target-selection problem: a continuing
anchor is necessary, but labeling every unseen action as failure on every
failed online trajectory may cause the longer-run erosion. Positive-return
gating is a distinct offline-to-online mechanism here. It remains one C51 Q
objective: target choice is a deterministic function of reward-to-go, with no
BC, action likelihood, margin, demo mask, actor, or policy pretraining.

### 3. Next-stage decision

Both seeds branch from the exact same Stage-38 raw-10k offline snapshots,
optimizer/RNG states, and replay episodes. Stage 41 differs from Stage 38 only
in `method.dense_return_positive_only=true`. All matched settings remain
batch/demo-batch 256/256, K=16, nstep=1, critic lambda 0.1, learning rate
5e-5, tau 0.02, UTD 1, noise 0.01, and a single dueling critic.

Selection is 50 episodes on seeds 400--449 at post-handoff raw
12.5/15/17.5/20k; the exact shared offline boundary uses the already frozen
Stage-38 52%/50% values instead of a noisy duplicate rollout. Earliest
checkpoint breaks ties. A bounded raw-30k extension requires: each seed's
post-best within 2pp of its Stage-38 control; treatment mean post-best at
least the Stage-38 54% mean; both raw-20k endpoints at least 40%; and endpoint
mean at least 50%. Full 101k and held-out seeds 800--999 remain sealed.

### 4. Execution

Added the Stage-41 launch config, branch runner, frozen summarizer, and focused
tests. Seven relevant tests pass and config composition proves the treatment
diff is exactly `method.dense_return_positive_only`. The controller has no
full-run path. The durable controller registered the protocol at 16:10:47 BST
in
`exp_local/cqn_no_bc/stage41_offline_dense_online_positive_gpu3_20260801161047`.
Both exact raw-10k replay branches and CUDA/EGL probes completed; seed 1 loaded
the shared snapshot at 16:11:20 BST. Its runtime config records
`dense_return_q_target=true`, `dense_return_positive_only=true`, batch/demo
batch 256/256, and zero BC/margin/FOSD. Seed 2 is launched by the same
controller after the registered 120-second stagger and restored its paired
snapshot at 16:13:20 BST. At the first online boundary rows the reward gate is
active on 90.2%/90.0% of the two mixed batches, both critic losses are finite,
and the two contexts occupy about 28.0 GiB together on GPU 3.

### Stage 41 initial result and raw-30k authorization (2026-08-01 16:53 BST)

#### 1. Previous-stage result

The fixed post-handoff curves at raw 12.5/15/17.5/20k are
56/50/62/62% for seed 1 and 50/56/58/48% for seed 2. Validation-selected
best is 62%/58%, mean 60%, versus the same-seed Stage-38 full-dense-online
best 58%/50%, mean 54%. The paired gains are +4pp/+8pp and mean gain is
+6pp. Raw-20k endpoints are 62%/48%, mean 55%. Stage 40 was 0% for both
seeds at every corresponding post-handoff checkpoint.

#### 2. Interpretation

Positive-return dense gating prevents the immediate hard-handoff collapse and
improves both seeds over continuing dense labels on every trajectory. Unlike
the shared raw-10k offline peak, both treatment seeds improve at later online
checkpoints, so this is positive initial scaling evidence. It does not yet
establish persistence to 30k, much less official-budget parity; held-out data
remain unopened.

#### 3. Next-stage decision

The registered initial gate passes all conditions: both paired post-best
values are noninferior, mean best is 6pp higher, both endpoints exceed 40%,
and endpoint mean exceeds 50%. Authorize only a separately capped raw-30k
extension on the same two seeds/configuration. Evaluate raw
22.5/25/27.5/30k on 50 episodes, seeds 400--449. A later raw-50k sentinel is
authorized only if both extension bests beat the Stage-38 extension controls
(52%/44%), extension mean best is at least 58%, at least one seed reaches its
own Stage-41 initial best (62%/58%), both raw-30k endpoints are at least 45%,
and endpoint mean is at least 52%. This gate still cannot authorize full 101k
or held-out evaluation.

#### 4. Execution

`stage41_summary.json` was independently recomputed byte-identically and
records `mechanism_pass=true` and
`eligible_for_bounded_raw30_extension=true`. Added the bounded extension
runner, frozen summarizer, and four passing focused summary tests. The runner
has no raw-50k/full/held-out path and will resume the two raw-20k snapshots on
physical GPU 3 immediately.

### Stage 41 raw-30k scaling result and raw-50k sentinel authorization (2026-08-01 17:36 BST)

#### 1. Previous-stage result

The fixed raw 22.5/25/27.5/30k curves are 64/64/54/66% for seed 1 and
46/46/52/46% for seed 2, each measured on 50 episodes with evaluation seeds
400--449. Validation-selected extension best is 66% at raw 30k and 52% at
raw 27.5k, mean 59%. These beat the exact Stage-38 extension controls of
52%/44% by +14pp/+8pp. The raw-30k endpoints are 66%/46%, mean 56%.
Independent summary recomputation is byte-identical to the controller output.

#### 2. Interpretation

The positive-return dense target remains task-effective through a second 10k
online block rather than following Stage 38's erosion or Stage 40's collapse.
Both seeds remain nonzero at every checkpoint and both improve over their
matched dense-online controls. The lower seed still fluctuates between 46%
and 52%, so this establishes bounded scaling support, not full-budget parity;
training loss is not used as policy-quality evidence and held-out remains
unopened.

#### 3. Next-stage decision

All preregistered raw-30k conditions pass, so authorize only a raw-50k
sentinel on the same two seeds and exact configuration. Resume raw 30k and
evaluate raw 32.5/35/37.5/40/42.5/45/47.5/50k, always 50 episodes on seeds
400--449 with earliest-checkpoint tie breaking. Batch/demo-batch remain
256/256 and every baseline setting remains unchanged; the only mechanism is
the already isolated positive-return choice inside the reward-derived C51
target.

The hypothesis is persistence rather than a transient selected peak. A full
raw-101k protocol can be designed only if: each seed's sentinel best is within
4pp of its own raw-30k-block best (66%/52%); sentinel mean best is at least
58%; the mean across all 16 fixed evaluations is at least 52%; each seed's
best in the late 45/47.5/50k window is at least 48% and their mean is at least
56%; and both raw-50k endpoints are at least 45% with endpoint mean at least
52%. Stage 38 through raw 30k and the frozen Stage-41 raw-30k block are the
matched references for this persistence question. The official 101k BC
endpoint is reserved for the later full comparison; seeds 800--999 stay
sealed. Passing does not automatically start full training.

#### 4. Execution

Added `run_cqn_no_bc_stage41_raw50_sentinel.sh`, the frozen raw-50k
summarizer, and focused pass/erosion/selection-bias/authorization tests. The
runner has no full-run or held-out path and is restricted to physical GPU 3,
two staggered runs, and the registered raw-50k cap. Shell syntax, Python
compilation, scoped diff checks, and all six Stage-41 extension/sentinel tests
pass. The durable controller launched at 17:40:15 BST in the existing Stage-41
run directory. CUDA and EGL probes passed on physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`. Both process environments contain
that UUID; the live composed configs record batch/demo-batch 256/256,
K=16, nstep=1, critic lambda 0.1, `bc_lambda=0`, `bc_margin=0`, and the
positive-return reward-Q target. At 17:44 BST seed 1 had advanced past the
restored boundary to raw 31k and seed 2 had written its restored raw-30k row.
The pair occupied 25,842 MiB on GPU 3 at 96--97% utilization with no failure
marker. Training is capped at raw 50k; fixed validation and the frozen summary
will run only after both training processes exit.

The strict-objective audit was also rerun against the exact Stage-41 code path
on CPU: 12 focused tests pass. They establish that strict mode rejects every
BC/margin/FOSD/separate-policy/flow-policy/value-policy/self-imitation path;
the critic is the only parameterized action selector; changing only the demo
mask leaves the reward-Q update identical; zero-return dense targets have
identical loss and gradients for different replay actions; failed trajectories
match canonical C51 parameter updates exactly; and Stage 41 differs from its
matched Stage-38 configuration only in the positive-return gate. The gate
reads completed `mc_return > 0`, never demonstration identity.

### Stage 41 raw-50k sentinel result and stop (2026-08-01 18:58 BST)

#### 1. Previous-stage result

The complete raw 32.5/35/37.5/40/42.5/45/47.5/50k curves are
64/68/66/60/64/72/58/76% for seed 1 and
44/42/26/36/36/42/32/40% for seed 2, each on the fixed 50 episodes with
seeds 400--449. Sentinel validation-best is 76%/44%, mean 60%; the late-window
best is 76%/42%, mean 59%; all 16 evaluations average 51.625%; and raw-50k
endpoints are 76%/40%, mean 58%. The controller summary was independently
recomputed byte-identically (SHA256
`55e98688016dd0ff3820546e0e775ccf7de22095446b41cd1c6ea54fe3daf63b`).

#### 2. Interpretation

The mechanism can scale very strongly in seed 1, including a new raw-50k
best, but it is not seed-robust: seed 2's best moves back to the first
sentinel checkpoint and its late-window values remain 32--42%. The registered
full gate fails because seed 2 is below the 48% sentinel/late-best floors and
45% endpoint floor, while the all-checkpoint mean is 0.375pp below 52%.
Therefore the attractive 60% validation-selected mean is insufficient to
justify full training. This rules out Stage 41 as the current full-run
candidate; it does not rule out the reward-Q mechanism after removing the
identified replay feedback. Held-out remains sealed.

#### 3. Next-stage decision

Artifact diagnosis exposes a distinct replay mechanism inherited from the
official baseline: `use_self_imitation=true` appends successful online
episodes to the protected demo buffer. By raw 49k the two buffers had diverged
to 25,335 transitions for seed 1 and 21,664 for seed 2, while both began from
the same 9,253-transition expert buffer. This produces seed-specific positive
feedback and also weakens the clean claim that the prior half-batch is the
fixed expert dataset.

Stage 42 will test only fixed expert replay: branch both seeds from the exact
same Stage-38 raw-10k offline snapshots/replay/RNG states as Stage 41, retain
the positive-return reward-Q target, and set `use_self_imitation=false` plus
its now-unused strict permission flag false. All numerical/model/optimizer
settings remain matched, including batch/demo-batch 256/256. Train only to raw
30k and evaluate raw 12.5/15/17.5/20/22.5/25/27.5/30k on seeds 400--449,
50 episodes each, earliest tie break. Compare against the frozen Stage-41
curve at the same steps and Stage-38 full-dense control.

Passing requires both Stage-42 per-seed bests at least 55%, mean best at least
60%, both raw-30k endpoints at least 50%, endpoint mean at least 55%, and the
mean over all 16 fixed evaluations at least 54%. This stage can authorize only
a separately designed raw-50k replication; it cannot authorize full or open
held-out. Failure stops fixed expert replay and moves to the next action-value
stability mechanism.

#### 4. Execution

All Stage-41 training/evaluation processes exited, all 16 snapshots and both
CSVs are present, completion sentinels were written, and GPU 3 returned to
3 MiB. `eligible_for_full_run_protocol=false` and
`next_decision=stop_stage41_scaling_after_raw50k` are frozen in the summary.
Stage 42 implementation and launch follow immediately on GPU 3.

### Stage 42 fixed-expert replay launch (2026-08-01 19:09 BST)

#### 1. Previous-stage result

Stage 41 is frozen as ineligible for full: despite seed 1 reaching 76%, seed
2's sentinel best/late best/endpoint are 44/42/40%, and the all-checkpoint mean
is 51.625%. The protected replay grew differently by seed, providing a
specific matched replay-feedback hypothesis rather than a generic tuning
attempt.

#### 2. Interpretation

The Stage-41 result establishes that the positive-return reward-Q objective is
capable of strong scaling, but not that its growing online-success replay is
stable or necessary. Holding the expert prior fixed is both a cleaner
no-imitation claim and a direct test of the observed seed feedback. It does
not change the Q objective, model, batch, optimizer, exploration, or offline
initialization.

#### 3. Next-stage decision

Use the preregistered Stage-42 raw-30k curve and gate above. Matched primary
control is the frozen Stage-41 growing-success-replay curve at the same eight
steps; Stage 38 is the secondary full-dense control. Selection and held-out
splits, metrics, thresholds, and stop behavior are frozen before rollout
evaluation.

#### 4. Execution

Added `cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate.yaml`, the capped
runner, frozen summarizer, and three summary plus one exact-config-diff test;
all four tests, shell syntax, Python compilation, and scoped diff checks pass.
The durable controller launched at 19:04:58 BST in
`exp_local/cqn_no_bc/stage42_fixed_expert_replay_gpu3_20260801190458`.
CUDA/EGL probes passed on physical GPU 3. Each branch manifest records the
exact raw-10k snapshot, 10,953 main-replay transitions, and 9,253 expert
transitions. Both runtime configs record batch/demo-batch 256/256,
`use_self_imitation=false`, its strict permission false, and BC/margin zero.
At 19:09 seed 1/2 had genuinely advanced to raw 11k/10k; both processes were
bound to GPU UUID `GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`, occupying about
27,998 MiB at 96% utilization. The runner is capped at raw 30k and has no
full-run or held-out path.

### Stage 42 raw-30k result and raw-50k replication authorization (2026-08-01 20:22 BST)

#### 1. Previous-stage result

The fixed raw 12.5/15/17.5/20/22.5/25/27.5/30k curves are
62/56/62/66/56/74/56/54% for seed 1 and
46/52/48/66/54/50/56/62% for seed 2. Validation-best is 74%/66%, mean 70%,
versus Stage 41's 66%/58%, and versus Stage 38's 58%/50%. Paired best gains
are +8pp/+8pp over Stage 41 and +16pp/+16pp over Stage 38. Raw-30k endpoints
are 54%/62%, mean 58%; all 16 checkpoints average 57.5%. Both protected
buffers remained exactly 9,253 transitions. The summary recomputes
byte-identically with SHA256
`e8b1d8230f255c714ddc585741c229e23c7339715a0d81b1a407a54ff7ffafb3`.

#### 2. Interpretation

Fixing the prior replay to expert trajectories removes the Stage-41 seed-2
erosion through raw 30k while improving both validation-selected bests. The
effect is not merely one selected peak: the all-checkpoint mean and both
raw-30k endpoints pass their preregistered floors. This supports the replay
feedback hypothesis and yields a cleaner reward-only method. It still does
not prove persistence through raw 50k or official-budget parity.

#### 3. Next-stage decision

Authorize only continuation of the same two seeds/configuration from raw 30k
to raw 50k. Evaluate raw 32.5/35/37.5/40/42.5/45/47.5/50k on the unchanged
50 episodes, seeds 400--449. A matched raw-101k full protocol may be designed
only if: each replication best is within 6pp of its own Stage-42 raw-30k-block
best (74%/66%); mean best is at least 65%; all 16 replication evaluations
average at least 55%; each seed's late-window best is at least 55% with mean
at least 60%; both raw-50k endpoints are at least 50% with mean at least 58%;
and expert replay remains exactly fixed. Stage 41's raw-50k block is the
matched growing-replay control. Passing cannot itself open held-out or start
full.

#### 4. Execution

Added the capped Stage-42 raw-50k runner, frozen summarizer, and focused
robustness/late-erosion/buffer-growth tests. The runner has no full or held-out
path. Shell syntax, Python compilation, scoped diff checks, and all six
Stage-42/raw-50k summary tests pass. The durable controller launched at
20:27:06 BST in the existing Stage-42 run directory. CUDA and EGL probes
passed on physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`; seed 2 was staggered by 120
seconds, so at most two runs share the card. At 20:31 BST both training
processes were live and bound to that UUID: seed 1 had advanced from raw 30k
to raw 31k and seed 2 had produced its restored raw-30k row. GPU 3 held
27,998 MiB at 93% utilization, both runtime rows still reported exactly 9,253
expert transitions, and no failure marker existed. Training is hard-capped at
raw 50k; the fixed 50-episode validation sweep and frozen gate summary start
only after both trainers exit.

### Stage 42 raw-50k replication result and Stage 43 full protocol (2026-08-01 21:45 BST)

#### 1. Previous-stage result

The complete fixed raw 32.5/35/37.5/40/42.5/45/47.5/50k curves are
64/62/66/64/68/66/62/62% for seed 1 and
58/62/70/70/80/72/74/68% for seed 2, each measured on 50 episodes with
seeds 400--449. Replication best is 68%/80%, mean 74%; all 16 evaluations
average 66.75%; the late 45/47.5/50k best is 66%/74%, mean 70%; and the fixed
raw-50k endpoints are 62%/68%, mean 65%. Both expert buffers remained exactly
9,253 transitions. Every preregistered full-protocol condition passes.

All 16 checkpoint files and both CSVs are complete, both evaluators exited,
and GPU 3 returned to 3 MiB. The controller summary was independently
recomputed byte-identically with SHA256
`19c08cc815e451f07765906ae441d30cae3a153afa853644b50cf102969510ac`.
It freezes `eligible_for_matched_raw101k_full_protocol=true`,
`heldout_opened=false`, and
`next_decision=design_matched_raw101k_full_protocol`.

#### 2. Interpretation

Fixed expert replay replicates robust scaling rather than merely preserving a
single selected early peak. Seed 2 no longer follows Stage 41's 44% best and
40% endpoint erosion; it reaches 80% and ends at 68%. Seed 1 gives up the
Stage-41 growing-replay spike of 76% but remains within the frozen 6pp
tolerance of its own Stage-42 raw-30k-block best and ends at 62%. Thus the
cleaner fixed-prior reward-only method earns official-scale compute. This
still does not establish persistence to 101k online interactions, four-seed
robustness, or held-out parity; training loss is not used as policy evidence.

#### 3. Next-stage decision

Register Stage 43 as a four-training-seed, 101k-online-interaction full
protocol, staged on the single allocated physical GPU 3 with at most two
runs. The environment-budget match is explicit: the No-BC method receives
10k demo-only reward-Q updates followed by 101k online interactions, so its
fixed endpoint is raw 111k; official CQN-AS receives no offline phase and its
matched online endpoint is raw 101k. Both retain batch/demo-batch 256/256 and
the same K=16 single-dueling-C51 architecture, optimizer, target update,
replanning, exploration noise, and one update per step. The Stage-42 method
continues unchanged: BC/margin/FOSD zero, no actor or policy pretraining,
positive-return reward-derived dense C51 target, and fixed 9,253-transition
expert replay.

Stage 43A first branches seeds 1/2 exactly from their frozen raw-50k
snapshots, replay states, and RNG states, without mutating Stage 42, and runs
to raw 111k. Validation appends raw 60/70/80/90/100/110/111k to the already
frozen raw 12.5--50k curves, always 50 episodes on seeds 400--449 with
earliest-checkpoint tie breaking. Expansion to fresh full seeds 3/4 requires:
fixed expert replay; each overall validation best at least 60% and mean at
least 70%; each online-50k-or-later best at least 55% and mean at least 65%;
all 14 late evaluations averaging at least 55%; and both fixed raw-111k
endpoints at least 45% with mean at least 55%. Failure stops this recipe and
does not spend fresh full seeds or held-out episodes.

On a Stage-43A pass, fresh seeds 3/4 run the identical 10k-offline plus
101k-online recipe. The four-seed held-out gate is frozen now: each
validation-selected best at least 55% with mean at least the official fixed
validation-endpoint mean of 68%; each online-50k-or-later best at least 55%
with mean at least 65%; each fixed raw-111k validation endpoint at least 45%
with mean at least 60%; and all expert buffers fixed. Only then may the four
fixed raw-111k endpoints be evaluated for 200 episodes on sealed seeds
800--999. No held-out checkpoint selection is allowed. The matched official
fixed raw-101k references are validation 76/62/68/66% (mean 68%) and held-out
62/60.5/62/74% (mean 64.625%) for training seeds 1--4. Final empirical parity
requires the No-BC four-seed held-out mean at least 64.625%; validation-best
results are reported separately and never substituted for the matched fixed
endpoint comparison. Stage 37's 28.5% seed-1 held-out endpoint is the matched
No-BC full-budget negative control, not the target baseline.

#### 4. Execution

Added an exact online-snapshot branch mode, Stage-43A capped runner, frozen
summarizer, and online-branch/persistence/endpoint-collapse/buffer-growth
tests. The runner can launch only seeds 1/2, has no fresh-seed or held-out
path, keeps snapshot cadence at the baseline 2.5k, and hard-codes physical
GPU 3 defaults, batch/demo-batch 256/256, raw-111k endpoint, and a 120-second
two-run stagger. Twelve focused Stage-42/43 tests pass; shell syntax and
Python compilation pass. The repository-wide diff check is blocked only by
pre-existing unrelated trailing whitespace in `cqn-flow.md`; the scoped
Stage-43 diff check is clean.

The durable controller launched at 21:54:56 BST in
`exp_local/cqn_no_bc/stage43_seed12_full_gpu3_20260801215456`. Both branch
manifests record raw step 50k, 10k offline updates, and exactly 40k online
iterations. Their main replay counts are 50,894/50,656 and both expert replay
counts are exactly 9,253; all selected immutable files were hard-linked while
the Stage-42 source summary retained the registered SHA256. CUDA and EGL
probes passed on physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`. After the 120-second stagger,
both live configs differ from their respective Stage-42 source configs only
in `num_train_frames: 50000 -> 111000`; each still records batch/demo-batch
256/256, strict reward-only mode, BC/margin zero, FOSD false, and
`use_self_imitation=false`. At 21:58 BST seed 1 had genuinely advanced to raw
51k and seed 2 had written its restored raw-50k boundary row, both with expert
buffer 9,253. The two processes were UUID-bound and occupied 25,842 MiB at
97% utilization. This verifies the full-scale stage has started rather than
merely been queued; held-out remains sealed.

While Stage 43A runs, the downstream guarded runners and frozen decision
summaries were implemented without launching them. The fresh-seed runner
cannot start unless `stage43_seed12_summary.json` records the registered pass;
it resolves seed 3/4 offline as full-dense reward-Q with zero online
interaction, then online as positive-return-only fixed-expert reward-Q. The
held-out runner cannot start unless the subsequent four-seed validation
summary passes, evaluates only raw 111k, uses exactly 200 episodes beginning
at seed 800, and runs seeds in two waves of two on GPU 3. Resolved-config
inspection confirms both fresh-seed phases retain batch/demo-batch 256/256,
strict reward-only mode, BC/margin/FOSD zero, and fixed replay; only the dense
positive-return gate changes at the registered offline-to-online boundary.
Twelve focused seed-expansion tests and six four-seed/held-out decision tests
pass, and both guarded runners pass shell and compilation checks. These
prepared paths do not open held-out or bypass either report-before-launch
gate.

The first registered full-scale checkpoint pair, raw 60k (50k online after
the offline phase), is complete: seed 1/2 snapshots are 448,855,407 and
448,855,418 bytes, written at 22:26:55/22:31:01 BST. The trainers had advanced
to raw 61k/60k with both expert buffers still 9,253, GPU 3 at 92% and 25,842
MiB, and no failure marker. The frozen Stage-42 source-summary SHA256 remains
unchanged, confirming the exact branches have not mutated development
evidence. At the measured paired throughput, raw-111k training is expected
near 01:10--01:20 BST and the fixed seven-checkpoint sweep near 01:20--01:30;
artifacts supersede this estimate. This checkpoint is execution/scaling
evidence only, not a policy-quality result.

The raw-70k pair is also complete. Seed 1/2 snapshots are 448,855,415 and
448,855,425 bytes, written at 22:59:43/23:03:52 BST. At the 23:05 health
check the trainers had advanced to raw 71k/70k, both expert buffers remained
exactly 9,253, both Python processes were live, and physical GPU 3 was at 93%
utilization with 25,842 MiB allocated. A bounded scan of both training logs
and CSVs found no NaN, infinity, traceback, OOM, CUDA error, or failure
marker. The measured raw-60k-to-70k interval was 32:48/32:51, giving a
current artifact-based raw-111k ETA of approximately 01:14--01:19 BST and a
seven-checkpoint validation completion estimate around 01:25--01:30. This is
again execution evidence only: no interim policy evaluation was run, and
fresh seeds 3/4 and held-out evaluation remain gated and unlaunched.

The raw-80k pair is complete as well. Seed 1/2 snapshots are 448,855,416 and
448,855,426 bytes, written at 23:32:26/23:36:38 BST. At 23:37 the trainers
were live at raw 81k/80k on physical GPU 3 (90%, 25,842 MiB), and both expert
buffers were still exactly 9,253. The bounded numerical/error scan again
found no NaN, infinity, traceback, OOM, CUDA error, or failure marker. The
raw-70k-to-80k intervals were 32:43/32:46, consistent with the prior interval
and the approximately 01:14--01:19 raw-111k ETA. No raw-80k policy evaluation
was run during training; the registered seven-checkpoint sweep remains after
both trainers finish, and neither downstream gate has been opened.

The raw-90k pair completed at 00:05:13/00:09:35 BST with snapshot sizes
448,855,417/448,855,426 bytes. At 00:10 the live trainers had reached raw
91k/90k, physical GPU 3 was at 97% with 25,842 MiB allocated, and both expert
buffers remained exactly 9,253. The bounded numerical/error scan remained
clean. Raw 80k--90k took 32:47/32:56, so the observed throughput still
projects the raw-111k endpoint near 01:14--01:19. No interim evaluation or
downstream launch occurred.

The raw-100k pair completed at 00:38:31/00:42:54 BST, again with complete
448,855,417/448,855,426-byte snapshots. At 00:43 the trainers were live at
raw 101k/100k, GPU 3 was at 96% with 25,842 MiB allocated, both expert
buffers remained 9,253, and the bounded numerical/error scan remained clean.
Raw 90k--100k took 33:17/33:19; extrapolating only the remaining 11k steps
puts raw 111k near 01:15--01:20. The registered validation sweep remains
unopened until both fixed endpoints exist.

### Stage 43A qualified-seed full result and Stage 43B fresh-seed expansion (2026-08-02 01:33 BST)

#### 1. Previous-stage result

Stage 43A completed in
`exp_local/cqn_no_bc/stage43_seed12_full_gpu3_20260801215456`. Seed 1/2 raw
111k endpoint snapshots are 448,855,416/448,855,426 bytes and were written at
01:15:17/01:17:43 BST. The registered raw 60/70/80/90/100/110/111k
validation curves, each measured on 50 episodes with seeds 400--449, are
60/64/70/60/64/62/62% for seed 1 and
62/62/62/64/60/68/64% for seed 2. Thus the 14 full-scale evaluations average
63.14%.

After joining those frozen later points to the Stage-42 development curves,
the validation-selected overall best checkpoints are raw 25k at 74% and raw
42.5k at 80% (mean 77%). The online-50k-or-later best checkpoints are raw
80k at 70% and raw 110k at 68% (mean 69%). Fixed raw-111k endpoints are
62%/64% (mean 63%). Both expert buffers remained exactly 9,253 transitions.
Every preregistered seed-3/4 expansion condition passes, and held-out remains
sealed. The controller summary was independently recomputed byte-identically
with SHA256
`c46d7b6d108f53cd2ad5a92a1af1448e98ff3e7163734f007cc81135c06ff61d`;
the frozen Stage-42 source summary also retained SHA256
`19c08cc815e451f07765906ae441d30cae3a153afa853644b50cf102969510ac`.

#### 2. Interpretation

The reward-only fixed-expert method does not show the all-zero early curve or
the full-budget collapse that would have made this full run wasteful. On both
qualified seeds it maintains useful policy success throughout 101k online
interactions, and its 63% fixed-endpoint mean is 5pp below the official
four-seed validation-endpoint reference of 68%. The 77% validation-selected
mean is not substituted for that fixed-endpoint comparison because it uses
checkpoint selection. These results establish persistence and authorize a
fresh-seed robustness test; they do not establish four-seed robustness,
sealed held-out parity, or superiority. Seeds 1/2 were already development-
qualified, so running only them would retain selection bias.

#### 3. Next-stage decision

Stage 43B tests the single hypothesis that the same no-BC mechanism is robust
on fresh training seeds 3/4. Each seed starts from random parameters, receives
10k demo-only full-dense reward-Q updates, then 101k online interactions with
positive-return-only reward-Q targets and fixed expert replay. Batch and demo
batch remain 256/256; architecture, optimizer, target update, K=16 replanning,
baseline exploration, and one update per interaction remain matched. BC,
margin, FOSD, self-imitation, actor/policy pretraining, and action-regression
objectives remain absent.

Selection remains exactly 50 episodes on seeds 400--449 at the frozen 23 raw
steps from 12.5k through 111k, with earliest-checkpoint tie breaking. The
four-seed pass requires every overall selected best at least 55% and mean at
least 68%; every online-50k-or-later best at least 55% and mean at least 65%;
every fixed raw-111k endpoint at least 45% and mean at least 60%; and all four
expert buffers fixed at 9,253. Failure stops before held-out. Only a pass may
open the four fixed raw-111k endpoints for 200 episodes each on sealed seeds
800--999; the official held-out reference remains 64.625%.

#### 4. Execution

The Stage-43A controller exited cleanly after writing `training_complete`,
`validation_complete`, and `complete`; GPU 3 returned to 3 MiB. The guarded
Stage-43B runner rechecked the Stage-43A authorization and is the only next
execution path; it runs at most seeds 3/4 concurrently on physical GPU 3 and
contains no held-out call. Launch verification is recorded below after the
two-run stagger completes.

Stage 43B launched durably at 01:35:14 BST from the same Stage-43 directory.
CUDA/EGL probes passed, seed 3/4 offline processes started after the registered
120-second stagger, and `seed34_offline_pair_started` was written at 01:37:16.
Both process environments resolve physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`; at 01:37:59 they occupied 11,488
MiB at 97% utilization. Both resolved configs record 10k updates,
batch/demo-batch 256/256, demo-only replay, `use_self_imitation=false`,
`bc_lambda=0`, `bc_margin=0`, full-dense reward-Q targets, and no positive-
return filter during the offline phase. Seed 3 had already written 1,600
offline updates with expert buffer 9,253; seed 4 was still in its staggered
initialization window. This verifies that both runs genuinely started. A
reliable online/full-stage ETA will be recorded after both 2.5k offline
checkpoints establish paired throughput; held-out remains sealed.

The offline pair completed at 01:52:01/01:54:17 BST. Both raw-10k snapshots
are complete (448,855,572/448,855,583 bytes) and independently decode to
`pretrain_step=10000`, `main_loop_iterations=0`, and demo replay 9,253; the
runner then wrote `seed34_offline_complete`. Online seed 3/4 started after a
second 120-second stagger and `seed34_online_pair_started` was present by
01:56:29. At 01:57:59 they had genuinely advanced to raw 11k/10k with demo
buffers still 9,253, occupying 27,998 MiB on GPU 3. Both online configs record
raw limit 111k, batch/demo-batch 256/256, mixed replay, positive-return-only
dense reward-Q, BC/margin zero, `demo_fosd=false`, no self-imitation, and zero
demo-behavior forcing. Both process environments resolve the registered GPU 3
UUID. The next ETA calibration uses paired online checkpoints; no validation
or held-out evaluation has started.

The first fresh-seed online checkpoint pair, raw 12.5k, is complete. Seed 3/4
snapshots are 448,855,604/448,855,614 bytes, written at 02:01:56/02:06:02
BST. At 02:09 the trainers had advanced to raw 14k/13k with both demo buffers
still 9,253; GPU 3 held 27,998 MiB at 97%, and the bounded numerical/error
scan was clean. Handoff-to-checkpoint time was 7:37/9:44 including compile.
Combining this with Stage-43A steady-state throughput gives a provisional raw-
111k ETA of 07:25--07:45 and a 23-checkpoint validation ETA of 08:05--08:25;
raw-20k artifacts will supersede this estimate. These are execution artifacts,
not policy-quality results, and no downstream held-out action is possible from
this runner.

The paired raw-20k checkpoints are complete at 02:26:36/02:30:52 BST with
sizes 448,855,604/448,855,614 bytes. At 02:31 the trainers were live at raw
21k/20k, both demo buffers remained 9,253, GPU 3 held 27,998 MiB at 95%, and
the bounded error scan remained clean. Raw 12.5k--20k took 24:40/24:49, or
197.4/198.6 seconds per 1k steady-state interactions. Projecting that measured
rate gives raw-111k ETAs near 07:26/07:32 and a 23-checkpoint validation
completion estimate around 08:05--08:15. Artifacts supersede these estimates;
the job remains inside Stage 43B and held-out is still sealed.

The raw-30k checkpoint pair is complete at 02:59:43/03:03:59 BST with sizes
448,855,603/448,855,614 bytes. At 03:04 the live trainers were at raw 31k/30k,
both demo buffers remained exactly 9,253, GPU 3 held 27,998 MiB at 97%, and
the bounded numerical/error scan remained clean. Raw 20k--30k took
33:06/33:06, or 198.6 seconds per 1k for both seeds. The stable projection is
therefore raw-111k near 07:28/07:32 and validation completion near 08:10.
This remains execution evidence only; fresh-seed policy results are still
unobserved and held-out remains sealed.

The raw-40k checkpoint pair is complete at 03:33:02/03:37:11 BST with sizes
448,855,604/448,855,615 bytes. At 03:37 the trainers were live at raw 41k/40k,
both demo buffers remained 9,253, GPU 3 held 28,000 MiB at 97%, and the
bounded numerical/error scan remained clean. Raw 30k--40k took 33:19/33:12,
confirming the approximately 199 seconds-per-1k steady-state rate. Current
raw-111k ETAs are about 07:30/07:33 and validation completion remains around
08:10--08:15. No fresh-seed policy result has been inspected and held-out is
still sealed.

The raw-50k checkpoint pair is complete at 04:06:03/04:10:03 BST with sizes
448,855,604/448,855,614 bytes. At 04:10:56 the trainers were live at raw
51k/50k, both demo buffers remained exactly 9,253, GPU 3 held 28,000 MiB at
91%, and the bounded NaN/Inf, exception, OOM, and CUDA-error scan was clean.
Raw 40k--50k took 33:01/32:52, or 198.1/197.2 seconds per 1k. This projects
raw-111k near 07:27/07:31; validation remains a downstream artifact-backed
milestone rather than a reason to inspect policy early. No fresh-seed policy
result has been inspected and held-out remains sealed.

The raw-60k checkpoint pair is complete at 04:38:45/04:42:47 BST with sizes
448,855,605/448,855,616 bytes. At 04:44 the trainers were live at raw 61k/60k,
both demo buffers remained exactly 9,253, GPU 3 held 28,002 MiB at 94%, and
the bounded numerical/error scan was empty. Raw 50k--60k took 32:42/32:44,
or 196.2/196.4 seconds per 1k. Projecting the remaining 51k gives raw-111k
near 07:25/07:30. This remains execution evidence only: no fresh-seed policy
result has been inspected, validation has not started, and held-out remains
sealed.

The raw-70k checkpoint pair is complete at 05:11:29/05:15:34 BST with sizes
448,855,611/448,855,623 bytes. At 05:16:26 the trainers were live at raw
71k/70k, both demo buffers remained exactly 9,253, GPU 3 held 28,002 MiB at
98%, and the bounded numerical/error scan was empty. Raw 60k--70k took
32:44/32:47, or 196.4/196.7 seconds per 1k. Projecting the remaining 41k
keeps raw-111k near 07:26/07:30. This is still execution evidence only;
validation has not started, no fresh-seed policy result has been inspected,
and held-out remains sealed.

The raw-80k checkpoint pair is complete at 05:44:08/05:48:18 BST with sizes
448,855,614/448,855,625 bytes. At 05:48:56 the trainers were live at raw
81k/80k, both demo buffers remained exactly 9,253, GPU 3 held 28,002 MiB at
97%, and the bounded numerical/error scan was empty. Raw 70k--80k took
32:40/32:44, or 196.0/196.4 seconds per 1k. Projecting the remaining 31k
keeps raw-111k near 07:25/07:30. This remains execution evidence only;
validation has not started, no fresh-seed policy result has been inspected,
and held-out remains sealed.

The raw-90k checkpoint pair is complete at 06:16:55/06:21:06 BST with sizes
448,855,613/448,855,625 bytes. At 06:21:31 the trainers were live at raw
91k/90k, both demo buffers remained exactly 9,253, GPU 3 held 28,002 MiB at
96%, and the bounded numerical/error scan was empty. Raw 80k--90k took
32:47/32:48, or 196.7/196.8 seconds per 1k. Projecting the remaining 21k
keeps raw-111k near 07:26/07:30. This remains execution evidence only;
validation has not started, no fresh-seed policy result has been inspected,
and held-out remains sealed.

The raw-100k checkpoint pair is complete at 06:49:37/06:53:52 BST with sizes
448,855,614/448,855,624 bytes. At 06:55 the trainers were live at raw
101k/100k, both demo buffers remained exactly 9,253, GPU 3 held 28,002 MiB at
94%, and the bounded numerical/error scan was empty. Raw 90k--100k took
32:42/32:46, or 196.2/196.6 seconds per 1k. Projecting the remaining 11k
keeps raw-111k near 07:26/07:30. This remains execution evidence only;
validation has not started, no fresh-seed policy result has been inspected,
and held-out remains sealed.

Fresh-seed training completed cleanly. The raw-110k checkpoints were written
at 07:22:42/07:26:39 with sizes 448,855,614/448,855,624 bytes, and the final
raw-111k checkpoints were written at 07:26:04/07:28:31 with sizes
448,855,613/448,855,625 bytes. The controller created
`seed34_training_complete` at 07:28:34 only after asserting every one of the
23 registered checkpoints exists, batch/demo-batch are 256/256,
`use_self_imitation=false`, `bc_lambda=0.0`, and every recorded demo-buffer
size is exactly 9,253. Thus Stage 43B now has the complete matched training
artifacts, but no fresh-seed policy result yet.

The pre-registered validation stage started immediately at 07:28:47. Its two
live evaluators use validation seeds 400--449, 50 episodes per checkpoint, 25
parallel environments, and all 23 frozen steps from raw 12.5k through 111k.
Both process environments resolve physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`; they disable JAX preallocation so
the two validation runs share the card. The seed3/4 runner contains no
held-out invocation: it can only finish validation and create the registered
four-seed summary. Held-out therefore remains sealed pending that gate.

At 07:37 both validation curves had produced 3/23 rows. Seed 3 measured
50/58/60% and seed 4 measured 48/62/52% at raw 12.5k/15k/17.5k,
respectively. Each 50-episode point took 126--129 seconds, giving an
artifact-calibrated completion ETA around 08:18--08:20. These are partial
scaling-curve points only: checkpoint selection and the four-seed gate remain
deferred until every registered row exists, and held-out remains sealed.

**Previous-stage result.** Stage 43B validation completed at 08:16 with all
23 checkpoints and 50 episodes from seeds 400--449 for every training seed.
The four validation-selected best results are 74/80/80/66% at raw
25k/42.5k/37.5k/42.5k, for a 75% mean. The online-50k--101k best results are
70/68/74/64%, for a 69% mean. Fixed raw-111k endpoints are 62/64/74/64%, for
a 66% mean. All four demo buffers remained exactly 9,253. The generated
summary has SHA-256
`b15f523742bb11babf169e9e0c1ce16e8824ef373dc61bc6f9c9ffd05cbe9169`.
An independent computation from the seed3/4 raw CSVs plus the locked seed1/2
artifact reproduced every value and the positive held-out gate exactly.

The controller initially failed after validation because direct execution of
`summarize_cqn_no_bc_stage43_full.py` could not resolve its `scripts.*`
import. This did not affect either validation CSV. The CLI/module import was
made entrypoint-safe, a real subprocess regression test was added, and the
focused suite passes 4/4. The repaired CLI then generated the summary above;
`seed34_complete` was restored only after the independent recomputation.

**Interpretation.** The full fresh-seed evidence rules out an all-zero or
small-budget-only effect: late-training success remains nonzero on all four
seeds, every registered validation gate passes, and the matched raw-111k
endpoint mean is 66%. This establishes eligibility for sealed evaluation. It
does not yet establish parity with official CQN-AS because the 66% number is
on validation seeds and the official 64.625% reference is on sealed seeds.

**Next-stage decision.** Evaluate only the four fixed no-BC raw-111k
endpoints against the four official raw-101k endpoints. Each evaluation uses
200 episodes with seeds 800--999 and no checkpoint reselection. The pass
criterion is no-BC four-seed held-out mean at least the locked official
64.625% mean. GPU 3 runs at most two evaluations concurrently.

**Execution.** The separate held-out controller started at 08:19:32 with PID
416109 after CUDA and EGL probes passed. Its registered protocol locks 200
episodes, seeds 800--999, fixed endpoint only, no checkpoint selection, and
official mean 0.64625. The first wave is seed 1/2 (PIDs 416431/416432); both
process environments resolve physical GPU 3 UUID
`GPU-03f1431f-36c0-b258-6ca1-05007175e3eb`, use EGL device 2, and disable JAX
preallocation. The second wave is seed 3/4 and cannot start before this wave
finishes. No held-out result exists yet.

### Stage 43 sealed held-out result and goal completion (2026-08-02 08:36 BST)

#### 1. Previous-stage result

The sealed evaluation completed with all four fixed raw-111k no-BC endpoints,
200 episodes each, and seeds 800--999. No held-out checkpoint was selected or
re-evaluated. The raw results are:

| training seed | no-BC raw 111k | official BC raw 101k | paired delta |
|---:|---:|---:|---:|
| 1 | 64.5% (129/200) | 62.0% (124/200) | +2.5pp |
| 2 | 74.5% (149/200) | 60.5% (121/200) | +14.0pp |
| 3 | 68.0% (136/200) | 62.0% (124/200) | +6.0pp |
| 4 | 58.5% (117/200) | 74.0% (148/200) | -15.5pp |
| **mean / total** | **66.375% (531/800)** | **64.625% (517/800)** | **+1.75pp** |

The pre-held-out validation evidence remains separate: validation-selected
best checkpoints were 74/80/80/66% (75% mean), online-50k-or-later best values
were 70/68/74/64% (69% mean), and fixed raw-111k validation endpoints were
62/64/74/64% (66% mean). The validation summary SHA-256 is
`b15f523742bb11babf169e9e0c1ce16e8824ef373dc61bc6f9c9ffd05cbe9169`;
the held-out summary SHA-256 is
`b35ba75311dd9c6f6a6c07f7a92e5dcfeb51672d578d511c577172c69a5a827d`.
An independent read of all eight raw CSVs reproduced 531/800 versus 517/800,
the two means, and the +1.75pp delta exactly.

#### 2. Interpretation

This passes the preregistered empirical parity criterion: the four-seed no-BC
fixed-endpoint mean is at least the official four-seed fixed-endpoint mean.
It establishes that, on MovePlate under this protocol, CQN-AS can remove BC,
FOSD, and margin losses and still match or exceed the official point estimate.
It rules out the earlier conclusion drawn from all-zero or 10k-only curves
that BC is intrinsically required. It does not establish statistically
significant superiority, universal task generalization, or per-seed dominance:
seed 4 is 15.5pp below its official counterpart and the training-seed variance
is material.

The successful method is an offline-to-online, critic-only RL method. It first
uses 10k demo-only updates, then 101k online interactions with a fixed 9,253-
transition successful-expert replay. At the bootstrap state, the Bellman
candidate set contains both the critic-greedy chunk and the recorded next
behavior chunk; offline demo updates force the recorded continuation, while
online updates use critic candidate-max. The single optimized loss is a C51
critic cross-entropy over scalar-return targets: the executed bin receives its
reward-derived Bellman/Monte-Carlo lower-bound distribution and counterfactual
bins receive the task-valid failure-return distribution. During online
training, dense all-bin targets are enabled only on positive-return samples;
zero-return samples use the canonical Bellman target. Consequently a zero-
return sample is exactly invariant to the recorded action label, while reward
is the only source of a positive action preference.

There is no action likelihood, action-distance regression, expert-margin,
FOSD, actor, flow policy, AWR, policy pretraining, or auxiliary imitation
gradient. The historical seed-1/2 offline configs contain
`use_self_imitation=true`, but the raw-10k snapshots have
`main_loop_iterations=0`, every pretrain CSV row has `env_episodes=0`, and the
flag is only consulted when an online episode ends. It was therefore
operationally unreachable; all four online configs set it false and all four
expert buffers remain fixed at 9,253. The Stage-42 archive had copied the
shared run directory's later online config under an offline-looking filename;
the audit resolves the actual phase-specific Stage-38 configs and the runner
has been corrected for future provenance.

#### 3. Next-stage decision

The registered terminal criterion is met, so no further performance-tuning
stage or held-out reuse is authorized for this goal. The final decision is to
stop the MovePlate search and complete the no-BC CQN-AS goal. Replication on
additional tasks, confidence intervals from more training seeds, or a paper
ablation of candidate backup versus dense return targets would be a new
research protocol with new selection and held-out splits, not a condition for
this completed result.

#### 4. Execution

The held-out controller wrote `heldout_evaluation_complete` and
`heldout_complete` at 08:35:45 BST and exited. The reproducible no-imitation
audit is
`exp_local/cqn_no_bc/stage43_seed12_full_gpu3_20260801215456/stage43_no_imitation_audit.json`
with SHA-256
`9925e116161ff2dd46e6294ab9bf20bcec5e8809e6c2c8148358d5b229794159`;
it checks every phase config, actual offline runtime CSV, fixed online demo
buffer, held-out summary, and source-snapshot provenance. The focused CPU test
suite covering strict objective rejection, zero-return action-label
invariance, single-critic-loss wiring, candidate backup, replay, branch
handoff, validation selection, held-out protocol, and the audit itself passes
207/207 in 379.59 seconds. `git diff --check` passes for the research subset,
and the exact fatal-pattern scan finds no traceback, CUDA error, OOM, NaN, or
Inf in the Stage-43 logs. All Stage-43 processes have exited. Stage-43 released
GPU 3 after evaluation; a separate Stage-171 evaluator subsequently acquired
the card and is not part of this result.
