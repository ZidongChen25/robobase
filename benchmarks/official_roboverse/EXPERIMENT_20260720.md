# A2A RoboVerse reproduction audit - 2026-07-20

Status: complete for the executable two-task proxy. All 18 native 50-episode
evaluations passed `results.py` validation. The exact five-task reproduction
remains blocked by the public-artifact and simulator constraints below.

## Scope boundary

This is not an exact five-task reproduction. The public artifacts do not expose
the two LIBERO paper task IDs or any 100-demo LIBERO set; all 65 public LIBERO90
trajectory files contain exactly 50 episodes. The public Close Box source has
exactly 100 trajectories, but six fail deterministic MuJoCo replay, leaving 94.
Isaac Sim requires an NVIDIA EULA that was not accepted on the user's behalf.

The executable subset is therefore a clearly labelled MuJoCo proxy on Pick Cube
and Stack Cube. The paper does not explicitly state each simulator backend; the
Isaac/MuJoCo reference comes from the public launchers.

## Frozen inputs

- Initial source commit: `596f6220f87734c39dd1e7598bda05b83690a3f7`
- Public data revision: `1133c84a9d5624b7670a75d4043992c57d09b5cd`
- Hardware: six NVIDIA GeForce RTX 5090 GPUs
- Dataset: 100 unique demonstrations per executable proxy task
- Pick logical dataset SHA256: `cfd2bc1bc7e763d0ea6f8171af26c3363d9d722731ab32d8a79778b65f0170ff`
- Stack logical dataset SHA256: `6d5de1caf3d40d43e26b9f60a3a7eb11a886e1d978d4755dae851437dd38a0c6`

The dataset audit decoded every raw MP4 and verified state, action, and RGB
against every Zarr episode bit-for-bit. All 200 complete episode hashes are
unique within and across tasks.

## Controlled settings

- batch 32, seed 42, RGB 256x256, joint-position dimension 9
- horizon 16, observation history 8, prediction 8, execution 8
- A2A 6 Euler NFE; FM-UNet 10 midpoint steps (20 model NFE)
- maximum 250 training batches per epoch
- fresh30: independent 30-epoch cosine horizon
- long200: uninterrupted 200 epochs; E30 and E200 from the same run
- eval: task indices 0 through 49, maximum 300 steps. The paper does not state
  the simulation evaluation count explicitly. The initial official launcher
  sets `max_demo=50`, and all published percentages have 2-point granularity,
  so 50 is strongly supported by the code and table values rather than by an
  explicit paper statement.

## Evaluation limitations

These are fixed-state, in-distribution proxy evaluations, not held-out
generalization tests. The evaluator uses task IDs 0 through 49. The Stack proxy
training source contains all 50 corresponding source indices; Pick contains 40
of those 50. The comparison also has one training seed (`42`). Exact McNemar
tests below pair the 50 fixed evaluation states for one checkpoint, but do not
measure training-seed variance and are not corrected for multiple comparisons.

## Validated proxy results

| Task | Method | Train arm / checkpoint | Success | Paper E30 target | Mean `get_action` / control step |
| --- | --- | --- | ---: | ---: | ---: |
| Pick Cube | initial OT A2A | fresh30 / E30 | 34/50 (68%) | 92% | 1.12 ms |
| Pick Cube | initial OT A2A | long200 / E30 | 29/50 (58%) | 92% | 1.16 ms |
| Pick Cube | initial OT A2A | long200 / E200 | 43/50 (86%) | n/a | 1.04 ms |
| Pick Cube | current Conditional A2A | fresh30 / E30 | 31/50 (62%) | n/a | 1.15 ms |
| Pick Cube | current Conditional A2A | long200 / E30 | 23/50 (46%) | n/a | 1.21 ms |
| Pick Cube | current Conditional A2A | long200 / E200 | 42/50 (84%) | n/a | 0.98 ms |
| Pick Cube | FM-UNet | fresh30 / E30 | 37/50 (74%) | 70% | 8.87 ms |
| Pick Cube | FM-UNet | long200 / E30 | 21/50 (42%) | 70% | 8.53 ms |
| Pick Cube | FM-UNet | long200 / E200 | 45/50 (90%) | n/a | 8.47 ms |
| Stack Cube | initial OT A2A | fresh30 / E30 | 8/50 (16%) | 86% | 1.10 ms |
| Stack Cube | initial OT A2A | long200 / E30 | 12/50 (24%) | 86% | 1.14 ms |
| Stack Cube | initial OT A2A | long200 / E200 | 26/50 (52%) | n/a | 1.17 ms |
| Stack Cube | current Conditional A2A | fresh30 / E30 | 12/50 (24%) | n/a | 1.15 ms |
| Stack Cube | current Conditional A2A | long200 / E30 | 9/50 (18%) | n/a | 1.22 ms |
| Stack Cube | current Conditional A2A | long200 / E200 | 47/50 (94%) | n/a | 1.01 ms |
| Stack Cube | FM-UNet | fresh30 / E30 | 0/50 (0%) | 28% | 8.72 ms |
| Stack Cube | FM-UNet | long200 / E30 | 0/50 (0%) | 28% | 8.74 ms |
| Stack Cube | FM-UNet | long200 / E200 | 21/50 (42%) | n/a | 8.92 ms |

Paper targets are context only: every row above has
`exact_paper_protocol=false` and is not used to claim a paper-result delta.
The evaluator replans only when its eight-action cache is empty. The latency
column is therefore amortized over all control steps, including seven cheap
cache hits per model call; it is not directly comparable to the paper's
single-policy-head latency measurement.

The E30 reproduction verdict is negative. On Pick, initial-OT A2A is 68%
versus the paper's 92%, while FM is 74% versus 70%; even the paper's A2A-over-FM
direction is not reproduced. On Stack, A2A's direction over FM is reproduced
(16% versus 0%), but both absolute values are far below 86% and 28%. These two
proxy tasks cannot establish the paper's five-task result.

On the paired 50 initial states, Pick fresh30 A2A versus FM is not significant
(McNemar p=0.629). Stack fresh30 favors A2A (8 A2A-only successes, zero FM-only;
exact McNemar p=0.0078), but its absolute success is far below the paper table.

The current Conditional setting does not change that direction. On Pick it is
31/50 versus FM's 37/50 (exact McNemar p=0.238); on Stack it is 12/50 versus
FM's 0/50 (12 Conditional-only successes, exact McNemar p=0.00049). Conditional
versus initial OT is not significant on either proxy task (Pick p=0.375, Stack
p=0.424).

The paper's E200 table covers only Close Box Levels 0-3, not all five Table 1
tasks: A2A-S6 reports 100/38/42/38 and FM reports 96/6/6/4 after training on
Level 0. The uninterrupted E200 runs here are therefore a new epoch-sensitivity
comparison, not a direct reproduction of a published five-task E200 table.
The paper's Close Box learning curve and its longer FM/DDPM training discussion
already predict that the baselines improve with more epochs, so baseline
catch-up here is consistent with the published long-training trend.
The completed E200 arms already show material changes. Stack FM rises from 0/50
at long-run E30 to 21/50 at E200, with 21 E200-only successes on the paired
initial states (exact McNemar p=9.54e-7). Stack initial-OT A2A rises from 12/50
to 26/50 (4 E30-only and 18 E200-only; p=0.00434), and remains 10 percentage
points above FM at E200. That E200 cross-method difference is not significant
on the paired starts (13 A2A-only and 8 FM-only; p=0.383). Pick FM rises from
21/50 to 45/50 (2 E30-only and 26 E200-only; p=3.03e-6). Pick initial-OT A2A
rises from 29/50 to 43/50 (1 E30-only and 15 E200-only; p=0.000519). At Pick
E200, FM is 45/50 and A2A is 43/50; the two extra FM successes are not a
significant paired difference (p=0.5).

Current-Conditional A2A also changes materially with training length. Pick
rises from 23/50 at long-run E30 to 42/50 at E200 (1 E30-only and 20 E200-only;
p=2.10e-5). At E200 it is statistically indistinguishable from initial OT
(42/50 versus 43/50; p=1.0) and FM (42/50 versus 45/50; p=0.25). Stack rises
from 9/50 to 47/50 (38 E200-only; p=7.28e-12), and at E200 exceeds both initial
OT (47/50 versus 26/50; p=1.94e-5) and FM (47/50 versus 21/50; p=2.16e-7) on
these paired states. This large task-specific matcher interaction reinforces
that the unpublished Table 1 code identity cannot be inferred from the repo's
current config.

The E200 sensitivity verdict is therefore different from the 30-epoch view:
all six task-method long runs improve significantly, the Pick methods converge
to 84-90%, and initial-OT versus FM is not significantly different on either
task at E200. Current-Conditional remains similar on Pick but reaches 94% on
Stack. This is evidence that the short training budget and matcher version both
affect the ranking; it is not evidence of multi-seed or held-out superiority.

## Source-disjoint reevaluation

The same 18 checkpoints were reevaluated without retraining on later public
trajectory initializations. Pick uses IDs 125 through 174; Stack uses IDs 100
through 149. The Pick training source IDs are a 100-element successful subset
of 1 through 124, and the Stack training source IDs are 0 through 99, so both
evaluation ranges have zero source-ID overlap. Canonical hashes of the public
initial-state payloads also show 50 unique evaluation states per task and zero
content duplicates with the corresponding 100 training initial states.
After all evaluations, a fresh full-array Zarr hash reproduced the frozen Pick
and Stack logical digests and every array digest. The newest dataset file also
predates the earliest checkpoint. This supports the current post-hoc audit,
although the original train manifests still lack an embedded dataset digest as
described under Evidence-chain limitations.

| Method | Train arm / checkpoint | Pick held-out (change from IDs 0-49) | Stack held-out (change from IDs 0-49) |
| --- | --- | ---: | ---: |
| initial OT A2A | fresh30 / E30 | 15/50 (30%, -38 pp) | 2/50 (4%, -12 pp) |
| initial OT A2A | long200 / E30 | 17/50 (34%, -24 pp) | 3/50 (6%, -18 pp) |
| initial OT A2A | long200 / E200 | 13/50 (26%, -60 pp) | 1/50 (2%, -50 pp) |
| current Conditional A2A | fresh30 / E30 | 16/50 (32%, -30 pp) | 3/50 (6%, -18 pp) |
| current Conditional A2A | long200 / E30 | 14/50 (28%, -18 pp) | 2/50 (4%, -14 pp) |
| current Conditional A2A | long200 / E200 | 15/50 (30%, -54 pp) | 1/50 (2%, -92 pp) |
| FM-UNet | fresh30 / E30 | 28/50 (56%, -18 pp) | 0/50 (0%, +0 pp) |
| FM-UNet | long200 / E30 | 17/50 (34%, -8 pp) | 0/50 (0%, +0 pp) |
| FM-UNet | long200 / E200 | 31/50 (62%, -28 pp) | 4/50 (8%, -34 pp) |

The held-out result reverses the earlier in-distribution E200 interpretation.
On Pick, FM-UNet improves from 17/50 at E30 to 31/50 at E200 (paired exact
McNemar p=0.00936) and exceeds initial-OT A2A 31/50 to 13/50 at E200
(p=0.000277). Current A2A is also below FM at E200, 15/50 to 31/50
(p=0.000145). Neither A2A variant improves significantly from E30 to E200.
Stack is near floor for every method: the E200 values are 1/50, 1/50, and 4/50,
so the observed differences are not statistically resolved.

These are source-disjoint fixed-state proxy results, not an IID estimate of the
paper task distribution. The later contiguous IDs may be harder than IDs 0
through 49. As a replay control, the public expert actions were executed from
the same states with the same MuJoCo reset, camera, lighting, one-environment,
and 300-step evaluator semantics. Stack replayed successfully on 50/50 states;
Pick replayed on 38/50. Repeating each of the 12 failed Pick trajectories three
times produced 0/3 successes for every ID, so that gap is deterministic under
this setup. The 76% Pick replay rate is a proxy transfer reference, not a true
task-solvability ceiling: a learned policy can take a different path.

Old-versus-held-out percentage-point changes are therefore descriptive cohort
shifts, not paired model regressions. The fair comparison is between methods on
the same held-out IDs. Stack's near-zero policy result cannot be attributed to
unreplayable expert trajectories, while Pick combines policy generalization
error with a measurable expert-to-proxy replay gap. It is nevertheless clear
that the previous 40/50 Pick and 50/50 Stack training-state overlap made those
results too optimistic to support a generalization claim.

## Model-size control

The official checkpoints are not parameter matched. Excluding optimizer state,
both A2A variants have 34,657,036 tensor elements in each model state, while
FM-UNet has 133,922,637 (3.864x as many). A2A/current have identical state-dict
structure. The public FM trainer itself logs 122,746,000 UNet parameters, while
the paper appendix describes an approximately 28M-parameter DDPM-UNet and says
architectures were kept comparable where possible. Together with A2A's 6 Euler
model calls versus FM's 10 midpoint steps (20 model calls), this means neither
success nor latency differences can be attributed to the source distribution
alone. Evaluation also shared the host with training jobs and was not a
dedicated warmed latency benchmark.

## Public-code ambiguity

The initial commit and paper text use Exact OT. Official `main` changed standard
A2A to `ConditionalFlowMatcher` in commit `131d493` before paper v2. The initial
OT wrapper re-pairs source/target minibatch rows without re-pairing
`global_cond`; a seed-42 probe retained only 2/32 condition-aligned positions.
Both implementations are therefore tracked separately:

- `a2a`: initial-release Exact OT matcher;
- `a2a_current`: current-main Conditional matcher sensitivity arm.

No checkpoint, tagged paper code release, or training Zarr is public, so Table 1
cannot currently be assigned to either implementation with confidence.
There is also a paper/code loss mismatch: the paper writes latent consistency
as L1, while the released trainer uses MSE for the latent term and L1 only for
the decoded action term.

## Upstream quirks retained

- A hard-coded validation index leaves 99 of 100 episodes in the train sampler;
  normalization still uses all 100.
- Scheduler length uses the uncapped loader while each epoch stops at 250
  batches. Both fresh30 and long200 therefore traverse about 89.3% of the Pick
  cosine horizon and 55.2% of the Stack horizon.
- A2A supervises the executed 8-step slice; FM trains the full 16-step horizon.
- The official requirements do not pin torchcfm or diffusers. The isolated
  environment uses torchcfm 1.0.7 and diffusers 0.39.0.

## Evidence-chain limitations

The result aggregator strictly checks 50 native episode records, final counts,
checkpoint hashes, method identity, and that E30/E200 belong to the same
uninterrupted long run. The dataset logical hashes come from separate full
raw-to-Zarr audits, however; current train/eval manifests bind the dataset path
but not that logical hash. The launcher now separates declared
task/demo-count/simulator agreement from a true reproduction claim:
`declared_paper_controls_match` records the former, while
`exact_paper_protocol` remains false until the unpublished paper assets and
evaluation identity can be bound. These proxy rows were already `false`, so the
stricter rule does not change their label or counts.

## Artifact roots

- Training: `/home/zc1525/.local/share/a2a-roboverse-paper/proxy_runs_20260720`
- Evaluation: `/home/zc1525/.local/share/a2a-roboverse-paper/proxy_evals_20260720`
- Source-disjoint evaluation: `/home/zc1525/.local/share/a2a-roboverse-paper/proxy_evals_heldout_20260720`
- Proxy data: `/home/zc1525/.local/share/a2a-roboverse-paper/proxy_data`
- Pick provenance: `proxy_data/audits/pick_cube_proxy_provenance.json`
- Stack provenance: `proxy_data/audits/stack_cube_proxy_provenance.json`
- Final JSON: `proxy_evals_20260720/comparison.json`
- Final CSV: `proxy_evals_20260720/comparison.csv`
- Source-disjoint JSON: `proxy_evals_heldout_20260720/comparison.json`
- Source-disjoint CSV: `proxy_evals_heldout_20260720/comparison.csv`
- Combined cohort JSON: `proxy_evals_heldout_20260720/comparison_official_and_heldout.json`
- Combined cohort CSV: `proxy_evals_heldout_20260720/comparison_official_and_heldout.csv`
- Pick expert replay: `proxy_evals_heldout_20260720/pick_cube/expert_replay_evaluator_semantics.log`
- Pick failed-ID repeat replay: `proxy_evals_heldout_20260720/pick_cube/expert_replay_failed_ids_3attempts.log`
- Stack expert replay: `proxy_evals_heldout_20260720/stack_cube/expert_replay_evaluator_semantics.log`
