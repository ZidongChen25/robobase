# CQNAS without imitation loss

## Goal and non-negotiable boundary

The only research goal is to remove BC/imitation from original CQNAS and learn
MovePlate from expert demonstrations through RL targets alone, while matching
or exceeding the original policy. There is no actor pretraining, action
cross-entropy, FOSD, large-margin imitation, AWR-weighted cloning, flow-policy
fitting, replay-next expert target, or frozen behavior policy. The deployed
policy is the canonical CQNAS `argmax Q`.

The launch enables `method.strict_demo_rl_only=true`. Configuration composition
must fail if any forbidden path is enabled. Demonstrations may contribute only
ordinary RL transition information `(s, a, r, s')` and completed returns in a
stage that explicitly preregisters them.

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
- Final goal gate: validation-selected no-BC models must reach the original
  three-seed mean `72.0%`, then reach or exceed the official held-out mean
  `64.6%` on seeds 800--999 without checkpoint reselection.

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

Exact Stage-1 run paths and measured results will be appended after launch.
