# Progress-potential shaping for CQN-AS — implementation report (2026-08-18)

Implements the blueprint's *Recommended composition*: **item 4** (raw
per-transition progress labels in replay) + **insertion C** (auxiliary
progress head) + **insertion B, variant B1** (target-side shaped reward
scalar). Insertion A (baking shaped rewards into replay) was explicitly
rejected and is **not** implemented — stored rewards and `mc_return` stay raw.

All knobs default to zero/null, and the default configuration is the exact
legacy graph, parameter tree, RNG stream and metric set.

---

## 1. What the mechanism is

For each replay transition the C51 target's reward scalar becomes

```
r~ = r + lambda * (gamma_n * bootstrap * Phi(s') - Phi(s))          (Ng et al. 1999)
target_z = r~ * q_reward_scale + gamma_n * bootstrap * support
```

* `Phi(s) = clip(V_prog(s), 0, 1)`, `V_prog` = a state-only `ExpectileValueHead`
  (the same class AWR uses) reading **stop-gradient** encoder features. It
  never sees an action, so it cannot introduce an action-label objective.
* `bootstrap` (= `1 - terminal`, unless `always_bootstrap`) supplies
  `Phi(terminal) = 0` for free, so a terminal success target is exactly
  `1 - lambda * Phi(s_{T-1})`.
* `gamma_n` is the *same* `discounts` array the projection already multiplies
  into the support, so the shaping telescopes exactly at n-step / chunk
  granularity.
* `Phi` is trained by expectile regression (tau = `progress_expectile_tau`)
  onto the **raw** stored label `p_t = (t+1)/T`, optionally masked to
  successful episodes / genuine demos.

Nothing gamma- or lambda-dependent is ever written to replay, so a
lambda/gamma/tau sweep never invalidates the shared demo cache
(`demo_cache_key` hashes the element set, not method knobs).

---

## 2. Files and insertion points

### 2.1 Config — `robobase/cfgs/method/cqn_as.yaml` (new block above `awr_beta`)

| knob | default | meaning |
|---|---|---|
| `progress_potential_weight` | `0.0` | `lambda` for the target-side potential. `0.0` = exact legacy target. Also the enable gate for the shaping path and the C51 bound check. |
| `progress_potential_schedule` | `null` | schedule string (`"linear(0.25,0.0,50000)"`), same convention as `bc_lambda_schedule` / `bin_explore_schedule`. Evaluated with `utils.schedule(sched, step)` and passed as a traced scalar so it does not retrigger jit. Requires `progress_potential_weight > 0`. |
| `progress_head_weight` | `0.0` | weight of the expectile regression loss on `V_prog`. |
| `progress_expectile_tau` | `0.9` | tau in `|tau - 1{u<0}| * u^2`, `u = p_t - V_prog(s_t)`. |
| `progress_success_gated` | `true` | regress only where `progress_valid = 1`. |

The head parameter group is created when **either** `progress_head_weight > 0`
**or** `progress_potential_weight > 0`. Unlike `awr_beta` it does **not**
require `separate_bc_policy`.

### 2.2 Raw labels in replay

* `robobase/workspace.py`
  * `_progress_label_enabled(cfg)` (new, next to `_mc_return_anchor_enabled`) —
    true for `cqn_as`/`cqn_flow` when either progress knob is positive.
  * `_create_default_replay_buffer`: registers
    `progress` (`Box(0,1,(),float32)`) and
    `progress_valid` (`Box(0,1,(),uint8)`) alongside `mc_return`.
  * lazy-replay guard extended: `NotImplementedError` when progress labels are
    requested with `lazy_replay` active (labels are stamped at episode close).
  * `_add_to_replay` episode-close block: `store_progress` flag,
    `episode_length = max(len(ep), 1)`,
    `progress_valid = uint8(task_success)`, and per transition
    `info["progress"] = float32((transition_index + 1) / episode_length)`
    written next to `info["mc_return"]`. `mc_return` continues to read the
    **raw** discounted-return array.
* `robobase/utils.py` — `add_demo_to_replay_buffer`: same closed form,
  `extra["progress"] = float32((index + 1) / len(ep))`.
  **Validity is derived from the raw reward sum** (`sum(rew) > 0.25`), never
  from `info["demo"]`, because `env.treat_all_demos_as_expert` relabels failed
  demos as expert and a failed demo's time index is not progress.

Demo and online paths use one identical formula `(t+1)/T`, so the two label
distributions live on one scale (test asserts both against the same array).

### 2.3 Progress head + target-side potential

* `robobase/method/cqn.py`
  * `progress_shaped_rewards(rewards, discounts, bootstrap, phi, phi_next, weight)`
    — new module-level helper (exported in `__all__`), the single definition of
    the Ng form.
  * `_build_update_fn`: new closure flags `use_progress_head`,
    `use_progress_shaping`, `progress_head_weight`, `progress_expectile_tau`,
    `progress_success_gated`.
  * `update_impl(...)` gained three positional args
    `progress_labels, progress_valid, progress_lambda`, threaded **after**
    `support_mask_ce_weight` and **before** `action_key`.
  * `loss_fn`: `progress_phi_raw` (unclipped head output, used by the loss),
    `progress_phi` / `progress_phi_next` (clipped + `stop_gradient`, used by the
    potential); shaped scalar fed into `project_categorical` at **both** call
    sites — the main target and the token-split auxiliary target. The auxiliary
    target uses `Phi(aux_next_obs)` with `aux_discounts`/`aux_bootstrap`, which
    is the correct endpoint pair for the long-horizon backup.
  * Progress head loss added into `critic_loss` (only when
    `progress_head_weight > 0`).
  * `split_progress_args` added to the variadic dispatcher; every existing arm
    (`use_mc_returns × use_bc_schedule`, support-mask gate, token-split aux
    tail) keeps its positional alignment. When the gate is off the impl gets
    zero-filled placeholders, i.e. the identical legacy computation.
  * `_progress_update_args(batch, step)` — reads `progress`/`progress_valid`
    (raises `KeyError` naming the missing element) and resolves the scheduled
    lambda; appended to `canonical_mc_args` in `CQN.update`.
* `robobase/method/cqn_as.py`
  * `CQNASpec` fields + `cqn_as_spec_from_cfg` readers + `CQNAS.__init__` kwargs
    (mirroring the `awr_beta` pattern).
  * Head creation next to the flow/AWR head block. The init key is
    `jax.random.fold_in(self.rng_key, 0x9209)` rather than a `split`, so adding
    the head **does not consume the rollout/update RNG stream** — a
    progress-enabled arm keeps byte-identical action keys to its control.
  * `separate_bc_policy` update graph: same Phi/shaping/head-loss block, three
    new positional args and a `split_progress_args` dispatcher for both the
    `cv_rct` and plain wrappers; `update()` appends `*progress_args` after
    `*direct_q_extra_args`.
* `robobase/factory.py` — five new kwargs forwarded from the spec.

### 2.4 Validation (raises)

At `cqn_as_spec_from_cfg` (config-visible checks):

1. `method.name != cqn_as` → `NotImplementedError` (cqn_flow's graph would
   silently ignore the potential).
2. `direct_scalar_q=true` → `NotImplementedError`.
3. `env.env_name == bigym` and `env.truncate_demo_at_success != true` →
   `ValueError` (96% post-success-tail artifact makes `(t+1)/T` flat).
4. `lazy_replay_enabled(cfg)` → `ValueError`.

At `CQNAS.__init__` (mirroring the block at the `awr_*` asserts):

5. non-negative weights, `0 < progress_expectile_tau < 1`, parsable schedule
   string (fails fast at construction, not at the first update).
6. `pessimistic_twin_critic` → `NotImplementedError` (that graph is untouched).
7. `progress_potential_weight > 0` together with **any** of
   `mc_lower_bound_target`, `episodic_success_q_target`,
   `ordered_success_return_mix > 0` → `ValueError`. Raw MC returns and shaped
   Bellman targets must not mix; the MC-shift variant (`mc - lambda*Phi(s_t)`)
   is deliberately left for later.
8. C51 support bound: `(1 + progress_potential_weight) * q_reward_scale <= v_max`,
   the same "scaling must fit the configured C51 support" contract as
   `q_reward_scale` (yaml lines 127-128). With `v_max = 2.0`,
   `q_reward_scale = 1.0` this admits `lambda <= 1.0`.

### 2.5 Diagnostics (only emitted when the corresponding gate is on)

| metric | when | meaning |
|---|---|---|
| `progress_head_loss` | head enabled | weighted expectile loss |
| `progress_head_value_mean` | head enabled | mean `clip(Phi, 0, 1)` over the batch |
| `progress_label_mean` | head enabled | mean stored `(t+1)/T` |
| `progress_valid_fraction` | head enabled | fraction of the batch the head trains on |
| `progress_potential_lambda` | shaping on | the *scheduled* lambda actually used this step |
| `progress_shaping_clip_frac` | shaping on | fraction of `(sample, atom)` shaped targets outside `[v_min, v_max]` — clipping breaks the telescope and destroys invariance, so this must stay ~0 |

---

## 3. Tests

New tests in `tests/unit/test_cqn_as.py` (14) and
`tests/unit/utils/test_add_demo_to_replay_buffer.py` (2):

| test | covers |
|---|---|
| `test_progress_knobs_default_to_exact_legacy_and_add_no_params` | (a) defaults, no `progress_value` param, no `progress*` metrics |
| `test_progress_label_gate_tracks_either_consumer` | gate predicate |
| `test_progress_shaped_rewards_terminal_drops_the_next_potential` | (d) terminal: `r - lambda*Phi(s)`; `lambda=0` is identity |
| `test_progress_shaping_telescopes_over_a_whole_episode` | (e) `sum_t gamma^t F_t = -lambda*Phi(s_0)` with `Phi(terminal)=0` |
| `test_cqn_as_progress_head_is_created_on_the_canonical_platform` | (c) head exists **without** `separate_bc_policy`, trains, logs |
| `test_cqn_as_progress_head_leaves_the_legacy_critic_update_bitwise` | head loss does not perturb critic params (stop-grad features, separate param group, `fold_in` RNG) |
| `test_cqn_as_zero_initialized_potential_is_the_exact_legacy_target` | (a) `lambda=0.25` with a zero-init head (`Phi≡0`) is bitwise legacy |
| `test_cqn_as_constant_potential_equals_pre_shifted_rewards` | end-to-end: constant `Phi=0.4`, `lambda=0.25` reproduces a legacy update on pre-shifted rewards (covers terminal handling in the real graph) |
| `test_cqn_as_progress_success_gate_censors_failed_episodes` | gating on → zero loss, head untouched; gating off → trains |
| `test_cqn_as_progress_potential_schedule_anneals_lambda` | schedule string wiring |
| `test_cqn_as_progress_potential_rejects_raw_mc_targets` (×2) | (f) mutual exclusion |
| `test_cqn_as_progress_potential_must_fit_the_c51_support` | support bound |
| `test_cqn_as_progress_requires_truncated_demo_tails_on_bigym` | env-coupled assert |
| `test_cqn_as_progress_launch_composes_on_the_demo_driven_platform` | (g) config composition on the real launch |
| `test_online_progress_labels_are_monotone_and_success_gated` | (b) online labels `(t+1)/T`, strictly increasing, ends at 1.0, `valid=0` on failure |
| `test_add_demo_to_replay_buffer_stores_monotone_raw_progress_labels` | (b) demo labels, same closed form = demo/online parity |
| `test_add_demo_progress_validity_reads_raw_reward_not_the_demo_label` | `treat_all_demos_as_expert` risk: `demo=1` but `progress_valid=0` |

### Results

```
pytest tests/unit/test_cqn_as.py \
       tests/unit/utils/test_add_demo_to_replay_buffer.py \
       tests/unit/replay_buffer/ tests/unit/test_factory.py \
       tests/unit/test_workspace_fast.py -q
-> 317 passed, 11 failed  (7m27s, JAX_PLATFORMS=cpu)

pytest tests/unit/test_cqn_as.py -k progress   -> 14 passed
pytest tests/unit/utils/test_add_demo_to_replay_buffer.py -> 11 passed
```

All 16 new tests pass. The 11 failures are **pre-existing and unrelated**: the
`*_is_one_demo_agnostic_*` / `mc_lower_bound_is_reward_only` family, all failing
on the same two BC-anchor diagnostics (`bc_agreement` vs `bc_online_agreement`)
which are demo-identity-keyed by construction and therefore cannot be equal
between a demo batch and an online batch. Verified byte-identical failures on a
clean `git worktree` at `HEAD` (`0198600`), i.e. before any change in this
work:

```
10 failed, 176 deselected   (-k "demo_agnostic or reward_gated_and_demo_agnostic")
 1 failed                   (test_cqn_as_mc_lower_bound_is_reward_only_...)
```

`tests/unit/test_cqn_flow.py` was also checked because the shared
`cqn.py` update dispatcher is on its call path: **11 failed / 121 passed** in
the working tree vs **12 failed / 120 passed** at `HEAD` — one fewer failure,
none added. (Those failures are a separate, pre-existing
`legacy_update_fn() takes 12 positional arguments but 13 were given` signature
drift in `cqn_flow.py`, unrelated to progress shaping.)

Extra manual check (not a unit test): with `Phi = 1` everywhere, `lambda = 0.5`,
51 atoms on `[-2, 2]` and sparse `{0,1}` rewards, `progress_shaping_clip_frac`
is exactly `0.0` — the worst realistic case stays inside the support.

---

## 4. Phi quality gate — `scripts/probe_progress_sandwich.py`

CPU-only, no training. Loads a run's `.hydra/config.yaml`, rebuilds an
eval-shaped `Workspace` (no envs, no persistence), re-loads the demos, and for
each snapshot computes `Phi_hat = clip(V_prog(features), 0, 1)` against the
stored `mc_return` (for a demo truncated at first success, `mc_return` *is* the
realised discounted value of that state under the demo policy). Reports the
Gupta sandwich constants

```
c1 = min_s Phi(s)/V(s),   c2 = max_s Phi(s)/V(s),   gate: c2/c1 >> 3 → no benefit
```

plus an outlier-robust `c1_p05` / `c2_p95` pair (`min`/`max` are single-sample
statistics), the Spearman rank correlation of `Phi` vs `V`, and
`mean |Phi - progress_label|`. It force-enables `mc_return_weight` in the probe
config so the reference is available even when the run itself stored no MC
returns (replay element only; no loss runs).

```
python scripts/probe_progress_sandwich.py --run-dir exp_local/<run> \
    [--only-steps 50000,100000] [--batches 8] [--output reports/<run>_sandwich.csv]
```

Caveat printed by the script: a normal CQN-AS run trains on every demo it
loads, so the default probe batch is in-sample and `c2/c1` is an optimistic
bound. `--demos N --holdout-only` loads extra demos but the replay sampler does
not expose the source demo index, so per-transition hold-out is reported as not
enforceable rather than silently faked.

---

## 5. Known limitations

1. **Nonstationary Phi breaks strict policy invariance.** Ng's theorem assumes a
   fixed `Phi`. Here `Phi` is trained online, so the shaped MDP drifts.
   Devlin & Kudenko (2012) license shaping at *sample* time with the current
   `Phi` (drift becomes variance, not bias, in the tabular limit), but the deep
   case has no such guarantee. **Mitigation: anneal `lambda` to 0** with
   `progress_potential_schedule`; the recommended first arm does exactly this.
   The auxiliary head warm-up arm (`progress_head_weight` only) has no
   invariance exposure at all because the target is untouched.
2. **The label is a task clock, not a task state.** Two states at the same `t`
   with different real progress get the same label. `tau = 0.9` fits the
   optimistic envelope, which partially absorbs this, but it is a modelling
   assumption, not a fix.
3. **Failed online episodes are censored, not labelled.** With
   `progress_success_gated=true` the head trains only on successes + genuine
   demos, which re-couples the signal to the success-gating dependency (F3).
   Setting it false teaches "the last step of a failure is progress 1", which
   is worse.
4. **Raw-MC consumers are refused, not shifted.** `mc_lower_bound_target`,
   `episodic_success_q_target` and `ordered_success_return_mix` raise instead of
   applying `mc - lambda*Phi(s_t)`. Implementing the shift is the natural
   follow-up if a shaped arm needs an MC lower bound.
5. **Only two update graphs are wired**: the canonical CQN-AS path and the
   `separate_bc_policy` path. `pessimistic_twin_critic`, `direct_scalar_q` and
   `cqn_flow` raise rather than silently ignoring the knobs.
6. **Cache-key discipline holds for lambda/gamma/tau but not for the label
   rule.** Adding `progress` to the storage signature mints a new demo-cache key
   automatically. If the *rule* ever changes (`(t+1)/T` → remaining-time,
   different validity test), `CACHE_SCHEMA_VERSION` must be bumped or a
   `progress_label_rule` string added to `demo_cache_key`.
7. **Chunk granularity.** Online labels index executed-action transitions
   (receding-horizon chunks), demo labels index primitive demo steps.
   Normalising by `T` puts both on `[0,1]`, but the two clocks are not the same
   physical time; this is the same granularity mismatch `mc_return` already has.
8. **Weight decay touches an unused head.** With `progress_potential_weight > 0`
   and `progress_head_weight == 0`, the head receives no gradient but adamw's
   decoupled weight decay still shrinks it (it stays at ~0, so `Phi ≈ 0` and the
   shaping is inert). Do not run that combination expecting a nonzero potential.

---

## 6. First 2-seed arms (wide-first, per the seed-budget policy)

Both arms sit on the `sandwich_remove` task and the standard demo-driven
platform. Nothing below has been launched.

**Arm A — auxiliary head warm-up only (no target change).**
Pure representation probe: measures whether `Phi` is learnable and passes the
sandwich gate before spending any invariance risk.

```
python train.py \
  launch=cqn_as_pixel_bigym_demo_driven \
  env=bigym/sandwich_remove \
  method.progress_head_weight=1.0 \
  method.progress_expectile_tau=0.9 \
  method.progress_success_gated=true \
  seed=1        # and seed=2
```

**Arm B — head + annealed target-side potential.**
Only launch this if Arm A's `progress_head_value_mean` is nonzero, its
`progress_head_loss` is falling, and `probe_progress_sandwich.py` reports
`ratio_p90` below ~3.

```
python train.py \
  launch=cqn_as_pixel_bigym_demo_driven \
  env=bigym/sandwich_remove \
  method.progress_head_weight=1.0 \
  method.progress_expectile_tau=0.9 \
  method.progress_success_gated=true \
  method.progress_potential_weight=0.25 \
  'method.progress_potential_schedule="linear(0.25,0.0,50000)"' \
  seed=1        # and seed=2
```

`progress_potential_schedule` must be quoted on the CLI — hydra parses bare
`linear(...)` as a grammar function (same as `bc_lambda_schedule`).

Both arms pair against the existing 75.25% `sandwich_remove` baseline gate.

**Watch during the run**
* `progress_shaping_clip_frac` — must stay ~0. Any sustained nonzero value
  means the C51 projection is clipping and the telescope (hence the invariance
  argument) is broken; drop `lambda` or raise `v_max`.
* `progress_head_value_mean` — flat at 0 means the head never left its zero
  init and the shaping is inert.
* `progress_valid_fraction` — with success gating this tracks the success rate
  of the batch; near 0 early means the head is training on demos only.
* Trott (NeurIPS 2019) loitering signature: rollouts stalling at high `Phi`
  without success. Monitor rollout `Phi` traces before extending seeds.
