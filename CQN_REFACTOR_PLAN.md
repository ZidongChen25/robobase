# CQN-AS Decoupling Refactor Plan

Branch: `refactor/cqn-as-decouple` (worktree `/home/zc1525/robobase_jaxflat_refactor`, base `bdc28d0`).
Owner rule (permanent, from the user): **original/official algorithm files are frozen —
never edit them in place. Every modification lives in its own file, created by
copy-paste (or subclass-with-copied-overrides) and then modified there.**

## Why

`robobase/method/cqn_as.py` grew 811 → 6706 lines and `cqn.py` 793 → 2683 by in-place
research edits on top of the official-fidelity JAX port (import commit `173a01f`).
A 2026-08-19 two-agent audit confirmed the baseline default path at HEAD is math- and
RNG-equivalent to the pristine import (only: NaN-guard on the failure path, ulp-level
dueling reassociation, extra diagnostics). This refactor separates the two cleanly.

## Phase R1 — two-tier split (single agent)

| File | Action |
|---|---|
| `robobase/method/cqn.py` | RESTORE byte-exact from `git show 173a01f:robobase/method/cqn.py`. FROZEN afterwards. |
| `robobase/method/cqn_as.py` | RESTORE byte-exact from `git show 173a01f:robobase/method/cqn_as.py`. FROZEN afterwards. |
| `robobase/method/cqn_research.py` | NEW = byte-exact copy of `git show bdc28d0:robobase/method/cqn.py`. Expected zero edits (it has no intra-cqn imports). |
| `robobase/method/cqn_as_research.py` | NEW = byte-exact copy of `git show bdc28d0:robobase/method/cqn_as.py`, then ONLY its `from robobase.method.cqn import ...` lines re-pointed to `robobase.method.cqn_research`. No other edits. |
| `robobase/cfgs/method/cqn_as.yaml` | Change ONLY the `_target_` line → `robobase.method.cqn_as_research.CQNAS`. All flags/values untouched → every existing experiment config keeps working bit-identically. |
| `robobase/cfgs/method/cqn.yaml` | Same: `_target_` → `robobase.method.cqn_research.CQN`. |
| `robobase/cfgs/method/cqn_as_official.yaml` | NEW = byte-exact `git show 173a01f:robobase/cfgs/method/cqn_as.yaml` (its `_target_` already points at `robobase.method.cqn_as.CQNAS` = the restored pristine class). Change only `name:` → `cqn_as_official`. |
| `robobase/cfgs/method/cqn_official.yaml` | NEW analog for plain CQN (from `git show 173a01f:robobase/cfgs/method/cqn.yaml`, `name:` → `cqn_official`). |
| `robobase/factory.py` | (a) Re-point the research imports (`cqn_spec_from_cfg`, `cqn_as_spec_from_cfg`, and the `from robobase.method.cqn_as import CQNAS` at ~line 308) to the `*_research` modules — the existing `"robobase.method.cqn_as.CQNAS" → "cqn_as"` mapping entry changes its KEY to the research target string so the whole existing construction path is reused unchanged. (b) ADD mappings `"robobase.method.cqn_as.CQNAS" → "cqn_as_official"` and `"robobase.method.cqn.CQN" → "cqn_official"` with construction branches transplanted from `git show 173a01f:robobase/factory.py` (pristine spec builders + pristine kwargs from the pristine modules). |
| `robobase/method/__init__.py` | Route both: existing names keep resolving as today (research), add official exports. |
| Importer re-pointing (mechanical, `cqn`/`cqn_as` → `*_research`) | `robobase/method/cqn_flow.py`, `cqn_direct_q.py`, `djcqn.py`; `scripts/t1_value_probes.py`, `dueling_stream_transplant.py`, `dueling_stream_autopsy.py`, `analyze_cqn_value_fidelity.py`, `analyze_cqn_branch_counterfactual.py`; `tests/unit/test_cqn_as.py`, `test_rl.py`, `test_fscqn_support_mask.py`, `test_cqn_exploration_checkpoint_compat.py`, `test_cqn_as_satisficing_floor.py`, `test_cqn_as_return_gated_margin.py`, `test_cqn_as_relative_floor.py`, `test_cqn_as_label_smoothing.py`. Point everything at the research modules (superset of pristine symbols). |
| `scripts/refactor_equivalence_check.py` | NEW CPU-only harness: synthetic `spaces.Dict` obs (rgb `[V,C,H,W]` uint8 + `low_dim_state`) + `spaces.Box` action `[T,D]`; instantiate official `CQNAS` (pristine yaml defaults) and research `CQNAS` (current cqn_as.yaml defaults); run `act()` and one `update()` on a synthetic batch each; assert finiteness and matching output shapes. Research-vs-official numeric equality is NOT asserted (research default path adds NaN-guard/diagnostics; ulp reassociation) — shape/finite/smoke only. |

### R1 verification protocol (all must pass, include outputs in report)
1. `diff <(git show 173a01f:robobase/method/cqn.py) robobase/method/cqn.py` → empty. Same for `cqn_as.py`.
2. `diff <(git show bdc28d0:robobase/method/cqn.py) robobase/method/cqn_research.py` → empty. `diff <(git show bdc28d0:robobase/method/cqn_as.py) robobase/method/cqn_as_research.py` → ONLY the import lines.
3. `git diff bdc28d0 -- robobase/cfgs/method/cqn_as.yaml` → only the `_target_` line.
4. CPU smoke: `JAX_PLATFORMS=cpu PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor /home/zc1525/robobase_jaxflat/.venv/bin/python scripts/refactor_equivalence_check.py` passes.
5. `JAX_PLATFORMS=cpu ... -m pytest tests/unit/test_cqn_as.py tests/unit/test_rl.py -x -q` (and the other re-pointed cqn tests) passes.
6. `python -c "import robobase.factory"` and hydra-load of both method yamls succeed.

## Phase R2 — one file per research line (parallel agents, after R1)

Pattern (mandatory): `robobase/method/cqn_as_<line>.py` **subclasses the pristine
classes** (`from robobase.method.cqn_as import CQNAS`, `from robobase.method.cqn import CQN`)
and overrides methods by COPY-PASTING the pristine method body into the subclass and
applying only that line's changes. No file may import from `*_research`. Each line also
ships `robobase/cfgs/method/cqn_as_<line>.yaml` (= pristine official yaml + `_target_` +
only that line's keys) and a minimal CPU test. Agents DO NOT edit shared files
(`factory.py`, `method/__init__.py`, plan doc) — registration snippets go in their
reports and are integrated centrally afterwards.

Line taxonomy (flag families; source of truth = `robobase/cfgs/method/cqn_as.yaml`
comments + git history of each flag):
1. `structured-exploration`: structured_exploration_*, bin_flip_*, bin_explore_*, low_dim_mask_*, post_ensemble_*_flip*, level_override/random_levels_from
2. `dense-return` (no-BC line): dense_return_*, q_reward_scale, return_gated_margin*, episodic_success_q_target, ordered_success_return_mix, unseen_return_floor_*, sequence_aligned_mc_discount, strict_demo_rl_only, strict_allow_reward_only_success_replay
3. `frozen-support-mask` (FS-CQN): use_frozen_support_mask + subkeys, support_mask_decode
4. `token-split`: token_split_*
5. `mc-rct`: mc_return_*, mc_lower_bound_target, causal_rct_*, cv_rct_*
6. `progress-shaping`: progress_* (Ng-form potential, expectile head)
7. `awr`: awr_*
8. `flow-policy`: flow_policy*, coarse_flow*, policy_value_beta, td_target_policy_value_beta
9. `bc-policy`: separate_bc_policy, bc_policy_*, freeze_bc_policy, frozen_policy_snapshot, distinct_policy_encoder, demo_behavior_force_probability
10. `twin-critic`: pessimistic_twin_critic, auxiliary_td_loss_weight, episodic_twin_head_exploration, twin_rollout_beam_width
11. `td-variants`: td_target_action_source, critic_sequence_mode, autoregressive_action_dims
12. `guards-schedules`: NaN-guard + nan_diag + bc diagnostics, bc_lambda_schedule, demo_fosd toggle
(direct_scalar_q already lives in `cqn_direct_q.py`; `cqn_flow.py`/`djcqn.py` already separate.)

R2 verification per line: flags-off update ≡ official class output (numeric, CPU,
same seed, `allclose` atol 1e-5); flags-on runs finite; coupling to other lines
documented, never hacked around.

## Phase R3 — integration
Central registration of all variants, full pytest, commit(s) on this branch.
Merge to `jaxflat` only after checking no live training process is reading the
main worktree, or with the user's go-ahead.

## Environment constraints (all agents)
- Work ONLY in `/home/zc1525/robobase_jaxflat_refactor`. Never touch `/home/zc1525/robobase_jaxflat` (live experiments).
- Python: `/home/zc1525/robobase_jaxflat/.venv/bin/python` with `PYTHONPATH=/home/zc1525/robobase_jaxflat_refactor`.
- CPU ONLY: always `JAX_PLATFORMS=cpu` (and `CUDA_VISIBLE_DEVICES=""`). GPUs belong to live experiments.
- After R1 lands, `robobase/method/cqn.py` and `cqn_as.py` are FROZEN — any agent needing a change there is making a mistake; report instead.
