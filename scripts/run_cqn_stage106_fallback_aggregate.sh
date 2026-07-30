#!/usr/bin/env bash
set -euo pipefail

# Merge the two independent post-Stage97 fallbacks.  Each route may finish,
# fail its discovery gate, or be skipped because an earlier candidate already
# passed.  Only actual task/causal artifacts are appended to the common base.

cd /home/zc1525/robobase_jaxflat

a_master=exp_local/cqn_flow_high_utd/stage98_101_h1_cf_fqe_master
b_master=exp_local/cqn_flow_high_utd/stage102_105_floq_full_interaction_master
control=exp_local/cqn_flow_high_utd/stage106_fallback_aggregate_controller
base_summary=exp_local/cqn_flow_high_utd/stage97_bc_policy_final_summary_20260724.json
output=exp_local/cqn_flow_high_utd/stage106_fallback_autoresearch_summary_20260724.json

a_task=exp_local/cqn_flow_high_utd/stage100_h1_cf_fqe_task_seed166000_20260724/summary.json
a_causal=exp_local/cqn_flow_high_utd/stage101_h1_cf_fqe_causal_seed169000_20260724/summary.json
b_task=exp_local/cqn_flow_high_utd/stage104_floq_full_interaction_task_seed172000_20260724/summary.json
b_causal=exp_local/cqn_flow_high_utd/stage105_floq_full_interaction_causal_seed175000_20260724/summary.json

mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/controller.pid"
trap 'touch "$control/failed"' ERR

a_pid=$(sed -n '1p' "$a_master/controller.pid")
b_pid=$(sed -n '1p' "$b_master/controller.pid")
tail --pid="$a_pid" -f /dev/null
tail --pid="$b_pid" -f /dev/null
test -f "$a_master/complete"
test -f "$b_master/complete"
test -f "$base_summary"

candidate_args=()
if test -f "$a_task" && test "$(jq -r .status "$a_task")" = ok; then
  if test "$(jq -r .gate "$a_task")" = pass; then
    test -f "$a_causal"
    candidate_args+=(
      --a-candidate
      "direct_q_h1_cf_fqe=$a_task,$a_causal"
    )
  else
    candidate_args+=(
      --a-candidate
      "direct_q_h1_cf_fqe=$a_task"
    )
  fi
fi
if test -f "$b_task" && test "$(jq -r .status "$b_task")" = ok; then
  if test "$(jq -r .gate "$b_task")" = pass; then
    test -f "$b_causal"
    candidate_args+=(
      --b-candidate
      "flow_full_official_interaction=$b_task,$b_causal"
    )
  else
    candidate_args+=(
      --b-candidate
      "flow_full_official_interaction=$b_task"
    )
  fi
fi

.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  "${candidate_args[@]}" \
  --output "$output" \
  > "$control/summary.log" 2>&1

if test "$(jq -r .research_goal_gate "$output")" = pass; then
  touch "$control/research_goal_pass"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
