#!/usr/bin/env bash
set -euo pipefail

# Final falsification gate for the CQN-AS "value is only imitation" concern.
# Earlier causal probes select the action dimension with maximum Q span and are
# discovery-only.  This stage rechecks every task+causal pass on fresh seeds
# with a dimension schedule independent of Q, BC, and realized return.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage127_133_evor_flowtd_master
control=exp_local/cqn_flow_high_utd/stage134_unbiased_dimension_confirmation_controller
output_root=exp_local/cqn_flow_high_utd/stage134_unbiased_dimension_confirmation_seed209000_20260724
final_output=exp_local/cqn_flow_high_utd/stage134_strict_autoresearch_summary_20260724.json

mkdir -p "$control" "$output_root"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"

base_summary=
for candidate in \
  exp_local/cqn_flow_high_utd/stage133_evor_flowtd_autoresearch_summary_20260724.json \
  exp_local/cqn_flow_high_utd/stage126_action_centered_autoresearch_summary_20260724.json \
  exp_local/cqn_flow_high_utd/stage118_flowcritic_truncated_autoresearch_summary_20260724.json
do
  if test -f "$candidate"; then
    base_summary=$candidate
    break
  fi
done
test -n "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

jq -n \
  --arg base_summary "$base_summary" \
  '{
    status: "running",
    selection_use_forbidden: true,
    base_summary: $base_summary,
    eval_seed_start: 209000,
    num_eval_seeds: 32,
    anchor_steps: [30, 75, 120],
    intervention_horizon: 1,
    dimension_selection: "round_robin",
    min_informative_dimensions: 8,
    min_informative_states_per_dimension: 2,
    required_positive_training_seeds: 2
  }' > "$output_root/manifest.json"

overrides=()
while IFS=$'\t' read -r route label task_summary; do
  test -n "$route"
  test -n "$label"
  test -f "$task_summary"
  candidate_output="$output_root/$route/$label"
  mkdir -p "$candidate_output"
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_unbiased_causal_from_task.py \
    --task-summary "$task_summary" \
    --output-dir "$candidate_output" \
    --gpu-id 1 --gpu-id 5 \
    --eval-seed-start 209000 --num-eval-seeds 32 \
    --anchor-steps 30,75,120 --force-level 1 \
    --max-continuation-steps 300 \
    --bootstrap-replicates 20000 --bootstrap-seed 209100 \
    --min-informative-states 24 \
    --min-informative-dimensions 8 \
    --min-informative-states-per-dimension 2 \
    --required-positive-training-seeds 2 \
    > "$candidate_output/controller.log" 2>&1
  overrides+=(
    --causal-override
    "$label=$candidate_output/summary.json"
  )
done < <(
  jq -r '
    ["route_a", "route_b"][] as $route
    | .[$route].candidates[]
    | select(.task.gate == "pass")
    | select(.causal_value.gate == "pass")
    | [$route, .label, .task.artifact]
    | @tsv
  ' "$base_summary"
)

.venv/bin/python scripts/revalidate_cqn_autoresearch_summary.py \
  --base-summary "$base_summary" \
  "${overrides[@]}" \
  --output "$final_output" \
  > "$control/summary.log" 2>&1

jq \
  --arg final_output "$final_output" \
  '.status = "ok" | .strict_summary = $final_output' \
  "$output_root/manifest.json" \
  > "$output_root/manifest.complete.json"
mv \
  "$output_root/manifest.complete.json" \
  "$output_root/manifest.json"

if test "$(jq -r .research_goal_gate "$final_output")" = pass; then
  touch "$control/research_goal_pass"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
