#!/usr/bin/env bash
set -euo pipefail

# CPU-only numerical-value audit over the exact raw branch artifacts produced
# by Stage-134. It can run alongside later GPU work, but it never opens the
# Stage-134 confirmation data before the ranking gate has completed.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage134_unbiased_dimension_confirmation_master
strict_summary=exp_local/cqn_flow_high_utd/stage134_strict_autoresearch_summary_20260724.json
control=exp_local/cqn_flow_high_utd/stage134b_numeric_calibration_controller
output_root=exp_local/cqn_flow_high_utd/stage134b_numeric_calibration_seed209000_20260724
summary_output="$output_root/summary.json"
rows="$output_root/candidates.jsonl"

mkdir -p "$control" "$output_root"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$strict_summary"
test "$(jq -r .status "$strict_summary")" = ok

: > "$rows"
while IFS=$'\t' read -r route label causal_artifact; do
  test -n "$route"
  test -n "$label"
  test -f "$causal_artifact"
  case "$label" in
    *[!A-Za-z0-9_.-]*)
      echo "unsafe candidate label: $label" >&2
      exit 1
      ;;
  esac
  candidate_dir="$output_root/$route/$label"
  calibration_artifact="$candidate_dir/calibration.json"
  mkdir -p "$candidate_dir"
  .venv/bin/python scripts/analyze_cqn_branch_calibration.py \
    --causal-summary "$causal_artifact" \
    --output "$calibration_artifact" \
    --bootstrap-replicates 20000 --bootstrap-seed 209200 \
    --permutation-replicates 2000 --permutation-seed 209201 \
    --min-training-seeds 3 \
    --min-informative-states-per-split 12 \
    --native-slope-lower 0.5 --native-slope-upper 2.0 \
    > "$candidate_dir/calibration.log" 2>&1
  jq -n \
    --arg route "$route" \
    --arg label "$label" \
    --arg causal_artifact "$causal_artifact" \
    --arg calibration_artifact "$calibration_artifact" \
    --slurpfile calibration "$calibration_artifact" \
    '{
      route: $route,
      label: $label,
      causal_artifact: $causal_artifact,
      calibration_artifact: $calibration_artifact,
      calibration: $calibration[0]
    }' >> "$rows"
done < <(
  jq -r '
    ["route_a", "route_b"][] as $route
    | .[$route].candidates[]
    | select(.checks.task_requirement == true)
    | select(.checks.causally_meaningful_value == true)
    | [$route, .label, .causal_value.artifact]
    | @tsv
  ' "$strict_summary"
)

jq -s \
  --arg upstream "$strict_summary" \
  '{
    status: "ok",
    stage: "stage134b_numeric_value_calibration",
    upstream_strict_summary: $upstream,
    selection_use_forbidden: true,
    split: {
      calibration_seed_count: 16,
      heldout_seed_count: 16,
      split_rule: "first_half_calibration_second_half_sealed_heldout"
    },
    candidates: .,
    route_a_gate: (
      if any(.[]; .route == "route_a" and .calibration.gate == "pass")
      then "pass"
      else "fail"
      end
    ),
    route_b_gate: (
      if any(.[]; .route == "route_b" and .calibration.gate == "pass")
      then "pass"
      else "fail"
      end
    ),
    interpretation: (
      if length == 0
      then "No task-and-ranking-qualified candidate existed, so numerical calibration was not claimable."
      else "Only candidates already passing task and unbiased causal-ranking gates were tested; calibration failure does not erase ranking evidence, but blocks a native-return-unit value claim."
      end
    )
  }' "$rows" > "$summary_output"

if jq -e 'any(.candidates[]; .calibration.gate == "pass")' \
  "$summary_output" >/dev/null
then
  touch "$control/calibration_candidate_pass"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
