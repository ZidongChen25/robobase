#!/usr/bin/env bash
set -euo pipefail

# Run only after the currently preregistered Route-B chain has released the
# two user-assigned GPUs.  Stage-76 already freezes the arm/checkpoint on its
# validation split; this diagnostic consumes that choice and cannot select it.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage114_118_flowcritic_truncated_master
stage76_summary=exp_local/cqn_flow_high_utd/stage76_floq_fidelity_arm_gate_seed114000_20260724/summary.json
control=exp_local/cqn_flow_high_utd/stage119_flow_utilization_controller
output_dir=exp_local/cqn_flow_high_utd/stage119_flow_utilization_seed194000_20260724

mkdir -p "$control" "$output_dir"
printf "%s\n" "$BASHPID" > "$control/stage.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$stage76_summary"
test "$(jq -r .status "$stage76_summary")" = ok

selected_arm=$(jq -r .selection.selected_arm "$stage76_summary")
selected_step=$(jq -r .selection.selected_step "$stage76_summary")
case "$selected_arm" in
  legacy)
    selected_run=exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724
    ;;
  source01)
    selected_run=exp_local/cqn_flow_high_utd/stage75_floq_source01_utd4_seed1_gpu1_20260724
    ;;
  bcfm8)
    selected_run=exp_local/cqn_flow_high_utd/stage75_floq_bcfm8_utd4_seed1_gpu5_20260724
    ;;
  source01_bcfm8)
    selected_run=exp_local/cqn_flow_high_utd/stage75_floq_source01_bcfm8_utd4_seed1_20260724
    ;;
  *)
    printf "unknown Stage-76 selected arm: %s\n" "$selected_arm" >&2
    exit 1
    ;;
esac
selected_snapshot="$selected_run/snapshots/${selected_step}_snapshot.pkl"
test -f "$selected_snapshot"

jq -n \
  --arg selected_arm "$selected_arm" \
  --argjson selected_step "$selected_step" \
  --arg selected_run "$selected_run" \
  --arg selected_snapshot "$selected_snapshot" \
  --arg stage76_summary "$stage76_summary" \
  '{
    status: "running",
    diagnostic_only: true,
    selection_use_forbidden: true,
    selected_arm: $selected_arm,
    selected_step: $selected_step,
    selected_run: $selected_run,
    selected_snapshot: $selected_snapshot,
    selection_artifact: $stage76_summary,
    condition_action_source: "independent_bc_policy",
    eval_seed_start: 194000,
    num_observations: 16,
    num_source_samples: 8,
    step_counts: [1, 2, 4, 8],
    critic: "target"
  }' > "$output_dir/manifest.json"

MUJOCO_GL=egl .venv/bin/python \
  scripts/analyze_cqn_flow_utilization.py \
  --run-dir "$selected_run" \
  --snapshot "$selected_snapshot" \
  --output "$output_dir/probe.json" \
  --gpu-id 1 \
  --num-observations 16 \
  --num-source-samples 8 \
  --step-counts 1,2,4,8 \
  --eval-seed-start 194000 \
  --probe-seed 194100 \
  --critic target \
  > "$control/probe.log" 2>&1

.venv/bin/python scripts/summarize_cqn_flow_utilization.py \
  --input "$output_dir/probe.json" \
  --output "$output_dir/summary.json" \
  > "$control/summary.log" 2>&1

jq '.status = "ok" | . + {
  probe: "'"$output_dir"'/probe.json",
  summary: "'"$output_dir"'/summary.json"
}' "$output_dir/manifest.json" > "$output_dir/manifest.complete.json"
mv "$output_dir/manifest.complete.json" "$output_dir/manifest.json"
touch "$control/complete"
