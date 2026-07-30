#!/usr/bin/env bash
set -euo pipefail

# Conditional Route-B continuation after every scalar-FLOQ/PCBF fallback.
# It tests the previously unrun CQN-conditioned Value Flows core, with the
# official source-JVP confidence weight as a single-variable treatment.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage106_fallback_aggregate_master
base_summary=exp_local/cqn_flow_high_utd/stage106_fallback_autoresearch_summary_20260724.json

stage107=exp_local/cqn_flow_high_utd/stage107_value_flows_confidence_smoke_controller
stage108=exp_local/cqn_flow_high_utd/stage108_value_flows_confidence_seed1_controller
stage109=exp_local/cqn_flow_high_utd/stage109_value_flows_confidence_arm_gate_controller
stage110=exp_local/cqn_flow_high_utd/stage110_value_flows_confidence_multiseed_controller
stage111=exp_local/cqn_flow_high_utd/stage111_value_flows_confidence_task_controller
stage112=exp_local/cqn_flow_high_utd/stage112_value_flows_confidence_causal_controller
stage113=exp_local/cqn_flow_high_utd/stage113_value_flows_confidence_summary_controller

control_smoke=exp_local/cqn_flow_high_utd/stage107_value_flows_control_smoke_seed1_20260724
weighted_smoke=exp_local/cqn_flow_high_utd/stage107_value_flows_weighted_smoke_seed1_20260724
control_seed1=exp_local/cqn_flow_high_utd/stage108_value_flows_control_utd4_seed1_20260724
weighted_seed1=exp_local/cqn_flow_high_utd/stage108_value_flows_weighted_utd4_seed1_20260724
weighted_seed2=exp_local/cqn_flow_high_utd/stage110_value_flows_weighted_utd4_seed2_20260724
weighted_seed3=exp_local/cqn_flow_high_utd/stage110_value_flows_weighted_utd4_seed3_20260724

arm_output=exp_local/cqn_flow_high_utd/stage109_value_flows_confidence_arm_seed178000_20260724
arm_work=exp_local/cqn_flow_high_utd/stage109_value_flows_confidence_arm_work_seed178000_20260724
task_output=exp_local/cqn_flow_high_utd/stage111_value_flows_confidence_task_seed182000_20260724
task_work=exp_local/cqn_flow_high_utd/stage111_value_flows_confidence_task_work_seed182000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage112_value_flows_confidence_causal_seed185000_20260724
final_output=exp_local/cqn_flow_high_utd/stage113_value_flows_confidence_autoresearch_summary_20260724.json

control_launch=cqn_value_flows_bc_target_two_tower_high_utd4_gate
weighted_launch=cqn_value_flows_confidence_bc_target_two_tower_high_utd4_gate

mkdir -p \
  "$stage107" "$stage108" "$stage109" "$stage110" \
  "$stage111" "$stage112" "$stage113"
printf "%s\n" "$BASHPID" > "$stage107/controller.pid"
current_control=$stage107
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

mark_remaining_skipped() {
  local reason=$1
  for control in \
    "$stage107" "$stage108" "$stage109" "$stage110" \
    "$stage111" "$stage112" "$stage113"; do
    touch "$control/$reason"
    touch "$control/complete"
  done
}

if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  mark_remaining_skipped skipped_route_b_already_passed
  exit 0
fi

train_one() {
  local launch=$1
  local seed=$2
  local gpu=$3
  local frames=$4
  local destination=$5
  local log=$6
  if ! test -f "$destination/snapshots/$((frames / 1000 * 1000))_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      launch="$launch" env=bigym/move_plate \
      seed="$seed" gpu_id="$gpu" num_train_frames="$frames" \
      wandb.use=false save_csv=true log_eval_video=false \
      hydra.run.dir="$destination" \
      > "$log" 2>&1
  fi
}

check_one() {
  local destination=$1
  local output=$2
  local required_step=$3
  local min_rows=$4
  local confidence=${5:-}
  local confidence_args=()
  if test -n "$confidence"; then
    confidence_args=(--expected-confidence-temp "$confidence")
  fi
  .venv/bin/python scripts/check_cqn_value_flows_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --required-snapshot-step "$required_step" \
    --min-log-rows "$min_rows" \
    --expected-flow-steps 10 \
    --expected-flow-samples 4 \
    "${confidence_args[@]}"
}

# Stage 107: exact no-op control and JVP-weighted treatment in parallel.
train_one "$control_launch" 1 1 1500 "$control_smoke" "$stage107/control.log" &
control_pid=$!
train_one "$weighted_launch" 1 5 1500 "$weighted_smoke" "$stage107/weighted.log" &
weighted_pid=$!
printf "%s\n" "$control_pid" > "$stage107/control.pid"
printf "%s\n" "$weighted_pid" > "$stage107/weighted.pid"
wait "$control_pid"
wait "$weighted_pid"
check_one "$control_smoke" "$stage107/control_gate.json" 1000 2
check_one "$weighted_smoke" "$stage107/weighted_gate.json" 1000 2 0.3
test "$(jq -r .gate "$stage107/control_gate.json")" = pass
test "$(jq -r .gate "$stage107/weighted_gate.json")" = pass
touch "$stage107/complete"

# Stage 108: seed1 matched full training.
current_control=$stage108
printf "%s\n" "$BASHPID" > "$stage108/controller.pid"
train_one "$control_launch" 1 1 10500 "$control_seed1" "$stage108/control.log" &
control_pid=$!
train_one "$weighted_launch" 1 5 10500 "$weighted_seed1" "$stage108/weighted.log" &
weighted_pid=$!
printf "%s\n" "$control_pid" > "$stage108/control.pid"
printf "%s\n" "$weighted_pid" > "$stage108/weighted.pid"
wait "$control_pid"
wait "$weighted_pid"
check_one "$control_seed1" "$stage108/control_gate.json" 10000 10
check_one "$weighted_seed1" "$stage108/weighted_gate.json" 10000 10 0.3
test "$(jq -r .gate "$stage108/control_gate.json")" = pass
test "$(jq -r .gate "$stage108/weighted_gate.json")" = pass
touch "$stage108/complete"

# Stage 109: control is declared first and wins ties. Weighted must be selected
# on validation, then beat both clean and control on sealed common seeds.
current_control=$stage109
printf "%s\n" "$BASHPID" > "$stage109/controller.pid"
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_floq_fidelity_arm_gate.py \
  --clean-run-dir exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724 \
  --clean-snapshot exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl \
  --candidate "control=$control_seed1" \
  --candidate "weighted=$weighted_seed1" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$arm_output" --work-root "$arm_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout integrated --num-flow-steps 10 \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 178000 \
  --validation-episodes 50 --validation-seed-start 179000 \
  --confirmation-episodes 200 --confirmation-seed-start 180000 \
  --bootstrap-replicates 20000 --bootstrap-seed 180200 \
  --min-validation-delta 0 \
  --min-confirmation-ci-lower -0.05 \
  --required-selected-arm weighted \
  --confirmation-baseline-arm control \
  --min-arm-confirmation-ci-lower -0.05 \
  > "$stage109/gate.log" 2>&1
touch "$stage109/complete"

if test "$(jq -r .promotion "$arm_output/summary.json")" != pass; then
  for control in "$stage110" "$stage111" "$stage112"; do
    touch "$control/skipped_seed1_mechanism_fail"
    touch "$control/complete"
  done
  current_control=$stage113
  printf "%s\n" "$BASHPID" > "$stage113/controller.pid"
  jq \
    --arg arm_gate "$arm_output/summary.json" \
    '. + {
      stage107_112_value_flows: {
        status: "seed1_mechanism_fail",
        arm_gate: $arm_gate
      }
    }' \
    "$base_summary" > "$final_output"
  touch "$stage113/next_gate_required"
  touch "$stage113/complete"
  exit 0
fi

selected_beta=$(jq -r .selection.selected_beta "$arm_output/summary.json")

# Stage 110: replicate only the promoted treatment.
current_control=$stage110
printf "%s\n" "$BASHPID" > "$stage110/controller.pid"
train_one "$weighted_launch" 2 1 10500 "$weighted_seed2" "$stage110/seed2.log" &
seed2_pid=$!
train_one "$weighted_launch" 3 5 10500 "$weighted_seed3" "$stage110/seed3.log" &
seed3_pid=$!
printf "%s\n" "$seed2_pid" > "$stage110/seed2.pid"
printf "%s\n" "$seed3_pid" > "$stage110/seed3.pid"
wait "$seed2_pid"
wait "$seed3_pid"
check_one "$weighted_seed2" "$stage110/seed2_gate.json" 10000 10 0.3
check_one "$weighted_seed3" "$stage110/seed3_gate.json" 10000 10 0.3
touch "$stage110/complete"

# Stage 111: strict three-training-seed task superiority.
current_control=$stage111
printf "%s\n" "$BASHPID" > "$stage111/controller.pid"
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_floq_checkpoint_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$weighted_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$weighted_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$weighted_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$task_output" --work-root "$task_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout integrated --num-flow-steps 10 \
  --policy-value-beta "$selected_beta" \
  --screen-episodes 10 --screen-seed-start 182000 \
  --validation-episodes 50 --validation-seed-start 183000 \
  --confirmation-episodes 200 --confirmation-seed-start 184000 \
  --bootstrap-replicates 20000 --bootstrap-seed 184200 \
  > "$stage111/gate.log" 2>&1
touch "$stage111/complete"

task_summary="$task_output/summary.json"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage112/skipped_task_fail"
  touch "$stage112/complete"
  current_control=$stage113
  printf "%s\n" "$BASHPID" > "$stage113/controller.pid"
  .venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
    --base-summary "$base_summary" \
    --b-candidate "value_flows_confidence=$task_summary" \
    --output "$final_output" \
    > "$stage113/summary.log" 2>&1
  touch "$stage113/next_gate_required"
  touch "$stage113/complete"
  exit 0
fi

# Stage 112: fresh H=1 anti-cheat branches for the task-qualified checkpoints.
current_control=$stage112
printf "%s\n" "$BASHPID" > "$stage112/controller.pid"
step1=$(jq -r .selected_steps.seed1 "$task_summary")
step2=$(jq -r .selected_steps.seed2 "$task_summary")
step3=$(jq -r .selected_steps.seed3 "$task_summary")
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$weighted_seed1,$weighted_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$weighted_seed2,$weighted_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$weighted_seed3,$weighted_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$causal_output" \
  --eval-seed-start 185000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout integrated --policy-value-beta "$selected_beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 185100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage112/gate.log" 2>&1
touch "$stage112/complete"

current_control=$stage113
printf "%s\n" "$BASHPID" > "$stage113/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --b-candidate \
  "value_flows_confidence=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage113/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" = pass; then
  touch "$stage113/research_goal_pass"
else
  touch "$stage113/next_gate_required"
fi
touch "$stage113/complete"
