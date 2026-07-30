#!/usr/bin/env bash
set -euo pipefail

# Conditional Route-B continuation after the CQN-adapted Value Flows stages.
# It isolates FlowCritic's pessimistic return readout: for the same trained
# return flow, checkpoint, BC/value beta, and simulator seeds, compare a
# 10-sample mean with a 10-sample mean after dropping the largest sample.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage107_113_value_flows_confidence_master
base_summary=exp_local/cqn_flow_high_utd/stage113_value_flows_confidence_autoresearch_summary_20260724.json
source_arm_summary=exp_local/cqn_flow_high_utd/stage109_value_flows_confidence_arm_seed178000_20260724/summary.json

stage114=exp_local/cqn_flow_high_utd/stage114_flowcritic_truncated_seed1_controller
stage115=exp_local/cqn_flow_high_utd/stage115_flowcritic_truncated_multiseed_controller
stage116=exp_local/cqn_flow_high_utd/stage116_flowcritic_truncated_task_controller
stage117=exp_local/cqn_flow_high_utd/stage117_flowcritic_truncated_causal_controller
stage118=exp_local/cqn_flow_high_utd/stage118_flowcritic_truncated_summary_controller

arm_output=exp_local/cqn_flow_high_utd/stage114_flowcritic_truncated_arm_seed187000_20260724
arm_work=exp_local/cqn_flow_high_utd/stage114_flowcritic_truncated_arm_work_seed187000_20260724
task_output=exp_local/cqn_flow_high_utd/stage116_flowcritic_truncated_task_seed190000_20260724
task_work=exp_local/cqn_flow_high_utd/stage116_flowcritic_truncated_task_work_seed190000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage117_flowcritic_truncated_causal_seed193000_20260724
final_output=exp_local/cqn_flow_high_utd/stage118_flowcritic_truncated_autoresearch_summary_20260724.json

mkdir -p "$stage114" "$stage115" "$stage116" "$stage117" "$stage118"
printf "%s\n" "$BASHPID" > "$stage114/controller.pid"
current_control=$stage114
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

mark_remaining_skipped() {
  local reason=$1
  for control in "$stage114" "$stage115" "$stage116" "$stage117" "$stage118"; do
    touch "$control/$reason"
    touch "$control/complete"
  done
}

if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  mark_remaining_skipped skipped_route_b_already_passed
  exit 0
fi

test -f "$source_arm_summary"
selected_training_arm=$(jq -r .selection.selected_arm "$source_arm_summary")
case "$selected_training_arm" in
  control)
    selected_launch=cqn_value_flows_bc_target_two_tower_high_utd4_gate
    selected_seed1=exp_local/cqn_flow_high_utd/stage108_value_flows_control_utd4_seed1_20260724
    selected_seed2=exp_local/cqn_flow_high_utd/stage115_value_flows_control_utd4_seed2_20260724
    selected_seed3=exp_local/cqn_flow_high_utd/stage115_value_flows_control_utd4_seed3_20260724
    confidence_args=()
    ;;
  weighted)
    selected_launch=cqn_value_flows_confidence_bc_target_two_tower_high_utd4_gate
    selected_seed1=exp_local/cqn_flow_high_utd/stage108_value_flows_weighted_utd4_seed1_20260724
    selected_seed2=exp_local/cqn_flow_high_utd/stage110_value_flows_weighted_utd4_seed2_20260724
    selected_seed3=exp_local/cqn_flow_high_utd/stage110_value_flows_weighted_utd4_seed3_20260724
    confidence_args=(--expected-confidence-temp 0.3)
    ;;
  *)
    printf "unknown Stage-109 selected training arm: %s\n" \
      "$selected_training_arm" >&2
    exit 1
    ;;
esac
test -f "$selected_seed1/snapshots/10000_snapshot.pkl"
printf "%s\n" "$selected_training_arm" > "$stage114/selected_training_arm"

# Stage 114: single-training-seed mechanism isolation. Mean is declared first
# and wins ties; truncated must beat clean and its independently selected mean
# checkpoint on fresh confirmation seeds.
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_floq_fidelity_arm_gate.py \
  --clean-run-dir exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724 \
  --clean-snapshot exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl \
  --candidate "mean=$selected_seed1" \
  --candidate "truncated=$selected_seed1" \
  --candidate-return-sample-aggregation mean=mean \
  --candidate-return-sample-aggregation truncated=truncated_mean \
  --candidate-action-flow-samples mean=10 \
  --candidate-action-flow-samples truncated=10 \
  --candidate-return-sample-truncate-top truncated=1 \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$arm_output" --work-root "$arm_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout integrated --num-flow-steps 10 \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 187000 \
  --validation-episodes 50 --validation-seed-start 188000 \
  --confirmation-episodes 200 --confirmation-seed-start 189000 \
  --bootstrap-replicates 20000 --bootstrap-seed 189200 \
  --min-validation-delta 0 \
  --min-confirmation-ci-lower -0.05 \
  --required-selected-arm truncated \
  --confirmation-baseline-arm mean \
  --min-arm-confirmation-ci-lower -0.05 \
  > "$stage114/gate.log" 2>&1
touch "$stage114/complete"

if test "$(jq -r .promotion "$arm_output/summary.json")" != pass; then
  for control in "$stage115" "$stage116" "$stage117"; do
    touch "$control/skipped_seed1_readout_fail"
    touch "$control/complete"
  done
  current_control=$stage118
  printf "%s\n" "$BASHPID" > "$stage118/controller.pid"
  jq \
    --arg arm_gate "$arm_output/summary.json" \
    --arg training_arm "$selected_training_arm" \
    '. + {
      stage114_117_flowcritic_truncated: {
        status: "seed1_readout_fail",
        selected_training_arm: $training_arm,
        arm_gate: $arm_gate
      }
    }' \
    "$base_summary" > "$final_output"
  touch "$stage118/next_gate_required"
  touch "$stage118/complete"
  exit 0
fi

selected_beta=$(jq -r .selection.selected_beta "$arm_output/summary.json")

train_one() {
  local seed=$1
  local gpu=$2
  local destination=$3
  local log=$4
  if ! test -f "$destination/snapshots/10000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      launch="$selected_launch" env=bigym/move_plate \
      seed="$seed" gpu_id="$gpu" num_train_frames=10500 \
      wandb.use=false save_csv=true log_eval_video=false \
      hydra.run.dir="$destination" \
      > "$log" 2>&1
  fi
}

check_one() {
  local destination=$1
  local output=$2
  .venv/bin/python scripts/check_cqn_value_flows_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --required-snapshot-step 10000 \
    --min-log-rows 10 \
    --expected-flow-steps 10 \
    --expected-flow-samples 4 \
    "${confidence_args[@]}"
}

# Stage 115: replicate only the training arm selected before this readout
# experiment. Existing Stage-110 weighted checkpoints are reused when present.
current_control=$stage115
printf "%s\n" "$BASHPID" > "$stage115/controller.pid"
train_one 2 1 "$selected_seed2" "$stage115/seed2.log" &
seed2_pid=$!
train_one 3 5 "$selected_seed3" "$stage115/seed3.log" &
seed3_pid=$!
printf "%s\n" "$seed2_pid" > "$stage115/seed2.pid"
printf "%s\n" "$seed3_pid" > "$stage115/seed3.pid"
wait "$seed2_pid"
wait "$seed3_pid"
check_one "$selected_seed2" "$stage115/seed2_gate.json"
check_one "$selected_seed3" "$stage115/seed3_gate.json"
test "$(jq -r .gate "$stage115/seed2_gate.json")" = pass
test "$(jq -r .gate "$stage115/seed3_gate.json")" = pass
touch "$stage115/complete"

# Stage 116: strict task superiority over validation-selected clean CQN-AS.
current_control=$stage116
printf "%s\n" "$BASHPID" > "$stage116/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_floq_checkpoint_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$selected_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$selected_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$selected_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$task_output" --work-root "$task_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout integrated --num-flow-steps 10 \
  --return-sample-aggregation truncated_mean \
  --num-action-flow-samples 10 \
  --return-sample-truncate-top 1 \
  --policy-value-beta "$selected_beta" \
  --screen-episodes 10 --screen-seed-start 190000 \
  --validation-episodes 50 --validation-seed-start 191000 \
  --confirmation-episodes 200 --confirmation-seed-start 192000 \
  --bootstrap-replicates 20000 --bootstrap-seed 192200 \
  > "$stage116/gate.log" 2>&1
touch "$stage116/complete"

task_summary="$task_output/summary.json"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage117/skipped_task_fail"
  touch "$stage117/complete"
  current_control=$stage118
  printf "%s\n" "$BASHPID" > "$stage118/controller.pid"
  .venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
    --base-summary "$base_summary" \
    --b-candidate "flowcritic_truncated=$task_summary" \
    --output "$final_output" \
    > "$stage118/summary.log" 2>&1
  touch "$stage118/next_gate_required"
  touch "$stage118/complete"
  exit 0
fi

# Stage 117: the task-qualified readout must also predict H=1 same-state
# sibling interventions and beat BC/path/nearness proxies.
current_control=$stage117
printf "%s\n" "$BASHPID" > "$stage117/controller.pid"
step1=$(jq -r .selected_steps.seed1 "$task_summary")
step2=$(jq -r .selected_steps.seed2 "$task_summary")
step3=$(jq -r .selected_steps.seed3 "$task_summary")
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$selected_seed1,$selected_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$selected_seed2,$selected_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$selected_seed3,$selected_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$causal_output" \
  --eval-seed-start 193000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout integrated --num-flow-steps 10 \
  --return-sample-aggregation truncated_mean \
  --num-action-flow-samples 10 \
  --return-sample-truncate-top 1 \
  --policy-value-beta "$selected_beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 193100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage117/gate.log" 2>&1
touch "$stage117/complete"

current_control=$stage118
printf "%s\n" "$BASHPID" > "$stage118/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --b-candidate "flowcritic_truncated=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage118/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" != pass; then
  touch "$stage118/next_gate_required"
fi
touch "$stage118/complete"
