#!/usr/bin/env bash
set -euo pipefail

# Deployable Route-A fallback. It uses only logged randomized interventions,
# their known propensities, and completed replay returns. Simulator branches
# are reserved for the final held-out authenticity audit, never for training.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage119_flow_utilization_master
base_summary=exp_local/cqn_flow_high_utd/stage118_flowcritic_truncated_autoresearch_summary_20260724.json

stage120=exp_local/cqn_flow_high_utd/stage120_action_centered_smoke_controller
stage121=exp_local/cqn_flow_high_utd/stage121_action_centered_seed1_controller
stage122=exp_local/cqn_flow_high_utd/stage122_action_centered_discovery_controller
stage123=exp_local/cqn_flow_high_utd/stage123_action_centered_replication_controller
stage124=exp_local/cqn_flow_high_utd/stage124_action_centered_task_controller
stage125=exp_local/cqn_flow_high_utd/stage125_action_centered_causal_controller
stage126=exp_local/cqn_flow_high_utd/stage126_action_centered_summary_controller

smoke_control=exp_local/cqn_flow_high_utd/stage120_direct_q_h1_rct_control_smoke_seed1_gpu1_20260724
smoke_rct=exp_local/cqn_flow_high_utd/stage120_direct_q_h1_rct_w0p1_smoke_seed1_gpu5_20260724
control_seed1=exp_local/cqn_flow_high_utd/stage121_direct_q_h1_rct_control_utd4_seed1_20260724
rct_seed1=exp_local/cqn_flow_high_utd/stage121_direct_q_h1_rct_w0p1_utd4_seed1_20260724
rct_seed2=exp_local/cqn_flow_high_utd/stage123_direct_q_h1_rct_w0p1_utd4_seed2_20260724
rct_seed3=exp_local/cqn_flow_high_utd/stage123_direct_q_h1_rct_w0p1_utd4_seed3_20260724

arm_output=exp_local/cqn_flow_high_utd/stage122_action_centered_arm_seed195000_20260724
arm_work=exp_local/cqn_flow_high_utd/stage122_action_centered_arm_work_seed195000_20260724
task_output=exp_local/cqn_flow_high_utd/stage124_action_centered_task_seed198000_20260724
task_work=exp_local/cqn_flow_high_utd/stage124_action_centered_task_work_seed198000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage125_action_centered_causal_seed201000_20260724
final_output=exp_local/cqn_flow_high_utd/stage126_action_centered_autoresearch_summary_20260724.json

mkdir -p \
  "$stage120" "$stage121" "$stage122" "$stage123" \
  "$stage124" "$stage125" "$stage126"
printf "%s\n" "$BASHPID" > "$stage120/controller.pid"
current_control=$stage120
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
# Stage-119 is diagnostic-only. Its exit releases GPU even if the diagnostic
# itself fails; Stage-118 is the research-summary dependency for this route.
if ! test -f "$upstream_master/complete"; then
  touch "$stage120/upstream_diagnostic_failed_ignored"
fi
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

all_controls=(
  "$stage120" "$stage121" "$stage122" "$stage123"
  "$stage124" "$stage125" "$stage126"
)
mark_remaining_skipped() {
  local reason=$1
  for control in "${all_controls[@]}"; do
    touch "$control/$reason"
    touch "$control/complete"
  done
}

if test "$(jq -r .route_a.overall_gate "$base_summary")" = pass; then
  mark_remaining_skipped skipped_route_a_already_passed
  exit 0
fi

train_one() {
  local launch=$1
  local seed=$2
  local gpu=$3
  local frames=$4
  local destination=$5
  local log=$6
  local required_step=$7
  if ! test -f "$destination/snapshots/${required_step}_snapshot.pkl"; then
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
  local weight=$3
  local step=$4
  local min_rows=$5
  local min_starts=$6
  local min_per_dimension=$7
  .venv/bin/python scripts/check_cqn_direct_q_rct_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --expected-causal-rct-weight "$weight" \
    --expected-exploration-prob 0.2 \
    --expected-level 1 \
    --required-snapshot-step "$step" \
    --min-log-rows "$min_rows" \
    --min-online-starts "$min_starts" \
    --min-starts-per-dimension "$min_per_dimension"
}

# Stage 120: matched 1.5k smoke. Both runs must contain real randomized online
# transitions; treatment additionally needs finite, nonzero causal advantage.
train_one \
  cqn_direct_q_h1_rct_control_two_tower_high_utd4_gate \
  1 1 1500 "$smoke_control" "$stage120/control.log" 1000 &
control_pid=$!
train_one \
  cqn_direct_q_h1_rct_two_tower_high_utd4_gate \
  1 5 1500 "$smoke_rct" "$stage120/rct.log" 1000 &
rct_pid=$!
printf "%s\n" "$control_pid" > "$stage120/control.pid"
printf "%s\n" "$rct_pid" > "$stage120/rct.pid"
wait "$control_pid"
wait "$rct_pid"
check_one "$smoke_control" "$stage120/control_gate.json" 0 1000 1 30 1
check_one "$smoke_rct" "$stage120/rct_gate.json" 0.1 1000 1 30 1
test "$(jq -r .gate "$stage120/control_gate.json")" = pass
test "$(jq -r .gate "$stage120/rct_gate.json")" = pass
touch "$stage120/complete"

# Stage 121: full matched seed-1 mechanism experiment.
current_control=$stage121
printf "%s\n" "$BASHPID" > "$stage121/controller.pid"
train_one \
  cqn_direct_q_h1_rct_control_two_tower_high_utd4_gate \
  1 1 10500 "$control_seed1" "$stage121/control.log" 10000 &
control_pid=$!
train_one \
  cqn_direct_q_h1_rct_two_tower_high_utd4_gate \
  1 5 10500 "$rct_seed1" "$stage121/rct.log" 10000 &
rct_pid=$!
printf "%s\n" "$control_pid" > "$stage121/control.pid"
printf "%s\n" "$rct_pid" > "$stage121/rct.pid"
wait "$control_pid"
wait "$rct_pid"
check_one "$control_seed1" "$stage121/control_gate.json" 0 10000 10 200 5
check_one "$rct_seed1" "$stage121/rct_gate.json" 0.1 10000 10 200 5
test "$(jq -r .gate "$stage121/control_gate.json")" = pass
test "$(jq -r .gate "$stage121/rct_gate.json")" = pass
touch "$stage121/complete"

# Stage 122: control is declared first and wins ties. RCT must be selected on
# validation and then beat both clean CQN-AS and the matched no-loss control on
# never-used confirmation seeds.
current_control=$stage122
printf "%s\n" "$BASHPID" > "$stage122/controller.pid"
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_floq_fidelity_arm_gate.py \
  --clean-run-dir exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724 \
  --clean-snapshot exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl \
  --candidate "control=$control_seed1" \
  --candidate "rct=$rct_seed1" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$arm_output" --work-root "$arm_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout distill \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 195000 \
  --validation-episodes 50 --validation-seed-start 196000 \
  --confirmation-episodes 200 --confirmation-seed-start 197000 \
  --bootstrap-replicates 20000 --bootstrap-seed 197200 \
  --min-validation-delta 0 \
  --min-confirmation-ci-lower 0 \
  --required-selected-arm rct \
  --confirmation-baseline-arm control \
  --min-arm-confirmation-ci-lower 0 \
  > "$stage122/gate.log" 2>&1
touch "$stage122/complete"

if test "$(jq -r .promotion "$arm_output/summary.json")" != pass; then
  for control in "$stage123" "$stage124" "$stage125"; do
    touch "$control/skipped_seed1_discovery_fail"
    touch "$control/complete"
  done
  current_control=$stage126
  printf "%s\n" "$BASHPID" > "$stage126/controller.pid"
  jq \
    --arg arm_gate "$arm_output/summary.json" \
    '. + {
      stage120_125_action_centered_rct: {
        status: "seed1_discovery_fail",
        training: "replay_only",
        simulator_branch_training: false,
        arm_gate: $arm_gate
      }
    }' "$base_summary" > "$final_output"
  touch "$stage126/next_gate_required"
  touch "$stage126/complete"
  exit 0
fi

# Stage 123: replicate the discovered treatment on two new algorithm seeds.
current_control=$stage123
printf "%s\n" "$BASHPID" > "$stage123/controller.pid"
train_one \
  cqn_direct_q_h1_rct_two_tower_high_utd4_gate \
  2 1 10500 "$rct_seed2" "$stage123/seed2.log" 10000 &
seed2_pid=$!
train_one \
  cqn_direct_q_h1_rct_two_tower_high_utd4_gate \
  3 5 10500 "$rct_seed3" "$stage123/seed3.log" 10000 &
seed3_pid=$!
printf "%s\n" "$seed2_pid" > "$stage123/seed2.pid"
printf "%s\n" "$seed3_pid" > "$stage123/seed3.pid"
wait "$seed2_pid"
wait "$seed3_pid"
check_one "$rct_seed2" "$stage123/seed2_gate.json" 0.1 10000 10 200 5
check_one "$rct_seed3" "$stage123/seed3_gate.json" 0.1 10000 10 200 5
test "$(jq -r .gate "$stage123/seed2_gate.json")" = pass
test "$(jq -r .gate "$stage123/seed3_gate.json")" = pass
touch "$stage123/complete"

# Stage 124: new splits select one global beta and per-seed checkpoint, then
# require strict non-inferiority to each clean CQN-AS best checkpoint.
current_control=$stage124
printf "%s\n" "$BASHPID" > "$stage124/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$rct_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$rct_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$rct_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$task_output" --work-root "$task_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --candidate-readout auto \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 198000 \
  --validation-episodes 50 --validation-seed-start 199000 \
  --confirmation-episodes 200 --confirmation-seed-start 200000 \
  --bootstrap-replicates 20000 --bootstrap-seed 200200 \
  --min-mean-delta 0 --min-ci-lower 0 \
  > "$stage124/gate.log" 2>&1
touch "$stage124/complete"

task_summary="$task_output/summary.json"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage125/skipped_task_fail"
  touch "$stage125/complete"
  current_control=$stage126
  printf "%s\n" "$BASHPID" > "$stage126/controller.pid"
  .venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
    --base-summary "$base_summary" \
    --a-candidate "direct_q_action_centered_rct=$task_summary" \
    --output "$final_output" \
    > "$stage126/summary.log" 2>&1
  touch "$stage126/next_gate_required"
  touch "$stage126/complete"
  exit 0
fi

# Stage 125: task-qualified checkpoints must predict fresh H=1 do-action
# returns beyond action nearness and the frozen independent-BC prior.
current_control=$stage125
printf "%s\n" "$BASHPID" > "$stage125/controller.pid"
beta=$(jq -r .selection.selected_global_beta "$task_summary")
step1=$(jq -r .selection.selected_steps.seed1 "$task_summary")
step2=$(jq -r .selection.selected_steps.seed2 "$task_summary")
step3=$(jq -r .selection.selected_steps.seed3 "$task_summary")
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$rct_seed1,$rct_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$rct_seed2,$rct_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$rct_seed3,$rct_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$causal_output" \
  --eval-seed-start 201000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout auto --policy-value-beta "$beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 201100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage125/gate.log" 2>&1
touch "$stage125/complete"

current_control=$stage126
printf "%s\n" "$BASHPID" > "$stage126/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --a-candidate \
  "direct_q_action_centered_rct=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage126/summary.log" 2>&1
if test "$(jq -r .route_a.overall_gate "$final_output")" = pass; then
  touch "$stage126/route_a_pass"
else
  touch "$stage126/next_gate_required"
fi
touch "$stage126/complete"
