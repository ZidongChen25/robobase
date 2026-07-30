#!/usr/bin/env bash
set -euo pipefail

# Recover the FLOQ key-mechanism fidelity experiment that was skipped when the old
# Stage-64 controller referenced a nonexistent clean seed2 path.  The three
# new arms form the non-base cells of a 2x2 design:
# source interval [0, 0.1], official flow:distill ratio, and both together.
# It is a CQN-AS integration rather than a line-for-line FQL/FLOQ reproduction.
# This script runs only if fixed-budget, integrated, and validation-best legacy
# Flow task gates have all failed.

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage65_67_recovery_master/controller.pid
stage64_summary=exp_local/cqn_flow_high_utd/stage64_floq_multiseed_final_seed92000_20260724/summary.json
stage66_summary=exp_local/cqn_flow_high_utd/stage66_integrated_multiseed_final_seed99000_20260724/summary.json
stage67_summary=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724/summary.json

stage74_control=exp_local/cqn_flow_high_utd/stage74_floq_fidelity_smoke_controller_r1
stage75_control=exp_local/cqn_flow_high_utd/stage75_floq_fidelity_seed1_controller_r1
stage76_control=exp_local/cqn_flow_high_utd/stage76_floq_fidelity_arm_gate_controller_r1
stage77_control=exp_local/cqn_flow_high_utd/stage77_floq_fidelity_multiseed_controller_r1
stage78_control=exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_gate_controller_r1
stage80_control=exp_local/cqn_flow_high_utd/stage80_floq_fidelity_causal_controller_r1

source_smoke=exp_local/cqn_flow_high_utd/stage74_floq_source01_smoke_seed1_gpu1_20260724
bcfm_smoke=exp_local/cqn_flow_high_utd/stage74_floq_bcfm8_smoke_seed1_gpu5_20260724
combo_smoke=exp_local/cqn_flow_high_utd/stage74_floq_source01_bcfm8_smoke_seed1_20260724

source_seed1=exp_local/cqn_flow_high_utd/stage75_floq_source01_utd4_seed1_gpu1_20260724
bcfm_seed1=exp_local/cqn_flow_high_utd/stage75_floq_bcfm8_utd4_seed1_gpu5_20260724
combo_seed1=exp_local/cqn_flow_high_utd/stage75_floq_source01_bcfm8_utd4_seed1_20260724
legacy_seed1=exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724
direct_seed1=exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed1_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed2_20260724
direct_seed3=exp_local/cqn_flow_high_utd/stage85_direct_q_replay_utd4_seed3_20260724

stage76_output=exp_local/cqn_flow_high_utd/stage76_floq_fidelity_arm_gate_seed114000_20260724
stage76_work=exp_local/cqn_flow_high_utd/stage76_floq_fidelity_arm_gate_work_seed114000_20260724
stage78_output=exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_seed118000_20260724
stage78_work=exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_work_seed118000_20260724
stage80_output=exp_local/cqn_flow_high_utd/stage80_floq_fidelity_causal_seed122000_20260724

mkdir -p \
  "$stage74_control" \
  "$stage75_control" \
  "$stage76_control" \
  "$stage77_control" \
  "$stage78_control" \
  "$stage80_control"
printf "%s\n" "$BASHPID" > "$stage74_control/controller.pid"
current_control=$stage74_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test -f exp_local/cqn_flow_high_utd/stage67_best_checkpoint_controller_r1/complete

gate_pass() {
  local artifact=$1
  test -f "$artifact" && test "$(jq -r .gate "$artifact")" = pass
}

mark_all_skipped() {
  local reason=$1
  for control in \
    "$stage74_control" \
    "$stage75_control" \
    "$stage76_control" \
    "$stage77_control" \
    "$stage78_control" \
    "$stage80_control"; do
    touch "$control/$reason"
    touch "$control/complete"
  done
}

write_failed_task_summary() {
  local reason=$1
  mkdir -p "$stage78_output"
  jq -n \
    --arg reason "$reason" \
    --arg stage76 "$stage76_output/summary.json" \
    '{
      status: "ok",
      gate: "fail",
      reason: $reason,
      upstream_fidelity_selection: $stage76,
      mean_baseline_success: null,
      mean_candidate_success: null,
      mean_paired_delta: null,
      crossed_bootstrap_ci95: null,
      aggregate_paired_wins: null,
      aggregate_paired_losses: null,
      per_training_seed: null,
      selected_steps: null,
      candidate_readout: "distill",
      gate_checks: {
        seed1_fidelity_arm_promoted: false,
        strict_three_training_seed_task_gate: false
      }
    }' > "$stage78_output/summary.json"
}

if gate_pass "$stage64_summary" \
  || gate_pass "$stage66_summary" \
  || gate_pass "$stage67_summary"; then
  mark_all_skipped skipped_existing_b_task_pass
  exit 0
fi

run_flow_training() {
  local launch=$1
  local seed=$2
  local gpu=$3
  local frames=$4
  local run_dir=$5
  local log=$6
  local required_step=$((frames / 1000 * 1000))
  if test -f "$run_dir/snapshots/${required_step}_snapshot.pkl"; then
    # Preserve completed artifacts on controller recovery while retaining the
    # asynchronous PID contract expected by the scheduling helpers below.
    (:) &
    LAST_PID=$!
    return
  fi
  MUJOCO_GL=egl .venv/bin/python train.py \
    "launch=$launch" env=bigym/move_plate \
    "seed=$seed" "gpu_id=$gpu" "num_train_frames=$frames" \
    wandb.use=false save_csv=true log_eval_video=false \
    "hydra.run.dir=$run_dir" \
    > "$log" 2>&1 &
  LAST_PID=$!
}

run_direct_training() {
  local seed=$1
  local gpu=$2
  local run_dir=$3
  local log=$4
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate "seed=$seed" "gpu_id=$gpu" \
    wandb.use=false save_csv=true log_eval_video=false \
    "hydra.run.dir=$run_dir" \
    > "$log" 2>&1 &
  LAST_PID=$!
}

run_two_then_third() {
  local frames=$1
  local source_dir=$2
  local bcfm_dir=$3
  local combo_dir=$4
  local log_dir=$5

  run_flow_training \
    cqn_flow_floq_source01_distill_two_tower_high_utd4_gate \
    1 1 "$frames" "$source_dir" "$log_dir/source.log"
  local source_pid=$LAST_PID
  run_flow_training \
    cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate \
    1 5 "$frames" "$bcfm_dir" "$log_dir/bcfm.log"
  local bcfm_pid=$LAST_PID
  printf "%s\n" "$source_pid" > "$log_dir/source.pid"
  printf "%s\n" "$bcfm_pid" > "$log_dir/bcfm.pid"

  local finished_pid
  if ! wait -n -p finished_pid "$source_pid" "$bcfm_pid"; then
    return 1
  fi
  local combo_gpu
  local remaining_pid
  if test "$finished_pid" = "$source_pid"; then
    combo_gpu=1
    remaining_pid=$bcfm_pid
    touch "$log_dir/source_training_complete"
  else
    combo_gpu=5
    remaining_pid=$source_pid
    touch "$log_dir/bcfm_training_complete"
  fi
  printf "%s\n" "$combo_gpu" > "$log_dir/combo_gpu"
  run_flow_training \
    cqn_flow_floq_source01_bcfm8_distill_two_tower_high_utd4_gate \
    1 "$combo_gpu" "$frames" "$combo_dir" "$log_dir/combo.log"
  local combo_pid=$LAST_PID
  printf "%s\n" "$combo_pid" > "$log_dir/combo.pid"

  wait "$remaining_pid"
  if test "$remaining_pid" = "$source_pid"; then
    touch "$log_dir/source_training_complete"
  else
    touch "$log_dir/bcfm_training_complete"
  fi
  wait "$combo_pid"
  touch "$log_dir/combo_training_complete"
}

run_two_then_third \
  1500 "$source_smoke" "$bcfm_smoke" "$combo_smoke" "$stage74_control"

.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$source_smoke" --output "$stage74_control/source_gate.json" \
  --expected-bcfm-lambda 1 \
  --expected-source-min 0 --expected-source-max 0.1
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$bcfm_smoke" --output "$stage74_control/bcfm_gate.json" \
  --expected-bcfm-lambda 8
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$combo_smoke" --output "$stage74_control/combo_gate.json" \
  --expected-bcfm-lambda 8 \
  --expected-source-min 0 --expected-source-max 0.1
touch "$stage74_control/complete"

current_control=$stage75_control
printf "%s\n" "$BASHPID" > "$stage75_control/controller.pid"
for gate in source bcfm combo; do
  test "$(jq -r .gate "$stage74_control/${gate}_gate.json")" = pass
done

# First train the two single-variable Flow cells in parallel.
run_flow_training \
  cqn_flow_floq_source01_distill_two_tower_high_utd4_gate \
  1 1 10500 "$source_seed1" "$stage75_control/source.log"
source_pid=$LAST_PID
run_flow_training \
  cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate \
  1 5 10500 "$bcfm_seed1" "$stage75_control/bcfm.log"
bcfm_pid=$LAST_PID
printf "%s\n" "$source_pid" > "$stage75_control/source.pid"
printf "%s\n" "$bcfm_pid" > "$stage75_control/bcfm.pid"
status=0
if wait "$source_pid"; then
  touch "$stage75_control/source_training_complete"
else
  status=1
fi
if wait "$bcfm_pid"; then
  touch "$stage75_control/bcfm_training_complete"
else
  status=1
fi
test "$status" -eq 0

# The full-fidelity combination occupies GPU1.  Use GPU5 for the three
# independent Route-A replay-next direct-Q training seeds.  This is only
# resource co-scheduling: runs, replay, checkpoints, task splits, and gates
# remain disjoint.  Any filler failure is retried later by Stage-84/85 and
# therefore must not invalidate the B-route combination run.
run_flow_training \
  cqn_flow_floq_source01_bcfm8_distill_two_tower_high_utd4_gate \
  1 1 10500 "$combo_seed1" "$stage75_control/combo.log"
combo_pid=$LAST_PID
printf "%s\n" "$combo_pid" > "$stage75_control/combo.pid"
printf "%s\n" 1 > "$stage75_control/combo_gpu"

test "$(jq -r .gate exp_local/cqn_flow_high_utd/stage70_preflight_direct_q_smoke_controller/gate.json)" = pass
route_a_fill_status=0
for item in \
  "seed1:1:$direct_seed1" \
  "seed2:2:$direct_seed2" \
  "seed3:3:$direct_seed3"; do
  label=${item%%:*}
  remainder=${item#*:}
  seed=${remainder%%:*}
  run_dir=${remainder#*:}
  if ! test -f "$run_dir/snapshots/10000_snapshot.pkl"; then
    run_direct_training \
      "$seed" 5 "$run_dir" "$stage75_control/route_a_${label}.log"
    filler_pid=$LAST_PID
    printf "%s\n" "$filler_pid" \
      > "$stage75_control/route_a_${label}.pid"
    if wait "$filler_pid"; then
      touch "$stage75_control/route_a_${label}_training_complete"
    else
      touch "$stage75_control/route_a_${label}_training_failed"
      route_a_fill_status=1
      break
    fi
  else
    touch "$stage75_control/route_a_${label}_reused"
  fi
  if .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
    --run-dir "$run_dir" \
    --output "$stage75_control/route_a_${label}_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-td-target-action-source replay_next; then
    touch "$stage75_control/route_a_${label}_gate_pass"
  else
    touch "$stage75_control/route_a_${label}_gate_failed"
    route_a_fill_status=1
    break
  fi
done
if test "$route_a_fill_status" -eq 0; then
  touch "$stage75_control/route_a_prefill_complete"
else
  touch "$stage75_control/route_a_prefill_incomplete"
fi

wait "$combo_pid"
touch "$stage75_control/combo_training_complete"

.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$source_seed1" --output "$stage75_control/source_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-bcfm-lambda 1 \
  --expected-source-min 0 --expected-source-max 0.1
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$bcfm_seed1" --output "$stage75_control/bcfm_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-bcfm-lambda 8
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$combo_seed1" --output "$stage75_control/combo_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-bcfm-lambda 8 \
  --expected-source-min 0 --expected-source-max 0.1
touch "$stage75_control/complete"

current_control=$stage76_control
printf "%s\n" "$BASHPID" > "$stage76_control/controller.pid"
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_floq_fidelity_arm_gate.py \
  --clean-run-dir exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724 \
  --clean-snapshot exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl \
  --candidate "legacy=$legacy_seed1" \
  --candidate "source01=$source_seed1" \
  --candidate "bcfm8=$bcfm_seed1" \
  --candidate "source01_bcfm8=$combo_seed1" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage76_output" --work-root "$stage76_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout distill \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 114000 \
  --validation-episodes 50 --validation-seed-start 115000 \
  --confirmation-episodes 200 --confirmation-seed-start 116000 \
  --bootstrap-replicates 20000 --bootstrap-seed 116200 \
  --min-validation-delta 0.02 \
  --min-confirmation-ci-lower -0.05 \
  > "$stage76_control/gate.log" 2>&1
touch "$stage76_control/complete"

current_control=$stage77_control
printf "%s\n" "$BASHPID" > "$stage77_control/controller.pid"
selected_arm=$(jq -r .selection.selected_arm "$stage76_output/summary.json")
if test "$(jq -r .promotion "$stage76_output/summary.json")" != pass; then
  touch "$stage77_control/skipped_no_seed1_promotion"
  touch "$stage77_control/complete"
  write_failed_task_summary no_seed1_fidelity_promotion
  touch "$stage78_control/skipped_no_seed1_promotion"
  touch "$stage78_control/complete"
  touch "$stage80_control/skipped_no_b_task_pass"
  touch "$stage80_control/complete"
  exit 0
fi
if test "$selected_arm" = legacy; then
  touch "$stage77_control/skipped_legacy_won_joint_selection"
  touch "$stage77_control/complete"
  write_failed_task_summary no_fidelity_arm_beats_legacy
  touch "$stage78_control/skipped_legacy_won_joint_selection"
  touch "$stage78_control/complete"
  touch "$stage80_control/skipped_no_b_task_pass"
  touch "$stage80_control/complete"
  exit 0
fi

selected_beta=$(jq -r .selection.selected_beta "$stage76_output/summary.json")
case "$selected_arm" in
  source01)
    selected_launch=cqn_flow_floq_source01_distill_two_tower_high_utd4_gate
    selected_seed1=$source_seed1
    expected_lambda=1
    source_args=(--expected-source-min 0 --expected-source-max 0.1)
    ;;
  bcfm8)
    selected_launch=cqn_flow_floq_bcfm8_distill_two_tower_high_utd4_gate
    selected_seed1=$bcfm_seed1
    expected_lambda=8
    source_args=()
    ;;
  source01_bcfm8)
    selected_launch=cqn_flow_floq_source01_bcfm8_distill_two_tower_high_utd4_gate
    selected_seed1=$combo_seed1
    expected_lambda=8
    source_args=(--expected-source-min 0 --expected-source-max 0.1)
    ;;
  *)
    printf 'unknown selected arm: %s\n' "$selected_arm" >&2
    exit 1
    ;;
esac
printf "%s\n" "$selected_arm" > "$stage77_control/selected_arm"
printf "%s\n" "$selected_beta" > "$stage77_control/selected_beta"

selected_seed2="exp_local/cqn_flow_high_utd/stage77_floq_fidelity_${selected_arm}_seed2_gpu1_20260724"
selected_seed3="exp_local/cqn_flow_high_utd/stage77_floq_fidelity_${selected_arm}_seed3_gpu5_20260724"
run_flow_training \
  "$selected_launch" 2 1 10500 "$selected_seed2" "$stage77_control/seed2.log"
seed2_pid=$LAST_PID
run_flow_training \
  "$selected_launch" 3 5 10500 "$selected_seed3" "$stage77_control/seed3.log"
seed3_pid=$LAST_PID
printf "%s\n" "$seed2_pid" > "$stage77_control/seed2.pid"
printf "%s\n" "$seed3_pid" > "$stage77_control/seed3.pid"
status=0
if wait "$seed2_pid"; then
  touch "$stage77_control/seed2_training_complete"
else
  status=1
fi
if wait "$seed3_pid"; then
  touch "$stage77_control/seed3_training_complete"
else
  status=1
fi
test "$status" -eq 0

for item in "seed2:$selected_seed2" "seed3:$selected_seed3"; do
  label=${item%%:*}
  run_dir=${item#*:}
  .venv/bin/python scripts/check_cqn_floq_training_gate.py \
    --run-dir "$run_dir" \
    --output "$stage77_control/${label}_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-bcfm-lambda "$expected_lambda" \
    "${source_args[@]}"
done
touch "$stage77_control/complete"

current_control=$stage78_control
printf "%s\n" "$BASHPID" > "$stage78_control/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_floq_checkpoint_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$selected_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$selected_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$selected_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage78_output" --work-root "$stage78_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout distill \
  --policy-value-beta "$selected_beta" \
  --screen-episodes 10 --screen-seed-start 118000 \
  --validation-episodes 50 --validation-seed-start 119000 \
  --confirmation-episodes 200 --confirmation-seed-start 120000 \
  --bootstrap-replicates 20000 --bootstrap-seed 120200 \
  > "$stage78_control/gate.log" 2>&1
touch "$stage78_control/complete"

current_control=$stage80_control
printf "%s\n" "$BASHPID" > "$stage80_control/controller.pid"
if test "$(jq -r .gate "$stage78_output/summary.json")" != pass; then
  touch "$stage80_control/skipped_no_b_task_pass"
  touch "$stage80_control/complete"
  exit 0
fi

step1=$(jq -r .selected_steps.seed1 "$stage78_output/summary.json")
step2=$(jq -r .selected_steps.seed2 "$stage78_output/summary.json")
step3=$(jq -r .selected_steps.seed3 "$stage78_output/summary.json")
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$selected_seed1,$selected_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$selected_seed2,$selected_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$selected_seed3,$selected_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 --output-dir "$stage80_output" \
  --eval-seed-start 122000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout distill --policy-value-beta "$selected_beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 122100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage80_control/gate.log" 2>&1
touch "$stage80_control/complete"
