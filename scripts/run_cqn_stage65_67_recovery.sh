#!/usr/bin/env bash
set -euo pipefail

# Recover the Stage-65--67 Flow readout/checkpoint gates after the original
# controller chain referenced the wrong clean seed-2 run directory.  This
# script is deliberately event driven: it sleeps inside tail --pid until the
# corrected Stage-64 controller exits, then runs only the gates still needed.

cd /home/zc1525/robobase_jaxflat

stage64_control=exp_local/cqn_flow_high_utd/stage64_multiseed_final_controller_r1
stage64_summary=exp_local/cqn_flow_high_utd/stage64_floq_multiseed_final_seed92000_20260724/summary.json
stage65_control=exp_local/cqn_flow_high_utd/stage65_readout_mechanism_controller_r1
stage65_output=exp_local/cqn_flow_high_utd/stage65_readout_mechanism_seed96000_20260724
stage65_work=exp_local/cqn_flow_high_utd/stage65_readout_mechanism_work_seed96000_20260724
stage66_control=exp_local/cqn_flow_high_utd/stage66_integrated_task_controller_r1
stage66_output=exp_local/cqn_flow_high_utd/stage66_integrated_multiseed_final_seed99000_20260724
stage66_work=exp_local/cqn_flow_high_utd/stage66_integrated_multiseed_final_work_seed99000_20260724
stage67_control=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_controller_r1
stage67_output=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724
stage67_work=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_work_seed101000_20260724

mkdir -p \
  "$stage65_control" \
  "$stage66_control" \
  "$stage67_control"
printf "%s\n" "$BASHPID" > "$stage65_control/controller.pid"

stage64_pid=$(sed -n '1p' "$stage64_control/controller.pid")
tail --pid="$stage64_pid" -f /dev/null
test -f "$stage64_control/complete"
test -f "$stage64_summary"

current_control=$stage65_control
trap 'touch "$current_control/failed"' ERR

MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_readout_multiseed_gate.py \
  --checkpoint seed1=exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724,exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724/snapshots/10000_snapshot.pkl \
  --checkpoint seed2=exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724/snapshots/10000_snapshot.pkl \
  --checkpoint seed3=exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724/snapshots/10000_snapshot.pkl \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage65_output" --work-root "$stage65_work" \
  --steps 2 8 --policy-value-beta 1 \
  --selection-episodes 50 --selection-seed-start 96000 \
  --confirmation-episodes 200 --confirmation-seed-start 97000 \
  --bootstrap-replicates 20000 --bootstrap-seed 97200 \
  > "$stage65_control/gate.log" 2>&1
touch "$stage65_control/complete"

current_control=$stage66_control
printf "%s\n" "$BASHPID" > "$stage66_control/controller.pid"
stage65_summary="$stage65_output/summary.json"
test -f "$stage65_summary"
if test "$(jq -r .gate "$stage65_summary")" = pass; then
  steps=$(jq -r .selected_readout.steps "$stage65_summary")
  test "$steps" -ge 1
  printf "%s\n" "$steps" > "$stage66_control/selected_steps"
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_floq_multiseed_paired_gate.py \
    --pair seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724,exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724/snapshots/10000_snapshot.pkl \
    --pair seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724/snapshots/10000_snapshot.pkl \
    --pair seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724/snapshots/10000_snapshot.pkl \
    --gpu-id 1 --gpu-id 5 \
    --output-dir "$stage66_output" --work-root "$stage66_work" \
    --num-eval-episodes 200 --eval-seed-start 99000 \
    --policy-value-beta 1 --flow-readout integrated \
    --num-flow-steps "$steps" \
    --bootstrap-replicates 20000 --bootstrap-seed 99200 \
    --min-mean-delta 0 --min-ci-lower 0 \
    > "$stage66_control/gate.log" 2>&1
else
  touch "$stage66_control/skipped_no_integrated_promotion"
fi
touch "$stage66_control/complete"

current_control=$stage67_control
printf "%s\n" "$BASHPID" > "$stage67_control/controller.pid"
stage66_summary="$stage66_output/summary.json"
if test "$(jq -r .gate "$stage64_summary")" = pass \
  || { test -f "$stage66_summary" \
    && test "$(jq -r .gate "$stage66_summary")" = pass; }; then
  touch "$stage67_control/skipped_task_gate_already_passed"
  touch "$stage67_control/complete"
  exit 0
fi

readout=distill
step_args=()
if test "$(jq -r .gate "$stage65_summary")" = pass; then
  readout=integrated
  steps=$(jq -r .selected_readout.steps "$stage65_summary")
  step_args=(--num-flow-steps "$steps")
  printf "%s\n" "$steps" > "$stage67_control/selected_steps"
fi
printf "%s\n" "$readout" > "$stage67_control/selected_readout"

MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_floq_checkpoint_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724 \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724 \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724 \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage67_output" --work-root "$stage67_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout "$readout" "${step_args[@]}" \
  --policy-value-beta 1 \
  --screen-episodes 10 --screen-seed-start 101000 \
  --validation-episodes 50 --validation-seed-start 102000 \
  --confirmation-episodes 200 --confirmation-seed-start 103000 \
  --bootstrap-replicates 20000 --bootstrap-seed 103200 \
  > "$stage67_control/gate.log" 2>&1
touch "$stage67_control/complete"
