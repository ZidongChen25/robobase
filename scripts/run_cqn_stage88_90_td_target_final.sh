#!/usr/bin/env bash
set -euo pipefail

# Final three-training-seed task and causal gates for the target-only
# policy-value mechanism.  Flow and monolithic scalar-Q use the same
# screen/validation/confirmation simulator seeds so their effects relative to
# clean CQN-AS can also be compared without environment-seed noise.

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage84_87_route_a_master/controller.pid
target_stage83=exp_local/cqn_flow_high_utd/stage83_td_target_mechanism_seed12_controller
target_stage85=exp_local/cqn_flow_high_utd/stage85_direct_q_replay_seed3_controller

flow_seed1=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed1_gpu1_20260724
flow_seed2=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed2_gpu5_20260724
flow_seed3=exp_local/cqn_flow_high_utd/stage85_floq_td_policy_value_utd4_seed3_gpu5_20260724
direct_seed1=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed1_gpu5_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed2_gpu1_20260724
direct_seed3=exp_local/cqn_flow_high_utd/stage85_direct_q_td_policy_value_utd4_seed3_gpu1_20260724

stage88_control=exp_local/cqn_flow_high_utd/stage88_floq_td_target_task_controller
stage88_output=exp_local/cqn_flow_high_utd/stage88_floq_td_target_task_seed134000_20260724
stage88_work=exp_local/cqn_flow_high_utd/stage88_floq_td_target_task_work_seed134000_20260724
stage89_control=exp_local/cqn_flow_high_utd/stage89_direct_q_td_target_task_controller
stage89_output=exp_local/cqn_flow_high_utd/stage89_direct_q_td_target_task_seed134000_20260724
stage89_work=exp_local/cqn_flow_high_utd/stage89_direct_q_td_target_task_work_seed134000_20260724
stage90_control=exp_local/cqn_flow_high_utd/stage90_td_target_causal_controller
flow_causal=exp_local/cqn_flow_high_utd/stage90_floq_td_target_causal_seed137000_20260724
direct_causal=exp_local/cqn_flow_high_utd/stage90_direct_q_td_target_causal_seed137000_20260724

mkdir -p "$stage88_control" "$stage89_control" "$stage90_control"
printf "%s\n" "$BASHPID" > "$stage88_control/controller.pid"
current_control=$stage88_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test -f "$target_stage85/complete"
test -f exp_local/cqn_flow_high_utd/stage87_direct_q_replay_causal_controller/complete

if test -f "$target_stage83/skipped_existing_b_task_pass" \
  || test -f "$target_stage85/target_seed3_skipped"; then
  touch "$stage88_control/skipped_existing_b_task_pass"
  touch "$stage88_control/complete"
  touch "$stage89_control/skipped_existing_b_task_pass"
  touch "$stage89_control/complete"
  touch "$stage90_control/skipped_existing_b_task_pass"
  touch "$stage90_control/complete"
  exit 0
fi

for path in \
  "$flow_seed1/snapshots/10000_snapshot.pkl" \
  "$flow_seed2/snapshots/10000_snapshot.pkl" \
  "$flow_seed3/snapshots/10000_snapshot.pkl" \
  "$direct_seed1/snapshots/10000_snapshot.pkl" \
  "$direct_seed2/snapshots/10000_snapshot.pkl" \
  "$direct_seed3/snapshots/10000_snapshot.pkl"; do
  test -f "$path"
done

MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$flow_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$flow_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$flow_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage88_output" --work-root "$stage88_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --candidate-readout integrated --num-flow-steps 8 \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 134000 \
  --validation-episodes 50 --validation-seed-start 135000 \
  --confirmation-episodes 200 --confirmation-seed-start 136000 \
  --bootstrap-replicates 20000 --bootstrap-seed 136200 \
  --min-mean-delta 0 --min-ci-lower 0 \
  > "$stage88_control/gate.log" 2>&1
touch "$stage88_control/complete"

current_control=$stage89_control
printf "%s\n" "$BASHPID" > "$stage89_control/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$direct_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$direct_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$direct_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage89_output" --work-root "$stage89_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --candidate-readout auto \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 134000 \
  --validation-episodes 50 --validation-seed-start 135000 \
  --confirmation-episodes 200 --confirmation-seed-start 136000 \
  --bootstrap-replicates 20000 --bootstrap-seed 136200 \
  --min-mean-delta 0 --min-ci-lower 0 \
  > "$stage89_control/gate.log" 2>&1
touch "$stage89_control/complete"

current_control=$stage90_control
printf "%s\n" "$BASHPID" > "$stage90_control/controller.pid"

run_causal_gate() {
  local task_summary=$1
  local output=$2
  local prefix=$3
  local readout=$4
  local flow_steps=${5:-}
  local beta
  local step1
  local step2
  local step3
  beta=$(jq -r .selection.selected_global_beta "$task_summary")
  step1=$(jq -r .selection.selected_steps.seed1 "$task_summary")
  step2=$(jq -r .selection.selected_steps.seed2 "$task_summary")
  step3=$(jq -r .selection.selected_steps.seed3 "$task_summary")
  printf "%s\n" "$beta" > "$stage90_control/${prefix}_selected_beta"
  printf "%s %s %s\n" "$step1" "$step2" "$step3" \
    > "$stage90_control/${prefix}_selected_steps"

  local step_args=()
  if test -n "$flow_steps"; then
    step_args=(--num-flow-steps "$flow_steps")
  fi
  local run1 run2 run3
  if test "$prefix" = flow; then
    run1=$flow_seed1
    run2=$flow_seed2
    run3=$flow_seed3
  else
    run1=$direct_seed1
    run2=$direct_seed2
    run3=$direct_seed3
  fi
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_flow_branch_multiseed_gate.py \
    --checkpoint "seed1=$run1,$run1/snapshots/${step1}_snapshot.pkl" \
    --checkpoint "seed2=$run2,$run2/snapshots/${step2}_snapshot.pkl" \
    --checkpoint "seed3=$run3,$run3/snapshots/${step3}_snapshot.pkl" \
    --gpu-id 1 --gpu-id 5 --output-dir "$output" \
    --eval-seed-start 137000 --num-eval-seeds 32 \
    --anchor-steps 30,75,120 --force-level 1 \
    --intervention-mode sibling_horizon --intervention-horizon 1 \
    --max-continuation-steps 300 \
    --flow-readout "$readout" "${step_args[@]}" \
    --policy-value-beta "$beta" \
    --bootstrap-replicates 20000 --bootstrap-seed 137100 \
    --min-informative-states 24 \
    --required-positive-training-seeds 2 \
    --require-anti-cheat-proxies
}

flow_summary="$stage88_output/summary.json"
direct_summary="$stage89_output/summary.json"
if test "$(jq -r .gate "$flow_summary")" = pass; then
  run_causal_gate "$flow_summary" "$flow_causal" flow integrated 8 \
    > "$stage90_control/flow_gate.log" 2>&1
else
  touch "$stage90_control/flow_skipped_task_fail"
fi
if test "$(jq -r .gate "$direct_summary")" = pass; then
  run_causal_gate "$direct_summary" "$direct_causal" direct auto \
    > "$stage90_control/direct_gate.log" 2>&1
else
  touch "$stage90_control/direct_skipped_task_fail"
fi
touch "$stage90_control/complete"
