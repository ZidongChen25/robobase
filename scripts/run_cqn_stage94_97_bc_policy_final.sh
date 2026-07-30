#!/usr/bin/env bash
set -euo pipefail

# Select and confirm the BC-policy-target candidates, run causal branch audits
# only for task-qualified policies, then append them to the Stage-91 evidence.

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage92_93_bc_policy_target_master/controller.pid
upstream_control=exp_local/cqn_flow_high_utd/stage93_bc_policy_target_training_controller
base_summary=exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_20260724.json
stage94_control=exp_local/cqn_flow_high_utd/stage94_flow_bc_policy_task_controller
stage95_control=exp_local/cqn_flow_high_utd/stage95_direct_q_bc_policy_task_controller
stage96_control=exp_local/cqn_flow_high_utd/stage96_bc_policy_causal_controller
stage97_control=exp_local/cqn_flow_high_utd/stage97_bc_policy_final_summary_controller

flow_seed1=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed1_gpu1_20260724
flow_seed2=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed2_gpu5_20260724
flow_seed3=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed3_gpu1_20260724
direct_seed1=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed1_gpu5_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed2_gpu1_20260724
direct_seed3=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed3_gpu5_20260724

stage94_output=exp_local/cqn_flow_high_utd/stage94_flow_bc_policy_task_seed144000_20260724
stage94_work=exp_local/cqn_flow_high_utd/stage94_flow_bc_policy_task_work_seed144000_20260724
stage95_output=exp_local/cqn_flow_high_utd/stage95_direct_q_bc_policy_task_seed144000_20260724
stage95_work=exp_local/cqn_flow_high_utd/stage95_direct_q_bc_policy_task_work_seed144000_20260724
flow_causal=exp_local/cqn_flow_high_utd/stage96_flow_bc_policy_causal_seed147000_20260724
direct_causal=exp_local/cqn_flow_high_utd/stage96_direct_q_bc_policy_causal_seed147000_20260724
final_output=exp_local/cqn_flow_high_utd/stage97_bc_policy_final_summary_20260724.json

mkdir -p \
  "$stage94_control" \
  "$stage95_control" \
  "$stage96_control" \
  "$stage97_control"
printf "%s\n" "$BASHPID" > "$stage94_control/controller.pid"
current_control=$stage94_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_control/complete"
test -f "$base_summary"

need_a=false
need_b=false
if test -f "$upstream_control/route_a_enabled"; then
  need_a=true
fi
if test -f "$upstream_control/route_b_enabled"; then
  need_b=true
fi

clean_seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724
clean_seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724
clean_seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724

if test "$need_b" = true; then
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_checkpoint_beta_selection_gate.py \
    --training-seed seed1="$clean_seed1","$clean_seed1/snapshots/5000_snapshot.pkl","$flow_seed1" \
    --training-seed seed2="$clean_seed2","$clean_seed2/snapshots/5000_snapshot.pkl","$flow_seed2" \
    --training-seed seed3="$clean_seed3","$clean_seed3/snapshots/2500_snapshot.pkl","$flow_seed3" \
    --gpu-id 1 --gpu-id 5 \
    --output-dir "$stage94_output" --work-root "$stage94_work" \
    --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --screen-top-k 2 --candidate-readout distill \
    --beta 0.3 1 3 --screen-beta 1 \
    --screen-episodes 10 --screen-seed-start 144000 \
    --validation-episodes 50 --validation-seed-start 145000 \
    --confirmation-episodes 200 --confirmation-seed-start 146000 \
    --bootstrap-replicates 20000 --bootstrap-seed 146200 \
    --min-mean-delta 0 --min-ci-lower 0 \
    > "$stage94_control/gate.log" 2>&1
else
  touch "$stage94_control/skipped_route_b_already_passed"
fi
touch "$stage94_control/complete"

current_control=$stage95_control
printf "%s\n" "$BASHPID" > "$stage95_control/controller.pid"
if test "$need_a" = true; then
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_checkpoint_beta_selection_gate.py \
    --training-seed seed1="$clean_seed1","$clean_seed1/snapshots/5000_snapshot.pkl","$direct_seed1" \
    --training-seed seed2="$clean_seed2","$clean_seed2/snapshots/5000_snapshot.pkl","$direct_seed2" \
    --training-seed seed3="$clean_seed3","$clean_seed3/snapshots/2500_snapshot.pkl","$direct_seed3" \
    --gpu-id 1 --gpu-id 5 \
    --output-dir "$stage95_output" --work-root "$stage95_work" \
    --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
    --screen-top-k 2 --candidate-readout auto \
    --beta 0.3 1 3 --screen-beta 1 \
    --screen-episodes 10 --screen-seed-start 144000 \
    --validation-episodes 50 --validation-seed-start 145000 \
    --confirmation-episodes 200 --confirmation-seed-start 146000 \
    --bootstrap-replicates 20000 --bootstrap-seed 146200 \
    --min-mean-delta 0 --min-ci-lower 0 \
    > "$stage95_control/gate.log" 2>&1
else
  touch "$stage95_control/skipped_route_a_already_passed"
fi
touch "$stage95_control/complete"

current_control=$stage96_control
printf "%s\n" "$BASHPID" > "$stage96_control/controller.pid"

run_causal() {
  local task_summary=$1
  local output=$2
  local readout=$3
  local run1=$4
  local run2=$5
  local run3=$6
  local beta
  local step1
  local step2
  local step3
  beta=$(jq -r .selection.selected_global_beta "$task_summary")
  step1=$(jq -r .selection.selected_steps.seed1 "$task_summary")
  step2=$(jq -r .selection.selected_steps.seed2 "$task_summary")
  step3=$(jq -r .selection.selected_steps.seed3 "$task_summary")
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_flow_branch_multiseed_gate.py \
    --checkpoint "seed1=$run1,$run1/snapshots/${step1}_snapshot.pkl" \
    --checkpoint "seed2=$run2,$run2/snapshots/${step2}_snapshot.pkl" \
    --checkpoint "seed3=$run3,$run3/snapshots/${step3}_snapshot.pkl" \
    --gpu-id 1 --gpu-id 5 --output-dir "$output" \
    --eval-seed-start 147000 --num-eval-seeds 32 \
    --anchor-steps 30,75,120 --force-level 1 \
    --intervention-mode sibling_horizon --intervention-horizon 1 \
    --max-continuation-steps 300 \
    --flow-readout "$readout" --policy-value-beta "$beta" \
    --bootstrap-replicates 20000 --bootstrap-seed 147100 \
    --min-informative-states 24 \
    --required-positive-training-seeds 2 \
    --require-anti-cheat-proxies
}

flow_task="$stage94_output/summary.json"
direct_task="$stage95_output/summary.json"
if test "$need_b" = true; then
  test -f "$flow_task"
  if test "$(jq -r .gate "$flow_task")" = pass; then
    run_causal \
      "$flow_task" "$flow_causal" distill \
      "$flow_seed1" "$flow_seed2" "$flow_seed3" \
      > "$stage96_control/flow.log" 2>&1
  else
    touch "$stage96_control/flow_skipped_task_fail"
  fi
fi
if test "$need_a" = true; then
  test -f "$direct_task"
  if test "$(jq -r .gate "$direct_task")" = pass; then
    run_causal \
      "$direct_task" "$direct_causal" auto \
      "$direct_seed1" "$direct_seed2" "$direct_seed3" \
      > "$stage96_control/direct.log" 2>&1
  else
    touch "$stage96_control/direct_skipped_task_fail"
  fi
fi
touch "$stage96_control/complete"

current_control=$stage97_control
printf "%s\n" "$BASHPID" > "$stage97_control/controller.pid"
candidate_args=()
if test "$need_a" = true; then
  if test "$(jq -r .gate "$direct_task")" = pass; then
    test -f "$direct_causal/summary.json"
    candidate_args+=(
      --a-candidate
      "direct_q_bc_policy_td=$direct_task,$direct_causal/summary.json"
    )
  else
    candidate_args+=(--a-candidate "direct_q_bc_policy_td=$direct_task")
  fi
fi
if test "$need_b" = true; then
  if test "$(jq -r .gate "$flow_task")" = pass; then
    test -f "$flow_causal/summary.json"
    candidate_args+=(
      --b-candidate
      "flow_bc_policy_td=$flow_task,$flow_causal/summary.json"
    )
  else
    candidate_args+=(--b-candidate "flow_bc_policy_td=$flow_task")
  fi
fi

.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  "${candidate_args[@]}" \
  --output "$final_output" \
  > "$stage97_control/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" = pass; then
  touch "$stage97_control/research_goal_pass"
else
  touch "$stage97_control/next_gate_required"
fi
touch "$stage97_control/complete"
