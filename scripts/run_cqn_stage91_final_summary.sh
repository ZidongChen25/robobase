#!/usr/bin/env bash
set -euo pipefail

# Complete causal audits for any task-qualified legacy Flow candidate, then
# combine every predeclared Route-A and Route-B candidate into one final
# reproducibility-oriented verdict.

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage88_90_td_target_final_master/controller.pid
control=exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_controller
output=exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_20260724.json
mkdir -p "$control"
printf "%s\n" "$BASHPID" > "$control/controller.pid"
trap 'touch "$control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test -f exp_local/cqn_flow_high_utd/stage90_td_target_causal_controller/complete
test -f exp_local/cqn_flow_high_utd/stage87_direct_q_replay_causal_controller/complete

stage64=exp_local/cqn_flow_high_utd/stage64_floq_multiseed_final_seed92000_20260724/summary.json
stage65=exp_local/cqn_flow_high_utd/stage65_readout_mechanism_seed96000_20260724/summary.json
stage66=exp_local/cqn_flow_high_utd/stage66_integrated_multiseed_final_seed99000_20260724/summary.json
stage67=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724/summary.json
stage67_control=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_controller_r1
stage78=exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_seed118000_20260724/summary.json
stage80_fidelity=exp_local/cqn_flow_high_utd/stage80_floq_fidelity_causal_seed122000_20260724/summary.json
stage88=exp_local/cqn_flow_high_utd/stage88_floq_td_target_task_seed134000_20260724/summary.json
stage90_flow=exp_local/cqn_flow_high_utd/stage90_floq_td_target_causal_seed137000_20260724/summary.json
stage89=exp_local/cqn_flow_high_utd/stage89_direct_q_td_target_task_seed134000_20260724/summary.json
stage90_direct=exp_local/cqn_flow_high_utd/stage90_direct_q_td_target_causal_seed137000_20260724/summary.json

flow_run1=exp_local/cqn_flow_high_utd/stage24_floq_distill_utd4_seed1_gpu5_20260724
flow_run2=exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed2_gpu5_20260724
flow_run3=exp_local/cqn_flow_high_utd/stage62_floq_distill_utd4_seed3_gpu1_20260724

run_causal() {
  local destination=$1
  local readout=$2
  local beta=$3
  local step1=$4
  local step2=$5
  local step3=$6
  local flow_steps=${7:-}
  if test -f "$destination/summary.json" \
    && test "$(jq -r .status "$destination/summary.json")" = ok; then
    return
  fi
  local step_args=()
  if test -n "$flow_steps"; then
    step_args=(--num-flow-steps "$flow_steps")
  fi
  MUJOCO_GL=egl .venv/bin/python \
    scripts/run_cqn_flow_branch_multiseed_gate.py \
    --checkpoint "seed1=$flow_run1,$flow_run1/snapshots/${step1}_snapshot.pkl" \
    --checkpoint "seed2=$flow_run2,$flow_run2/snapshots/${step2}_snapshot.pkl" \
    --checkpoint "seed3=$flow_run3,$flow_run3/snapshots/${step3}_snapshot.pkl" \
    --gpu-id 1 --gpu-id 5 --output-dir "$destination" \
    --eval-seed-start 142000 --num-eval-seeds 32 \
    --anchor-steps 30,75,120 --force-level 1 \
    --intervention-mode sibling_horizon --intervention-horizon 1 \
    --max-continuation-steps 300 \
    --flow-readout "$readout" "${step_args[@]}" \
    --policy-value-beta "$beta" \
    --bootstrap-replicates 20000 --bootstrap-seed 142100 \
    --min-informative-states 24 \
    --required-positive-training-seeds 2 \
    --require-anti-cheat-proxies
}

b_args=()
test -f "$stage64"
if test "$(jq -r .gate "$stage64")" = pass; then
  legacy_distill=exp_local/cqn_flow_high_utd/stage91_legacy_distill_causal_seed142000_20260724
  run_causal "$legacy_distill" distill 1 10000 10000 10000 \
    > "$control/legacy_distill_causal.log" 2>&1
  b_args+=(
    --b-candidate
    "legacy_distill=$stage64,$legacy_distill/summary.json"
  )
else
  b_args+=(--b-candidate "legacy_distill=$stage64")
fi

if test -f "$stage66"; then
  if test "$(jq -r .gate "$stage66")" = pass; then
    integrated_steps=$(jq -r .selected_readout.steps "$stage65")
    legacy_integrated=exp_local/cqn_flow_high_utd/stage91_legacy_integrated_causal_seed142000_20260724
    run_causal \
      "$legacy_integrated" integrated 1 10000 10000 10000 \
      "$integrated_steps" \
      > "$control/legacy_integrated_causal.log" 2>&1
    b_args+=(
      --b-candidate
      "legacy_integrated=$stage66,$legacy_integrated/summary.json"
    )
  else
    b_args+=(--b-candidate "legacy_integrated=$stage66")
  fi
fi

if test -f "$stage67"; then
  if test "$(jq -r .gate "$stage67")" = pass; then
    readout=$(sed -n '1p' "$stage67_control/selected_readout")
    # The fixed-beta Flow checkpoint runner stores the frozen checkpoint map
    # at top-level ``selected_steps``; the checkpoint×beta runner used by
    # later stages nests it under ``selection``.
    step1=$(jq -r .selected_steps.seed1 "$stage67")
    step2=$(jq -r .selected_steps.seed2 "$stage67")
    step3=$(jq -r .selected_steps.seed3 "$stage67")
    step_args=()
    if test "$readout" = integrated; then
      integrated_steps=$(sed -n '1p' "$stage67_control/selected_steps")
      step_args=("$integrated_steps")
    fi
    legacy_best=exp_local/cqn_flow_high_utd/stage91_legacy_best_checkpoint_causal_seed142000_20260724
    run_causal \
      "$legacy_best" "$readout" 1 "$step1" "$step2" "$step3" \
      "${step_args[@]}" \
      > "$control/legacy_best_causal.log" 2>&1
    b_args+=(
      --b-candidate
      "legacy_validation_best=$stage67,$legacy_best/summary.json"
    )
  else
    b_args+=(--b-candidate "legacy_validation_best=$stage67")
  fi
fi

if test -f "$stage78"; then
  if test "$(jq -r .gate "$stage78")" = pass; then
    test -f "$stage80_fidelity"
    b_args+=(
      --b-candidate
      "official_fidelity_flow=$stage78,$stage80_fidelity"
    )
  else
    b_args+=(--b-candidate "official_fidelity_flow=$stage78")
  fi
fi

if test -f "$stage88"; then
  if test "$(jq -r .gate "$stage88")" = pass; then
    test -f "$stage90_flow"
    b_args+=(
      --b-candidate
      "flow_policy_value_td=$stage88,$stage90_flow"
    )
  else
    b_args+=(--b-candidate "flow_policy_value_td=$stage88")
  fi
fi

a_args=(
  --a-candidate
  direct_q_replay_next=exp_local/cqn_flow_high_utd/stage86_direct_q_replay_task_seed130000_20260724/summary.json,exp_local/cqn_flow_high_utd/stage87_direct_q_replay_causal_seed133000_20260724/summary.json
)
if test -f "$stage89"; then
  if test "$(jq -r .gate "$stage89")" = pass; then
    test -f "$stage90_direct"
    a_args+=(
      --a-candidate
      "direct_q_policy_value_td=$stage89,$stage90_direct"
    )
  else
    a_args+=(--a-candidate "direct_q_policy_value_td=$stage89")
  fi
fi

.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  "${a_args[@]}" "${b_args[@]}" --output "$output" \
  > "$control/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$output")" = pass; then
  touch "$control/research_goal_pass"
else
  touch "$control/next_gate_required"
fi
touch "$control/complete"
