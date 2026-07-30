#!/usr/bin/env bash
set -euo pipefail

# Conditional matched mechanism arm for any route still open after Stage 91.
# The only algorithmic change relative to the replay-next parents is that the
# independently trained BC policy supplies the Bellman next action.  Rollout
# remains exact BC, so the critic cannot improve task behavior during training
# by selecting its own TD target action.

cd /home/zc1525/robobase_jaxflat

upstream_control=exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_controller
base_summary=exp_local/cqn_flow_high_utd/stage91_final_autoresearch_summary_20260724.json
stage92_control=exp_local/cqn_flow_high_utd/stage92_bc_policy_target_smoke_controller
stage93_control=exp_local/cqn_flow_high_utd/stage93_bc_policy_target_training_controller

flow_smoke=exp_local/cqn_flow_high_utd/stage92_floq_td_bc_policy_smoke_seed1_gpu1_20260724
direct_smoke=exp_local/cqn_flow_high_utd/stage92_direct_q_td_bc_policy_smoke_seed1_gpu5_20260724
flow_seed1=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed1_gpu1_20260724
flow_seed2=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed2_gpu5_20260724
flow_seed3=exp_local/cqn_flow_high_utd/stage93_floq_td_bc_policy_utd4_seed3_gpu1_20260724
direct_seed1=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed1_gpu5_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed2_gpu1_20260724
direct_seed3=exp_local/cqn_flow_high_utd/stage93_direct_q_td_bc_policy_utd4_seed3_gpu5_20260724

mkdir -p "$stage92_control" "$stage93_control"
printf "%s\n" "$BASHPID" > "$stage92_control/controller.pid"
current_control=$stage92_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_control/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_control/complete"
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

if test "$(jq -r .research_goal_gate "$base_summary")" = pass; then
  touch "$stage92_control/skipped_research_goal_already_passed"
  touch "$stage92_control/complete"
  printf "%s\n" "$BASHPID" > "$stage93_control/controller.pid"
  touch "$stage93_control/skipped_research_goal_already_passed"
  touch "$stage93_control/complete"
  exit 0
fi

need_a=false
need_b=false
if test "$(jq -r .route_a.overall_gate "$base_summary")" != pass; then
  need_a=true
  touch "$stage92_control/route_a_enabled"
fi
if test "$(jq -r .route_b.overall_gate "$base_summary")" != pass; then
  need_b=true
  touch "$stage92_control/route_b_enabled"
fi
test "$need_a" = true || test "$need_b" = true

smoke_status=0
if test "$need_b" = true; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate \
    env=bigym/move_plate seed=1 gpu_id=1 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$flow_smoke" \
    > "$stage92_control/flow.log" 2>&1 &
  flow_smoke_pid=$!
  printf "%s\n" "$flow_smoke_pid" > "$stage92_control/flow.pid"
fi
if test "$need_a" = true; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=1 gpu_id=5 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$direct_smoke" \
    > "$stage92_control/direct.log" 2>&1 &
  direct_smoke_pid=$!
  printf "%s\n" "$direct_smoke_pid" > "$stage92_control/direct.pid"
fi
if test "$need_b" = true; then
  if wait "$flow_smoke_pid"; then
    touch "$stage92_control/flow_training_complete"
  else
    touch "$stage92_control/flow_failed"
    smoke_status=1
  fi
fi
if test "$need_a" = true; then
  if wait "$direct_smoke_pid"; then
    touch "$stage92_control/direct_training_complete"
  else
    touch "$stage92_control/direct_failed"
    smoke_status=1
  fi
fi
test "$smoke_status" -eq 0

if test "$need_b" = true; then
  .venv/bin/python scripts/check_cqn_floq_training_gate.py \
    --run-dir "$flow_smoke" \
    --output "$stage92_control/flow_gate.json" \
    --expected-bcfm-lambda 1 \
    --expected-td-target-action-source bc_policy
  test "$(jq -r .gate "$stage92_control/flow_gate.json")" = pass
fi
if test "$need_a" = true; then
  .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
    --run-dir "$direct_smoke" \
    --output "$stage92_control/direct_gate.json" \
    --expected-td-target-action-source bc_policy
  test "$(jq -r .gate "$stage92_control/direct_gate.json")" = pass
fi
touch "$stage92_control/complete"

current_control=$stage93_control
printf "%s\n" "$BASHPID" > "$stage93_control/controller.pid"
if test "$need_a" = true; then
  touch "$stage93_control/route_a_enabled"
fi
if test "$need_b" = true; then
  touch "$stage93_control/route_b_enabled"
fi

run_training() {
  local label=$1
  local launch=$2
  local seed=$3
  local gpu=$4
  local run_dir=$5
  if test -f "$run_dir/snapshots/10000_snapshot.pkl"; then
    touch "$stage93_control/${label}_snapshot_reused"
    return
  fi
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch="$launch" env=bigym/move_plate seed="$seed" gpu_id="$gpu" \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$run_dir" \
    > "$stage93_control/${label}.log" 2>&1
  touch "$stage93_control/${label}_training_complete"
}

worker_gpu1() {
  if test "$need_b" = true; then
    run_training \
      flow_seed1 cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate \
      1 1 "$flow_seed1"
  fi
  if test "$need_a" = true; then
    run_training \
      direct_seed2 cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate \
      2 1 "$direct_seed2"
  fi
  if test "$need_b" = true; then
    run_training \
      flow_seed3 cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate \
      3 1 "$flow_seed3"
  fi
}

worker_gpu5() {
  if test "$need_a" = true; then
    run_training \
      direct_seed1 cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate \
      1 5 "$direct_seed1"
  fi
  if test "$need_b" = true; then
    run_training \
      flow_seed2 cqn_flow_floq_td_bc_policy_two_tower_high_utd4_gate \
      2 5 "$flow_seed2"
  fi
  if test "$need_a" = true; then
    run_training \
      direct_seed3 cqn_direct_q_td_bc_policy_two_tower_coherent_mc_high_utd4_gate \
      3 5 "$direct_seed3"
  fi
}

worker_gpu1 &
gpu1_worker_pid=$!
worker_gpu5 &
gpu5_worker_pid=$!
printf "%s\n" "$gpu1_worker_pid" > "$stage93_control/gpu1_worker.pid"
printf "%s\n" "$gpu5_worker_pid" > "$stage93_control/gpu5_worker.pid"
training_status=0
if ! wait "$gpu1_worker_pid"; then
  touch "$stage93_control/gpu1_worker_failed"
  training_status=1
fi
if ! wait "$gpu5_worker_pid"; then
  touch "$stage93_control/gpu5_worker_failed"
  training_status=1
fi
test "$training_status" -eq 0

if test "$need_b" = true; then
  for item in \
    "flow_seed1:$flow_seed1" \
    "flow_seed2:$flow_seed2" \
    "flow_seed3:$flow_seed3"; do
    label=${item%%:*}
    run_dir=${item#*:}
    .venv/bin/python scripts/check_cqn_floq_training_gate.py \
      --run-dir "$run_dir" \
      --output "$stage93_control/${label}_gate.json" \
      --required-snapshot-step 10000 --min-log-rows 10 \
      --expected-bcfm-lambda 1 \
      --expected-td-target-action-source bc_policy
    test "$(jq -r .gate "$stage93_control/${label}_gate.json")" = pass
  done
fi
if test "$need_a" = true; then
  for item in \
    "direct_seed1:$direct_seed1" \
    "direct_seed2:$direct_seed2" \
    "direct_seed3:$direct_seed3"; do
    label=${item%%:*}
    run_dir=${item#*:}
    .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
      --run-dir "$run_dir" \
      --output "$stage93_control/${label}_gate.json" \
      --required-snapshot-step 10000 --min-log-rows 10 \
      --expected-td-target-action-source bc_policy
    test "$(jq -r .gate "$stage93_control/${label}_gate.json")" = pass
  done
fi
touch "$stage93_control/complete"
