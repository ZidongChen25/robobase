#!/usr/bin/env bash
set -euo pipefail

# Conditional mechanism experiment motivated by FLOQ's non-stationary-TD
# analysis.  It runs only if all prior Flow task gates fail.  Rollout remains
# exact BC; only Bellman next-action selection changes to online
# critic+BC-prior selection, with target-network evaluation (Double-Q).

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage74_80_fidelity_recovery_master/controller.pid
stage64_summary=exp_local/cqn_flow_high_utd/stage64_floq_multiseed_final_seed92000_20260724/summary.json
stage66_summary=exp_local/cqn_flow_high_utd/stage66_integrated_multiseed_final_seed99000_20260724/summary.json
stage67_summary=exp_local/cqn_flow_high_utd/stage67_best_checkpoint_multiseed_seed101000_20260724/summary.json
stage78_summary=exp_local/cqn_flow_high_utd/stage78_floq_fidelity_multiseed_seed118000_20260724/summary.json

stage82_control=exp_local/cqn_flow_high_utd/stage82_td_target_mechanism_smoke_controller
stage83_control=exp_local/cqn_flow_high_utd/stage83_td_target_mechanism_seed12_controller

flow_smoke=exp_local/cqn_flow_high_utd/stage82_floq_td_policy_value_smoke_seed1_gpu1_20260724
direct_smoke=exp_local/cqn_flow_high_utd/stage82_direct_q_td_policy_value_smoke_seed1_gpu5_20260724
flow_seed1=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed1_gpu1_20260724
flow_seed2=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed2_gpu5_20260724
direct_seed1=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed1_gpu5_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed2_gpu1_20260724

mkdir -p "$stage82_control" "$stage83_control"
printf "%s\n" "$BASHPID" > "$stage82_control/controller.pid"
current_control=$stage82_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test -f "$stage64_summary"
test -f exp_local/cqn_flow_high_utd/stage80_floq_fidelity_causal_controller_r1/complete

gate_pass() {
  local artifact=$1
  test -f "$artifact" && test "$(jq -r .gate "$artifact")" = pass
}

if gate_pass "$stage64_summary" \
  || gate_pass "$stage66_summary" \
  || gate_pass "$stage67_summary" \
  || gate_pass "$stage78_summary"; then
  touch "$stage82_control/skipped_existing_b_task_pass"
  touch "$stage82_control/complete"
  touch "$stage83_control/skipped_existing_b_task_pass"
  touch "$stage83_control/complete"
  exit 0
fi

MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate \
  env=bigym/move_plate seed=1 gpu_id=1 num_train_frames=1500 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$flow_smoke" \
  > "$stage82_control/flow.log" 2>&1 &
flow_smoke_pid=$!

MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate \
  env=bigym/move_plate seed=1 gpu_id=5 num_train_frames=1500 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$direct_smoke" \
  > "$stage82_control/direct.log" 2>&1 &
direct_smoke_pid=$!

printf "%s\n" "$flow_smoke_pid" > "$stage82_control/flow.pid"
printf "%s\n" "$direct_smoke_pid" > "$stage82_control/direct.pid"
status=0
if wait "$flow_smoke_pid"; then
  touch "$stage82_control/flow_training_complete"
else
  touch "$stage82_control/flow_failed"
  status=1
fi
if wait "$direct_smoke_pid"; then
  touch "$stage82_control/direct_training_complete"
else
  touch "$stage82_control/direct_failed"
  status=1
fi
test "$status" -eq 0

.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$flow_smoke" \
  --output "$stage82_control/flow_gate.json" \
  --expected-bcfm-lambda 1 \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

.venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
  --run-dir "$direct_smoke" \
  --output "$stage82_control/direct_gate.json" \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

touch "$stage82_control/complete"

current_control=$stage83_control
printf "%s\n" "$BASHPID" > "$stage83_control/controller.pid"
test "$(jq -r .gate "$stage82_control/flow_gate.json")" = pass
test "$(jq -r .gate "$stage82_control/direct_gate.json")" = pass

# Pipeline the unequal-duration jobs to keep both GPUs useful:
# Flow-1 || Direct-1; when Direct-1 ends start Flow-2 on GPU5; when Flow-1
# ends start Direct-2 on GPU1.
MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate \
  env=bigym/move_plate seed=1 gpu_id=1 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$flow_seed1" \
  > "$stage83_control/flow_seed1.log" 2>&1 &
flow1_pid=$!

MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate \
  env=bigym/move_plate seed=1 gpu_id=5 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$direct_seed1" \
  > "$stage83_control/direct_seed1.log" 2>&1 &
direct1_pid=$!

printf "%s\n" "$flow1_pid" > "$stage83_control/flow_seed1.pid"
printf "%s\n" "$direct1_pid" > "$stage83_control/direct_seed1.pid"

wait "$direct1_pid"
touch "$stage83_control/direct_seed1_training_complete"
.venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
  --run-dir "$direct_seed1" \
  --output "$stage83_control/direct_seed1_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate \
  env=bigym/move_plate seed=2 gpu_id=5 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$flow_seed2" \
  > "$stage83_control/flow_seed2.log" 2>&1 &
flow2_pid=$!
printf "%s\n" "$flow2_pid" > "$stage83_control/flow_seed2.pid"

wait "$flow1_pid"
touch "$stage83_control/flow_seed1_training_complete"
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$flow_seed1" \
  --output "$stage83_control/flow_seed1_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-bcfm-lambda 1 \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

MUJOCO_GL=egl .venv/bin/python train.py \
  launch=cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate \
  env=bigym/move_plate seed=2 gpu_id=1 \
  wandb.use=false save_csv=true log_eval_video=false \
  hydra.run.dir="$direct_seed2" \
  > "$stage83_control/direct_seed2.log" 2>&1 &
direct2_pid=$!
printf "%s\n" "$direct2_pid" > "$stage83_control/direct_seed2.pid"

wait "$flow2_pid"
touch "$stage83_control/flow_seed2_training_complete"
wait "$direct2_pid"
touch "$stage83_control/direct_seed2_training_complete"

.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$flow_seed2" \
  --output "$stage83_control/flow_seed2_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-bcfm-lambda 1 \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

.venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
  --run-dir "$direct_seed2" \
  --output "$stage83_control/direct_seed2_gate.json" \
  --required-snapshot-step 10000 --min-log-rows 10 \
  --expected-td-target-action-source policy_value \
  --expected-td-target-policy-value-beta 1

touch "$stage83_control/complete"
