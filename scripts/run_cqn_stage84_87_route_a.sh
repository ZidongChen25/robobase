#!/usr/bin/env bash
set -euo pipefail

# Route A: train a matched monolithic scalar-Q CQN-AS over three independent
# training seeds, select one global deployment beta plus per-seed checkpoints
# on disjoint data, then audit the frozen winners with causal branch
# interventions.  The script starts only after the conditional Stage-82/83
# mechanism jobs release both GPUs.

cd /home/zc1525/robobase_jaxflat

upstream_pid_file=exp_local/cqn_flow_high_utd/stage82_83_td_target_mechanism_master/controller.pid
preflight=exp_local/cqn_flow_high_utd/stage70_preflight_direct_q_smoke_controller/gate.json

stage84_control=exp_local/cqn_flow_high_utd/stage84_direct_q_replay_seed12_controller
stage85_control=exp_local/cqn_flow_high_utd/stage85_direct_q_replay_seed3_controller
stage86_control=exp_local/cqn_flow_high_utd/stage86_direct_q_replay_task_controller
stage87_control=exp_local/cqn_flow_high_utd/stage87_direct_q_replay_causal_controller

direct_seed1=exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed1_20260724
direct_seed2=exp_local/cqn_flow_high_utd/stage84_direct_q_replay_utd4_seed2_20260724
direct_seed3=exp_local/cqn_flow_high_utd/stage85_direct_q_replay_utd4_seed3_20260724

target_stage83=exp_local/cqn_flow_high_utd/stage83_td_target_mechanism_seed12_controller
target_flow_seed1=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed1_gpu1_20260724
target_flow_seed2=exp_local/cqn_flow_high_utd/stage83_floq_td_policy_value_utd4_seed2_gpu5_20260724
target_direct_seed1=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed1_gpu5_20260724
target_direct_seed2=exp_local/cqn_flow_high_utd/stage83_direct_q_td_policy_value_utd4_seed2_gpu1_20260724
target_flow_seed3=exp_local/cqn_flow_high_utd/stage85_floq_td_policy_value_utd4_seed3_gpu5_20260724
target_direct_seed3=exp_local/cqn_flow_high_utd/stage85_direct_q_td_policy_value_utd4_seed3_gpu1_20260724

fallback_c51_seed2=exp_local/cqn_flow_high_utd/stage85_direct_c51_utd4_seed2_20260724

stage86_output=exp_local/cqn_flow_high_utd/stage86_direct_q_replay_task_seed130000_20260724
stage86_work=exp_local/cqn_flow_high_utd/stage86_direct_q_replay_task_work_seed130000_20260724
stage87_output=exp_local/cqn_flow_high_utd/stage87_direct_q_replay_causal_seed133000_20260724

mkdir -p \
  "$stage84_control" \
  "$stage85_control" \
  "$stage86_control" \
  "$stage87_control"
printf "%s\n" "$BASHPID" > "$stage84_control/controller.pid"
current_control=$stage84_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_pid_file")
tail --pid="$upstream_pid" -f /dev/null
test "$(jq -r .gate "$preflight")" = pass
test -f "$target_stage83/complete"

prefill_control=exp_local/cqn_flow_high_utd/stage75_floq_fidelity_seed1_controller_r1
prefill_passes() {
  local label=$1
  local artifact="$prefill_control/route_a_${label}_gate.json"
  test -f "$artifact" && test "$(jq -r .gate "$artifact")" = pass
}

# Do not overwrite an incomplete prefill.  A failed or interrupted filler is
# retried in a fresh run directory, while every downstream reference follows
# the updated variable in this same controller.
if test -d "$direct_seed1" && ! prefill_passes seed1; then
  direct_seed1="${direct_seed1}_retry_${BASHPID}"
fi
if test -d "$direct_seed2" && ! prefill_passes seed2; then
  direct_seed2="${direct_seed2}_retry_${BASHPID}"
fi
if test -d "$direct_seed3" && ! prefill_passes seed3; then
  direct_seed3="${direct_seed3}_retry_${BASHPID}"
fi

seed1_pid=
seed2_pid=
if test -f "$direct_seed1/snapshots/10000_snapshot.pkl"; then
  touch "$stage84_control/seed1_reused_prefill"
else
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=1 gpu_id=1 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$direct_seed1" \
    > "$stage84_control/seed1.log" 2>&1 &
  seed1_pid=$!
  printf "%s\n" "$seed1_pid" > "$stage84_control/seed1.pid"
fi

if test -f "$direct_seed2/snapshots/10000_snapshot.pkl"; then
  touch "$stage84_control/seed2_reused_prefill"
else
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=2 gpu_id=5 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$direct_seed2" \
    > "$stage84_control/seed2.log" 2>&1 &
  seed2_pid=$!
  printf "%s\n" "$seed2_pid" > "$stage84_control/seed2.pid"
fi

status=0
if test -n "$seed1_pid"; then
  if wait "$seed1_pid"; then
    touch "$stage84_control/seed1_training_complete"
  else
    touch "$stage84_control/seed1_failed"
    status=1
  fi
fi
if test -n "$seed2_pid"; then
  if wait "$seed2_pid"; then
    touch "$stage84_control/seed2_training_complete"
  else
    touch "$stage84_control/seed2_failed"
    status=1
  fi
fi
test "$status" -eq 0

for item in "seed1:$direct_seed1" "seed2:$direct_seed2"; do
  label=${item%%:*}
  run_dir=${item#*:}
  .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
    --run-dir "$run_dir" \
    --output "$stage84_control/${label}_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-td-target-action-source replay_next
done
touch "$stage84_control/complete"

current_control=$stage85_control
printf "%s\n" "$BASHPID" > "$stage85_control/controller.pid"

replay3_pid=
if test -f "$direct_seed3/snapshots/10000_snapshot.pkl"; then
  touch "$stage85_control/replay_seed3_reused_prefill"
else
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=3 gpu_id=1 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$direct_seed3" \
    > "$stage85_control/replay_seed3.log" 2>&1 &
  replay3_pid=$!
  printf "%s\n" "$replay3_pid" > "$stage85_control/replay_seed3.pid"
fi

replay3_checked=false
finish_replay3() {
  if test "$replay3_checked" = true; then
    return
  fi
  if test -n "$replay3_pid"; then
    wait "$replay3_pid"
    touch "$stage85_control/replay_seed3_training_complete"
  fi
  .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
    --run-dir "$direct_seed3" \
    --output "$stage85_control/replay_seed3_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-td-target-action-source replay_next
  replay3_checked=true
}

target_ready=false
if test -f "$target_stage83/complete" \
  && ! test -f "$target_stage83/skipped_existing_b_task_pass"; then
  for path in \
    "$target_flow_seed1/snapshots/10000_snapshot.pkl" \
    "$target_flow_seed2/snapshots/10000_snapshot.pkl" \
    "$target_direct_seed1/snapshots/10000_snapshot.pkl" \
    "$target_direct_seed2/snapshots/10000_snapshot.pkl"; do
    test -f "$path"
  done
  target_ready=true
fi

if test "$target_ready" = true; then
  touch "$stage85_control/target_seed3_enabled"
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_flow_floq_td_policy_value_two_tower_high_utd4_gate \
    env=bigym/move_plate seed=3 gpu_id=5 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$target_flow_seed3" \
    > "$stage85_control/target_flow_seed3.log" 2>&1 &
  target_flow3_pid=$!
  printf "%s\n" "$target_flow3_pid" \
    > "$stage85_control/target_flow_seed3.pid"

  finish_replay3

  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_direct_q_td_policy_value_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=3 gpu_id=1 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$target_direct_seed3" \
    > "$stage85_control/target_direct_seed3.log" 2>&1 &
  target_direct3_pid=$!
  printf "%s\n" "$target_direct3_pid" \
    > "$stage85_control/target_direct_seed3.pid"

  wait "$target_flow3_pid"
  touch "$stage85_control/target_flow_seed3_training_complete"
  wait "$target_direct3_pid"
  touch "$stage85_control/target_direct_seed3_training_complete"

  .venv/bin/python scripts/check_cqn_floq_training_gate.py \
    --run-dir "$target_flow_seed3" \
    --output "$stage85_control/target_flow_seed3_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-bcfm-lambda 1 \
    --expected-td-target-action-source policy_value \
    --expected-td-target-policy-value-beta 1
  .venv/bin/python scripts/check_cqn_direct_q_training_gate.py \
    --run-dir "$target_direct_seed3" \
    --output "$stage85_control/target_direct_seed3_gate.json" \
    --required-snapshot-step 10000 --min-log-rows 10 \
    --expected-td-target-action-source policy_value \
    --expected-td-target-policy-value-beta 1
else
  touch "$stage85_control/target_seed3_skipped"
  fallback_gpu=5
  if test -z "$replay3_pid"; then
    fallback_gpu=1
  fi
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch=cqn_as_pixel_bigym_two_tower_coherent_mc_high_utd4_gate \
    env=bigym/move_plate seed=2 "gpu_id=$fallback_gpu" \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$fallback_c51_seed2" \
    > "$stage85_control/fallback_c51_seed2.log" 2>&1 &
  fallback_pid=$!
  printf "%s\n" "$fallback_pid" > "$stage85_control/fallback_c51_seed2.pid"
  printf "%s\n" "$fallback_gpu" > "$stage85_control/fallback_c51_gpu"
  finish_replay3
  wait "$fallback_pid"
  touch "$stage85_control/fallback_c51_seed2_training_complete"
  test -f "$fallback_c51_seed2/snapshots/10000_snapshot.pkl"
fi
touch "$stage85_control/complete"

current_control=$stage86_control
printf "%s\n" "$BASHPID" > "$stage86_control/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$direct_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$direct_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$direct_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$stage86_output" --work-root "$stage86_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --candidate-readout auto \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 130000 \
  --validation-episodes 50 --validation-seed-start 131000 \
  --confirmation-episodes 200 --confirmation-seed-start 132000 \
  --bootstrap-replicates 20000 --bootstrap-seed 132200 \
  --min-mean-delta 0 --min-ci-lower 0 \
  > "$stage86_control/gate.log" 2>&1
touch "$stage86_control/complete"

current_control=$stage87_control
printf "%s\n" "$BASHPID" > "$stage87_control/controller.pid"
stage86_summary="$stage86_output/summary.json"
test -f "$stage86_summary"
beta=$(jq -r .selection.selected_global_beta "$stage86_summary")
step1=$(jq -r .selection.selected_steps.seed1 "$stage86_summary")
step2=$(jq -r .selection.selected_steps.seed2 "$stage86_summary")
step3=$(jq -r .selection.selected_steps.seed3 "$stage86_summary")
printf "%s\n" "$beta" > "$stage87_control/selected_beta"
printf "%s %s %s\n" "$step1" "$step2" "$step3" \
  > "$stage87_control/selected_steps"

MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$direct_seed1,$direct_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$direct_seed2,$direct_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$direct_seed3,$direct_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 --output-dir "$stage87_output" \
  --eval-seed-start 133000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout auto --policy-value-beta "$beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 133100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage87_control/gate.log" 2>&1
touch "$stage87_control/complete"
