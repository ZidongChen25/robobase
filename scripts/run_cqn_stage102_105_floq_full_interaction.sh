#!/usr/bin/env bash
set -euo pipefail

# Route-B fallback after the factorial main effects: combine the official
# MovePlate FLOQ source interval, lambda=8 BCFM weighting, and a fixed
# independent-BC Bellman target.  This is the previously untested interaction,
# not another sweep of flow steps or deployment beta.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage94_97_bc_policy_final_master
base_summary=exp_local/cqn_flow_high_utd/stage97_bc_policy_final_summary_20260724.json
stage102_control=exp_local/cqn_flow_high_utd/stage102_floq_full_interaction_smoke_controller
stage103_control=exp_local/cqn_flow_high_utd/stage103_floq_full_interaction_training_controller
stage104_control=exp_local/cqn_flow_high_utd/stage104_floq_full_interaction_task_controller
stage105_control=exp_local/cqn_flow_high_utd/stage105_floq_full_interaction_causal_controller
stage105_summary_control=exp_local/cqn_flow_high_utd/stage105_floq_full_interaction_summary_controller

smoke_base=exp_local/cqn_flow_high_utd/stage102_floq_full_interaction_smoke_seed1_20260724
prefill_gate=exp_local/cqn_flow_high_utd/stage102_full_interaction_prefill_controller/gate.json
run1=exp_local/cqn_flow_high_utd/stage103_floq_full_interaction_utd4_seed1_20260724
run2=exp_local/cqn_flow_high_utd/stage103_floq_full_interaction_utd4_seed2_20260724
run3=exp_local/cqn_flow_high_utd/stage103_floq_full_interaction_utd4_seed3_20260724
task_output=exp_local/cqn_flow_high_utd/stage104_floq_full_interaction_task_seed172000_20260724
task_work=exp_local/cqn_flow_high_utd/stage104_floq_full_interaction_task_work_seed172000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage105_floq_full_interaction_causal_seed175000_20260724
final_output=exp_local/cqn_flow_high_utd/stage105_floq_full_interaction_autoresearch_summary_20260724.json

mkdir -p \
  "$stage102_control" \
  "$stage103_control" \
  "$stage104_control" \
  "$stage105_control" \
  "$stage105_summary_control"
printf "%s\n" "$BASHPID" > "$stage102_control/controller.pid"
current_control=$stage102_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  for control in \
    "$stage102_control" \
    "$stage103_control" \
    "$stage104_control" \
    "$stage105_control" \
    "$stage105_summary_control"; do
    touch "$control/skipped_route_b_already_passed"
    touch "$control/complete"
  done
  exit 0
fi

launch=cqn_flow_floq_source01_bcfm8_td_bc_policy_two_tower_high_utd4_gate
smoke=$smoke_base
if test -d "$smoke_base" && (
  ! test -f "$prefill_gate" \
    || test "$(jq -r .gate "$prefill_gate" 2>/dev/null || true)" != pass
); then
  smoke="${smoke_base}_retry_${BASHPID}"
  printf "%s\n" "$smoke" > "$stage102_control/retry_run"
elif test -f "$prefill_gate"; then
  touch "$stage102_control/reused_prefill"
fi
if ! test -f "$smoke/snapshots/1000_snapshot.pkl"; then
  MUJOCO_GL=egl .venv/bin/python train.py \
    launch="$launch" env=bigym/move_plate \
    seed=1 gpu_id=5 num_train_frames=1500 \
    wandb.use=false save_csv=true log_eval_video=false \
    hydra.run.dir="$smoke" \
    > "$stage102_control/smoke.log" 2>&1
fi
.venv/bin/python scripts/check_cqn_floq_training_gate.py \
  --run-dir "$smoke" \
  --output "$stage102_control/gate.json" \
  --required-snapshot-step 1000 \
  --min-log-rows 2 \
  --expected-bcfm-lambda 8 \
  --expected-source-min 0 \
  --expected-source-max 0.1 \
  --expected-td-target-action-source bc_policy \
  > "$stage102_control/gate.log" 2>&1
test "$(jq -r .gate "$stage102_control/gate.json")" = pass
touch "$stage102_control/complete"

current_control=$stage103_control
printf "%s\n" "$BASHPID" > "$stage103_control/controller.pid"

train_one() {
  local label=$1
  local seed=$2
  local gpu=$3
  local destination=$4
  if ! test -f "$destination/snapshots/10000_snapshot.pkl"; then
    MUJOCO_GL=egl .venv/bin/python train.py \
      launch="$launch" env=bigym/move_plate \
      seed="$seed" gpu_id="$gpu" \
      wandb.use=false save_csv=true log_eval_video=false \
      hydra.run.dir="$destination" \
      > "$stage103_control/${label}.log" 2>&1
  fi
  .venv/bin/python scripts/check_cqn_floq_training_gate.py \
    --run-dir "$destination" \
    --output "$stage103_control/${label}_gate.json" \
    --required-snapshot-step 10000 \
    --min-log-rows 10 \
    --expected-bcfm-lambda 8 \
    --expected-source-min 0 \
    --expected-source-max 0.1 \
    --expected-td-target-action-source bc_policy \
    > "$stage103_control/${label}_gate.log" 2>&1
  touch "$stage103_control/${label}_complete"
}

# Route A owns GPU1 while still open.  If A has already passed, use both cards.
if test "$(jq -r .route_a.overall_gate "$base_summary")" = pass; then
  (
    train_one seed1 1 1 "$run1"
    train_one seed3 3 1 "$run3"
  ) &
  worker1=$!
  train_one seed2 2 5 "$run2" &
  worker5=$!
  printf "%s\n" "$worker1" > "$stage103_control/gpu1_worker.pid"
  printf "%s\n" "$worker5" > "$stage103_control/gpu5_worker.pid"
  wait "$worker1"
  wait "$worker5"
else
  train_one seed1 1 5 "$run1"
  route_a_fallback_master=exp_local/cqn_flow_high_utd/stage98_101_h1_cf_fqe_master
  if test -f "$route_a_fallback_master/complete"; then
    train_one seed2 2 5 "$run2" &
    worker5=$!
    train_one seed3 3 1 "$run3" &
    worker1=$!
    printf "%s\n" "$worker1" > "$stage103_control/gpu1_worker.pid"
    printf "%s\n" "$worker5" > "$stage103_control/gpu5_worker.pid"
    wait "$worker1"
    wait "$worker5"
  else
    train_one seed2 2 5 "$run2"
    train_one seed3 3 5 "$run3"
  fi
fi
touch "$stage103_control/complete"

current_control=$stage104_control
printf "%s\n" "$BASHPID" > "$stage104_control/controller.pid"
gpu_args=(--gpu-id 5)
if test "$(jq -r .route_a.overall_gate "$base_summary")" = pass; then
  gpu_args=(--gpu-id 1 --gpu-id 5)
fi
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$run1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$run2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$run3" \
  "${gpu_args[@]}" \
  --output-dir "$task_output" \
  --work-root "$task_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 \
  --candidate-readout distill \
  --beta 0.3 1 3 \
  --screen-beta 1 \
  --screen-episodes 10 \
  --screen-seed-start 172000 \
  --validation-episodes 50 \
  --validation-seed-start 173000 \
  --confirmation-episodes 200 \
  --confirmation-seed-start 174000 \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 174200 \
  --min-mean-delta 0 \
  --min-ci-lower 0 \
  > "$stage104_control/gate.log" 2>&1
touch "$stage104_control/complete"

current_control=$stage105_control
printf "%s\n" "$BASHPID" > "$stage105_control/controller.pid"
task_summary="$task_output/summary.json"
test -f "$task_summary"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage105_control/skipped_task_fail"
  touch "$stage105_control/complete"
  current_control=$stage105_summary_control
  printf "%s\n" "$BASHPID" > "$stage105_summary_control/controller.pid"
  .venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
    --base-summary "$base_summary" \
    --b-candidate "flow_full_official_interaction=$task_summary" \
    --output "$final_output" \
    > "$stage105_summary_control/summary.log" 2>&1
  touch "$stage105_summary_control/next_gate_required"
  touch "$stage105_summary_control/complete"
  exit 0
fi

beta=$(jq -r .selection.selected_global_beta "$task_summary")
step1=$(jq -r .selection.selected_steps.seed1 "$task_summary")
step2=$(jq -r .selection.selected_steps.seed2 "$task_summary")
step3=$(jq -r .selection.selected_steps.seed3 "$task_summary")
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$run1,$run1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$run2,$run2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$run3,$run3/snapshots/${step3}_snapshot.pkl" \
  "${gpu_args[@]}" \
  --output-dir "$causal_output" \
  --eval-seed-start 175000 \
  --num-eval-seeds 32 \
  --anchor-steps 30,75,120 \
  --force-level 1 \
  --intervention-mode sibling_horizon \
  --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout distill \
  --policy-value-beta "$beta" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 175100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage105_control/gate.log" 2>&1
touch "$stage105_control/complete"

current_control=$stage105_summary_control
printf "%s\n" "$BASHPID" > "$stage105_summary_control/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --b-candidate \
  "flow_full_official_interaction=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage105_summary_control/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" = pass; then
  touch "$stage105_summary_control/research_goal_pass"
else
  touch "$stage105_summary_control/next_gate_required"
fi
touch "$stage105_summary_control/complete"
