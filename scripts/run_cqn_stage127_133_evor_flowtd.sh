#!/usr/bin/env bash
set -euo pipefail

# Conditional Route-B continuation after all currently queued flow-value
# candidates, the diagnostic-only Stage-119 probe, and the independent
# action-centered Route-A continuation.  Serializing the two routes prevents
# both event-driven controllers from claiming GPU1/GPU5 at the same time.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage120_126_action_centered_master
action_summary=exp_local/cqn_flow_high_utd/stage126_action_centered_autoresearch_summary_20260724.json
stage113_summary=exp_local/cqn_flow_high_utd/stage113_value_flows_confidence_autoresearch_summary_20260724.json
stage118_summary=exp_local/cqn_flow_high_utd/stage118_flowcritic_truncated_autoresearch_summary_20260724.json

stage127=exp_local/cqn_flow_high_utd/stage127_evor_flowtd_smoke_controller
stage128=exp_local/cqn_flow_high_utd/stage128_evor_flowtd_two_seed_controller
stage129=exp_local/cqn_flow_high_utd/stage129_evor_flowtd_two_seed_gate_controller
stage130=exp_local/cqn_flow_high_utd/stage130_evor_flowtd_seed3_controller
stage131=exp_local/cqn_flow_high_utd/stage131_evor_flowtd_task_controller
stage132=exp_local/cqn_flow_high_utd/stage132_evor_flowtd_causal_controller
stage133=exp_local/cqn_flow_high_utd/stage133_evor_flowtd_summary_controller

smoke_seed1=exp_local/cqn_flow_high_utd/stage127_evor_flowtd_smoke_seed1_20260724
smoke_seed2=exp_local/cqn_flow_high_utd/stage127_evor_flowtd_smoke_seed2_20260724
evor_seed1=exp_local/cqn_flow_high_utd/stage128_evor_flowtd_utd4_seed1_20260724
evor_seed2=exp_local/cqn_flow_high_utd/stage128_evor_flowtd_utd4_seed2_20260724
evor_seed3=exp_local/cqn_flow_high_utd/stage130_evor_flowtd_utd4_seed3_20260724

two_seed_output=exp_local/cqn_flow_high_utd/stage129_evor_flowtd_two_seed_gate_seed202000_20260724
two_seed_work=exp_local/cqn_flow_high_utd/stage129_evor_flowtd_two_seed_work_seed202000_20260724
task_output=exp_local/cqn_flow_high_utd/stage131_evor_flowtd_task_seed205000_20260724
task_work=exp_local/cqn_flow_high_utd/stage131_evor_flowtd_task_work_seed205000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage132_evor_flowtd_causal_seed208000_20260724
final_output=exp_local/cqn_flow_high_utd/stage133_evor_flowtd_autoresearch_summary_20260724.json

launch=cqn_evor_flowtd_bc_target_two_tower_high_utd4_gate

mkdir -p \
  "$stage127" "$stage128" "$stage129" "$stage130" \
  "$stage131" "$stage132" "$stage133"
printf "%s\n" "$BASHPID" > "$stage127/controller.pid"
current_control=$stage127
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
if ! test -f "$upstream_master/complete"; then
  touch "$stage127/upstream_route_a_failed_ignored"
fi

if test -f "$action_summary"; then
  base_summary=$action_summary
elif test -f "$stage118_summary"; then
  base_summary=$stage118_summary
elif test -f "$stage113_summary"; then
  base_summary=$stage113_summary
else
  printf "no post-Value-Flows Route-B summary exists\n" >&2
  exit 1
fi
test "$(jq -r .status "$base_summary")" = ok

mark_remaining_skipped() {
  local reason=$1
  for control in \
    "$stage127" "$stage128" "$stage129" "$stage130" \
    "$stage131" "$stage132" "$stage133"; do
    touch "$control/$reason"
    touch "$control/complete"
  done
}

if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  mark_remaining_skipped skipped_route_b_already_passed
  exit 0
fi

train_one() {
  local seed=$1
  local gpu=$2
  local frames=$3
  local destination=$4
  local log=$5
  if ! test -f "$destination/snapshots/$((frames / 1000 * 1000))_snapshot.pkl"; then
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
  local required_step=$3
  local min_rows=$4
  .venv/bin/python scripts/check_cqn_evor_training_gate.py \
    --run-dir "$destination" \
    --output "$output" \
    --required-snapshot-step "$required_step" \
    --min-log-rows "$min_rows" \
    --expected-flow-steps 10 \
    --expected-action-flow-samples 16
}

write_preliminary_failure() {
  local status=$1
  local artifact=$2
  jq \
    --arg status "$status" \
    --arg artifact "$artifact" \
    '. + {
      stage127_132_evor_flowtd: {
        status: $status,
        artifact: $artifact,
        conclusion_scope: "preliminary_only",
        route_b_update_forbidden: true
      }
    }' \
    "$base_summary" > "$final_output"
  touch "$stage133/next_gate_required"
  touch "$stage133/complete"
}

# Stage 127: two independent compilation/training smokes use both assigned
# GPUs.  These are health checks only and are never eligible for task claims.
train_one 1 1 1500 "$smoke_seed1" "$stage127/seed1.log" &
smoke1_pid=$!
train_one 2 5 1500 "$smoke_seed2" "$stage127/seed2.log" &
smoke2_pid=$!
printf "%s\n" "$smoke1_pid" > "$stage127/seed1.pid"
printf "%s\n" "$smoke2_pid" > "$stage127/seed2.pid"
wait "$smoke1_pid"
wait "$smoke2_pid"
smoke_failed=0
if ! check_one "$smoke_seed1" "$stage127/seed1_gate.json" 1000 2; then
  smoke_failed=1
fi
if ! check_one "$smoke_seed2" "$stage127/seed2_gate.json" 1000 2; then
  smoke_failed=1
fi
touch "$stage127/complete"
if test "$smoke_failed" != 0; then
  for control in "$stage128" "$stage129" "$stage130" "$stage131" "$stage132"; do
    touch "$control/skipped_smoke_fail"
    touch "$control/complete"
  done
  current_control=$stage133
  printf "%s\n" "$BASHPID" > "$stage133/controller.pid"
  write_preliminary_failure \
    smoke_health_fail "$stage127/seed1_gate.json,$stage127/seed2_gate.json"
  exit 0
fi

# Stage 128: train two algorithm seeds in parallel before any task selection.
current_control=$stage128
printf "%s\n" "$BASHPID" > "$stage128/controller.pid"
train_one 1 1 10500 "$evor_seed1" "$stage128/seed1.log" &
seed1_pid=$!
train_one 2 5 10500 "$evor_seed2" "$stage128/seed2.log" &
seed2_pid=$!
printf "%s\n" "$seed1_pid" > "$stage128/seed1.pid"
printf "%s\n" "$seed2_pid" > "$stage128/seed2.pid"
wait "$seed1_pid"
wait "$seed2_pid"
check_one "$evor_seed1" "$stage128/seed1_gate.json" 10000 10
check_one "$evor_seed2" "$stage128/seed2_gate.json" 10000 10
test "$(jq -r .gate "$stage128/seed1_gate.json")" = pass
test "$(jq -r .gate "$stage128/seed2_gate.json")" = pass
touch "$stage128/complete"

# Stage 129: select one global BC-prior beta plus one checkpoint per seed on
# disjoint screen/validation splits.  The two-seed confirmation is only a
# compute-promotion gate; it cannot satisfy the final Route-B claim.
current_control=$stage129
printf "%s\n" "$BASHPID" > "$stage129/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$evor_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$evor_seed2" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$two_seed_output" --work-root "$two_seed_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --candidate-readout integrated --num-flow-steps 10 \
  --beta 0.3 1 3 --screen-beta 1 \
  --screen-episodes 10 --screen-seed-start 202000 \
  --validation-episodes 50 --validation-seed-start 203000 \
  --confirmation-episodes 200 --confirmation-seed-start 204000 \
  --bootstrap-replicates 20000 --bootstrap-seed 204200 \
  --min-mean-delta 0 --min-ci-lower -0.05 \
  > "$stage129/gate.log" 2>&1
touch "$stage129/complete"

two_seed_summary="$two_seed_output/summary.json"
if test "$(jq -r .gate "$two_seed_summary")" != pass; then
  for control in "$stage130" "$stage131" "$stage132"; do
    touch "$control/skipped_two_seed_promotion_fail"
    touch "$control/complete"
  done
  current_control=$stage133
  printf "%s\n" "$BASHPID" > "$stage133/controller.pid"
  write_preliminary_failure two_seed_promotion_fail "$two_seed_summary"
  exit 0
fi
selected_beta=$(jq -r .selection.selected_global_beta "$two_seed_summary")

# Stage 130: only a two-seed-promoted method earns the third algorithm seed.
current_control=$stage130
printf "%s\n" "$BASHPID" > "$stage130/controller.pid"
train_one 3 1 10500 "$evor_seed3" "$stage130/seed3.log"
check_one "$evor_seed3" "$stage130/seed3_gate.json" 10000 10
test "$(jq -r .gate "$stage130/seed3_gate.json")" = pass
touch "$stage130/complete"

# Stage 131: final task-quality claim uses fresh selection and sealed common
# confirmation seeds over all three independently trained models.
current_control=$stage131
printf "%s\n" "$BASHPID" > "$stage131/controller.pid"
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_floq_checkpoint_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$evor_seed1" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$evor_seed2" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$evor_seed3" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$task_output" --work-root "$task_work" \
  --checkpoint-step 1000 2000 3000 4000 5000 6000 7000 8000 9000 10000 \
  --screen-top-k 2 --flow-readout integrated --num-flow-steps 10 \
  --return-sample-aggregation entropic \
  --num-action-flow-samples 16 \
  --policy-value-beta "$selected_beta" \
  --screen-episodes 10 --screen-seed-start 205000 \
  --validation-episodes 50 --validation-seed-start 206000 \
  --confirmation-episodes 200 --confirmation-seed-start 207000 \
  --bootstrap-replicates 20000 --bootstrap-seed 207200 \
  > "$stage131/gate.log" 2>&1
touch "$stage131/complete"

task_summary="$task_output/summary.json"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage132/skipped_task_fail"
  touch "$stage132/complete"
  current_control=$stage133
  printf "%s\n" "$BASHPID" > "$stage133/controller.pid"
  .venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
    --base-summary "$base_summary" \
    --b-candidate "evor_flowtd=$task_summary" \
    --output "$final_output" \
    > "$stage133/summary.log" 2>&1
  touch "$stage133/next_gate_required"
  touch "$stage133/complete"
  exit 0
fi

# Stage 132: task superiority is not used as a proxy for value authenticity.
# The same selected checkpoints must predict H=1 same-state interventions and
# beat BC/path/action-nearness proxies on fresh simulator branches.
current_control=$stage132
printf "%s\n" "$BASHPID" > "$stage132/controller.pid"
step1=$(jq -r .selected_steps.seed1 "$task_summary")
step2=$(jq -r .selected_steps.seed2 "$task_summary")
step3=$(jq -r .selected_steps.seed3 "$task_summary")
MUJOCO_GL=egl .venv/bin/python scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$evor_seed1,$evor_seed1/snapshots/${step1}_snapshot.pkl" \
  --checkpoint "seed2=$evor_seed2,$evor_seed2/snapshots/${step2}_snapshot.pkl" \
  --checkpoint "seed3=$evor_seed3,$evor_seed3/snapshots/${step3}_snapshot.pkl" \
  --gpu-id 1 --gpu-id 5 \
  --output-dir "$causal_output" \
  --eval-seed-start 208000 --num-eval-seeds 32 \
  --anchor-steps 30,75,120 --force-level 1 \
  --intervention-mode sibling_horizon --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout integrated --num-flow-steps 10 \
  --return-sample-aggregation entropic \
  --num-action-flow-samples 16 \
  --policy-value-beta "$selected_beta" \
  --bootstrap-replicates 20000 --bootstrap-seed 208100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage132/gate.log" 2>&1
touch "$stage132/complete"

current_control=$stage133
printf "%s\n" "$BASHPID" > "$stage133/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --b-candidate "evor_flowtd=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage133/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" != pass; then
  touch "$stage133/next_gate_required"
fi
touch "$stage133/complete"
