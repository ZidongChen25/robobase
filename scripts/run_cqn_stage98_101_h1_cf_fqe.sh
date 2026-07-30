#!/usr/bin/env bash
set -euo pipefail

# Route-A fallback after the predeclared online-TD arms: fit direct scalar-Q
# from one-step same-state sibling returns under a frozen independent BC
# continuation.  A within-state label shuffle is the state-only shortcut
# control.  Only a discovery pass unlocks two additional training seeds and a
# fresh closed-loop task/causal confirmation.

cd /home/zc1525/robobase_jaxflat

upstream_master=exp_local/cqn_flow_high_utd/stage94_97_bc_policy_final_master
base_summary=exp_local/cqn_flow_high_utd/stage97_bc_policy_final_summary_20260724.json

stage98_control=exp_local/cqn_flow_high_utd/stage98_h1_cf_fqe_discovery_controller
stage99_control=exp_local/cqn_flow_high_utd/stage99_h1_cf_fqe_replication_controller
stage100_control=exp_local/cqn_flow_high_utd/stage100_h1_cf_fqe_task_controller
stage101_control=exp_local/cqn_flow_high_utd/stage101_h1_cf_fqe_causal_controller
stage101_summary_control=exp_local/cqn_flow_high_utd/stage101_h1_cf_fqe_summary_controller

root=exp_local/cqn_flow_high_utd/stage98_101_h1_cf_fqe_20260724
task_output=exp_local/cqn_flow_high_utd/stage100_h1_cf_fqe_task_seed166000_20260724
task_work=exp_local/cqn_flow_high_utd/stage100_h1_cf_fqe_task_work_seed166000_20260724
causal_output=exp_local/cqn_flow_high_utd/stage101_h1_cf_fqe_causal_seed169000_20260724
final_output=exp_local/cqn_flow_high_utd/stage101_h1_cf_fqe_autoresearch_summary_20260724.json

mkdir -p \
  "$stage98_control" \
  "$stage99_control" \
  "$stage100_control" \
  "$stage101_control" \
  "$stage101_summary_control" \
  "$root"
printf "%s\n" "$BASHPID" > "$stage98_control/controller.pid"
current_control=$stage98_control
trap 'touch "$current_control/failed"' ERR

upstream_pid=$(sed -n '1p' "$upstream_master/controller.pid")
tail --pid="$upstream_pid" -f /dev/null
test -f "$upstream_master/complete"
test -f "$base_summary"
test "$(jq -r .status "$base_summary")" = ok

if test "$(jq -r .route_a.overall_gate "$base_summary")" = pass; then
  for control in \
    "$stage98_control" \
    "$stage99_control" \
    "$stage100_control" \
    "$stage101_control" \
    "$stage101_summary_control"; do
    touch "$control/skipped_route_a_already_passed"
    touch "$control/complete"
  done
  exit 0
fi

# Prefer a task-qualified parent.  If all online-TD arms failed task, choose
# the least-bad validation-selected parent so causal fine-tuning still gets a
# fair opportunity to repair its critic.
selected_label=$(
  jq -r '
    .route_a.candidates
    | map(select(.task.artifact != null))
    | max_by([
        (if .task.gate == "pass" then 1 else 0 end),
        (.task.crossed_bootstrap_ci95[0] // -999),
        (.task.mean_paired_delta // -999)
      ])
    | .label
  ' "$base_summary"
)
selected_task=$(
  jq -r --arg label "$selected_label" '
    .route_a.candidates[]
    | select(.label == $label)
    | .task.artifact
  ' "$base_summary"
)
test -n "$selected_label"
test -f "$selected_task"
printf "%s\n" "$selected_label" > "$stage98_control/selected_parent"
printf "%s\n" "$selected_task" > "$stage98_control/selected_task_artifact"

step1=$(jq -r '.selection.selected_steps.seed1 // .selected_steps.seed1' "$selected_task")
step2=$(jq -r '.selection.selected_steps.seed2 // .selected_steps.seed2' "$selected_task")
step3=$(jq -r '.selection.selected_steps.seed3 // .selected_steps.seed3' "$selected_task")
for value in "$step1" "$step2" "$step3"; do
  test "$value" != null
done
snapshot1=$(jq -r .sources.seed1.candidate_snapshot "$selected_task")
snapshot2=$(jq -r .sources.seed2.candidate_snapshot "$selected_task")
snapshot3=$(jq -r .sources.seed3.candidate_snapshot "$selected_task")
test -f "$snapshot1"
test -f "$snapshot2"
test -f "$snapshot3"
run1=$(dirname "$(dirname "$snapshot1")")
run2=$(dirname "$(dirname "$snapshot2")")
run3=$(dirname "$(dirname "$snapshot3")")
for run_dir in "$run1" "$run2" "$run3"; do
  test -f "$run_dir/.hydra/config.yaml"
done
printf "%s %s %s\n" "$step1" "$step2" "$step3" \
  > "$stage98_control/selected_steps"

csv_range() {
  seq -s, "$1" "$2"
}

action_dimensions=$(csv_range 0 14)

fit_one() {
  local label=$1
  local gpu=$2
  local run_dir=$3
  local snapshot=$4
  local train_start=$5
  local heldout_start=$6
  local fit_seed=$7
  local destination="$root/$label"
  local cache="$destination/branches_h1.npz"
  local positive="$destination/positive.json"
  local shuffled="$destination/shuffle.json"
  local positive_snapshot="$destination/positive_snapshot.pkl"
  local gate="$destination/gate.json"
  local train_seeds
  local heldout_seeds
  train_seeds=$(csv_range "$train_start" "$((train_start + 4))")
  heldout_seeds=$(csv_range "$heldout_start" "$((heldout_start + 4))")
  mkdir -p "$destination"

  common_args=(
    --run-dir "$run_dir"
    --snapshot "$snapshot"
    --dataset-cache "$cache"
    --gpu-id "$gpu"
    --train-seeds "$train_seeds"
    --heldout-seeds "$heldout_seeds"
    --anchor-steps 30,75,120
    --action-dimensions "$action_dimensions"
    --candidate-mode sibling_bins
    --force-level 1
    --intervention-horizon 1
    --score-level 1
    --max-continuation-steps 300
    --updates 2000
    --batch-size 32
    --sampling-mode full_batch
    --learning-rate 3e-4
    --temperature 0.05
    --delta-regression-weight 10
    --weight-decay 1e-5
    --bootstrap-replicates 5000
    --seed "$fit_seed"
  )

  if ! test -f "$positive" || \
    test "$(jq -r .status "$positive" 2>/dev/null || true)" != ok || \
    ! test -f "$positive_snapshot"; then
    MUJOCO_GL=egl .venv/bin/python \
      scripts/finetune_cqn_branch_oracle.py \
      "${common_args[@]}" \
      --output "$positive" \
      --finetuned-snapshot "$positive_snapshot" \
      --train-return-shuffle none \
      > "$destination/positive.log" 2>&1
  fi

  if ! test -f "$shuffled" || \
    test "$(jq -r .status "$shuffled" 2>/dev/null || true)" != ok; then
    MUJOCO_GL=egl .venv/bin/python \
      scripts/finetune_cqn_branch_oracle.py \
      "${common_args[@]}" \
      --output "$shuffled" \
      --train-return-shuffle within_state \
      > "$destination/shuffle.log" 2>&1
  fi

  .venv/bin/python scripts/summarize_cqn_h1_cf_fqe_gate.py \
    --positive "$positive" \
    --shuffle-control "$shuffled" \
    --output "$gate" \
    --min-informative-states 24 \
    --min-eval-seeds 4 \
    --bootstrap-replicates 20000 \
    --bootstrap-seed "$((fit_seed + 1000))" \
    > "$destination/gate.log" 2>&1
}

# Discovery uses one algorithm-training seed and a completely new 5+5
# simulator-seed split.  Historical all-dimension collection measured about
# 60--75 min for ten seeds; this is the initial event-driven ETA.
fit_one seed1 1 "$run1" "$snapshot1" 150000 151000 201
touch "$stage98_control/fit_complete"
test -f "$root/seed1/gate.json"
if test "$(jq -r .gate "$root/seed1/gate.json")" != pass; then
  touch "$stage98_control/discovery_gate_failed"
  touch "$stage98_control/complete"
  for control in \
    "$stage99_control" \
    "$stage100_control" \
    "$stage101_control" \
    "$stage101_summary_control"; do
    touch "$control/skipped_discovery_fail"
    touch "$control/complete"
  done
  exit 0
fi
touch "$stage98_control/discovery_gate_passed"
touch "$stage98_control/complete"

current_control=$stage99_control
printf "%s\n" "$BASHPID" > "$stage99_control/controller.pid"

# Leave GPU5 available for the independent B fallback when B is still open.
if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  fit_one seed2 1 "$run2" "$snapshot2" 152000 153000 202 &
  seed2_pid=$!
  fit_one seed3 5 "$run3" "$snapshot3" 154000 155000 203 &
  seed3_pid=$!
  printf "%s\n" "$seed2_pid" > "$stage99_control/seed2.pid"
  printf "%s\n" "$seed3_pid" > "$stage99_control/seed3.pid"
  wait "$seed2_pid"
  wait "$seed3_pid"
else
  fit_one seed2 1 "$run2" "$snapshot2" 152000 153000 202
  fit_one seed3 1 "$run3" "$snapshot3" 154000 155000 203
fi

passed=1
for label in seed2 seed3; do
  test -f "$root/$label/gate.json"
  if test "$(jq -r .gate "$root/$label/gate.json")" = pass; then
    passed=$((passed + 1))
  fi
done
printf "%s\n" "$passed" > "$stage99_control/passed_training_seeds"
touch "$stage99_control/complete"
if test "$passed" -lt 2; then
  touch "$stage99_control/replication_gate_failed"
  for control in \
    "$stage100_control" \
    "$stage101_control" \
    "$stage101_summary_control"; do
    touch "$control/skipped_replication_fail"
    touch "$control/complete"
  done
  exit 0
fi
touch "$stage99_control/replication_gate_passed"

materialize_run() {
  local label=$1
  local source_run=$2
  local destination="$root/${label}_run"
  mkdir -p "$destination/.hydra" "$destination/snapshots"
  cp -a "$source_run/.hydra/." "$destination/.hydra/"
  cp -a "$root/$label/positive_snapshot.pkl" \
    "$destination/snapshots/1_snapshot.pkl"
}
materialize_run seed1 "$run1"
materialize_run seed2 "$run2"
materialize_run seed3 "$run3"

current_control=$stage100_control
printf "%s\n" "$BASHPID" > "$stage100_control/controller.pid"
gpu_args=(--gpu-id 1)
if test "$(jq -r .route_b.overall_gate "$base_summary")" = pass; then
  gpu_args+=(--gpu-id 5)
fi
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_checkpoint_beta_selection_gate.py \
  --training-seed seed1=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed1_gpu1_20260724/snapshots/5000_snapshot.pkl,"$root/seed1_run" \
  --training-seed seed2=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed2_gpu1_20260724/snapshots/5000_snapshot.pkl,"$root/seed2_run" \
  --training-seed seed3=exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724,exp_local/cqn_value_fidelity_stage22/clean_cqn_as_repro_seed3_gpu1_20260724/snapshots/2500_snapshot.pkl,"$root/seed3_run" \
  "${gpu_args[@]}" \
  --output-dir "$task_output" \
  --work-root "$task_work" \
  --checkpoint-step 1 \
  --screen-top-k 1 \
  --candidate-readout auto \
  --beta 0.3 1 3 \
  --screen-beta 1 \
  --screen-episodes 10 \
  --screen-seed-start 166000 \
  --validation-episodes 50 \
  --validation-seed-start 167000 \
  --confirmation-episodes 200 \
  --confirmation-seed-start 168000 \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 168200 \
  --min-mean-delta 0 \
  --min-ci-lower 0 \
  > "$stage100_control/gate.log" 2>&1
touch "$stage100_control/complete"

current_control=$stage101_control
printf "%s\n" "$BASHPID" > "$stage101_control/controller.pid"
task_summary="$task_output/summary.json"
test -f "$task_summary"
if test "$(jq -r .gate "$task_summary")" != pass; then
  touch "$stage101_control/skipped_task_fail"
  touch "$stage101_control/complete"
  touch "$stage101_summary_control/skipped_task_fail"
  touch "$stage101_summary_control/complete"
  exit 0
fi
beta=$(jq -r .selection.selected_global_beta "$task_summary")
MUJOCO_GL=egl .venv/bin/python \
  scripts/run_cqn_flow_branch_multiseed_gate.py \
  --checkpoint "seed1=$root/seed1_run,$root/seed1_run/snapshots/1_snapshot.pkl" \
  --checkpoint "seed2=$root/seed2_run,$root/seed2_run/snapshots/1_snapshot.pkl" \
  --checkpoint "seed3=$root/seed3_run,$root/seed3_run/snapshots/1_snapshot.pkl" \
  "${gpu_args[@]}" \
  --output-dir "$causal_output" \
  --eval-seed-start 169000 \
  --num-eval-seeds 32 \
  --anchor-steps 30,75,120 \
  --force-level 1 \
  --intervention-mode sibling_horizon \
  --intervention-horizon 1 \
  --max-continuation-steps 300 \
  --flow-readout auto \
  --policy-value-beta "$beta" \
  --bootstrap-replicates 20000 \
  --bootstrap-seed 169100 \
  --min-informative-states 24 \
  --required-positive-training-seeds 2 \
  --require-anti-cheat-proxies \
  > "$stage101_control/gate.log" 2>&1
touch "$stage101_control/complete"

current_control=$stage101_summary_control
printf "%s\n" "$BASHPID" > "$stage101_summary_control/controller.pid"
.venv/bin/python scripts/summarize_cqn_autoresearch_routes.py \
  --base-summary "$base_summary" \
  --a-candidate \
  "direct_q_h1_cf_fqe=$task_summary,$causal_output/summary.json" \
  --output "$final_output" \
  > "$stage101_summary_control/summary.log" 2>&1
if test "$(jq -r .research_goal_gate "$final_output")" = pass; then
  touch "$stage101_summary_control/research_goal_pass"
else
  touch "$stage101_summary_control/next_gate_required"
fi
touch "$stage101_summary_control/complete"
