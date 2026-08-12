#!/usr/bin/env bash
set -euo pipefail

# Does task success need the critic's L1/L2 choices, or only the geometry they
# provide?  Reuse the validation-selected baselines from the formal L2 audit
# and replace every post-ensemble action by the centre of its geometric L0
# cell.  For levels=3, bins=5 this keeps L0 and fixes the remaining 25 leaves
# to leaf 12 (the exact L0-cell centre).

cd "$(dirname "$0")/.."

artifact_root="exp_local/cqn_l0_only_center_ep200_800"
eval_seed_start=800
eval_episodes=200
eval_envs=25

declare -A expected_uuid=(
  [2]="GPU-80b9cc0d-df5c-be12-e848-042d37578544"
  [4]="GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08"
)

gpu_process_count() {
  local gpu_index="$1" uuid
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index")"
  nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader \
    | awk -v target="$uuid" '$0 == target { count++ } END { print count + 0 }'
}

validate_residents_are_eval_only() {
  local gpu_index="$1" uuid resident_pid resident_uuid resident_cmd
  uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index")"
  [[ "$uuid" == "${expected_uuid[$gpu_index]}" ]]
  while IFS=, read -r resident_uuid resident_pid; do
    resident_uuid="${resident_uuid// /}"
    resident_pid="${resident_pid// /}"
    [[ "$resident_uuid" == "$uuid" ]] || continue
    [[ -r "/proc/$resident_pid/cmdline" ]]
    resident_cmd="$(tr '\0' ' ' < "/proc/$resident_pid/cmdline")"
    if [[ "$resident_cmd" != *"eval_cqn"* \
          && "$resident_cmd" != *"eval_only=true"* ]]; then
      echo "[l0-center] GPU $gpu_index has non-eval PID $resident_pid; refusing" >&2
      exit 1
    fi
  done < <(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader)
}

wait_for_eval_slot() {
  local gpu_index="$1"
  while (( $(gpu_process_count "$gpu_index") >= 3 )); do
    sleep 10
  done
}

make_view() {
  local family="$1" task="$2" train_seed="$3" step="$4" view_dir="$5"
  local baseline source_cfg source_checkpoint checkpoint_dir checkpoint_name
  if [[ "$family" == "cross" ]]; then
    baseline="exp_local/cqn_l2_cross_task_ep200_800/$task/seed${train_seed}/step${step}/baseline"
    checkpoint_dir="eval_checkpoints"
    checkpoint_name="${step}_checkpoint.pkl"
  else
    baseline="exp_local/cqn_l2_move_plate_ep200_800/seed${train_seed}/step${step}/baseline"
    checkpoint_dir="snapshots"
    checkpoint_name="${step}_snapshot.pkl"
  fi
  source_cfg="$(readlink -f "$baseline/.hydra/config.yaml")"
  source_checkpoint="$(readlink -f "$baseline/$checkpoint_dir/$checkpoint_name")"
  [[ -f "$source_cfg" && -f "$source_checkpoint" ]]
  mkdir -p "$view_dir/.hydra" "$view_dir/$checkpoint_dir"
  ln -sfn "$source_cfg" "$view_dir/.hydra/config.yaml"
  ln -sfn "$source_checkpoint" "$view_dir/$checkpoint_dir/$checkpoint_name"
}

run_arm() {
  local gpu_index="$1" family="$2" task="$3" train_seed="$4" step="$5"
  local view_dir
  view_dir="$artifact_root/$task/seed${train_seed}/step${step}/l0_center"
  if [[ -s "$view_dir/result.csv" ]]; then
    echo "[l0-center] skip existing task=$task seed=$train_seed step=$step"
    return
  fi
  make_view "$family" "$task" "$train_seed" "$step" "$view_dir"
  wait_for_eval_slot "$gpu_index"
  echo "[l0-center] start task=$task seed=$train_seed step=$step gpu=$gpu_index"
  MUJOCO_GL=egl \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "$view_dir" \
      --gpu-id "$gpu_index" \
      --only-steps "$step" \
      --post-ensemble-keep-levels 1 \
      --post-ensemble-fixed-leaf 12 \
      --num-eval-episodes "$eval_episodes" \
      --eval-seed-start "$eval_seed_start" \
      --num-eval-envs "$eval_envs" \
      --csv-name result.csv > "$view_dir/eval.log" 2>&1
  tail -n 1 "$view_dir/result.csv"
  echo "[l0-center] done task=$task seed=$train_seed step=$step gpu=$gpu_index"
}

worker() {
  local gpu_index="$1"
  shift
  local spec family task train_seed step
  for spec in "$@"; do
    IFS='|' read -r family task train_seed step <<< "$spec"
    run_arm "$gpu_index" "$family" "$task" "$train_seed" "$step"
  done
}

for gpu_index in 2 4; do
  validate_residents_are_eval_only "$gpu_index"
done
mkdir -p "$artifact_root"

queue_a=(
  "cross|flip_cup|1|101000"
  "cross|sandwich_remove|1|85000"
  "move|move_plate|1|100000"
  "move|move_plate|3|95000"
)
queue_b=(
  "cross|flip_cup|2|101000"
  "cross|sandwich_remove|2|35000"
  "move|move_plate|2|100000"
)
queue_c=(
  "cross|wall_cupboard_open|1|5000"
  "cross|wall_cupboard_open|2|5000"
  "move|move_plate|4|45000"
)

worker 2 "${queue_a[@]}" & worker_a_pid=$!
sleep 20
worker 2 "${queue_b[@]}" & worker_b_pid=$!
sleep 20
worker 4 "${queue_c[@]}" & worker_c_pid=$!

status=0
wait "$worker_a_pid" || status=1
wait "$worker_b_pid" || status=1
wait "$worker_c_pid" || status=1
if (( status != 0 )); then
  echo "[l0-center] one or more worker queues failed" >&2
  exit "$status"
fi
echo "[l0-center] all 10 matched evaluations complete"
