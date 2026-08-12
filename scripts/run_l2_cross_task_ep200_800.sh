#!/usr/bin/env bash
set -euo pipefail

# Matched supplement for the original reporting protocol: validation-selected
# checkpoints, episode seeds 800--999, 200 episodes per arm.  These seeds have
# already been opened by the original endpoint evaluations; they are not used
# to select checkpoints here.

cd "$(dirname "$0")/.."

eval_seed_start=800
eval_episodes=200
eval_envs=25
artifact_root="exp_local/cqn_l2_cross_task_ep200_800"

best_step() {
  awk -F, '
    NR == 1 { next }
    !seen || $3 > best { best=$3; step=$1; seen=1 }
    END {
      if (!seen) exit 1
      print step
    }
  ' "$1"
}

make_eval_view() {
  local source_dir="$1"
  local step="$2"
  local view_dir="$3"
  local source_cfg source_checkpoint
  source_cfg="$(readlink -f "$source_dir/.hydra/config.yaml")"
  source_checkpoint="$(readlink -f "$source_dir/eval_checkpoints/${step}_checkpoint.pkl")"
  [[ -f "$source_cfg" && -f "$source_checkpoint" ]]
  mkdir -p "$view_dir/.hydra" "$view_dir/eval_checkpoints"
  [[ ! -e "$view_dir/.hydra/config.yaml" || -L "$view_dir/.hydra/config.yaml" ]]
  [[ ! -e "$view_dir/eval_checkpoints/${step}_checkpoint.pkl" || -L "$view_dir/eval_checkpoints/${step}_checkpoint.pkl" ]]
  ln -sfn "$source_cfg" "$view_dir/.hydra/config.yaml"
  ln -sfn "$source_checkpoint" "$view_dir/eval_checkpoints/${step}_checkpoint.pkl"
}

run_arm() {
  local task="$1"
  local train_seed="$2"
  local gpu_index="$3"
  local arm="$4"
  local source_dir step view_dir
  local -a extra_args=()
  source_dir="$(<"exp_local/cqn_trunc_arms/official_${task}/seed${train_seed}_latest.txt")"
  step="$(best_step "$source_dir/val50_seeds400.csv")"
  view_dir="$artifact_root/$task/seed${train_seed}/step${step}/$arm"
  if [[ "$arm" == "iid_l2" ]]; then
    extra_args=(--post-ensemble-keep-levels 2)
  fi
  make_eval_view "$source_dir" "$step" "$view_dir"
  echo "[l2-ep200-800] start task=$task train_seed=$train_seed step=$step arm=$arm gpu=$gpu_index"
  MUJOCO_GL=egl \
  JAX_PLATFORMS=cuda \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "$view_dir" \
      --gpu-id "$gpu_index" \
      --only-steps "$step" \
      --num-eval-episodes "$eval_episodes" \
      --eval-seed-start "$eval_seed_start" \
      --num-eval-envs "$eval_envs" \
      --csv-name result.csv \
      "${extra_args[@]}" 2>&1 | tee "$view_dir/eval.log"
}

run_task() {
  local task="$1"
  local gpu_index="$2"
  local train_seed arm
  for train_seed in 1 2; do
    for arm in baseline iid_l2; do
      run_arm "$task" "$train_seed" "$gpu_index" "$arm"
    done
  done
}

declare -A expected_uuid=(
  [1]="GPU-ce804993-c33e-3d10-5676-5bae093a7d96"
  [2]="GPU-80b9cc0d-df5c-be12-e848-042d37578544"
  [3]="GPU-03f1431f-36c0-b258-6ca1-05007175e3eb"
)

# A card may already host evaluations, but never a training process.  Refuse
# if process provenance cannot be read or if any resident compute process is
# not recognizably eval-only.
for gpu_index in 1 2 3; do
  actual_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index")"
  [[ "$actual_uuid" == "${expected_uuid[$gpu_index]}" ]]
  while IFS=, read -r resident_pid resident_uuid; do
    resident_pid="${resident_pid// /}"
    resident_uuid="${resident_uuid// /}"
    [[ "$resident_uuid" == "$actual_uuid" ]] || continue
    [[ -r "/proc/$resident_pid/cmdline" ]]
    resident_cmd="$(tr '\0' ' ' < "/proc/$resident_pid/cmdline")"
    if [[ "$resident_cmd" != *"eval_cqn_as_snapshot_sweep.py"* && "$resident_cmd" != *"eval_only=true"* ]]; then
      echo "[l2-ep200-800] GPU $gpu_index has non-eval PID $resident_pid; refusing" >&2
      exit 1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_index")"
  util_pct="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_index")"
  if (( used_mib >= 20000 || util_pct >= 80 )); then
    echo "[l2-ep200-800] GPU $gpu_index exceeds eval occupancy limit" >&2
    exit 1
  fi
done

mkdir -p "$artifact_root"
run_task flip_cup 1 &
flip_pid=$!
sleep 20
run_task sandwich_remove 2 &
sandwich_pid=$!
sleep 20
run_task wall_cupboard_open 3 &
wall_pid=$!

status=0
wait "$flip_pid" || status=1
wait "$sandwich_pid" || status=1
wait "$wall_pid" || status=1
if (( status != 0 )); then
  echo "[l2-ep200-800] one or more task pipelines failed" >&2
  exit "$status"
fi
echo "[l2-ep200-800] all matched evaluations complete"
