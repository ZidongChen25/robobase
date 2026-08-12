#!/usr/bin/env bash
set -euo pipefail

# Follow-up to run_l2_cross_task_eval.sh.  Reuse the exact same
# validation-selected checkpoints and diagnostic episode seeds, replacing the
# iid L2 draw with the fixed center leaf.  One task is assigned to each idle
# GPU, and the two train seeds run sequentially on that card.

cd "$(dirname "$0")/.."

eval_seed_start=600
eval_episodes=100
eval_envs=25
artifact_root="exp_local/cqn_l2_cross_task"

gpu_has_compute_process() {
  local wanted_uuid="$1"
  local apps
  if ! apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)"; then
    return 0
  fi
  awk -v wanted="$wanted_uuid" '$0 == wanted { found=1 } END { exit !found }' <<<"$apps"
}

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

run_task() {
  local task="$1"
  local gpu_index="$2"
  local train_seed source_dir step view_dir
  for train_seed in 1 2; do
    source_dir="$(<"exp_local/cqn_trunc_arms/official_${task}/seed${train_seed}_latest.txt")"
    step="$(best_step "$source_dir/val50_seeds400.csv")"
    view_dir="$artifact_root/$task/seed${train_seed}/step${step}/fixed_center"
    make_eval_view "$source_dir" "$step" "$view_dir"
    echo "[l2-center] start task=$task train_seed=$train_seed step=$step gpu=$gpu_index"
    MUJOCO_GL=egl \
    JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
      .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
        --run-dir "$view_dir" \
        --gpu-id "$gpu_index" \
        --only-steps "$step" \
        --post-ensemble-fixed-leaf 2 \
        --num-eval-episodes "$eval_episodes" \
        --eval-seed-start "$eval_seed_start" \
        --num-eval-envs "$eval_envs" \
        --csv-name result.csv 2>&1 | tee "$view_dir/eval.log"
  done
}

declare -A expected_uuid=(
  [1]="GPU-ce804993-c33e-3d10-5676-5bae093a7d96"
  [2]="GPU-80b9cc0d-df5c-be12-e848-042d37578544"
  [3]="GPU-03f1431f-36c0-b258-6ca1-05007175e3eb"
)

for gpu_index in 1 2 3; do
  actual_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index")"
  if [[ "$actual_uuid" != "${expected_uuid[$gpu_index]}" ]]; then
    echo "[l2-center] GPU mapping mismatch at index $gpu_index" >&2
    exit 1
  fi
  if gpu_has_compute_process "$actual_uuid"; then
    echo "[l2-center] GPU $gpu_index has a compute process; refusing to co-locate" >&2
    exit 1
  fi
done

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
  echo "[l2-center] one or more task pipelines failed" >&2
  exit "$status"
fi
echo "[l2-center] all matched fixed-center evaluations complete"
