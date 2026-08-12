#!/usr/bin/env bash
set -euo pipefail

# Cross-task confirmation of the post-temporal-ensemble L2 randomization
# diagnostic.  This is intentionally evaluation-only: each arm reads the
# same validation-selected checkpoint and the same non-sealed episode seeds.

cd "$(dirname "$0")/.."

gpu_index=2
gpu_uuid="GPU-80b9cc0d-df5c-be12-e848-042d37578544"
eval_seed_start=600
eval_episodes=100
eval_envs=25
artifact_root="exp_local/cqn_l2_cross_task"

actual_uuid="$({ nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index"; } 2>/dev/null)"
if [[ "$actual_uuid" != "$gpu_uuid" ]]; then
  echo "[l2-cross-task] GPU mapping mismatch: index $gpu_index is $actual_uuid" >&2
  exit 1
fi

gpu_has_compute_process() {
  local apps
  if ! apps="$(nvidia-smi --query-compute-apps=gpu_uuid --format=csv,noheader 2>/dev/null)"; then
    return 0
  fi
  awk -v wanted="$gpu_uuid" '$0 == wanted { found=1 } END { exit !found }' <<<"$apps"
}

echo "[l2-cross-task] waiting for an evaluation-safe GPU2 (no compute process)"
while gpu_has_compute_process; do
  sleep 60
done

# Require two clean observations so a transient process exit does not look
# like a free card while another job is constructing its CUDA context.
sleep 20
if gpu_has_compute_process; then
  echo "[l2-cross-task] GPU2 was claimed during the safety window; resuming wait"
  while gpu_has_compute_process; do
    sleep 60
  done
  sleep 20
  if gpu_has_compute_process; then
    echo "[l2-cross-task] GPU2 remained contested; refusing to launch" >&2
    exit 1
  fi
fi

best_step() {
  local csv_path="$1"
  awk -F, '
    NR == 1 { next }
    !seen || $3 > best { best=$3; step=$1; seen=1 }
    END {
      if (!seen) exit 1
      print step
    }
  ' "$csv_path"
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
  if [[ -e "$view_dir/.hydra/config.yaml" && ! -L "$view_dir/.hydra/config.yaml" ]]; then
    echo "[l2-cross-task] refusing to replace non-symlink config: $view_dir" >&2
    return 1
  fi
  if [[ -e "$view_dir/eval_checkpoints/${step}_checkpoint.pkl" && ! -L "$view_dir/eval_checkpoints/${step}_checkpoint.pkl" ]]; then
    echo "[l2-cross-task] refusing to replace non-symlink checkpoint: $view_dir" >&2
    return 1
  fi
  ln -sfn "$source_cfg" "$view_dir/.hydra/config.yaml"
  ln -sfn "$source_checkpoint" "$view_dir/eval_checkpoints/${step}_checkpoint.pkl"
}

run_arm() {
  local source_dir="$1"
  local task="$2"
  local train_seed="$3"
  local step="$4"
  local arm="$5"
  local view_dir="$artifact_root/$task/seed${train_seed}/step${step}/$arm"
  local -a extra_args=()
  if [[ "$arm" == "iid_l2" ]]; then
    extra_args=(--post-ensemble-keep-levels 2)
  fi

  make_eval_view "$source_dir" "$step" "$view_dir"
  echo "[l2-cross-task] start task=$task train_seed=$train_seed step=$step arm=$arm"
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
  local train_seed latest_file source_dir step arm
  for train_seed in 1 2; do
    latest_file="exp_local/cqn_trunc_arms/official_${task}/seed${train_seed}_latest.txt"
    source_dir="$(<"$latest_file")"
    step="$(best_step "$source_dir/val50_seeds400.csv")"
    for arm in baseline iid_l2; do
      run_arm "$source_dir" "$task" "$train_seed" "$step" "$arm"
    done
  done
}

mkdir -p "$artifact_root"

run_task flip_cup &
flip_pid=$!
sleep 20
run_task sandwich_remove &
sandwich_pid=$!
sleep 20
run_task wall_cupboard_open &
wall_pid=$!

status=0
wait "$flip_pid" || status=1
wait "$sandwich_pid" || status=1
wait "$wall_pid" || status=1

if (( status != 0 )); then
  echo "[l2-cross-task] one or more task pipelines failed" >&2
  exit "$status"
fi

echo "[l2-cross-task] all matched evaluations complete"
