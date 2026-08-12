#!/usr/bin/env bash
set -euo pipefail

# Independent four-train-seed verification of the MovePlate L2 result under
# the same protocol as the cross-task supplement: validation-selected best
# checkpoint, episode seeds 800--999, 200 episodes per arm.

cd "$(dirname "$0")/.."

eval_seed_start=800
eval_episodes=200
eval_envs=25
artifact_root="exp_local/cqn_l2_move_plate_ep200_800"

run_dir_for_seed() {
  find exp_local/cqn_official_truncated -maxdepth 1 -mindepth 1 -type d \
    -name "move_plate_trunc_seed${1}_*" | sort | tail -n 1
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
  local source_cfg source_snapshot
  source_cfg="$(readlink -f "$source_dir/.hydra/config.yaml")"
  source_snapshot="$(readlink -f "$source_dir/snapshots/${step}_snapshot.pkl")"
  [[ -f "$source_cfg" && -f "$source_snapshot" ]]
  mkdir -p "$view_dir/.hydra" "$view_dir/snapshots"
  [[ ! -e "$view_dir/.hydra/config.yaml" || -L "$view_dir/.hydra/config.yaml" ]]
  [[ ! -e "$view_dir/snapshots/${step}_snapshot.pkl" || -L "$view_dir/snapshots/${step}_snapshot.pkl" ]]
  ln -sfn "$source_cfg" "$view_dir/.hydra/config.yaml"
  ln -sfn "$source_snapshot" "$view_dir/snapshots/${step}_snapshot.pkl"
}

run_arm() {
  local train_seed="$1"
  local gpu_index="$2"
  local arm="$3"
  local source_dir step view_dir
  local -a extra_args=()
  source_dir="$(run_dir_for_seed "$train_seed")"
  step="$(best_step "$source_dir/val50_seeds400.csv")"
  view_dir="$artifact_root/seed${train_seed}/step${step}/$arm"
  if [[ "$arm" == "iid_l2" ]]; then
    extra_args=(--post-ensemble-keep-levels 2)
  fi
  make_eval_view "$source_dir" "$step" "$view_dir"
  echo "[l2-move-plate] start train_seed=$train_seed step=$step arm=$arm gpu=$gpu_index"
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

run_seed_group() {
  local gpu_index="$1"
  shift
  local train_seed arm
  for train_seed in "$@"; do
    for arm in baseline iid_l2; do
      run_arm "$train_seed" "$gpu_index" "$arm"
    done
  done
}

declare -A expected_uuid=(
  [0]="GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919"
  [5]="GPU-2f044e6a-9150-0e30-7d97-009bdd425b11"
)

for gpu_index in 0 5; do
  actual_uuid="$(nvidia-smi --query-gpu=uuid --format=csv,noheader -i "$gpu_index")"
  [[ "$actual_uuid" == "${expected_uuid[$gpu_index]}" ]]
  used_mib="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu_index")"
  util_pct="$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits -i "$gpu_index")"
  if (( used_mib >= 20000 || util_pct >= 80 )); then
    echo "[l2-move-plate] GPU $gpu_index exceeds eval occupancy limit" >&2
    exit 1
  fi
  while IFS=, read -r resident_pid resident_uuid; do
    resident_pid="${resident_pid// /}"
    resident_uuid="${resident_uuid// /}"
    [[ "$resident_uuid" == "$actual_uuid" ]] || continue
    [[ -r "/proc/$resident_pid/cmdline" ]]
    resident_cmd="$(tr '\0' ' ' < "/proc/$resident_pid/cmdline")"
    if [[ "$resident_cmd" != *"eval_cqn_as_snapshot_sweep.py"* && "$resident_cmd" != *"eval_only=true"* ]]; then
      echo "[l2-move-plate] GPU $gpu_index has non-eval PID $resident_pid; refusing" >&2
      exit 1
    fi
  done < <(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader)
done

mkdir -p "$artifact_root"
run_seed_group 0 1 3 &
gpu0_pid=$!
sleep 20
run_seed_group 5 2 4 &
gpu5_pid=$!

status=0
wait "$gpu0_pid" || status=1
wait "$gpu5_pid" || status=1
if (( status != 0 )); then
  echo "[l2-move-plate] one or more seed pipelines failed" >&2
  exit "$status"
fi
echo "[l2-move-plate] all matched evaluations complete"
