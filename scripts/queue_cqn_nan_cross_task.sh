#!/usr/bin/env bash
# Run a task-only NaN provenance control on the first safe GPU2/GPU4 slot.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:?artifact root}"
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08

gpu_process_count() {
  local gpu_uuid="${1:?gpu uuid}"
  nvidia-smi \
    --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits \
    | awk -v uuid="${gpu_uuid}" '$1 == uuid {count++} END {print count + 0}'
}

choose_stable_slot() {
  local first_uuid=""
  local second_uuid=""
  while true; do
    if [ "$(gpu_process_count "${U2}")" -lt 2 ]; then
      first_uuid="${U2}"
    elif [ "$(gpu_process_count "${U4}")" -lt 2 ]; then
      first_uuid="${U4}"
    else
      first_uuid=""
    fi
    if [ -z "${first_uuid}" ]; then
      sleep 20
      continue
    fi
    sleep 15
    if [ "$(gpu_process_count "${first_uuid}")" -lt 2 ]; then
      second_uuid="${first_uuid}"
    else
      second_uuid=""
    fi
    if [ -n "${second_uuid}" ]; then
      printf '%s\n' "${second_uuid}"
      return 0
    fi
  done
}

mkdir -p "${ROOT}"
{
  echo "[cross-task] queued $(date --iso-8601=seconds)"
  gpu_uuid=$(choose_stable_slot)
  if [ "${gpu_uuid}" = "${U2}" ]; then
    egl_id=2
  else
    egl_id=0
  fi
  run_dir="${ROOT}/dishwasher_close_trays_seed2"
  mkdir -p "${run_dir}"
  echo "[cross-task] launch gpu=${gpu_uuid} egl=${egl_id} $(date --iso-8601=seconds)"
  set +e
  env \
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    MUJOCO_EGL_DEVICE_ID="${egl_id}" \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py \
      launch=cqn_as_pixel_bigym_stage163c_official_qc8 \
      env=bigym/dishwasher_close_trays \
      seed=2 \
      xla_mem_fraction=0.45 \
      env.truncate_demo_at_success=true \
      env.obs_std_floor_relative=0.0 \
      eval_every_steps=1000000 \
      num_eval_episodes=0 \
      num_eval_envs=0 \
      log_eval_video=false \
      save_snapshot=true \
      snapshot_every_n=5000 \
      save_csv=true \
      wandb.use=false \
      nonfinite_dump=true \
      nonfinite_dump_keep_batches=3 \
      nonfinite_dump_include_uint8=true \
      nonfinite_dump_save_state=true \
      hydra.run.dir="${run_dir}" \
      > "${run_dir}/train_fast.log" 2>&1
  status=$?
  set -e
  echo "[cross-task] exit=${status} $(date --iso-8601=seconds)"
  exit "${status}"
} >> "${ROOT}/controller.log" 2>&1
