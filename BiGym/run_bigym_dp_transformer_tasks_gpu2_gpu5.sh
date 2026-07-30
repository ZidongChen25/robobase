#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
LOG_DIR="${ROOT_DIR}/BiGym/logs"
BATCH_SIZE="${BATCH_SIZE:-128}"
SKIP_CACHE="${SKIP_CACHE:-0}"
STATUS_FILE="${LOG_DIR}/bigym_dp_transformer_1000e_b${BATCH_SIZE}_${STAMP}.status"

mkdir -p "${LOG_DIR}"

export HYDRA_FULL_ERROR="${HYDRA_FULL_ERROR:-1}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

# episode_length values are already at each task's configured demo/control frequency.
CACHE_TASKS=(
  "flip_cup:2"
  "dishwasher_load_cups:5"
  "put_cups:2"
  "sandwich_remove:5"
  "dishwasher_open:2"
)

GPU2_TASKS=(
  "flip_cup:483"
  "put_cups:419"
  "dishwasher_open:345"
)

GPU5_TASKS=(
  "dishwasher_load_cups:704"
  "sandwich_remove:522"
)

log_status() {
  printf '[%s] %s\n' "$(date --iso-8601=seconds)" "$*" | tee -a "${STATUS_FILE}"
}

cache_one_task() {
  local task="$1"
  local egl_gpu="$2"
  local log_file="${LOG_DIR}/cache_${task}_pixel_256_3cam_${STAMP}.log"
  local summary_file="${LOG_DIR}/cache_${task}_pixel_256_3cam_${STAMP}.json"

  log_status "cache_start task=${task} egl_gpu=${egl_gpu} log=${log_file}"
  (
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${egl_gpu}"
    export JAX_CUDA_VISIBLE_DEVICES="${egl_gpu}"
    export MUJOCO_EGL_DEVICE_ID="${egl_gpu}"
    export OMP_NUM_THREADS="${CACHE_OMP_NUM_THREADS:-4}"
    export MKL_NUM_THREADS="${CACHE_MKL_NUM_THREADS:-4}"
    export OPENBLAS_NUM_THREADS="${CACHE_OPENBLAS_NUM_THREADS:-4}"
    exec uv run python scripts/cache_bigym_pixel_demos.py \
      --task "${task}" \
      --summary-file "${summary_file}" \
      >"${log_file}" 2>&1
  )
}

run_train_task() {
  local gpu="$1"
  local task="$2"
  local episode_length="$3"
  local run_dir="${ROOT_DIR}/exp_local/bigym_${task}_dp_transformer_trainable_lang_ddpm_1000e_b${BATCH_SIZE}_gpu${gpu}_${STAMP}"
  local log_file="${LOG_DIR}/${task}_dp_transformer_trainable_lang_ddpm_1000e_b${BATCH_SIZE}_gpu${gpu}_${STAMP}.log"

  mkdir -p "${run_dir}"
  ln -sfnT "${run_dir}" "${ROOT_DIR}/BiGym/latest_${task}_dp_transformer_run"
  log_status "train_start task=${task} gpu=${gpu} episode_length=${episode_length} run_dir=${run_dir} log=${log_file}"
  (
    cd "${ROOT_DIR}"
    export CUDA_VISIBLE_DEVICES="${gpu}"
    export JAX_CUDA_VISIBLE_DEVICES="${gpu}"
    export MUJOCO_EGL_DEVICE_ID="${gpu}"
    exec uv run python train.py \
      launch=dp_pixel_bigym_transformer_ddpm \
      env=bigym/"${task}" \
      gpu_id="${gpu}" \
      env.episode_length="${episode_length}" \
      batch_size="${BATCH_SIZE}" \
      hydra.run.dir="${run_dir}" \
      wandb.name="${task}_dp_transformer_ddpm_1000e_b${BATCH_SIZE}_gpu${gpu}" \
      >"${log_file}" 2>&1
  )
  log_status "train_done task=${task} gpu=${gpu}"
}

run_gpu_queue() {
  local gpu="$1"
  shift
  local item task episode_length
  for item in "$@"; do
    IFS=: read -r task episode_length <<<"${item}"
    run_train_task "${gpu}" "${task}" "${episode_length}"
  done
}

main() {
  log_status "pipeline_start stamp=${STAMP} batch_size=${BATCH_SIZE}"
  local cache_pids=()
  local item task egl_gpu
  if [[ "${SKIP_CACHE}" == "1" ]]; then
    log_status "cache_skipped"
  else
    for item in "${CACHE_TASKS[@]}"; do
      IFS=: read -r task egl_gpu <<<"${item}"
      cache_one_task "${task}" "${egl_gpu}" &
      cache_pids+=("$!")
    done

    local pid cache_rc=0
    for pid in "${cache_pids[@]}"; do
      if ! wait "${pid}"; then
        cache_rc=1
      fi
    done
    if [[ "${cache_rc}" -ne 0 ]]; then
      log_status "cache_failed"
      exit 1
    fi
    log_status "cache_all_done"
  fi

  run_gpu_queue 2 "${GPU2_TASKS[@]}" &
  local gpu2_pid="$!"
  run_gpu_queue 5 "${GPU5_TASKS[@]}" &
  local gpu5_pid="$!"

  echo "${gpu2_pid}" > "${ROOT_DIR}/BiGym/bigym_dp_transformer_gpu2_worker.pid"
  echo "${gpu5_pid}" > "${ROOT_DIR}/BiGym/bigym_dp_transformer_gpu5_worker.pid"
  log_status "train_workers_started gpu2_pid=${gpu2_pid} gpu5_pid=${gpu5_pid}"

  local train_rc=0
  if ! wait "${gpu2_pid}"; then
    log_status "train_worker_failed gpu=2"
    train_rc=1
  fi
  if ! wait "${gpu5_pid}"; then
    log_status "train_worker_failed gpu=5"
    train_rc=1
  fi
  if [[ "${train_rc}" -ne 0 ]]; then
    log_status "pipeline_failed"
    exit 1
  fi
  log_status "pipeline_done"
}

main "$@"
