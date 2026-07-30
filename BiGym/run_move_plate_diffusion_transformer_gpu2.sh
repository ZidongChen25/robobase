#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/exp_local/bigym_move_plate_dp_transformer_trainable_lang_ddpm_500e_gpu2_${STAMP}}"
LOG_DIR="${ROOT_DIR}/BiGym/logs"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/move_plate_dp_transformer_trainable_lang_ddpm_500e_gpu2_${STAMP}.log}"
PID_FILE="${ROOT_DIR}/BiGym/move_plate_dp_transformer_trainable_lang_gpu2.pid"

mkdir -p "${RUN_DIR}" "${LOG_DIR}"
ln -sfnT "${RUN_DIR}" "${ROOT_DIR}/BiGym/latest_move_plate_dp_transformer_run"
ln -sfnT "${RUN_DIR}" "${ROOT_DIR}/BiGym/latest_move_plate_dp_transformer_trainable_lang_run"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export JAX_CUDA_VISIBLE_DEVICES="${JAX_CUDA_VISIBLE_DEVICES:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT_DIR}"

rm -f "${PID_FILE}"
setsid -f bash -c '
  echo "$$" > "$1"
  exec uv run python train.py \
    launch=dp_pixel_bigym_move_plate_transformer \
    hydra.run.dir="$2" \
    >"$3" 2>&1
' _ "${PID_FILE}" "${RUN_DIR}" "${LOG_FILE}"

for _ in {1..50}; do
  if [[ -s "${PID_FILE}" ]]; then
    break
  fi
  sleep 0.1
done

PID="$(cat "${PID_FILE}")"

printf 'pid=%s\nrun_dir=%s\nlog_file=%s\n' "${PID}" "${RUN_DIR}" "${LOG_FILE}"
