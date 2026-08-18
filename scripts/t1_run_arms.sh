#!/usr/bin/env bash
# T1: train the value-target arms SEQUENTIALLY on GPU0 and probe each one.
# Usage: t1_run_arms.sh <STEPS> <SEED> <arm> [arm ...]
set -uo pipefail
cd "$(dirname "$0")/.."

STEPS="${1:?steps}"
SEED="${2:?seed}"
shift 2
ARMS=("$@")

SRC=exp_local/cqn_trunc_arms/official_sandwich_remove/seed2_20260805085944
FROZEN=exp_local/t1_td_mc/frozen_data
ROOT=exp_local/t1_td_mc/arms

E=(CUDA_VISIBLE_DEVICES=GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919
   MUJOCO_EGL_DEVICE_ID=4 MUJOCO_GL=egl
   XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
   ROBOBASE_HOST_MERGE=1)

mkdir -p "${ROOT}"
for ARM in "${ARMS[@]}"; do
  OUT="${ROOT}/${ARM}_s${SEED}"
  if [ -f "${OUT}/t1_done.json" ]; then
    echo "[t1] ${ARM} already trained, skipping"
  else
    echo "[t1] === train ${ARM} seed ${SEED} steps ${STEPS} $(date +%H:%M:%S)"
    free -g | sed -n 2p
    env "${E[@]}" .venv/bin/python scripts/t1_offline_train.py \
      --run-dir "${SRC}" --frozen-dir "${FROZEN}" --out-dir "${OUT}" \
      --arm "${ARM}" --steps "${STEPS}" --snapshot-every "${SNAP:-5000}" \
      --seed "${SEED}" > "${OUT}.train.log" 2>&1
    if [ ! -f "${OUT}/t1_done.json" ]; then
      echo "[t1] TRAIN FAILED ${ARM}; see ${OUT}.train.log"
      continue
    fi
  fi
  echo "[t1] === probe ${ARM} $(date +%H:%M:%S)"
  env "${E[@]}" .venv/bin/python scripts/t1_value_probes.py \
    --arm-dir "${OUT}" --frozen-dir "${FROZEN}" \
    --out-json "reports/t1_probes/${ARM}_s${SEED}.json" \
    > "${OUT}.probe.log" 2>&1 \
    || echo "[t1] PROBE FAILED ${ARM}; see ${OUT}.probe.log"
done
echo "[t1] all arms done $(date +%H:%M:%S)"
