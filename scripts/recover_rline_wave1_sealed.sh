#!/usr/bin/env bash
# Recovery for the wave-1 sealed stage (cqn-rline.md): the runner shells died
# at the SKIP sed ($# expansion bug, fixed in run_cqn_trunc_arm.sh) after
# val50 completed. This replays exactly the runner's sealed tail for all four
# runs, then adds the user-directed seeds-450 confirmation pass (n=100 rule)
# for the nstep3 curves.
set -uo pipefail
cd "$(dirname "$0")/.."

U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb   # GPU3 -> EGL 3
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08   # GPU4 -> EGL 0

sealed() {
  local RUN_DIR=$1 GPU_UUID=$2 EGL_ID=$3
  local E=(CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}"
           MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda)
  local CHECKPOINT_DIR="${RUN_DIR}/snapshots"
  local CHECKPOINT_SUFFIX="_snapshot.pkl"
  local FINALIZE=()
  if compgen -G "${RUN_DIR}/eval_checkpoints/*_checkpoint.pkl" > /dev/null; then
    CHECKPOINT_DIR="${RUN_DIR}/eval_checkpoints"
    CHECKPOINT_SUFFIX="_checkpoint.pkl"
    FINALIZE=(--finalize-artifacts --selection-csv "${RUN_DIR}/val50_seeds400.csv")
  fi
  local SKIP
  SKIP="$(find "${CHECKPOINT_DIR}" -maxdepth 1 \( -type f -o -type l \) \
          | sed -n "s#^.*/\([0-9][0-9]*\)${CHECKPOINT_SUFFIX}\$#\1#p" | sort -n -u \
          | awk '$0 != 100000 && $0 != 101000' | paste -sd, -)"
  echo "[sealed] ${RUN_DIR} ($(date +%H:%M:%S))"
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --num-eval-episodes 200 --eval-seed-start 800 \
    --num-eval-envs 25 --csv-name ep200_seeds800.csv --skip-steps "${SKIP}" \
    "${FINALIZE[@]}" > "${RUN_DIR}/ep200.log" 2>&1 \
    || { touch "${RUN_DIR}/sealed_failed"; echo "[sealed] FAILED ${RUN_DIR}"; return 1; }
  touch "${RUN_DIR}/complete"
  grep '^\(100000\|101000\),' "${RUN_DIR}/ep200_seeds800.csv" | sed "s#^#[sealed ${RUN_DIR##*/}] #"
}

confirm50() {
  local RUN_DIR=$1 GPU_UUID=$2 EGL_ID=$3
  local E=(CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}"
           MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda)
  echo "[confirm50] ${RUN_DIR} ($(date +%H:%M:%S))"
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --num-eval-episodes 50 --eval-seed-start 450 \
    --num-eval-envs 25 --csv-name val50_seeds450.csv \
    > "${RUN_DIR}/val50_seeds450.log" 2>&1 \
    || echo "[confirm50] FAILED ${RUN_DIR}"
}

(
  sealed exp_local/cqn_trunc_arms/rfloor_move_plate/seed1_20260809rline1 "${U3}" 3
  sealed exp_local/cqn_trunc_arms/rfloor_move_plate/seed2_20260809rline1 "${U3}" 3
) &
P3=$!
(
  sealed exp_local/cqn_trunc_arms/nstep3_move_plate/seed1_20260809rline1 "${U4}" 0
  sealed exp_local/cqn_trunc_arms/nstep3_move_plate/seed2_20260809rline1 "${U4}" 0
  confirm50 exp_local/cqn_trunc_arms/nstep3_move_plate/seed1_20260809rline1 "${U4}" 0
  confirm50 exp_local/cqn_trunc_arms/nstep3_move_plate/seed2_20260809rline1 "${U4}" 0
) &
P4=$!
wait ${P3} ${P4}
echo "[recover] all done $(date +%H:%M:%S)"
