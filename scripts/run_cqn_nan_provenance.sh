#!/usr/bin/env bash
# Capture the first non-finite stage in the recurrent JAX CQN-AS update.
# Two runs share each of the user-approved GPU2/GPU4 cards, with staggered JITs.
set -euo pipefail
cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d%H%M%S)}"
ROOT="exp_local/cqn_nan_provenance/qc_${STAMP}"
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08
mkdir -p "${ROOT}"

run_one() {
  local seed="${1:?seed}"
  local gpu_uuid="${2:?gpu uuid}"
  local egl_id="${3:?egl id}"
  local run_dir="${ROOT}/seed${seed}"
  mkdir -p "${run_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${gpu_uuid}" \
    MUJOCO_EGL_DEVICE_ID="${egl_id}" \
    MUJOCO_GL=egl \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py \
      launch=cqn_as_pixel_bigym_stage163c_official_qc8 \
      env=bigym/move_plate \
      seed="${seed}" \
      xla_mem_fraction=0.45 \
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
}

{
  echo "[nan-provenance] ${STAMP} start $(date --iso-8601=seconds)"
  run_one 1 "${U2}" 2 &
  p1=$!
  run_one 2 "${U4}" 0 &
  p2=$!
  sleep 120
  run_one 3 "${U2}" 2 &
  p3=$!
  run_one 4 "${U4}" 0 &
  p4=$!
  set +e
  wait "${p1}"; s1=$?
  wait "${p2}"; s2=$?
  wait "${p3}"; s3=$?
  wait "${p4}"; s4=$?
  set -e
  echo "[nan-provenance] exits seed1=${s1} seed2=${s2} seed3=${s3} seed4=${s4}"
  echo "[nan-provenance] end $(date --iso-8601=seconds)"
} > "${ROOT}/controller.log" 2>&1
