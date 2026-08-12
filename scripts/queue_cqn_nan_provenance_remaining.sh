#!/usr/bin/env bash
# Queue seeds 3/4 behind the already-running provenance seed1 on GPU2.
set -euo pipefail
cd "$(dirname "$0")/.."

ROOT="${1:?artifact root}"
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544

gpu2_process_count() {
  nvidia-smi \
    --query-compute-apps=gpu_uuid \
    --format=csv,noheader,nounits \
    | awk -v uuid="${U2}" '$1 == uuid {count++} END {print count + 0}'
}

wait_for_slot() {
  local consecutive_free=0
  while [ "${consecutive_free}" -lt 2 ]; do
    if [ "$(gpu2_process_count)" -lt 2 ]; then
      consecutive_free=$((consecutive_free + 1))
    else
      consecutive_free=0
    fi
    [ "${consecutive_free}" -ge 2 ] || sleep 15
  done
}

wait_until_registered() {
  local launch_pid="${1:?launch pid}"
  local before_count="${2:?pre-launch process count}"
  while kill -0 "${launch_pid}" 2>/dev/null; do
    if [ "$(gpu2_process_count)" -gt "${before_count}" ]; then
      return 0
    fi
    sleep 2
  done
  return 1
}

run_one() {
  local seed="${1:?seed}"
  local run_dir="${ROOT}/seed${seed}"
  mkdir -p "${run_dir}"
  env \
    CUDA_VISIBLE_DEVICES="${U2}" \
    MUJOCO_EGL_DEVICE_ID=2 \
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
  echo "[remaining] start $(date --iso-8601=seconds)"
  # Preserve the intended two-minute same-card JIT stagger from seed1.
  sleep 80
  wait_for_slot
  count_before_seed3=$(gpu2_process_count)
  run_one 3 &
  seed3_pid=$!
  echo "[remaining] seed3 pid=${seed3_pid} $(date --iso-8601=seconds)"
  # CUDA process registration trails fork/exec. Without this barrier the
  # second slot check can observe the old count and start a third trainer.
  wait_until_registered "${seed3_pid}" "${count_before_seed3}"
  # Never create a third training process on the card.
  wait_for_slot
  run_one 4 &
  seed4_pid=$!
  echo "[remaining] seed4 pid=${seed4_pid} $(date --iso-8601=seconds)"
  set +e
  wait "${seed3_pid}"; seed3_status=$?
  wait "${seed4_pid}"; seed4_status=$?
  set -e
  echo "[remaining] exits seed3=${seed3_status} seed4=${seed4_status}"
  echo "[remaining] end $(date --iso-8601=seconds)"
} >> "${ROOT}/controller.log" 2>&1
