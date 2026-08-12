#!/usr/bin/env bash
# Continue the qualified Stage-41 pair from raw 30k to raw 50k only.
# This runner cannot launch a full run or open held-out evaluation seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage41_latest.txt)}"
GPU="${2:-$(tr -d '\n' < "${BASE}/gpu.txt")}"
GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
case "${GPU}" in
  0) DEFAULT_EGL_DEVICE=0 ;; 2) DEFAULT_EGL_DEVICE=1 ;;
  3) DEFAULT_EGL_DEVICE=2 ;; 4) DEFAULT_EGL_DEVICE=3 ;;
  5) DEFAULT_EGL_DEVICE=4 ;; *) exit 2 ;;
esac
EGL_DEVICE="${STAGE41_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
LAUNCH="cqn_as_pixel_bigym_stage41_online_positive_dense_handoff_gate"
SENTINEL_STEPS="32500,35000,37500,40000,42500,45000,47500,50000"

run_dir () { printf '%s/seed%s/offline_dense_online_positive_dense' "${BASE}" "$1"; }

test -s "${BASE}/stage41_extension_summary.json"
if ! rg -q '"eligible_for_separately_designed_50k_sentinel": true' \
    "${BASE}/stage41_extension_summary.json"; then
  printf 'Stage-41 raw30 gate did not authorize raw50.\n' >&2; exit 2
fi
if [[ -e "${BASE}/raw50_complete" ]]; then exit 0; fi
for seed in 1 2; do test -s "$(run_dir "${seed}")/snapshots/30000_snapshot.pkl"; done

printf '%s\n' "${BASHPID}" > "${BASE}/raw50_controller.pid"
printf '%s objective=stage41_raw50_scaling full_run=false heldout=sealed gpu=%s uuid=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" "${GPU_UUID}" \
  > "${BASE}/raw50_protocol_registered.txt"

timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
  .venv/bin/python -c \
  "import jax; d=jax.devices(); assert d and d[0].platform == 'gpu', d" \
  > "${BASE}/raw50_cuda_probe.log" 2>&1
timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
  .venv/bin/python -c \
  "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
  > "${BASE}/raw50_egl_probe.log" 2>&1

train_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 num_pretrain_steps=10000 \
    num_train_frames=50000 replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    save_snapshot=true snapshot_every_n=2500 gpu_id=null xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_raw50_train.log" 2>&1
}

status=0
train_seed 1 & seed1_pid=$!
sleep 120
if ! kill -0 "${seed1_pid}" 2>/dev/null; then
  wait "${seed1_pid}" || status=$?; printf '%s\n' "${status}" > "${BASE}/raw50_training_failed"; exit "${status}"
fi
train_seed 2 & seed2_pid=$!
touch "${BASE}/raw50_training_pair_started"
wait "${seed1_pid}" || status=$?
seed2_status=0; wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/raw50_training_failed"; exit "${status}"
fi
for seed in 1 2; do
  run="$(run_dir "${seed}")"
  for step in 32500 35000 37500 40000 42500 45000 47500 50000; do
    test -s "${run}/snapshots/${step}_snapshot.pkl"
  done
done
touch "${BASE}/raw50_training_complete"

evaluate_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id -1 --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 \
    --only-steps "${SENTINEL_STEPS}" \
    --csv-name val50_seeds400_stage41_raw50_sentinel.csv \
    > "${BASE}/seed${seed}_raw50_val50.log" 2>&1
}

status=0
evaluate_seed 1 & eval1_pid=$!
evaluate_seed 2 & eval2_pid=$!
wait "${eval1_pid}" || status=$?
eval2_status=0; wait "${eval2_pid}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/raw50_validation_failed"; exit "${status}"
fi
touch "${BASE}/raw50_validation_complete"
.venv/bin/python scripts/summarize_cqn_no_bc_stage41_raw50_sentinel.py \
  --stage-dir "${BASE}" \
  --output "${BASE}/stage41_raw50_sentinel_summary.json" \
  > "${BASE}/stage41_raw50_sentinel_summary.log" 2>&1
touch "${BASE}/raw50_complete"
