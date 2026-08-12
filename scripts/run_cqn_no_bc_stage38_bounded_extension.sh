#!/usr/bin/env bash
# Continue the already-qualified Stage-38 pair from raw 20k to raw 30k.
# This script has no full-run or held-out-evaluation path.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage38_latest.txt)}"
GPU="${2:-$(tr -d '\n' < "${BASE}/gpu.txt")}"
GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
if [[ ! "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]+$ ]]; then
  printf 'Could not resolve physical GPU %s to a UUID.\n' "${GPU}" >&2
  exit 2
fi
# On this six-GPU host the boot-VGA card has no NVIDIA EGL device, so the
# NVIDIA EGL list is [physical 0,2,3,4,5].  Physical GPU 5 is therefore EGL
# device 4.  Keep this explicit so MuJoCo rendering and CUDA both stay on the
# user's assigned card instead of falling through to GPU 1's Mesa node.
EGL_DEVICE="${STAGE38_EGL_DEVICE_ID:-4}"
LAUNCH="cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate"
OFFLINE_UPDATES=10000
EXTENSION_GLOBAL_LIMIT=30000
EGL_RETRY_SECONDS="${STAGE38_EGL_RETRY_SECONDS:-60}"

run_dir () {
  printf '%s/dense_seed%s/offline_then_online' "${BASE}" "$1"
}

test -s "${BASE}/stage38_summary.json"
test -e "${BASE}/matched_validation_complete"
if ! rg -q '"eligible_for_20k_online_extension": true' \
    "${BASE}/stage38_summary.json"; then
  printf 'Stage-38 matched gate did not authorize a bounded extension.\n' >&2
  exit 2
fi
if [[ -e "${BASE}/extension_complete" ]]; then
  printf 'Stage-38 bounded extension is already complete: %s\n' "${BASE}"
  exit 0
fi
for seed in 1 2; do
  test -s "$(run_dir "${seed}")/snapshots/20000_snapshot.pkl"
done

printf '%s\n' "${BASHPID}" > "${BASE}/extension_controller.pid"
printf '%s\n' "${GPU}" > "${BASE}/extension_gpu.txt"
printf '%s\n' "${GPU_UUID}" > "${BASE}/extension_gpu_uuid.txt"
printf '%s\n' "${EGL_DEVICE}" > "${BASE}/extension_egl_device.txt"
printf '%s physical_gpu=%s uuid=%s egl_device=%s controller=%s\n' \
  "$(date --iso-8601=seconds)" "${GPU}" "${GPU_UUID}" "${EGL_DEVICE}" \
  "${BASHPID}" >> "${BASE}/extension_device_history.log"
touch "${BASE}/extension_started"

probe_egl () {
  timeout 30s env \
    MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python -c \
    "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
    > "${BASE}/extension_egl_probe_latest.log" 2>&1
}

probe_cuda () {
  timeout 30s env \
    CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    .venv/bin/python -c \
    "import jax; d=jax.devices(); assert d and d[0].platform == 'gpu', d; print(d)" \
    > "${BASE}/extension_cuda_probe_latest.log" 2>&1
}

wait_for_device () {
  local label="$1"
  local attempt=0
  while true; do
    local egl_status=available
    local cuda_status=available
    probe_egl || egl_status=unavailable
    probe_cuda || cuda_status=unavailable
    if [[ "${egl_status}" == available && "${cuda_status}" == available ]]; then
      break
    fi
    attempt=$((attempt + 1))
    printf '%s label=%s attempt=%s egl=%s cuda=%s\n' \
      "$(date --iso-8601=seconds)" "${label}" "${attempt}" \
      "${egl_status}" "${cuda_status}" \
      >> "${BASE}/extension_egl_probe_history.log"
    touch "${BASE}/extension_egl_waiting"
    sleep "${EGL_RETRY_SECONDS}"
  done
  printf '%s label=%s attempt=%s egl=available cuda=available\n' \
    "$(date --iso-8601=seconds)" "${label}" "$((attempt + 1))" \
    >> "${BASE}/extension_egl_probe_history.log"
}

train_with_retry () {
  local seed="$1"
  local attempt=0
  local dir
  dir="$(run_dir "${seed}")"
  while true; do
    attempt=$((attempt + 1))
    local log="${BASE}/dense_seed${seed}/extension_train_gpu${GPU}_attempt${attempt}.log"
    local status=0
    if XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cuda \
      CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
      .venv/bin/python train_fast.py \
      launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
      batch_size=256 demo_batch_size=256 \
      num_pretrain_steps="${OFFLINE_UPDATES}" \
      num_train_frames="${EXTENSION_GLOBAL_LIMIT}" \
      replay.demo_only_updates=false \
      method.demo_behavior_force_probability=0.0 \
      eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
      snapshot_every_n=2500 gpu_id=null xla_mem_fraction=0.45 \
      wandb.use=false hydra.run.dir="${dir}" > "${log}" 2>&1; then
      return 0
    else
      status=$?
    fi
    if rg -q "Cannot initialize a EGL device display|CUDA_ERROR_NO_DEVICE|CUDA_ERROR_NOT_INITIALIZED|Unable to initialize backend 'cuda'|Backend 'cuda' is not" "${log}"; then
      printf '%s seed=%s attempt=%s status=egl_retry\n' \
        "$(date --iso-8601=seconds)" "${seed}" "${attempt}" \
        >> "${BASE}/extension_train_history.log"
      wait_for_device "extension_train_seed${seed}"
      continue
    fi
    printf '%s seed=%s attempt=%s status=fatal exit=%s\n' \
      "$(date --iso-8601=seconds)" "${seed}" "${attempt}" "${status}" \
      >> "${BASE}/extension_train_history.log"
    return "${status}"
  done
}

wait_for_device extension_training
touch "${BASE}/extension_device_available"
status=0
train_with_retry 1 &
seed1_pid=$!
printf '%s\n' "${seed1_pid}" > "${BASE}/dense_seed1/extension_train.pid"
sleep 120
train_with_retry 2 &
seed2_pid=$!
printf '%s\n' "${seed2_pid}" > "${BASE}/dense_seed2/extension_train.pid"
wait "${seed1_pid}" || status=$?
seed2_status=0; wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/extension_training_failed"
  exit "${status}"
fi
for seed in 1 2; do
  dir="$(run_dir "${seed}")"
  test -s "${dir}/snapshots/30000_snapshot.pkl"
  cp "${dir}/.hydra/config.yaml" \
    "${BASE}/dense_seed${seed}/phase_configs/online_to_raw30k.yaml"
done
touch "${BASE}/extension_training_complete"

evaluate_with_retry () {
  local seed="$1"
  local attempt=0
  local dir
  dir="$(run_dir "${seed}")"
  while true; do
    attempt=$((attempt + 1))
    local log="${BASE}/dense_seed${seed}/extension_val50_gpu${GPU}_attempt${attempt}.log"
    local status=0
    if XLA_PYTHON_CLIENT_PREALLOCATE=false \
      MUJOCO_GL=egl PYOPENGL_PLATFORM=egl JAX_PLATFORMS=cuda \
      CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
      .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "${dir}" --gpu-id -1 \
      --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
      --only-steps "22500,25000,27500,30000" \
      --csv-name val50_seeds400_extension.csv > "${log}" 2>&1; then
      return 0
    else
      status=$?
    fi
    if rg -q "Cannot initialize a EGL device display" "${log}"; then
      wait_for_device "extension_eval_seed${seed}"
      continue
    fi
    return "${status}"
  done
}

# Training contexts are gone before the paired fixed-policy sweep starts.
sleep 60
wait_for_device extension_validation
status=0
evaluate_with_retry 1 &
eval1_pid=$!
printf '%s\n' "${eval1_pid}" > "${BASE}/dense_seed1/extension_val50.pid"
evaluate_with_retry 2 &
eval2_pid=$!
printf '%s\n' "${eval2_pid}" > "${BASE}/dense_seed2/extension_val50.pid"
wait "${eval1_pid}" || status=$?
eval2_status=0; wait "${eval2_pid}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/extension_validation_failed"
  exit "${status}"
fi

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage38_extension \
  --base "${BASE}" --output "${BASE}/stage38_extension_summary.json" \
  > "${BASE}/stage38_extension_summary.log" 2>&1
touch "${BASE}/extension_validation_complete"
touch "${BASE}/extension_complete"
