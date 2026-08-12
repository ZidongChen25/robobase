#!/usr/bin/env bash
# Evaluate only four fixed raw-111k endpoints after the four-seed gate passes.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage43_latest.txt)}"
GPU="${2:-3}"
GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
case "${GPU}" in
  0) DEFAULT_EGL_DEVICE=0 ;; 2) DEFAULT_EGL_DEVICE=1 ;;
  3) DEFAULT_EGL_DEVICE=2 ;; 4) DEFAULT_EGL_DEVICE=3 ;;
  5) DEFAULT_EGL_DEVICE=4 ;; *) exit 2 ;;
esac
EGL_DEVICE="${STAGE43_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
[[ "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]+$ ]]
test -e "${BASE}/seed34_complete"
test -s "${BASE}/stage43_full_summary.json"
rg -q '"eligible_for_sealed_heldout": true' "${BASE}/stage43_full_summary.json"
if [[ -e "${BASE}/heldout_complete" ]]; then exit 0; fi

run_dir () { printf '%s/seed%s/fixed_expert_101k_online' "${BASE}" "$1"; }
printf '%s\n' "${BASHPID}" > "${BASE}/heldout_controller.pid"
printf '%s comparison=four_fixed_raw111k_endpoints episodes=200 seeds=800-999 checkpoint_selection=false official_mean=0.64625\n' \
  "$(date --iso-8601=seconds)" > "${BASE}/heldout_protocol_registered.txt"

for seed in 1 2 3 4; do
  run="$(run_dir "${seed}")"
  test -s "${run}/snapshots/111000_snapshot.pkl"
  rg -q '^batch_size: 256$' "${run}/.hydra/config.yaml"
  rg -q '^demo_batch_size: 256$' "${run}/.hydra/config.yaml"
  rg -q '^use_self_imitation: false$' "${run}/.hydra/config.yaml"
  rg -q '^  bc_lambda: 0.0$' "${run}/.hydra/config.yaml"
done

timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
  .venv/bin/python -c \
  "import jax; d=jax.devices(); assert d and d[0].platform == 'gpu', d" \
  > "${BASE}/heldout_cuda_probe.log" 2>&1
timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
  .venv/bin/python -c \
  "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
  > "${BASE}/heldout_egl_probe.log" 2>&1

evaluate_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id -1 --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --only-steps 111000 \
    --csv-name heldout200_seeds800_stage43.csv \
    > "${BASE}/seed${seed}_heldout200.log" 2>&1
}

run_wave () {
  local first="$1" second="$2" status=0 second_status=0
  evaluate_seed "${first}" & first_pid=$!
  evaluate_seed "${second}" & second_pid=$!
  wait "${first_pid}" || status=$?
  wait "${second_pid}" || second_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${second_status}"; fi
  return "${status}"
}

status=0
run_wave 1 2 || status=$?
if [[ "${status}" -eq 0 ]]; then run_wave 3 4 || status=$?; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/heldout_failed"; exit "${status}"
fi
touch "${BASE}/heldout_evaluation_complete"
.venv/bin/python scripts/summarize_cqn_no_bc_stage43_heldout.py \
  --stage43-dir "${BASE}" --output "${BASE}/stage43_heldout_summary.json" \
  > "${BASE}/stage43_heldout_summary.log" 2>&1
touch "${BASE}/heldout_complete"
