#!/usr/bin/env bash
# Stage 43B: fresh seeds 3/4, full-dense offline then fixed-expert online.
# Requires Stage-43A pass and cannot open held-out seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage43_latest.txt)}"
GPU="${2:-3}"
OFFLINE_LAUNCH="cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate"
ONLINE_LAUNCH="cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=111000
EVAL_STEPS="12500,15000,17500,20000,22500,25000,27500,30000,32500,35000,37500,40000,42500,45000,47500,50000,60000,70000,80000,90000,100000,110000,111000"

GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
case "${GPU}" in
  0) DEFAULT_EGL_DEVICE=0 ;; 2) DEFAULT_EGL_DEVICE=1 ;;
  3) DEFAULT_EGL_DEVICE=2 ;; 4) DEFAULT_EGL_DEVICE=3 ;;
  5) DEFAULT_EGL_DEVICE=4 ;; *) exit 2 ;;
esac
EGL_DEVICE="${STAGE43_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
[[ "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]+$ ]]
test -e "${BASE}/complete"
test -s "${BASE}/stage43_seed12_summary.json"
rg -q '"eligible_for_fresh_seed34_full_runs": true' \
  "${BASE}/stage43_seed12_summary.json"
if [[ -e "${BASE}/seed34_complete" ]]; then exit 0; fi

run_dir () { printf '%s/seed%s/fixed_expert_101k_online' "${BASE}" "$1"; }
printf '%s\n' "${BASHPID}" > "${BASE}/seed34_controller.pid"
printf '%s objective=fresh_seed34_fixed_expert_reward_only_101k_online offline_updates=10000 online_interactions=101000 raw_endpoint=111000 batch=256 demo_batch=256 heldout=sealed\n' \
  "$(date --iso-8601=seconds)" > "${BASE}/seed34_protocol_registered.txt"

timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
  .venv/bin/python -c \
  "import jax; d=jax.devices(); assert d and d[0].platform == 'gpu', d" \
  > "${BASE}/seed34_cuda_probe.log" 2>&1
timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
  .venv/bin/python -c \
  "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
  > "${BASE}/seed34_egl_probe.log" 2>&1

train_seed () {
  local seed="$1" phase="$2" run launch demo_only force frames log
  run="$(run_dir "${seed}")"
  if [[ "${phase}" == offline ]]; then
    launch="${OFFLINE_LAUNCH}"; demo_only=true; force=1.0
    frames="${OFFLINE_UPDATES}"; log="${BASE}/seed${seed}_offline.log"
  else
    launch="${ONLINE_LAUNCH}"; demo_only=false; force=0.0
    frames="${GLOBAL_LIMIT}"; log="${BASE}/seed${seed}_online.log"
  fi
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python train_fast.py \
    launch="${launch}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${frames}" replay.demo_only_updates="${demo_only}" \
    use_self_imitation=false \
    method.strict_allow_reward_only_success_replay=false \
    method.demo_behavior_force_probability="${force}" \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    log_eval_video=false save_snapshot=true snapshot_every_n=2500 \
    gpu_id=null xla_mem_fraction=0.45 wandb.use=false hydra.run.dir="${run}" \
    > "${log}" 2>&1
}

run_pair () {
  local phase="$1" status=0 seed4_status=0
  train_seed 3 "${phase}" & seed3_pid=$!
  sleep 120
  if ! kill -0 "${seed3_pid}" 2>/dev/null; then
    wait "${seed3_pid}" || status=$?; return "${status}"
  fi
  train_seed 4 "${phase}" & seed4_pid=$!
  touch "${BASE}/seed34_${phase}_pair_started"
  wait "${seed3_pid}" || status=$?
  wait "${seed4_pid}" || seed4_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${seed4_status}"; fi
  return "${status}"
}

status=0
run_pair offline || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/seed34_offline_failed"; exit "${status}"
fi
for seed in 3 4; do
  run="$(run_dir "${seed}")"
  test -s "${run}/snapshots/10000_snapshot.pkl"
  mkdir -p "${BASE}/seed${seed}/phase_configs"
  cp "${run}/.hydra/config.yaml" "${BASE}/seed${seed}/phase_configs/offline.yaml"
  .venv/bin/python - "${run}/snapshots/10000_snapshot.pkl" <<'PY'
import pickle,sys
with open(sys.argv[1],"rb") as handle: payload=pickle.load(handle)
assert int(payload["_pretrain_step"]) == 10000
assert int(payload["_main_loop_iterations"]) == 0
assert int(payload["demo_replay_buffer"]["num_transitions"]) == 9253
PY
done
touch "${BASE}/seed34_offline_complete"

status=0
run_pair online || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/seed34_training_failed"; exit "${status}"
fi
for seed in 3 4; do
  run="$(run_dir "${seed}")"
  for step in 12500 15000 17500 20000 22500 25000 27500 30000 \
      32500 35000 37500 40000 42500 45000 47500 50000 \
      60000 70000 80000 90000 100000 110000 111000; do
    test -s "${run}/snapshots/${step}_snapshot.pkl"
  done
  cp "${run}/.hydra/config.yaml" "${BASE}/seed${seed}/phase_configs/online.yaml"
  rg -q '^batch_size: 256$' "${run}/.hydra/config.yaml"
  rg -q '^demo_batch_size: 256$' "${run}/.hydra/config.yaml"
  rg -q '^use_self_imitation: false$' "${run}/.hydra/config.yaml"
  rg -q '^  bc_lambda: 0.0$' "${run}/.hydra/config.yaml"
  .venv/bin/python - "${run}/train.csv" <<'PY'
import csv,sys
rows=list(csv.DictReader(open(sys.argv[1],newline="")))
sizes={int(float(row["demo_buffer_size"])) for row in rows}
assert sizes == {9253}, sizes
PY
done
touch "${BASE}/seed34_training_complete"

evaluate_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id -1 --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --only-steps "${EVAL_STEPS}" \
    --csv-name val50_seeds400_stage43_seed34.csv \
    > "${BASE}/seed${seed}_val50.log" 2>&1
}

status=0
evaluate_seed 3 & eval3_pid=$!
evaluate_seed 4 & eval4_pid=$!
wait "${eval3_pid}" || status=$?
eval4_status=0; wait "${eval4_pid}" || eval4_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval4_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/seed34_validation_failed"; exit "${status}"
fi
touch "${BASE}/seed34_validation_complete"
.venv/bin/python scripts/summarize_cqn_no_bc_stage43_full.py \
  --stage43-dir "${BASE}" --output "${BASE}/stage43_full_summary.json" \
  > "${BASE}/stage43_full_summary.log" 2>&1
touch "${BASE}/seed34_complete"
