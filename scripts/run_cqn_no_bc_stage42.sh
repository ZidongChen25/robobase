#!/usr/bin/env bash
# Stage 42: exact raw-10k reward-Q offline branch, fixed expert replay online.
# This runner is capped at raw 30k and cannot open held-out or launch full.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-3}"
STAGE38="${2:-exp_local/cqn_no_bc/stage38_offline_dense_b256_gpu5_20260801083024}"
STAGE41="${3:-$(tr -d '\n' < exp_local/cqn_no_bc/stage41_latest.txt)}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage42_fixed_expert_replay_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=30000
STEPS="12500,15000,17500,20000,22500,25000,27500,30000"

GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
case "${GPU}" in
  0) DEFAULT_EGL_DEVICE=0 ;; 2) DEFAULT_EGL_DEVICE=1 ;;
  3) DEFAULT_EGL_DEVICE=2 ;; 4) DEFAULT_EGL_DEVICE=3 ;;
  5) DEFAULT_EGL_DEVICE=4 ;; *) exit 2 ;;
esac
EGL_DEVICE="${STAGE42_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
[[ "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]+$ ]]
test -e "${STAGE38}/extension_complete"
test -s "${STAGE41}/stage41_raw50_sentinel_summary.json"
rg -q '"next_decision": "stop_stage41_scaling_after_raw50k"' \
  "${STAGE41}/stage41_raw50_sentinel_summary.json"

run_dir () { printf '%s/seed%s/offline_dense_online_positive_fixed_expert' "${BASE}" "$1"; }
source_dir () { printf '%s/dense_seed%s/offline_then_online' "${STAGE38}" "$1"; }

mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage42_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"
printf '%s\n' "${GPU_UUID}" > "${BASE}/gpu_uuid.txt"
printf '%s\n' "${EGL_DEVICE}" > "${BASE}/egl_device.txt"
printf '%s\n' "${STAGE38}" > "${BASE}/stage38_source.txt"
printf '%s\n' "${STAGE41}" > "${BASE}/stage41_source.txt"
printf '%s objective=fixed_expert_positive_return_q full_run=false heldout=sealed\n' \
  "$(date --iso-8601=seconds)" > "${BASE}/protocol_registered.txt"

for seed in 1 2; do
  source="$(source_dir "${seed}")"
  destination="$(run_dir "${seed}")"
  .venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
    --source-run "${source}" --destination-run "${destination}" \
    --snapshot-step "${OFFLINE_UPDATES}" \
    --manifest-name stage42_branch_manifest.json \
    > "${BASE}/seed${seed}_branch_prepare.log"
  mkdir -p "${BASE}/seed${seed}/phase_configs"
  # The shared run directory is reused for the online phase, so its current
  # .hydra/config.yaml no longer describes the raw-10k offline snapshot.
  # Preserve the phase-specific config captured by the Stage-38 runner.
  offline_phase_config="$(dirname "${source}")/phase_configs/offline.yaml"
  test -s "${offline_phase_config}"
  cp "${offline_phase_config}" \
    "${BASE}/seed${seed}/phase_configs/offline_dense_source.yaml"
done
touch "${BASE}/branches_prepared"

timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
  .venv/bin/python -c \
  "import jax; d=jax.devices(); assert d and d[0].platform == 'gpu', d" \
  > "${BASE}/cuda_probe.log" 2>&1
timeout 30s env CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_GL=egl \
  PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
  .venv/bin/python -c \
  "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><body><geom size=\"0.1\"/></body></worldbody></mujoco>'); r=mujoco.Renderer(m,84,84); r.render(); r.close()" \
  > "${BASE}/egl_probe.log" 2>&1
touch "${BASE}/device_probes_passed"

train_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python train_fast.py \
    launch="${LAUNCH}" env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${GLOBAL_LIMIT}" replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    log_eval_video=false save_snapshot=true snapshot_every_n=2500 \
    gpu_id=null xla_mem_fraction=0.45 wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_train.log" 2>&1
}

status=0
train_seed 1 & seed1_pid=$!
sleep 120
if ! kill -0 "${seed1_pid}" 2>/dev/null; then
  wait "${seed1_pid}" || status=$?; printf '%s\n' "${status}" > "${BASE}/training_failed"; exit "${status}"
fi
train_seed 2 & seed2_pid=$!
touch "${BASE}/training_pair_started"
wait "${seed1_pid}" || status=$?
seed2_status=0; wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/training_failed"; exit "${status}"
fi

for seed in 1 2; do
  run="$(run_dir "${seed}")"
  for step in 12500 15000 17500 20000 22500 25000 27500 30000; do
    test -s "${run}/snapshots/${step}_snapshot.pkl"
  done
  cp "${run}/.hydra/config.yaml" \
    "${BASE}/seed${seed}/phase_configs/online_fixed_expert.yaml"
  rg -q '^use_self_imitation: false$' "${run}/.hydra/config.yaml"
  rg -q '^  strict_allow_reward_only_success_replay: false$' \
    "${run}/.hydra/config.yaml"
  rg -q '^  bc_lambda: 0.0$' "${run}/.hydra/config.yaml"
done
touch "${BASE}/training_complete"

evaluate_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id -1 --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --only-steps "${STEPS}" \
    --csv-name val50_seeds400_stage42.csv \
    > "${BASE}/seed${seed}_val50.log" 2>&1
}

status=0
evaluate_seed 1 & eval1_pid=$!
evaluate_seed 2 & eval2_pid=$!
wait "${eval1_pid}" || status=$?
eval2_status=0; wait "${eval2_pid}" || eval2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${eval2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/validation_failed"; exit "${status}"
fi
touch "${BASE}/validation_complete"
.venv/bin/python scripts/summarize_cqn_no_bc_stage42.py \
  --stage-dir "${BASE}" --stage41-dir "${STAGE41}" \
  --stage38-dir "${STAGE38}" --output "${BASE}/stage42_summary.json" \
  > "${BASE}/stage42_summary.log" 2>&1
touch "${BASE}/complete"
