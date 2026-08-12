#!/usr/bin/env bash
# Stage 43A: continue qualified fixed-expert seeds 1/2 to 101k online steps.
# This runner cannot launch seeds 3/4 and cannot open held-out seeds.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-3}"
STAGE42="${2:-$(tr -d '\n' < exp_local/cqn_no_bc/stage42_latest.txt)}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage43_seed12_full_gpu${GPU}_${STAMP}"
LAUNCH="cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate"
OFFLINE_UPDATES=10000
SOURCE_RAW_STEP=50000
SOURCE_ONLINE_STEPS=40000
FULL_ONLINE_STEPS=101000
GLOBAL_LIMIT=111000
EVAL_STEPS="60000,70000,80000,90000,100000,110000,111000"

GPU_UUID="$(nvidia-smi -i "${GPU}" --query-gpu=uuid --format=csv,noheader | tr -d '[:space:]')"
case "${GPU}" in
  0) DEFAULT_EGL_DEVICE=0 ;; 2) DEFAULT_EGL_DEVICE=1 ;;
  3) DEFAULT_EGL_DEVICE=2 ;; 4) DEFAULT_EGL_DEVICE=3 ;;
  5) DEFAULT_EGL_DEVICE=4 ;; *) exit 2 ;;
esac
EGL_DEVICE="${STAGE43_EGL_DEVICE_ID:-${DEFAULT_EGL_DEVICE}}"
[[ "${GPU_UUID}" =~ ^GPU-[0-9a-fA-F-]+$ ]]
test -e "${STAGE42}/raw50_complete"
test -s "${STAGE42}/stage42_raw50_summary.json"
rg -q '"eligible_for_matched_raw101k_full_protocol": true' \
  "${STAGE42}/stage42_raw50_summary.json"

source_run () {
  printf '%s/seed%s/offline_dense_online_positive_fixed_expert' "${STAGE42}" "$1"
}
run_dir () { printf '%s/seed%s/fixed_expert_101k_online' "${BASE}" "$1"; }

mkdir -p "${BASE}"
printf '%s\n' "${BASHPID}" > "${BASE}/controller.pid"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage43_latest.txt
printf '%s\n' "${GPU}" > "${BASE}/gpu.txt"
printf '%s\n' "${GPU_UUID}" > "${BASE}/gpu_uuid.txt"
printf '%s\n' "${EGL_DEVICE}" > "${BASE}/egl_device.txt"
printf '%s\n' "${STAGE42}" > "${BASE}/stage42_source.txt"
sha256sum "${STAGE42}/stage42_raw50_summary.json" \
  > "${BASE}/stage42_summary.sha256"
printf '%s objective=fixed_expert_reward_only_101k_online seeds=1,2 offline_updates=10000 online_interactions=101000 raw_endpoint=111000 batch=256 demo_batch=256 heldout=sealed seed34_auto=false\n' \
  "$(date --iso-8601=seconds)" > "${BASE}/protocol_registered.txt"

for seed in 1 2; do
  source="$(source_run "${seed}")"
  destination="$(run_dir "${seed}")"
  test -s "${source}/snapshots/${SOURCE_RAW_STEP}_snapshot.pkl"
  rg -q '^batch_size: 256$' "${source}/.hydra/config.yaml"
  rg -q '^demo_batch_size: 256$' "${source}/.hydra/config.yaml"
  rg -q '^use_self_imitation: false$' "${source}/.hydra/config.yaml"
  rg -q '^  strict_demo_rl_only: true$' "${source}/.hydra/config.yaml"
  rg -q '^  strict_allow_reward_only_success_replay: false$' \
    "${source}/.hydra/config.yaml"
  rg -q '^  bc_lambda: 0.0$' "${source}/.hydra/config.yaml"
  rg -q '^  bc_margin: 0.0$' "${source}/.hydra/config.yaml"
  rg -q '^  demo_fosd: false$' "${source}/.hydra/config.yaml"
  .venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
    --source-run "${source}" --destination-run "${destination}" \
    --snapshot-step "${SOURCE_RAW_STEP}" \
    --expected-pretrain-step "${OFFLINE_UPDATES}" \
    --expected-main-loop-iterations "${SOURCE_ONLINE_STEPS}" \
    --manifest-name stage43_branch_manifest.json \
    > "${BASE}/seed${seed}_branch_prepare.log"
  mkdir -p "${BASE}/seed${seed}/phase_configs"
  cp "${source}/.hydra/config.yaml" \
    "${BASE}/seed${seed}/phase_configs/stage42_raw50_source.yaml"
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
  wait "${seed1_pid}" || status=$?
  printf '%s\n' "${status}" > "${BASE}/training_failed"
  exit "${status}"
fi
train_seed 2 & seed2_pid=$!
touch "${BASE}/training_pair_started"
printf '%s seed1_pid=%s seed2_pid=%s\n' \
  "$(date --iso-8601=seconds)" "${seed1_pid}" "${seed2_pid}" \
  > "${BASE}/training_start_verified.txt"
wait "${seed1_pid}" || status=$?
seed2_status=0; wait "${seed2_pid}" || seed2_status=$?
if [[ "${status}" -eq 0 ]]; then status="${seed2_status}"; fi
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/training_failed"; exit "${status}"
fi

for seed in 1 2; do
  run="$(run_dir "${seed}")"
  for step in 60000 70000 80000 90000 100000 110000 111000; do
    test -s "${run}/snapshots/${step}_snapshot.pkl"
  done
  cp "${run}/.hydra/config.yaml" \
    "${BASE}/seed${seed}/phase_configs/stage43_101k_online.yaml"
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
touch "${BASE}/training_complete"

evaluate_seed () {
  local seed="$1" run
  run="$(run_dir "${seed}")"
  env CUDA_VISIBLE_DEVICES="${GPU_UUID}" JAX_PLATFORMS=cuda \
    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
    MUJOCO_EGL_DEVICE_ID="${EGL_DEVICE}" \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id -1 --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --only-steps "${EVAL_STEPS}" \
    --csv-name val50_seeds400_stage43_seed12.csv \
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
.venv/bin/python scripts/summarize_cqn_no_bc_stage43_seed12.py \
  --stage43-dir "${BASE}" --stage42-dir "${STAGE42}" \
  --output "${BASE}/stage43_seed12_summary.json" \
  > "${BASE}/stage43_seed12_summary.log" 2>&1
touch "${BASE}/complete"
