#!/usr/bin/env bash
# Agent line Stage A15: single-seed screening arm. Branches the A1
# seed-2 raw-10k de-scaled offline state and runs the online phase with
# one arm-specific override. Usage:
#   run_cqn_no_bc_agent_a15_screen.sh GPU ARM PHASE LAUNCH [EXTRA...]
# PHASE=train branches+trains; PHASE=eval runs the fixed sweep.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="$1"
ARM="$2"
PHASE="$3"
LAUNCH="$4"
shift 4
EXTRA=("$@")
SEED="${A15_SEED:-2}"
A1_BASE="$(tr -d '\n' < exp_local/cqn_no_bc/agent_a1_latest.txt)"
A5_BASE="$(tr -d '\n' < exp_local/cqn_no_bc/agent_a5_latest.txt)"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=20000
LATEST="exp_local/cqn_no_bc/agent_a15_${ARM}_s${SEED}_latest.txt"
if [[ "${PHASE}" == "train" ]]; then
  STAMP="$(date +%Y%m%d%H%M%S)"
  BASE="exp_local/cqn_no_bc/agent_a15_${ARM}_s${SEED}_gpu${GPU}_${STAMP}"
  mkdir -p "${BASE}"
  printf '%s\n' "${BASE}" > "${LATEST}"
else
  BASE="$(tr -d '\n' < "${LATEST}")"
fi

RUN="${BASE}/online"
if [[ "${PHASE}" == "train" ]]; then
.venv/bin/python scripts/prepare_cqn_no_bc_stage40_branch.py \
  --source-run "$(if [[ "${SEED}" -le 2 ]]; then printf '%s/descale_seed%s/offline' "${A1_BASE}" "${SEED}"; else printf '%s/descale_seed%s/full' "${A5_BASE}" "${SEED}"; fi)" \
  --destination-run "${RUN}" \
  --snapshot-step "${OFFLINE_UPDATES}" \
  > "${BASE}/branch.log" 2>&1
test -s "${RUN}/snapshots/${OFFLINE_UPDATES}_snapshot.pkl"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch="${LAUNCH}" env=bigym/move_plate seed="${SEED}" \
  batch_size=256 demo_batch_size=256 \
  num_pretrain_steps="${OFFLINE_UPDATES}" \
  num_train_frames="${GLOBAL_LIMIT}" \
  replay.demo_only_updates=false \
  method.demo_behavior_force_probability=0.0 \
  eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
  snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
  wandb.use=false hydra.run.dir="${RUN}" \
  "${EXTRA[@]}" \
  > "${BASE}/online.log" 2>&1
test -s "${RUN}/snapshots/${GLOBAL_LIMIT}_snapshot.pkl"
cp "${RUN}/.hydra/config.yaml" "${BASE}/online_config.yaml"
touch "${BASE}/online_complete"
exit 0
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN}" --gpu-id "${GPU}" \
  --num-eval-episodes 50 --eval-seed-start 400 \
  --num-eval-envs 25 \
  --only-steps "12500,15000,17500,20000" \
  --csv-name val50_seeds400_online.csv > "${BASE}/val50.log" 2>&1
touch "${BASE}/complete"
