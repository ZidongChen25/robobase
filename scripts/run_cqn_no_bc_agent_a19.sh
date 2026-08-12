#!/usr/bin/env bash
# Agent line Stage A19: GPT Stage-42 recipe x de-saturation. One seed per
# card (solo run: avoids the co-residence CUDA race and shortens the
# clock). Phases: fresh offline 10k (full dense + scale) -> online to raw
# 30k (positive-only + frozen expert replay + scale) -> 8-point eval.
# Usage: run_cqn_no_bc_agent_a18.sh GPU SEED
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="$1"
SEED="$2"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=30000
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a19_s${SEED}_gpu${GPU}_${STAMP}"
RUN="${BASE}/run"
mkdir -p "${BASE}"
printf '%s\n' "${BASE}" > "exp_local/cqn_no_bc/agent_a19_s${SEED}_latest.txt"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_agent_a19_offline_gate \
  env=bigym/move_plate seed="${SEED}" \
  batch_size=256 demo_batch_size=256 \
  num_pretrain_steps="${OFFLINE_UPDATES}" \
  num_train_frames="${OFFLINE_UPDATES}" \
  replay.demo_only_updates=true \
  method.demo_behavior_force_probability=1.0 \
  eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
  snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
  wandb.use=false hydra.run.dir="${RUN}" \
  > "${BASE}/offline.log" 2>&1
test -s "${RUN}/snapshots/10000_snapshot.pkl"
cp "${RUN}/.hydra/config.yaml" "${BASE}/offline_config.yaml"
touch "${BASE}/offline_complete"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_nobc_agent_a19_online_gate \
  env=bigym/move_plate seed="${SEED}" \
  batch_size=256 demo_batch_size=256 \
  num_pretrain_steps="${OFFLINE_UPDATES}" \
  num_train_frames="${GLOBAL_LIMIT}" \
  replay.demo_only_updates=false \
  method.demo_behavior_force_probability=0.0 \
  eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
  snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
  wandb.use=false hydra.run.dir="${RUN}" \
  > "${BASE}/online.log" 2>&1
test -s "${RUN}/snapshots/${GLOBAL_LIMIT}_snapshot.pkl"
cp "${RUN}/.hydra/config.yaml" "${BASE}/online_config.yaml"
touch "${BASE}/online_complete"

XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN}" --gpu-id "${GPU}" \
  --num-eval-episodes 50 --eval-seed-start 400 \
  --num-eval-envs 25 \
  --only-steps "12500,15000,17500,20000,22500,25000,27500,30000" \
  --csv-name val50_seeds400_a19.csv > "${BASE}/val50.log" 2>&1
touch "${BASE}/complete"
