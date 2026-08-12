#!/usr/bin/env bash
# Agent line Stage A21: offline-only sweep to fix the operating point of the
# offline quality gate (A20's prospectively validated predictor). For each
# seed: 10k demo-only updates of the sibling Stage-38 offline recipe, then
# the reward-only CPU geometry probe. No online step, no env evaluation.
# Usage: run_cqn_no_bc_agent_a21_offline_sweep.sh GPU MEMFRAC SEED [SEED...]
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="$1"; MEMFRAC="$2"; shift 2
SEEDS=("$@")
OFFLINE_UPDATES=10000
BASE="exp_local/cqn_no_bc/agent_a21_gpu${GPU}_$(date +%Y%m%d%H%M%S)"
mkdir -p "${BASE}"
printf '%s\n' "${BASE}" > "exp_local/cqn_no_bc/agent_a21_gpu${GPU}_latest.txt"

export JAX_PLATFORMS=cuda
until timeout 90 .venv/bin/python -c "import jax; d = jax.devices(); assert d and d[0].platform == 'gpu'" > "${BASE}/cuda_probe.log" 2>&1; do
  sleep 30
done

offline_and_probe () {
  local seed="$1" run="${BASE}/seed$1/run"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage38_offline_nobc_dense256_gate \
    env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${OFFLINE_UPDATES}" \
    replay.demo_only_updates=true \
    method.demo_behavior_force_probability=1.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=10000 gpu_id="${GPU}" xla_mem_fraction="${MEMFRAC}" \
    wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_offline.log" 2>&1
  test -s "${run}/snapshots/${OFFLINE_UPDATES}_snapshot.pkl"
  JAX_PLATFORMS=cpu .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${run}" \
    --snapshot "${run}/snapshots/${OFFLINE_UPDATES}_snapshot.pkl" \
    --output "${BASE}/seed${seed}_offline_geometry.json" \
    --gpu-id -1 --seed 0 --samples-per-group 32 --groups demo_success \
    --offline-episode-count 60 --critic target \
    >> "${BASE}/probe.log" 2>&1
  touch "${BASE}/seed${seed}_done"
}

for seed in "${SEEDS[@]}"; do
  offline_and_probe "${seed}" || printf '%s\n' "${seed}" >> "${BASE}/failed_seeds"
done
touch "${BASE}/complete"
