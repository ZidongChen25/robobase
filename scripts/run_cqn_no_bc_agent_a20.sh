#!/usr/bin/env bash
# Agent line Stage A20: prospective test of the offline value-ordering
# predictor (and of the offline quality gate it implies). Runs the sibling
# project's Stage-38/42 recipe verbatim on FRESH seeds, records the
# reward-only offline geometry probe at the handoff point BEFORE any online
# step, then runs online to raw 30k and evaluates.
# Usage: run_cqn_no_bc_agent_a20.sh GPU SEED_A SEED_B
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="$1"; SA="$2"; SB="$3"
OFFLINE_UPDATES=10000
GLOBAL_LIMIT=30000
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/agent_a20_gpu${GPU}_${STAMP}"
mkdir -p "${BASE}"
printf '%s\n' "${BASE}" > "exp_local/cqn_no_bc/agent_a20_gpu${GPU}_latest.txt"

# Hard device gate (the sibling line's fix for the two failure modes that
# killed the first A20 launch): pin by UUID, force the CUDA platform so a
# transient cuInit failure can never silently fall back to CPU, and wait for
# both CUDA and EGL to be creatable before consuming any update.
# JAX_PLATFORMS=cuda is the load-bearing part: a transient cuInit failure
# must abort the run, never silently fall back to CPU (that is what wasted
# the first A20 launch). GPU selection stays with robobase's own gpu_id, and
# MUJOCO_EGL_DEVICE_ID is deliberately left unset — on this host EGL
# enumeration does not follow CUDA order and deriving it breaks rendering.
export JAX_PLATFORMS=cuda
until timeout 90 .venv/bin/python -c "import jax; d = jax.devices(); assert d and d[0].platform == 'gpu'" > "${BASE}/cuda_probe.log" 2>&1; do
  sleep 30
done
touch "${BASE}/device_probes_passed"

run_dir () { printf '%s/seed%s/run' "${BASE}" "$1"; }

offline () {
  local seed="$1" run; run="$(run_dir "${seed}")"
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
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_offline.log" 2>&1
  test -s "${run}/snapshots/${OFFLINE_UPDATES}_snapshot.pkl"
}

online () {
  local seed="$1" run; run="$(run_dir "${seed}")"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage42_fixed_expert_replay_gate \
    env=bigym/move_plate seed="${seed}" \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps="${OFFLINE_UPDATES}" \
    num_train_frames="${GLOBAL_LIMIT}" \
    replay.demo_only_updates=false \
    method.demo_behavior_force_probability=0.0 \
    eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    snapshot_every_n=2500 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run}" \
    > "${BASE}/seed${seed}_online.log" 2>&1
  test -s "${run}/snapshots/${GLOBAL_LIMIT}_snapshot.pkl"
}

evaluate () {
  local seed="$1" run; run="$(run_dir "${seed}")"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${run}" --gpu-id "${GPU}" \
    --num-eval-episodes 50 --eval-seed-start 400 --num-eval-envs 25 \
    --only-steps "12500,15000,17500,20000,22500,25000,27500,30000" \
    --csv-name val50_seeds400_a20.csv > "${BASE}/seed${seed}_val50.log" 2>&1
}

# Phase 1: offline pair (staggered).
offline "${SA}" & p1=$!
sleep 120
offline "${SB}" & p2=$!
st=0; wait "$p1" || st=$?; st2=0; wait "$p2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "$st/$st2" > "${BASE}/offline_failed"; exit 1; fi
touch "${BASE}/offline_complete"

# Phase 2: the prediction is recorded BEFORE any online step (CPU probe).
for seed in "${SA}" "${SB}"; do
  JAX_PLATFORMS=cpu .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "$(run_dir "${seed}")" \
    --snapshot "$(run_dir "${seed}")/snapshots/${OFFLINE_UPDATES}_snapshot.pkl" \
    --output "${BASE}/seed${seed}_offline_geometry.json" \
    --gpu-id -1 --seed 0 --samples-per-group 8 --groups demo_success \
    --offline-episode-count 60 --critic target \
    >> "${BASE}/probe.log" 2>&1
done
touch "${BASE}/prediction_recorded"

# Phase 3: online + eval.
online "${SA}" & p1=$!
sleep 120
online "${SB}" & p2=$!
st=0; wait "$p1" || st=$?; st2=0; wait "$p2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "$st/$st2" > "${BASE}/online_failed"; exit 1; fi
touch "${BASE}/online_complete"
evaluate "${SA}" & e1=$!
evaluate "${SB}" & e2=$!
st=0; wait "$e1" || st=$?; st2=0; wait "$e2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "$st/$st2" > "${BASE}/eval_failed"; exit 1; fi
touch "${BASE}/complete"
