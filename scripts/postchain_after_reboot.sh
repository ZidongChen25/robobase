#!/usr/bin/env bash
# Rebuilt post-processing chains for the resumed U/E/offmask trainings:
# when each run's 101000 snapshot appears, run its sibling probe and the
# 200-ep final eval on that run's own GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

F="exp_local/cqn_stage162_eps_ablation"
O="exp_local/cqn_stage161_official_mask"

chain () {
  local RUN_DIR="$1" ARM="$2" BASE="$3" GPU="$4"
  until [ -f "${RUN_DIR}/snapshots/101000_snapshot.pkl" ]; do sleep 300; done
  echo "[post] ${ARM} training finished, probing"
  MUJOCO_GL=egl .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${RUN_DIR}" --snapshot "${RUN_DIR}/snapshots/101000_snapshot.pkl" \
    --output "${BASE}/sibling_${ARM}_seed1.json" --gpu-id "${GPU}" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
    --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${BASE}/sibling_${ARM}_seed1.log" 2>&1
  echo "[post] ${ARM} probe done"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${RUN_DIR}" --gpu-id "${GPU}" --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv \
    --skip-steps "$(ls ${RUN_DIR}/snapshots/ | grep -oE '^[0-9]+' | sort -n | head -n -1 | tr '\n' ',' | sed 's/,$//')" \
    > "${RUN_DIR}/ep200.log" 2>&1
  echo "[post] ${ARM} eval200 done"
}

chain "${F}/move_plate_uniform_seed1_gpu0_20260728141948" uniform "${F}" 0 &
P1=$!
chain "${F}/move_plate_edecay_seed1_gpu2_20260728141948" edecay "${F}" 2 &
P2=$!
chain "${O}/move_plate_offmask_seed1_gpu3_20260728084320" offmask "${O}" 3 &
P3=$!
wait "${P1}" "${P2}" "${P3}"
echo "[post] all chains complete"
