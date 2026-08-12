#!/usr/bin/env bash
# Dose-response L1 sibling probe wave (primary endpoint of the bin-flip line).
# All on GPU3 (4 slots). Doses: 0 (63-dim baseline) / 0.03 / 0.06, seeds 1-2.
# Baseline uses params-only eval_checkpoints/100000_checkpoint.pkl (snapshots
# were cleaned); probe only needs params. Flip arms' slots WAIT for their
# 100000_snapshot.pkl to land, then start.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=exp_local/sibling_probe_powered
mkdir -p "${OUT}"
SEEDS=700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723
ANCH=20,40,60,80,100,120,140,160,180,200
U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb

probe() {  # label run_dir snapshot_path
  local label="$1" run="$2" snap="$3"
  CUDA_VISIBLE_DEVICES="${U3}" MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${run}" --snapshot "${snap}" \
    --output "${OUT}/${label}_L1.json" \
    --eval-seeds "${SEEDS}" --anchor-steps "${ANCH}" \
    --intervention-mode sibling_horizon --intervention-horizon 4 \
    --force-level 1 --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${OUT}/${label}_L1.log" 2>&1
  echo "[probe] ${label} L1 done $(date +%H:%M:%S)" >> "${OUT}/dose_wave.progress"
}

wait_snap() {  # run_dir -> echoes snapshot path once it exists
  local run="$1"
  until [ -f "${run}/snapshots/100000_snapshot.pkl" ]; do sleep 60; done
  echo "${run}/snapshots/100000_snapshot.pkl"
}

R() { cat "exp_local/cqn_trunc_arms/$1/seed$2_latest.txt"; }

BASE1=$(R official_basestate_move_plate 1)
BASE2=$(R official_basestate_move_plate 2)

# Slot A: baseline s1 -> flip03 s1 (waits for its 100k)
{ probe base63_s1 "${BASE1}" "${BASE1}/eval_checkpoints/100000_checkpoint.pkl"
  F1=$(R l1flip_move_plate 1); probe flip03_s1 "${F1}" "$(wait_snap "${F1}")"; } &
sleep 20
# Slot B: baseline s2 -> flip03 s2
{ probe base63_s2 "${BASE2}" "${BASE2}/eval_checkpoints/100000_checkpoint.pkl"
  F2=$(R l1flip_move_plate 2); probe flip03_s2 "${F2}" "$(wait_snap "${F2}")"; } &
sleep 20
# Slot C/D: flip06 s1/s2 as soon as their 100k lands
{ H1=$(R l1flip_hi_move_plate 1); probe flip06_s1 "${H1}" "$(wait_snap "${H1}")"; } &
sleep 20
{ H2=$(R l1flip_hi_move_plate 2); probe flip06_s2 "${H2}" "$(wait_snap "${H2}")"; } &
wait
echo "[probe] dose wave complete $(date +%H:%M:%S)" >> "${OUT}/dose_wave.progress"
