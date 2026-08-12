#!/usr/bin/env bash
# Sandwich dual-level flip probe wave: dose=0 (official_sandwich_remove) vs
# swflip (L1=0.015/L2=0.05), both seeds, force-level 1 AND 2 (sandwich is the
# task where BOTH levels matter: iid-L2 -24/-32pp, L0-only 4.5%).
# GPU2: baseline probes (checkpoints exist now). GPU5: swflip probes, waiting
# for the eval chain to mint eval_checkpoints/100000_checkpoint.pkl (the sweep
# converts snapshots to params-only checkpoints and deletes the originals).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=exp_local/sibling_probe_powered
mkdir -p "${OUT}"
SEEDS=700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723
ANCH=20,40,60,80,100,120,140,160,180,200
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
U5=GPU-2f044e6a-9150-0e30-7d97-009bdd425b11

probe() {  # uuid egl label run_dir snapshot_path level
  local uuid="$1" egl="$2" label="$3" run="$4" snap="$5" lvl="$6"
  CUDA_VISIBLE_DEVICES="${uuid}" MUJOCO_EGL_DEVICE_ID="${egl}" MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${run}" --snapshot "${snap}" \
    --output "${OUT}/${label}_L${lvl}.json" \
    --eval-seeds "${SEEDS}" --anchor-steps "${ANCH}" \
    --intervention-mode sibling_horizon --intervention-horizon 4 \
    --force-level "${lvl}" --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${OUT}/${label}_L${lvl}.log" 2>&1
  echo "[probe] ${label} L${lvl} done $(date +%H:%M:%S)" >> "${OUT}/sandwich_wave.progress"
}

wait_100k() {  # run_dir -> echoes whichever 100k artifact appears first
  local run="$1"
  while true; do
    [ -f "${run}/eval_checkpoints/100000_checkpoint.pkl" ] && { echo "${run}/eval_checkpoints/100000_checkpoint.pkl"; return; }
    [ -f "${run}/snapshots/100000_snapshot.pkl" ] && { echo "${run}/snapshots/100000_snapshot.pkl"; return; }
    sleep 60
  done
}

R() { cat "exp_local/cqn_trunc_arms/$1/seed$2_latest.txt"; }
SB1=$(R official_sandwich_remove 1); SB2=$(R official_sandwich_remove 2)
SF1=$(R swflip_sandwich_remove 1);  SF2=$(R swflip_sandwich_remove 2)

# GPU2: baseline, 4 slots
probe "$U2" 2 swbase_s1 "$SB1" "${SB1}/eval_checkpoints/100000_checkpoint.pkl" 1 &
sleep 20
probe "$U2" 2 swbase_s2 "$SB2" "${SB2}/eval_checkpoints/100000_checkpoint.pkl" 1 &
sleep 20
probe "$U2" 2 swbase_s1 "$SB1" "${SB1}/eval_checkpoints/100000_checkpoint.pkl" 2 &
sleep 20
probe "$U2" 2 swbase_s2 "$SB2" "${SB2}/eval_checkpoints/100000_checkpoint.pkl" 2 &
# GPU5: swflip, 4 slots, wait for the 100k artifact
{ S=$(wait_100k "$SF1"); probe "$U5" 1 swflip_s1 "$SF1" "$S" 1 & sleep 20
  probe "$U5" 1 swflip_s1 "$SF1" "$S" 2 & wait; } &
{ S=$(wait_100k "$SF2"); sleep 40; probe "$U5" 1 swflip_s2 "$SF2" "$S" 1 & sleep 20
  probe "$U5" 1 swflip_s2 "$SF2" "$S" 2 & wait; } &
wait
echo "[probe] sandwich wave complete $(date +%H:%M:%S)" >> "${OUT}/sandwich_wave.progress"
