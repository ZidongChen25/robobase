#!/usr/bin/env bash
# Sibling probe (causal same-state action-ranking audit) on the truncated-era
# arms, at the 100k checkpoint. Same protocol as the historical numbers:
# 12 probe seeds (700-711) x 3 anchors (30/75/120), sibling_horizon
# intervention, horizon 4, force-level 0, round-robin dims, 10k bootstrap.
#
# Historical reference (untruncated era, 101k):
#   official 0.567   imitation-prior 0.58   combined 0.644   QC 0.467   mask 0.616
#
# Pinned by UUID with an explicit EGL device (the script's --gpu-id would set
# CUDA and EGL to the same number, which do not match on this box).
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=exp_local/sibling_probe_trunc
mkdir -p "${OUT}"

probe() {  # $1 label  $2 run_dir  $3 uuid  $4 egl
  local label="$1" run="$2" uuid="$3" egl="$4"
  CUDA_VISIBLE_DEVICES="${uuid}" MUJOCO_EGL_DEVICE_ID="${egl}" MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${run}" --snapshot "${run}/snapshots/100000_snapshot.pkl" \
    --output "${OUT}/${label}.json" \
    --eval-seeds 700,701,702,703,704,705,706,707,708,709,710,711 \
    --anchor-steps 30,75,120 --intervention-mode sibling_horizon \
    --intervention-horizon 4 --force-level 0 --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${OUT}/${label}.log" 2>&1
  echo "[probe] ${label} done $(date +%H:%M:%S)"
}

U1=GPU-ce804993-c33e-3d10-5676-5bae093a7d96
U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08
T() { cat exp_local/cqn_official_truncated/seed$1_latest.txt; }
C() { cat exp_local/cqn_trunc_arms/combined/seed$1_latest.txt; }

{ probe trunc_s1    "$(T 1)" "$U1" 5; probe trunc_s2 "$(T 2)" "$U1" 5; probe combined_s1 "$(C 1)" "$U1" 5; } &
sleep 40
{ probe trunc_s3    "$(T 3)" "$U3" 3; probe trunc_s4 "$(T 4)" "$U3" 3; probe combined_s2 "$(C 2)" "$U3" 3; } &
sleep 40
{ probe combined_s3 "$(C 3)" "$U4" 0; probe combined_s4 "$(C 4)" "$U4" 0; } &
wait
echo "[probe] wave complete $(date +%H:%M:%S)"
