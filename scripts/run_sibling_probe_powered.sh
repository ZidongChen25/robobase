#!/usr/bin/env bash
# Powered sibling probe. The first pass (12 seeds x 3 anchors = 36 states,
# 27 informative) had a per-run 95% CI of +-0.094 -- wider than the between-
# seed SD, so measurement noise dominated and the 0.586 vs 0.621 arm gap
# (t~0.9) was unresolvable.
#
# Fix 1 (sample size): 24 probe seeds x 10 anchors = 240 states, ~6.7x.
#                      Expected per-run SE 0.048 -> ~0.018.
# Fix 2 (scope):       force_level 0 was the coarsest of THREE C2F levels.
#                      Also sweep the deepest level (2) to see which level's
#                      ordering is the one that binds.
# Kept: intervention_horizon 4, bins 5 (all of them), round_robin dims --
# changing those would break comparability with the 0.58 imitation prior.
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=exp_local/sibling_probe_powered
mkdir -p "${OUT}"
SEEDS=700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723
ANCH=20,40,60,80,100,120,140,160,180,200

probe() {  # label run uuid egl level
  local label="$1" run="$2" uuid="$3" egl="$4" lvl="$5"
  CUDA_VISIBLE_DEVICES="${uuid}" MUJOCO_EGL_DEVICE_ID="${egl}" MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "${run}" --snapshot "${run}/snapshots/100000_snapshot.pkl" \
    --output "${OUT}/${label}_L${lvl}.json" \
    --eval-seeds "${SEEDS}" --anchor-steps "${ANCH}" \
    --intervention-mode sibling_horizon --intervention-horizon 4 \
    --force-level "${lvl}" --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "${OUT}/${label}_L${lvl}.log" 2>&1
  echo "[probe] ${label} L${lvl} done $(date +%H:%M:%S)"
}

U1=GPU-ce804993-c33e-3d10-5676-5bae093a7d96
U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08
U5=GPU-2f044e6a-9150-0e30-7d97-009bdd425b11
T() { cat exp_local/cqn_official_truncated/seed$1_latest.txt; }
C() { cat exp_local/cqn_trunc_arms/combined/seed$1_latest.txt; }

# Level 0 first on all four cards so the comparable, powered number lands
# early; the deepest-level sweep follows on the same card.
{ probe trunc_s1 "$(T 1)" "$U1" 5 0; probe trunc_s2 "$(T 2)" "$U1" 5 0
  probe trunc_s1 "$(T 1)" "$U1" 5 2; probe trunc_s2 "$(T 2)" "$U1" 5 2; } &
sleep 30
{ probe trunc_s3 "$(T 3)" "$U3" 3 0; probe trunc_s4 "$(T 4)" "$U3" 3 0
  probe trunc_s3 "$(T 3)" "$U3" 3 2; probe trunc_s4 "$(T 4)" "$U3" 3 2; } &
sleep 30
{ probe combined_s1 "$(C 1)" "$U4" 0 0; probe combined_s2 "$(C 2)" "$U4" 0 0
  probe combined_s1 "$(C 1)" "$U4" 0 2; probe combined_s2 "$(C 2)" "$U4" 0 2; } &
sleep 30
{ probe combined_s3 "$(C 3)" "$U5" 1 0; probe combined_s4 "$(C 4)" "$U5" 1 0
  probe combined_s3 "$(C 3)" "$U5" 1 2; probe combined_s4 "$(C 4)" "$U5" 1 2; } &
wait
echo "[probe] powered wave complete $(date +%H:%M:%S)"
