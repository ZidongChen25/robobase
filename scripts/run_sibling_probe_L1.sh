#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
OUT=exp_local/sibling_probe_powered; mkdir -p "$OUT"
SEEDS=700,701,702,703,704,705,706,707,708,709,710,711,712,713,714,715,716,717,718,719,720,721,722,723
ANCH=20,40,60,80,100,120,140,160,180,200
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
probe() {
  CUDA_VISIBLE_DEVICES="$U2" MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_branch_counterfactual.py \
    --run-dir "$2" --snapshot "$2/snapshots/100000_snapshot.pkl" \
    --output "$OUT/$1_L1.json" --eval-seeds "$SEEDS" --anchor-steps "$ANCH" \
    --intervention-mode sibling_horizon --intervention-horizon 4 \
    --force-level 1 --dimension-selection round_robin \
    --bootstrap-replicates 10000 > "$OUT/$1_L1.log" 2>&1
  echo "[L1] $1 done $(date +%H:%M:%S)"
}
T() { cat exp_local/cqn_official_truncated/seed$1_latest.txt; }
C() { cat exp_local/cqn_trunc_arms/combined/seed$1_latest.txt; }
for b in 1 2; do
  if [ "$b" = 1 ]; then A="trunc_s1:$(T 1) trunc_s2:$(T 2) trunc_s3:$(T 3) trunc_s4:$(T 4)"
  else A="combined_s1:$(C 1) combined_s2:$(C 2) combined_s3:$(C 3) combined_s4:$(C 4)"; fi
  for x in $A; do probe "${x%%:*}" "${x#*:}" & sleep 20; done
  wait
done
echo "[L1] wave complete $(date +%H:%M:%S)"
