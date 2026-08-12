#!/usr/bin/env bash
# R-line wave-1 mid-training look (cqn-rline.md 30k gate): evaluate the 25k
# and 30k snapshots of all four live runs from a free card while training
# continues on GPUs 3/4. Uses the same csv as the final sweep so those steps
# are skipped later (resumable-sweep contract). Informational only — no
# checkpoint selection; kill line = both seeds of an arm < 40% at 30k.
#
# Usage: run_rline_midlook.sh [GPU_UUID] [EGL_ID]
set -uo pipefail
cd "$(dirname "$0")/.."

GPU_UUID="${1:-GPU-80b9cc0d-df5c-be12-e848-042d37578544}"   # GPU2
EGL_ID="${2:-2}"                                            # GPU2 -> EGL 2
E=(CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}"
   MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda)

for ARM in rfloor nstep3; do
  for SEED in 1 2; do
    D="exp_local/cqn_trunc_arms/${ARM}_move_plate/seed${SEED}_20260809rline1"
    [ -d "${D}" ] || { echo "missing ${D}"; continue; }
    echo "[midlook] ${ARM} s${SEED} ($(date +%H:%M:%S))"
    env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "${D}" --num-eval-episodes 50 --eval-seed-start 400 \
      --num-eval-envs 25 --csv-name val50_seeds400.csv \
      --only-steps 25000,30000 \
      > "${D}/midlook.log" 2>&1 || echo "[midlook] ${ARM} s${SEED} FAILED"
    grep -E '^(25000|30000),' "${D}/val50_seeds400.csv" 2>/dev/null \
      | sed "s/^/[${ARM} s${SEED}] /"
  done
done
echo "[midlook] baseline reference s1/s2 @25k: 74/?  @30k: 74/?  (val50 csvs)"
for SEED in 1 2; do
  B=$(cat exp_local/cqn_trunc_arms/official_basestate_move_plate/seed${SEED}_latest.txt)
  grep -E '^(25000|30000),' "${B}/val50_seeds400.csv" 2>/dev/null \
    | sed "s/^/[baseline s${SEED}] /"
done
