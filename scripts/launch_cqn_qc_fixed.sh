#!/usr/bin/env bash
# QC (chunking: nstep=8 x replan-8), 4 seeds, WITH the two 08-03/08-04 fixes:
#   env.truncate_demo_at_success=true   (default since 08-03)
#   env.obs_std_floor_relative=0.01     (degenerate-scale guard, 08-04)
# Previous wave without the std floor: seeds 1/2/3 all died non-finite at
# iterations 4000/1000/OOM. Forensic tap (nonfinite_dump) is on, so if any
# seed still goes non-finite the triggering batch lands in the run dir.
set -euo pipefail
cd "$(dirname "$0")/.."
{
  echo "[qc-fixed] 20260804001359 start $(date +%H:%M:%S)"
  bash scripts/run_cqn_trunc_arm.sh qc GPU-80b9cc0d-df5c-be12-e848-042d37578544 2 1 20260804001359 & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc GPU-80b9cc0d-df5c-be12-e848-042d37578544 2 2 20260804001359 & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc GPU-ce804993-c33e-3d10-5676-5bae093a7d96 5 3 20260804001359 & sleep 120
  bash scripts/run_cqn_trunc_arm.sh qc GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 0 4 20260804001359 &
  wait
  echo "[qc-fixed] all exited $(date +%H:%M:%S)"
} > exp_local/cqn_trunc_arms/qc_fixed_20260804001359.log 2>&1
