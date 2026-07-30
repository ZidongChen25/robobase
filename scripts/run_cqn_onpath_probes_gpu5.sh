#!/usr/bin/env bash
# A-0b / Route-B first phase: on-path vs off-path Q reliability, four
# existing 10.5k checkpoints, sequential on one GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
OUT="exp_local/cqn_zoom_coverage/onpath_probes"
mkdir -p "${OUT}"

declare -A RUNS=(
  [clean_full]=exp_local/cqn_value_fidelity_stage2/move_plate_full_first_success_seed1_gpu3_20260722_165946
  [floq_anchored]=exp_local/cqn_value_fidelity_stage2/move_plate_floq_anchored_first_success_seed1_gpu4_20260722_171637
  [decoupled_mc_w0p1]=exp_local/cqn_value_fidelity_stage4/move_plate_mc_return_w0p1_shared_seed1_gpu0_20260723_005100
  [coherent_L1_mc]=exp_local/cqn_value_fidelity_stage7/move_plate_coherent_mc_w0p1_audited_seed1_gpu3_20260723_021600
)

for label in clean_full floq_anchored decoupled_mc_w0p1 coherent_L1_mc; do
  echo "[probe] ${label}"
  .venv/bin/python scripts/analyze_cqn_onpath_q_reliability.py \
    --run-dir "${RUNS[$label]}" \
    --snapshot "${RUNS[$label]}/snapshots/10500_snapshot.pkl" \
    --gpu-id "${GPU}" \
    --samples-per-group 120 \
    --output "${OUT}/${label}_10500_onpath.json" \
    > "${OUT}/${label}_10500_onpath.log" 2>&1
  echo "[probe] ${label} done"
done
echo "[probe] all done"
