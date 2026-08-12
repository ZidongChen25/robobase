#!/usr/bin/env bash
# Read-only Stage-36 value/ranking probes on common replay anchors.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="${1:-$(tr -d '\n' < exp_local/cqn_no_bc/stage36_offline_gate_latest.txt)}"
DATA_RUN="exp_local/cqn_value_fidelity_stage2/move_plate_full_first_success_seed1_gpu3_20260722_165946"
OUT="${BASE}/mechanism_diagnostics"
mkdir -p "${OUT}"

probe () {
  local arm="$1"
  local step="$2"
  local run="${BASE}/${arm}/offline_then_online_seed1"
  local output="${OUT}/${arm}_${step}_value_fidelity.json"
  if [[ -s "${output}" ]] && rg -q '"status": "ok"' "${output}"; then
    return 0
  fi
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${run}" \
    --snapshot "${run}/snapshots/${step}_snapshot.pkl" \
    --data-run-dir "${DATA_RUN}" \
    --output "${output}" \
    --gpu-id -1 --samples-per-group 8 --batch-size 4 \
    --seed 0 --offline-episode-count 60 \
    > "${OUT}/${arm}_${step}.log" 2>&1
}

probe treatment 10000
probe treatment 20000
probe control 10000

# Repeat the treatment endpoints on the exact current replay and reward
# semantics.  Only demo groups are requested because this run collected no
# successful online episode; using an older replay would change the repeated
# success-reward convention and invalidate RTG calibration.
probe_local () {
  local step="$1"
  local run="${BASE}/treatment/offline_then_online_seed1"
  local output="${OUT}/treatment_${step}_local_reward_value_fidelity.json"
  if [[ -s "${output}" ]] && rg -q '"status": "ok"' "${output}"; then
    return 0
  fi
  JAX_PLATFORMS=cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${run}" \
    --snapshot "${run}/snapshots/${step}_snapshot.pkl" \
    --data-run-dir "${run}" \
    --output "${output}" \
    --gpu-id -1 --samples-per-group 8 --batch-size 4 \
    --seed 0 --offline-episode-count 60 \
    --groups demo_success,demo_failure \
    > "${OUT}/treatment_${step}_local_reward.log" 2>&1
}

probe_local 10000
probe_local 20000
touch "${OUT}/complete"
