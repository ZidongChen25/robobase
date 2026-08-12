#!/usr/bin/env bash
# Agent line: erosion diagnostic on the frozen Stage-38 snapshots.
# For each seed and snapshot 10k..30k, probe fixed replay states with the
# read-only value-fidelity tool on CPU. Same sampling seed everywhere, so the
# only varying factor is the critic parameters. No env seed is consumed.
set -euo pipefail
cd "$(dirname "$0")/.."

STAGE38="exp_local/cqn_no_bc/stage38_offline_dense_b256_gpu5_20260801083024"
OUT="exp_local/cqn_no_bc/agent_erosion_probe_stage38"
mkdir -p "${OUT}"

for seed in 1 2; do
  run="${STAGE38}/dense_seed${seed}/offline_then_online"
  for step in 10000 12500 15000 17500 20000 22500 25000 27500 30000; do
    out_json="${OUT}/seed${seed}_step${step}.json"
    if [[ -s "${out_json}" ]]; then
      continue
    fi
    JAX_PLATFORMS=cpu .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
      --run-dir "${run}" \
      --snapshot "${run}/snapshots/${step}_snapshot.pkl" \
      --output "${out_json}" \
      --gpu-id -1 --seed 0 --samples-per-group 8 \
      --offline-episode-count 60 --critic target \
      >> "${OUT}/probe.log" 2>&1
  done
done
touch "${OUT}/complete"
