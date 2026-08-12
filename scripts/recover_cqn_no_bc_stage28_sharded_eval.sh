#!/usr/bin/env bash
# Finish Stage 28 after splitting its fixed checkpoint sweep into two shards.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="$1"
LOWER_PID="$2"
UPPER_PID="$(cat "${BASE}/upper_eval.pid")"
RUN_DIR="${BASE}/reward_scale_seed3"
LOWER_CSV="${RUN_DIR}/val50_seeds400.csv"
UPPER_CSV="${RUN_DIR}/val50_upper_seeds400.csv"
TAIL_CSV="${RUN_DIR}/val50_tail_seeds400.csv"
STAGE27="$(cat "${BASE}/stage27_source.txt")"
BASELINE3="$(cat "${BASE}/baseline_seed3.txt")"

printf "%s\n" "${BASHPID}" > "${BASE}/shard_recovery.pid"

while kill -0 "${LOWER_PID}" 2>/dev/null; do
  if [[ -f "${LOWER_CSV}" ]] && grep -q '^10000,' "${LOWER_CSV}"; then
    kill -TERM "${LOWER_PID}"
    touch "${BASE}/lower_shard_complete"
    break
  fi
  sleep 5
done

if [[ ! -f "${BASE}/lower_shard_complete" ]]; then
  printf "%s\n" "lower evaluator exited before 10k" \
    > "${BASE}/sharded_validation_failed"
  exit 1
fi

XLA_PYTHON_CLIENT_PREALLOCATE=false \
  MUJOCO_GL=egl \
  .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" \
  --gpu-id 1 \
  --num-eval-episodes 50 \
  --eval-seed-start 400 \
  --num-eval-envs 25 \
  --csv-name val50_tail_seeds400.csv \
  --only-steps 17500,20000 \
  > "${BASE}/tail_eval.log" 2>&1 &
TAIL_PID=$!
printf "%s\n" "${TAIL_PID}" > "${BASE}/tail_eval.pid"

while kill -0 "${UPPER_PID}" 2>/dev/null; do
  if [[ -f "${UPPER_CSV}" ]] && grep -q '^15000,' "${UPPER_CSV}"; then
    kill -TERM "${UPPER_PID}"
    touch "${BASE}/upper_head_complete"
    break
  fi
  sleep 5
done

wait "${TAIL_PID}"
touch "${BASE}/tail_shard_complete"

lower_rows="$(awk -F, 'NR > 1 && $1 + 0 <= 10000 {n++} END {print n + 0}' "${LOWER_CSV}")"
upper_rows="$(awk -F, 'NR > 1 && $1 + 0 >= 12500 && $1 + 0 <= 15000 {n++} END {print n + 0}' "${UPPER_CSV}")"
tail_rows="$(awk -F, 'NR > 1 && $1 + 0 >= 17500 {n++} END {print n + 0}' "${TAIL_CSV}")"
if [[ "${lower_rows}" -ne 4 || "${upper_rows}" -ne 2 || "${tail_rows}" -ne 2 ]]; then
  printf "lower=%s upper=%s tail=%s\n" \
    "${lower_rows}" "${upper_rows}" "${tail_rows}" \
    > "${BASE}/sharded_validation_failed"
  exit 1
fi

touch "${BASE}/validation_complete"
.venv/bin/python -m scripts.summarize_cqn_no_bc_stage28 \
  --stage27-summary "${STAGE27}/stage27_summary.json" \
  --baseline-seed3 "${BASELINE3}" \
  --treatment-seed3 "${RUN_DIR}" \
  --output "${BASE}/stage28_summary.json" \
  > "${BASE}/stage28_summary.log" 2>&1
touch "${BASE}/complete"
