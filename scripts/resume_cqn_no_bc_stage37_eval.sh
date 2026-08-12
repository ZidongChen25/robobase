#!/usr/bin/env bash
# Resume only the Stage-37 full validation and held-out evaluations.  Each
# attempt runs in its own session and is aborted/retried if another process
# starts using the selected GPU.
set -euo pipefail
cd "$(dirname "$0")/.."

BASE="$(tr -d '\n' < exp_local/cqn_no_bc/stage37_latest.txt)"
RUN_BASE="$(tr -d '\n' < "${BASE}/run_base.txt")"
RUN1="${RUN_BASE}/dense_b256_seed1"
PREFERRED_GPU="${1:-5}"

printf '%s\n' "${BASHPID}" > "${BASE}/eval_recovery_controller.pid"
touch "${BASE}/eval_recovery_started"

choose_idle_gpu () {
  local gpu apps free_mb sample ok
  local -A seen=()
  for gpu in "${PREFERRED_GPU}" 0 1 2 3 4 5; do
    [[ "${gpu}" =~ ^[0-5]$ ]] || continue
    [[ -z "${seen[${gpu}]:-}" ]] || continue
    seen["${gpu}"]=1
    ok=1
    for sample in 1 2 3; do
      if ! apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null)"; then
        ok=0
        break
      fi
      if [[ -n "${apps//[[:space:]]/}" ]]; then
        ok=0
        break
      fi
      if ! free_mb="$(nvidia-smi -i "${gpu}" --query-gpu=memory.free \
          --format=csv,noheader,nounits 2>/dev/null)"; then
        ok=0
        break
      fi
      free_mb="${free_mb//[[:space:]]/}"
      if [[ ! "${free_mb}" =~ ^[0-9]+$ || "${free_mb}" -lt 2048 ]]; then
        ok=0
        break
      fi
      sleep 10
    done
    if [[ "${ok}" -eq 1 ]]; then
      printf '%s\n' "${gpu}"
      return 0
    fi
  done
  return 1
}

run_isolated_eval () {
  local label="$1" episodes="$2" seed_start="$3" only_steps="$4" csv_name="$5"
  local gpu eval_pid state apps app_pid app_sid foreign status attempt=0
  while true; do
    gpu=""
    until gpu="$(choose_idle_gpu)"; do sleep 30; done
    attempt=$((attempt + 1))
    printf '%s attempt=%s gpu=%s started=%s\n' \
      "${label}" "${attempt}" "${gpu}" "$(date --iso-8601=seconds)" \
      >> "${BASE}/eval_isolation_attempts.log"
    printf '%s\n' "${gpu}" > "${BASE}/${label}_gpu.txt"

    XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
      setsid .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
      --run-dir "${RUN1}" --gpu-id "${gpu}" \
      --num-eval-episodes "${episodes}" --eval-seed-start "${seed_start}" \
      --num-eval-envs 25 --only-steps "${only_steps}" \
      --csv-name "${csv_name}" \
      >> "${BASE}/${label}.log" 2>&1 &
    eval_pid=$!
    printf '%s\n' "${eval_pid}" > "${BASE}/${label}.pid"
    foreign=0
    while kill -0 "${eval_pid}" 2>/dev/null; do
      state="$(ps -o stat= -p "${eval_pid}" 2>/dev/null | tr -d ' ')"
      [[ "${state}" == Z* || -z "${state}" ]] && break
      if ! apps="$(nvidia-smi -i "${gpu}" --query-compute-apps=pid \
          --format=csv,noheader,nounits 2>/dev/null)"; then
        foreign=1
      else
        while read -r app_pid; do
          [[ "${app_pid}" =~ ^[0-9]+$ ]] || continue
          app_sid="$(ps -o sid= -p "${app_pid}" 2>/dev/null | tr -d ' ')"
          if [[ "${app_sid}" != "${eval_pid}" ]]; then
            foreign=1
            break
          fi
        done <<< "${apps}"
      fi
      if [[ "${foreign}" -eq 1 ]]; then
        printf '%s attempt=%s gpu=%s foreign_process=%s\n' \
          "${label}" "${attempt}" "${gpu}" "$(date --iso-8601=seconds)" \
          >> "${BASE}/eval_isolation_attempts.log"
        kill -TERM -- "-${eval_pid}" 2>/dev/null || true
        break
      fi
      sleep 5
    done
    status=0
    wait "${eval_pid}" || status=$?
    if [[ "${foreign}" -eq 0 && "${status}" -eq 0 ]]; then
      printf '%s completed attempt=%s gpu=%s at=%s\n' \
        "${label}" "${attempt}" "${gpu}" "$(date --iso-8601=seconds)" \
        >> "${BASE}/eval_isolation_attempts.log"
      return 0
    fi
    sleep 30
  done
}

run_isolated_eval seed1_full_val50 50 400 \
  "20000,30000,40000,50000,60000,70000,80000,90000,100000,101000" \
  val50_seeds400_full.csv
touch "${BASE}/full_validation_complete"

run_isolated_eval seed1_heldout200 200 800 "101000" \
  heldout200_seeds800_endpoint.csv
touch "${BASE}/heldout_evaluation_complete"

.venv/bin/python -m scripts.summarize_cqn_no_bc_stage37 \
  --run-base "${RUN_BASE}" --mode full \
  --output "${BASE}/stage37_full_summary.json" \
  > "${BASE}/stage37_full_summary.log" 2>&1
touch "${BASE}/complete"
