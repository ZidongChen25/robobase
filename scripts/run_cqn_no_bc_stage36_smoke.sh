#!/usr/bin/env bash
# Two-run GPU smoke for official 256/256 replay batches and resume to a real
# merged 512-sample online update. This is wiring/memory evidence only.
set -euo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-3}"
STAMP="$(date +%Y%m%d%H%M%S)"
BASE="exp_local/cqn_no_bc/stage36_batch256_pair_smoke_gpu${GPU}_${STAMP}"
CONTROL="${BASE}/control"
TREATMENT="${BASE}/treatment"
mkdir -p "${CONTROL}" "${TREATMENT}"
printf '%s\n' "${BASE}" > exp_local/cqn_no_bc/stage36_smoke_latest.txt

train () {
  local launch="$1"
  local run_dir="$2"
  local demo_only="$3"
  local force_probability="$4"
  local train_frames="$5"
  local log="$6"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    .venv/bin/python train_fast.py \
    launch="${launch}" env=bigym/move_plate seed=1 \
    batch_size=256 demo_batch_size=256 \
    num_pretrain_steps=100 num_train_frames="${train_frames}" \
    replay.demo_only_updates="${demo_only}" \
    method.demo_behavior_force_probability="${force_probability}" \
    log_every=1 snapshot_every_n=100 gpu_id="${GPU}" xla_mem_fraction=0.45 \
    wandb.use=false hydra.run.dir="${run_dir}" > "${log}" 2>&1
}

run_pair () {
  local phase="$1"
  local demo_only="$2"
  local train_frames="$3"
  local control_force="$4"
  local treatment_force="$5"
  train cqn_as_pixel_bigym_stage36_offline_bc256_control \
    "${CONTROL}" "${demo_only}" "${control_force}" "${train_frames}" \
    "${BASE}/${phase}_control.log" &
  local control_pid=$!
  sleep 30
  train cqn_as_pixel_bigym_stage36_offline_nobc_candidate256_gate \
    "${TREATMENT}" "${demo_only}" "${treatment_force}" "${train_frames}" \
    "${BASE}/${phase}_treatment.log" &
  local treatment_pid=$!
  printf '%s\n' "${control_pid}" > "${BASE}/${phase}_control.pid"
  printf '%s\n' "${treatment_pid}" > "${BASE}/${phase}_treatment.pid"
  local status=0
  wait "${control_pid}" || status=$?
  local treatment_status=0
  wait "${treatment_pid}" || treatment_status=$?
  if [[ "${status}" -eq 0 ]]; then status="${treatment_status}"; fi
  return "${status}"
}

status=0
run_pair offline true 100 0.0 1.0 || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/offline_failed"
  exit "${status}"
fi
touch "${BASE}/offline_complete"

run_pair online false 102 0.0 0.0 || status=$?
if [[ "${status}" -ne 0 ]]; then
  printf '%s\n' "${status}" > "${BASE}/online_failed"
  exit "${status}"
fi
touch "${BASE}/complete"
