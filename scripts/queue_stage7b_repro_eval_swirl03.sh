#!/usr/bin/env bash
# Wait for the corrected Stage-7b 5k checkpoints, then run only the
# preregistered direct-selector reproducibility gate.  Learned/exact selectors
# are deliberately not launched until this gate passes.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "usage: $0 SEED1_RUN_DIR SEED2_RUN_DIR OUTPUT_DIR" >&2
  exit 2
fi

cd "$(dirname "$0")/.."
repo_root="$(pwd)"
python_bin="${STAGE7B_PYTHON:-/home/zc1525/robobase_jaxflat/.venv/bin/python3}"
seed1_dir="$1"
seed2_dir="$2"
output_dir="$3"
gpu0_uuid="GPU-0c5453df-50e2-ae54-203e-147fdfca6aed"
gpu5_uuid="GPU-67303939-4420-e463-9701-b56a07bd6982"

abspath() {
  case "$1" in
    /*) printf '%s\n' "$1" ;;
    *) printf '%s/%s\n' "${repo_root}" "$1" ;;
  esac
}

seed1_dir="$(abspath "${seed1_dir}")"
seed2_dir="$(abspath "${seed2_dir}")"
output_dir="$(abspath "${output_dir}")"
seed1_snapshot="${seed1_dir}/eval_checkpoints/5000_checkpoint.pkl"
seed2_snapshot="${seed2_dir}/eval_checkpoints/5000_checkpoint.pkl"

mkdir -p "${output_dir}"
if compgen -G "${output_dir}/*.json" >/dev/null; then
  echo "refusing to overwrite an existing Stage-7b result in ${output_dir}" >&2
  exit 40
fi

run_is_live() {
  local run_dir="$1"
  pgrep -f "train_fast.py.*hydra.run.dir=${run_dir#${repo_root}/}" >/dev/null
}

snapshots_ready=0
while [ "${snapshots_ready}" -lt 2 ]; do
  if [ -s "${seed1_snapshot}" ] && [ -s "${seed2_snapshot}" ] \
    && ! run_is_live "${seed1_dir}" && ! run_is_live "${seed2_dir}"; then
    size1_a="$(stat -c %s "${seed1_snapshot}")"
    size2_a="$(stat -c %s "${seed2_snapshot}")"
    sleep 10
    size1_b="$(stat -c %s "${seed1_snapshot}")"
    size2_b="$(stat -c %s "${seed2_snapshot}")"
    if [ "${size1_a}" = "${size1_b}" ] && [ "${size2_a}" = "${size2_b}" ]; then
      snapshots_ready=2
      break
    fi
  fi
  sleep 30
done

gpu_used_mib() {
  local uuid="$1"
  nvidia-smi --query-gpu=uuid,memory.used --format=csv,noheader,nounits \
    | awk -F',' -v wanted="${uuid}" \
      '{gsub(/ /,"",$1); if ($1==wanted) {gsub(/ /,"",$2); print $2}}'
}

gpu_foreign_pids() {
  local index="$1"
  nvidia-smi pmon -i "${index}" -c 1 2>/dev/null \
    | awk '
        $1 !~ /^#/ && $2 ~ /^[0-9]+$/ &&
        $NF != "Xorg" && $NF != "gnome-shell" {print $2}
      ' \
    | sort -u
}

while true; do
  used0="$(gpu_used_mib "${gpu0_uuid}")"
  used5="$(gpu_used_mib "${gpu5_uuid}")"
  foreign0="$(gpu_foreign_pids 0 | paste -sd, -)"
  foreign5="$(gpu_foreign_pids 5 | paste -sd, -)"
  if [ "${used0}" -le 512 ] && [ "${used5}" -le 512 ] \
    && [ -z "${foreign0}" ] && [ -z "${foreign5}" ]; then
    break
  fi
  sleep 30
done

common=(
  --selector direct
  --num-eval-episodes 50
  --eval-seed-start 400
)

env XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  "${python_bin}" scripts/eval_cqn_as_latent_consequence_control.py \
  --run-dir "${seed1_dir}" --snapshot "${seed1_snapshot}" \
  "${common[@]}" --gpu-id "${gpu0_uuid}" --egl-device-id 0 \
  --work-dir "${output_dir}/seed1_5000_direct_workspace" \
  --output "${output_dir}/seed1_5000_direct.json" \
  > "${output_dir}/seed1_5000_direct.log" 2>&1 &
pid1=$!

# Avoid simultaneous MuJoCo/EGL environment construction on the host.
sleep 20

env XLA_FLAGS=--xla_gpu_deterministic_ops=true \
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
  "${python_bin}" scripts/eval_cqn_as_latent_consequence_control.py \
  --run-dir "${seed2_dir}" --snapshot "${seed2_snapshot}" \
  "${common[@]}" --gpu-id "${gpu5_uuid}" --egl-device-id 5 \
  --work-dir "${output_dir}/seed2_5000_direct_workspace" \
  --output "${output_dir}/seed2_5000_direct.json" \
  > "${output_dir}/seed2_5000_direct.log" 2>&1 &
pid2=$!

printf 'seed1\t%s\nseed2\t%s\n' "${pid1}" "${pid2}" > "${output_dir}/pids.tsv"

set +e
wait "${pid1}"
status1=$?
wait "${pid2}"
status2=$?
set -e

if [ "${status1}" -ne 0 ] || [ "${status2}" -ne 0 ]; then
  touch "${output_dir}/eval_failed"
  exit 50
fi
touch "${output_dir}/eval_complete"

"${python_bin}" - "${output_dir}" <<'PY'
import glob
import json
import os
import sys

for path in sorted(glob.glob(os.path.join(sys.argv[1], "*.json"))):
    result = json.load(open(path))
    print(os.path.basename(path), result["success_percent"])
PY
