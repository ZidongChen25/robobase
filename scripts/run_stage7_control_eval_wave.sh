#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
  echo "usage: $0 SEED1_RUN_DIR SEED2_RUN_DIR SNAPSHOT_STEP OUTPUT_DIR" >&2
  exit 2
fi

seed1_run_dir=$(realpath "$1")
seed2_run_dir=$(realpath "$2")
snapshot_step=$3
output_dir=$(realpath -m "$4")
repo_root=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
robobase_python=${ROBOBASE_PYTHON:-/home/zc1525/robobase_jaxflat/.venv/bin/python3}

mkdir -p "$output_dir"
seed1_snapshot="$seed1_run_dir/eval_checkpoints/${snapshot_step}_checkpoint.pkl"
seed2_snapshot="$seed2_run_dir/eval_checkpoints/${snapshot_step}_checkpoint.pkl"
for required in \
  "$seed1_run_dir/.hydra/config.yaml" \
  "$seed2_run_dir/.hydra/config.yaml" \
  "$seed1_snapshot" \
  "$seed2_snapshot"; do
  if [[ ! -f "$required" ]]; then
    echo "missing required artifact: $required" >&2
    exit 3
  fi
done

mapfile -t free_gpu_rows < <(
  nvidia-smi \
    --query-gpu=index,uuid,memory.used,utilization.gpu \
    --format=csv,noheader,nounits \
  | awk -F, '{
      gsub(/ /, "", $1);
      gsub(/ /, "", $2);
      gsub(/ /, "", $3);
      gsub(/ /, "", $4);
      if ($3 < 2000 && $4 < 20) print $1 " " $2;
    }'
)
if [[ ${#free_gpu_rows[@]} -lt 1 ]]; then
  echo "need at least one training-free GPU; found none" >&2
  nvidia-smi \
    --query-gpu=index,uuid,memory.used,memory.total,utilization.gpu \
    --format=csv,noheader >&2
  exit 4
fi

selectors=(direct learned ground_truth direct learned ground_truth)
seeds=(1 1 1 2 2 2)
run_dirs=(
  "$seed1_run_dir" "$seed1_run_dir" "$seed1_run_dir"
  "$seed2_run_dir" "$seed2_run_dir" "$seed2_run_dir"
)
snapshots=(
  "$seed1_snapshot" "$seed1_snapshot" "$seed1_snapshot"
  "$seed2_snapshot" "$seed2_snapshot" "$seed2_snapshot"
)

: > "$output_dir/pids.tsv"
failed=0
next_job=0
while [[ $next_job -lt ${#selectors[@]} ]]; do
  batch_pids=()
  for gpu_row in "${free_gpu_rows[@]}"; do
    if [[ $next_job -ge ${#selectors[@]} ]]; then
      break
    fi
    read -r egl_index gpu_uuid <<< "$gpu_row"
    selector=${selectors[$next_job]}
    seed=${seeds[$next_job]}
    job_name="seed${seed}_${snapshot_step}_${selector}"
    job_output="$output_dir/${job_name}.json"
    job_log="$output_dir/${job_name}.log"
    job_workspace="$output_dir/${job_name}_workspace"

    (
      cd "$repo_root"
      exec "$robobase_python" \
        scripts/eval_cqn_as_latent_consequence_control.py \
        --run-dir "${run_dirs[$next_job]}" \
        --snapshot "${snapshots[$next_job]}" \
        --selector "$selector" \
        --num-eval-episodes 50 \
        --eval-seed-start 400 \
        --gpu-id "$gpu_uuid" \
        --egl-device-id "$egl_index" \
        --work-dir "$job_workspace" \
        --output "$job_output"
    ) > "$job_log" 2>&1 &
    pid=$!
    batch_pids+=("$pid")
    printf '%s\t%s\t%s\t%s\t%s\n' \
      "$pid" "$job_name" "$egl_index" "$gpu_uuid" "$job_output" \
      >> "$output_dir/pids.tsv"
    next_job=$((next_job + 1))
    sleep 20
  done

  # If fewer than six cards were idle at launch, wait for this batch before
  # reusing only those same verified cards. Never co-locate two evals.
  for pid in "${batch_pids[@]}"; do
    if ! wait "$pid"; then
      failed=1
    fi
  done
done

if [[ $failed -ne 0 ]]; then
  touch "$output_dir/eval_failed"
  exit 5
fi
touch "$output_dir/eval_complete"
