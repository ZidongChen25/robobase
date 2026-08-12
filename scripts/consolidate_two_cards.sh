#!/usr/bin/env bash
# Consolidate all training onto TWO cards (GPU1 + GPU2), max 2 runs per card.
#
# Capacity: a training run peaks at ~13.1 GB, so 2 per 32.6 GB card is the
# hard ceiling (3 would OOM). Two cards therefore hold 4 concurrent runs;
# everything else waits in this queue.
#
# Steady state after this script:
#   GPU1: trunc_s1 + trunc_s2        (already running, untouched)
#   GPU2: stage177_s3 + stage178_178B
#   queue: stage178_178A -> starts as soon as any of the four slots frees
#
# stage178_178A/B are stopped only AFTER their 120k snapshot lands, and
# train_fast.py:39 auto-resumes from snapshots/latest_snapshot.pkl, so no
# training progress is lost -- only ~4 min of restart overhead each.
#
# a19_s1/a19_s2 are deliberately NOT touched: they finish training ~14:22,
# and killing a run 7 minutes from completion costs more than it frees.
set -uo pipefail
cd "$(dirname "$0")/.."

GPU1_UUID="GPU-ce804993-c33e-3d10-5676-5bae093a7d96"
GPU2_UUID="GPU-80b9cc0d-df5c-be12-e848-042d37578544"

A_DIR="exp_local/cqn_stage178_floor/seed1_178A"
B_DIR="exp_local/cqn_stage178_floor/seed1_178B"
A_PID="${1:-1692472}"
B_PID="${2:-1692474}"

LOG="exp_local/consolidate_two_cards_$(date +%Y%m%d%H%M%S).log"
say() { echo "[$(date +%H:%M:%S)] $*" >> "${LOG}"; }

COMMON=(launch=cqn_as_pixel_bigym_stage163b_qc_nstep8 env=bigym/move_plate
  seed=1 replay.nstep_explore_truncate=true num_train_frames=151000
  method.bc_lambda=0.0 method.bc_lambda_schedule=null xla_mem_fraction=0.45
  eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0
  log_eval_video=false save_snapshot=true snapshot_every_n=5000
  save_csv=true wandb.use=false)

start_178() {  # $1 = A|B, $2 = target gpu uuid
  local arm="$1" uuid="$2" dir extra=()
  if [ "${arm}" = "A" ]; then
    dir="${A_DIR}"
    extra=(method.unseen_return_floor_weight=0.1
      method.unseen_return_floor_value=0.0
      method.unseen_return_floor_reduction=mean)
  else
    dir="${B_DIR}"
  fi
  say "starting 178${arm} on ${uuid} (resume from $(readlink -f ${dir}/snapshots/latest_snapshot.pkl 2>/dev/null | xargs -r basename))"
  XLA_PYTHON_CLIENT_PREALLOCATE=false MUJOCO_GL=egl \
    CUDA_VISIBLE_DEVICES="${uuid}" \
    nohup setsid .venv/bin/python train_fast.py "${COMMON[@]}" "${extra[@]}" \
    "hydra.run.dir=${dir}" >> "${dir}.requeue.log" 2>&1 &
  say "178${arm} launched pid $!"
}

# Count train_fast.py processes resident on a given GPU uuid.
count_on() {
  local uuid="$1" n=0 pid
  for pid in $(nvidia-smi --query-compute-apps=gpu_uuid,pid --format=csv,noheader \
                 | awk -F', ' -v u="${uuid}" '$1==u {print $2}'); do
    if tr '\0' ' ' < "/proc/${pid}/cmdline" 2>/dev/null | grep -q train_fast.py; then
      n=$((n + 1))
    fi
  done
  echo "${n}"
}

say "=== consolidation start; waiting for 178A/178B 120k snapshots"

# 1. Wait for the next checkpoint on both arms.
until [ -f "${A_DIR}/snapshots/120000_snapshot.pkl" ] \
   && [ -f "${B_DIR}/snapshots/120000_snapshot.pkl" ]; do
  sleep 20
done
sleep 20  # let the snapshot writes flush
say "120k snapshots present on both arms"

# 2. Stop both arms.
kill "${A_PID}" "${B_PID}" 2>/dev/null
for _ in $(seq 1 60); do
  kill -0 "${A_PID}" 2>/dev/null || kill -0 "${B_PID}" 2>/dev/null || break
  sleep 5
done
kill -9 "${A_PID}" "${B_PID}" 2>/dev/null
sleep 30
say "178A/178B stopped; GPU3/GPU4 released"

# 3. 178B joins stage177 on GPU2 right away.
start_178 B "${GPU2_UUID}"
sleep 120

# 4. 178A waits for any of the four slots to free.
say "178A queued; polling for a free slot on GPU1/GPU2"
while true; do
  n1="$(count_on "${GPU1_UUID}")"
  n2="$(count_on "${GPU2_UUID}")"
  if [ "${n2}" -lt 2 ]; then
    start_178 A "${GPU2_UUID}"
    break
  elif [ "${n1}" -lt 2 ]; then
    start_178 A "${GPU1_UUID}"
    break
  fi
  sleep 60
done

say "=== consolidation done; steady state reached"
