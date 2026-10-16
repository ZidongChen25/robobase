#!/bin/bash
# Launch RLPD seeds, one run per GPU, seeds queued round-robin over the GPUs.
# Each run is wrapped by gpu_util_monitor (100 ms utilisation samples).
#   bash scripts/run_rlpd_seeds.sh <launch> <out_root> "<gpus>" "<seeds>" [hydra overrides...]
# Runtime paths the task needs (gymnasium-robotics / legacy D4RL overlays,
# LD_LIBRARY_PATH for mujoco-py) come from the caller's environment.
set -u
ROOT=$(cd "$(dirname "$0")/.." && pwd)
LAUNCH=$1; OUT=$2; GPUS=($3); SEEDS=($4); shift 4
EXTRA=("$@")
mkdir -p "$OUT"
cd "$ROOT"
run_queue() {
  local gpu=$1; shift
  for seed in "$@"; do
    local dir="$OUT/seed$seed"
    [ -f "$dir/done" ] && continue
    mkdir -p "$dir"
    MUJOCO_GL=egl CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=$gpu \
    XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/gpu_util_monitor.py --gpu "$gpu" --out "$dir/gpu_util.csv" --skip 120 --tail 5 -- \
      .venv/bin/python train_rlpd.py launch="$LAUNCH" seed="$seed" \
        hydra.run.dir="$ROOT/$dir" "${EXTRA[@]}" > "$dir/stdout.log" 2>&1
    echo "exit=$? $(date -Is)" > "$dir/done"
  done
}
for i in "${!GPUS[@]}"; do
  queue=()
  for j in "${!SEEDS[@]}"; do (( j % ${#GPUS[@]} == i )) && queue+=("${SEEDS[$j]}"); done
  run_queue "${GPUS[$i]}" "${queue[@]}" &
done
wait
