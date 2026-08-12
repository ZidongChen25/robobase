#!/usr/bin/env bash
# What is the critic's per-level bin ordering worth on task success?
# The powered sibling probe put its ordering at chance on the finest C2F
# level (sign acc 0.491/0.500) while forcing a different fine bin still moved
# realized return (regret 0.052-0.065 on 57% of states). If replacing the
# fine-level argmax with a uniform draw costs nothing on task success, then
# that unexploited ordering is worth nothing and fixing it buys nothing.
# Paired against the existing sealed numbers (same checkpoint, same 200
# episodes, seeds 800-999): 78.5 / 73.0 / 82.0 / 82.0 at 100k.
set -uo pipefail
cd "$(dirname "$0")/.."
U=(GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 GPU-2f044e6a-9150-0e30-7d97-009bdd425b11)
E=(0 1)
run() {  # seed lvl cardidx
  local s="$1" lvl="$2" c="$3"
  local D; D=$(cat exp_local/cqn_official_truncated/seed${s}_latest.txt)
  local SK; SK=$(ls "$D/snapshots" | grep -oE '^[0-9]+' | sort -n -u | grep -v '^100000$' | tr '\n' ',' | sed 's/,$//')
  CUDA_VISIBLE_DEVICES="${U[$c]}" MUJOCO_EGL_DEVICE_ID="${E[$c]}" MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir "$D" \
    --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
    --csv-name "ep200_randfrom${lvl}.csv" --skip-steps "$SK" \
    --random-levels-from "$lvl" > "$D/ep200_randfrom${lvl}.log" 2>&1
  echo "[rand] seed$s from-level$lvl done $(date +%H:%M:%S)"
}
{ for s in 1 2 3 4; do run "$s" 2 0; done; } &
sleep 30
{ for s in 1 2 3 4; do run "$s" 1 1; done; } &
wait
echo "[rand] complete $(date +%H:%M:%S)"
