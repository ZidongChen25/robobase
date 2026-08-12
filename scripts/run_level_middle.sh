#!/usr/bin/env bash
# "Delete the fine levels" test. random_levels_from + level_override_mode=middle
# emits the parent cell's centre at that level and below -- exactly what an
# agent with those levels removed would output. Contrast with the random
# override (same knob, uniform draw): random = cell centre + jitter inside the
# cell. If middle matches random, the fine levels contribute nothing and can be
# deleted; if middle is worse, what they contribute is dithering, not decision.
# Paired against sealed 100k: 78.5 / 73.0 / 82.0 / 82.0 (normal)
#                             79.5 / 74.5 / 84.5 / -    (random from L2)
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
    --csv-name "ep200_midfrom${lvl}.csv" --skip-steps "$SK" \
    --random-levels-from "$lvl" --level-override-mode middle \
    > "$D/ep200_midfrom${lvl}.log" 2>&1
  echo "[mid] seed$s from-level$lvl done $(date +%H:%M:%S)"
}
{ for s in 1 2 3 4; do run "$s" 2 0; done; } &
sleep 30
{ for s in 1 2 3 4; do run "$s" 1 1; done; } &
wait
echo "[mid] complete $(date +%H:%M:%S)"
