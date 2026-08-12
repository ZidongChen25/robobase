#!/usr/bin/env bash
# Eval-time temporal-ensemble gain scan on the 4 truncated-baseline 100k
# checkpoints. gain 0.01 (default) gives the newest plan 6.7% weight and
# attenuates fine-level (L2) influence to ~0.001 of action range; gain 5.0
# (=99.3%) collapsed to 0% when TRAINED that way (cqn-flow.md 20.5), but the
# middle was never scanned, and eval-time-only sharpening isolates the
# execution-side tolerance from the learning-side effect.
# newest-plan weight by gain: 0.05->9.7% 0.2->19% 0.5->40% 1.0->63% 5.0->99.3%
set -uo pipefail
cd "$(dirname "$0")/.."
U=(GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 GPU-2f044e6a-9150-0e30-7d97-009bdd425b11 GPU-80b9cc0d-df5c-be12-e848-042d37578544)
E=(0 1 2)
run() {  # seed gain cardidx
  local s="$1" g="$2" c="$3"
  local D; D=$(cat exp_local/cqn_official_truncated/seed${s}_latest.txt)
  local SK; SK=$(ls "$D/snapshots" | grep -oE '^[0-9]+' | sort -n -u | grep -v '^100000$' | tr '\n' ',' | sed 's/,$//')
  CUDA_VISIBLE_DEVICES="${U[$c]}" MUJOCO_EGL_DEVICE_ID="${E[$c]}" MUJOCO_GL=egl \
  XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir "$D" \
    --num-eval-episodes 200 --eval-seed-start 800 --num-eval-envs 25 \
    --csv-name "ep200_gain${g}.csv" --skip-steps "$SK" \
    --temporal-ensemble-gain "$g" > "$D/ep200_gain${g}.log" 2>&1
  echo "[gain] seed$s gain$g done $(date +%H:%M:%S)"
}
{ for g in 0.05 1.0; do for s in 1 2 3 4; do run "$s" "$g" 0; done; done; } &
sleep 30
{ for g in 0.2 5.0; do for s in 1 2 3 4; do run "$s" "$g" 1; done; done; } &
sleep 30
{ for g in 0.5; do for s in 1 2 3 4; do run "$s" "$g" 2; done; done; } &
wait
echo "[gain] scan complete $(date +%H:%M:%S)"
