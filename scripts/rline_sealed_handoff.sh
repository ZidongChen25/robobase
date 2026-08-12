#!/usr/bin/env bash
# Waits for all four wave-1 sealed csvs (100000+101000 rows), kills the
# recovery script's confirm50 tail (frees GPU4 for training), moves the
# confirm50 passes to GPU1 (eval card, EGL 5 probe-verified), and arms the
# next training controllers for GPUs 3/4.
set -uo pipefail
cd "$(dirname "$0")/.."

RECOVER_PID="${1:?recovery pid}"
U1=GPU-ce804993-c33e-3d10-5676-5bae093a7d96   # GPU1 -> EGL 5 (eval card)

done_run() {
  local f=$1
  grep -q '^100000,' "$f" 2>/dev/null && grep -q '^101000,' "$f" 2>/dev/null
}

while true; do
  ok=1
  for a in rfloor_move_plate/seed1 rfloor_move_plate/seed2 \
           nstep3_move_plate/seed1 nstep3_move_plate/seed2; do
    done_run "exp_local/cqn_trunc_arms/${a}_20260809rline1/ep200_seeds800.csv" || ok=0
  done
  [ "$ok" = 1 ] && break
  sleep 60
done
echo "[handoff] all four sealed csvs complete ($(date +%H:%M:%S))"

# Kill the recovery tree so its confirm50 tail never occupies GPU4.
pkill -TERM -P "${RECOVER_PID}" 2>/dev/null || true
kill -TERM "${RECOVER_PID}" 2>/dev/null || true
sleep 5
pkill -KILL -P "${RECOVER_PID}" 2>/dev/null || true

nohup bash scripts/launch_rline_next_arms.sh 20260809rline3 \
  > exp_local/cqn_trunc_arms/rline_next_ctrl.log 2>&1 &
echo "[handoff] next-arms controllers launched"

E=(CUDA_VISIBLE_DEVICES="${U1}" MUJOCO_EGL_DEVICE_ID=5 MUJOCO_GL=egl
   XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda)
for s in 1 2; do
  D="exp_local/cqn_trunc_arms/nstep3_move_plate/seed${s}_20260809rline1"
  echo "[handoff] confirm50 nstep3 s${s} on GPU1 ($(date +%H:%M:%S))"
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${D}" --num-eval-episodes 50 --eval-seed-start 450 \
    --num-eval-envs 25 --csv-name val50_seeds450.csv \
    > "${D}/val50_seeds450.log" 2>&1 || echo "[handoff] confirm50 s${s} FAILED"
done
echo "[handoff] complete ($(date +%H:%M:%S))"
