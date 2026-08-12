#!/usr/bin/env bash
# Observation-parity control: our stack + truncation (default) +
# append_floating_base_to_low_dim=true, restoring the OFFICIAL low-dim
# layout (proprioception + grippers + floating_base). Our port's default
# excludes the floating-base state from low_dim_state entirely --
# a silent observation-layout divergence from the official implementation
# (their cfgs/bigym_task/default.yaml state_keys includes
# proprioception_floating_base). If official-layout drops us from ~78.9
# to paper's ~64, the port's accidental base-state omission explains the
# cross-paper gap.
set -uo pipefail
cd "$(dirname "$0")/.."
U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 2)" -lt 2000 ]; do sleep 60; done
STAMP=$(date +%Y%m%d%H%M%S)
run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/official_basestate_move_plate
  local D=${OUT}/seed${s}_${STAMP}
  mkdir -p $OUT; echo $D > $OUT/seed${s}_latest.txt
  CUDA_VISIBLE_DEVICES=$U2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py launch=cqn_as_pixel_bigym_demo_driven env=bigym/move_plate \
    env.append_floating_base_to_low_dim=true \
    seed=$s xla_mem_fraction=0.45 eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    log_eval_video=false save_snapshot=true snapshot_every_n=5000 save_csv=true wandb.use=false \
    hydra.run.dir=$D > ${D}.launch.log 2>&1 || { touch $D/failed; return 1; }
  touch $D/train_complete
  CUDA_VISIBLE_DEVICES=$U2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --csv-name val50_seeds400.csv > $D/val50.log 2>&1
  CUDA_VISIBLE_DEVICES=$U2 MUJOCO_EGL_DEVICE_ID=2 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 100000 \
    > $D/ep200.log 2>&1
  touch $D/complete
}
run 1 & sleep 120; run 2 & wait
echo "[basestate] done $(date +%H:%M:%S)"
