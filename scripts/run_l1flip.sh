#!/usr/bin/env bash
# L1-flip exploration arm: post-ensemble, persistent (h=4), single-dimension,
# within-L0-cell flips at p=0.03/step, on top of the default recipe
# (truncated demos, no std floor, Gaussian 0.01 kept). Endpoints: powered L1
# sibling probe (primary) + sealed 200ep@100k (secondary).
set -uo pipefail
cd "$(dirname "$0")/.."
run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/l1flip_move_plate
  local D=${OUT}/seed${s}_20260806215017
  mkdir -p $OUT; echo $D > $OUT/seed${s}_latest.txt
  CUDA_VISIBLE_DEVICES=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py launch=cqn_as_pixel_bigym_demo_driven env=bigym/move_plate \
    method.post_ensemble_l1_flip_prob=0.03 method.post_ensemble_l1_flip_horizon=4 \
    seed=$s xla_mem_fraction=0.45 eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    log_eval_video=false save_snapshot=true snapshot_every_n=5000 save_csv=true wandb.use=false \
    hydra.run.dir=$D > ${D}.launch.log 2>&1 || { touch $D/failed; return 1; }
  touch $D/train_complete
  CUDA_VISIBLE_DEVICES=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --csv-name val50_seeds400.csv > $D/val50.log 2>&1
  CUDA_VISIBLE_DEVICES=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08 MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 100000 \
    > $D/ep200.log 2>&1
  touch $D/complete
}
run 1 & sleep 120; run 2 & wait
