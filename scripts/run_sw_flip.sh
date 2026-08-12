#!/usr/bin/env bash
# sandwich_remove two-level flip arm. Diagnostics (user's l2_cross_task line)
# measured BOTH levels outcome-relevant on this task (iid-L2 -24/-32pp,
# L0-only collapse) unlike move_plate, so both are worth exploring; doses are
# scaled by per-event behaviour tax (L1 events ~3x costlier) and by the 540-
# step episodes: L1 p=0.015 (~8 ev/ep), L2 p=0.05 (~27 ev/ep), horizon 4.
set -uo pipefail
cd "$(dirname "$0")/.."
run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/swflip_sandwich_remove
  local D=${OUT}/seed${s}_20260806232537
  mkdir -p $OUT; echo $D > $OUT/seed${s}_latest.txt
  CUDA_VISIBLE_DEVICES=GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919 MUJOCO_EGL_DEVICE_ID=4 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py launch=cqn_as_pixel_bigym_demo_driven env=bigym/sandwich_remove \
    replay_size_before_train=600 \
    method.post_ensemble_l1_flip_prob=0.015 method.post_ensemble_l2_flip_prob=0.05 \
    method.post_ensemble_l1_flip_horizon=4 \
    seed=$s xla_mem_fraction=0.45 eval_every_steps=1000000 num_eval_episodes=0 num_eval_envs=0 \
    log_eval_video=false save_snapshot=true snapshot_every_n=5000 save_csv=true wandb.use=false \
    hydra.run.dir=$D > ${D}.launch.log 2>&1 || { touch $D/failed; return 1; }
  touch $D/train_complete
  CUDA_VISIBLE_DEVICES=GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919 MUJOCO_EGL_DEVICE_ID=4 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --csv-name val50_seeds400.csv > $D/val50.log 2>&1
  CUDA_VISIBLE_DEVICES=GPU-79eb6469-87b1-9dd3-ca2d-da93a916e919 MUJOCO_EGL_DEVICE_ID=4 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 100000 \
    > $D/ep200.log 2>&1
  touch $D/complete
}
run 1 & sleep 120; run 2 & wait
