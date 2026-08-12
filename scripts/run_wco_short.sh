#!/usr/bin/env bash
# wall_cupboard_open at the PAPER budget: 26k train, sealed endpoint 25k.
# Paper Fig.5 (8 runs, 25 ep, 25k steps): CQN-AS ~75-90 wide CI, ACT ~75.
# Demo stats: 44 succ demos, ep median 184, clipped fraction 98.5%.
# Replaces wall_cupboard_close, which the paper shows at ~100% immediately
# (no headroom, cannot show a differential).
set -uo pipefail
cd "$(dirname "$0")/.."
run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/official_wall_cupboard_open
  local D=${OUT}/seed${s}_20260805051758
  mkdir -p $OUT; echo $D > $OUT/seed${s}_latest.txt
  CUDA_VISIBLE_DEVICES=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python train_fast.py launch=cqn_as_pixel_bigym_demo_driven env=bigym/wall_cupboard_open \
    seed=$s num_train_frames=26000 xla_mem_fraction=0.45 eval_every_steps=1000000 \
    num_eval_episodes=0 num_eval_envs=0 log_eval_video=false save_snapshot=true \
    snapshot_every_n=5000 save_csv=true wandb.use=false hydra.run.dir=$D \
    > ${D}.launch.log 2>&1 || { touch $D/failed; return 1; }
  touch $D/train_complete
  CUDA_VISIBLE_DEVICES=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 50 \
    --eval-seed-start 400 --num-eval-envs 25 --csv-name val50_seeds400.csv > $D/val50.log 2>&1
  CUDA_VISIBLE_DEVICES=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py --run-dir $D --num-eval-episodes 200 \
    --eval-seed-start 800 --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 25000 \
    > $D/ep200.log 2>&1
  touch $D/complete
}
run 1 & sleep 120; run 2 & wait
