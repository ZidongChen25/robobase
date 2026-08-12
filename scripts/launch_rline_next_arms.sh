#!/usr/bin/env bash
# R-line follow-up training (cqn-rline.md): keep all four granted training
# cards saturated after wave-1 evals finish.
#   GPU3: official-basestate seeds 3/4 (the missing 63-dim paired reference
#         for every 4-seed claim; exact replica of launch_basestate_ctrl.sh
#         contract).
#   GPU4: l1flip p=0.03 seeds 3/4 (HANDOFF 6.2 pending power confirmation,
#         n=2 mean 78.25 vs baseline 75.25).
# Each controller waits for its card to clear (<2 GB) before launching.
set -uo pipefail
cd "$(dirname "$0")/.."

U3=GPU-03f1431f-36c0-b258-6ca1-05007175e3eb   # GPU3 -> EGL 3
U4=GPU-d224bb2a-2407-1fae-e610-410b2f2bdd08   # GPU4 -> EGL 0
STAMP="${1:-$(date +%Y%m%d%H%M%S)}"

wait_free() {
  local idx=$1
  until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i ${idx})" -lt 2000 ]; do
    sleep 60
  done
}

basestate_run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/official_basestate_move_plate
  local D=${OUT}/seed${s}_${STAMP}
  mkdir -p ${OUT}; echo ${D} > ${OUT}/seed${s}_latest.txt
  local E=(CUDA_VISIBLE_DEVICES=${U3} MUJOCO_EGL_DEVICE_ID=3 MUJOCO_GL=egl
           XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
           ROBOBASE_HOST_MERGE=1)
  env "${E[@]}" .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_demo_driven env=bigym/move_plate \
    env.append_floating_base_to_low_dim=true \
    seed=${s} xla_mem_fraction=0.45 eval_every_steps=1000000 \
    num_eval_episodes=0 num_eval_envs=0 log_eval_video=false \
    save_snapshot=true snapshot_every_n=5000 save_csv=true wandb.use=false \
    hydra.run.dir=${D} > ${D}.launch.log 2>&1 || { touch ${D}/failed; return 1; }
  touch ${D}/train_complete
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir ${D} --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 --csv-name val50_seeds400.csv > ${D}/val50.log 2>&1
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir ${D} --num-eval-episodes 200 --eval-seed-start 800 \
    --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 100000,101000 \
    > ${D}/ep200.log 2>&1
  touch ${D}/complete
}

flip_run() {
  local s=$1
  local OUT=exp_local/cqn_trunc_arms/l1flip_move_plate
  local D=${OUT}/seed${s}_${STAMP}
  mkdir -p ${OUT}; echo ${D} > ${OUT}/seed${s}_latest.txt
  local E=(CUDA_VISIBLE_DEVICES=${U4} MUJOCO_EGL_DEVICE_ID=0 MUJOCO_GL=egl
           XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda
           ROBOBASE_HOST_MERGE=1)
  env "${E[@]}" .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_demo_driven env=bigym/move_plate \
    method.post_ensemble_l1_flip_prob=0.03 method.post_ensemble_l1_flip_horizon=4 \
    seed=${s} xla_mem_fraction=0.45 eval_every_steps=1000000 \
    num_eval_episodes=0 num_eval_envs=0 log_eval_video=false \
    save_snapshot=true snapshot_every_n=5000 save_csv=true wandb.use=false \
    hydra.run.dir=${D} > ${D}.launch.log 2>&1 || { touch ${D}/failed; return 1; }
  touch ${D}/train_complete
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir ${D} --num-eval-episodes 50 --eval-seed-start 400 \
    --num-eval-envs 25 --csv-name val50_seeds400.csv > ${D}/val50.log 2>&1
  env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir ${D} --num-eval-episodes 200 --eval-seed-start 800 \
    --num-eval-envs 25 --csv-name ep200_seeds800.csv --only-steps 100000,101000 \
    > ${D}/ep200.log 2>&1
  touch ${D}/complete
}

(
  wait_free 3
  echo "[next/basestate34] GPU3 clear, launching ($(date +%H:%M:%S))"
  basestate_run 3 & sleep 120; basestate_run 4 & wait
  echo "[next/basestate34] done ($(date +%H:%M:%S))"
) > exp_local/cqn_trunc_arms/rline_next_basestate34.log 2>&1 &

(
  wait_free 4
  echo "[next/flip34] GPU4 clear, launching ($(date +%H:%M:%S))"
  flip_run 3 & sleep 120; flip_run 4 & wait
  echo "[next/flip34] done ($(date +%H:%M:%S))"
) > exp_local/cqn_trunc_arms/rline_next_flip34.log 2>&1 &

echo "[next] controllers armed (basestate34 -> GPU3, flip34 -> GPU4)"
wait
