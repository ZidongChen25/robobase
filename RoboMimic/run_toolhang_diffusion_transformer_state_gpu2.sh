#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"
TRAIN_STEPS="${TRAIN_STEPS:-100000}"
ACTION_HORIZON="${ACTION_HORIZON:-20}"
RUN_DIR="${RUN_DIR:-${ROOT_DIR}/RoboMimic/runs/toolhang_cd_parity_transformer_state_gpu2_${TRAIN_STEPS}_${STAMP}}"
DATASET_PATH="${DATASET_PATH:-/home/zc1525/CleanDiffuser/dev/robomimic/datasets/tool_hang/ph/low_dim_abs.hdf5}"

mkdir -p "${RUN_DIR}"
ln -sfnT "${RUN_DIR}" "${ROOT_DIR}/RoboMimic/latest_toolhang_run"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2}"
export JAX_CUDA_VISIBLE_DEVICES="${JAX_CUDA_VISIBLE_DEVICES:-2}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"
export WANDB_MODE="${WANDB_MODE:-offline}"

cd "${ROOT_DIR}"

uv run python train.py \
  launch=dp_state_robomimic \
  env=robomimic/tool_hang \
  backend=jax \
  backend.replay_prefetch_size=4 \
  backend.replay_device_prefetch=true \
  backend.fused_update_steps=8 \
  backend.update_block_every_steps=8 \
  gpu_id=2 \
  seed=0 \
  pixels=false \
  env.dataset_path="${DATASET_PATH}" \
  env.filter_key=all \
  env.episode_length=700 \
  env.use_live_env=true \
  env.obs_keys="[object,robot0_eef_pos,robot0_eef_quat,robot0_gripper_qpos]" \
  env.abs_action=true \
  num_pretrain_steps="${TRAIN_STEPS}" \
  eval_every_steps=100000 \
  snapshot_every_n=100000 \
  snapshot_save_start_step=100000 \
  save_snapshot=true \
  save_csv=true \
  demos=.inf \
  batch_size=256 \
  action_sequence="${ACTION_HORIZON}" \
  frame_stack=2 \
  execution_length=8 \
  action_execution_start=1 \
  temporal_ensemble=false \
  num_eval_envs=10 \
  num_eval_episodes=50 \
  replay.nstep=1 \
  replay.epoch_style_sampling=true \
  replay.action_sequence_start_offset=1 \
  replay.action_padding=edge \
  norm_obs=true \
  obs_norm_type=min_max \
  use_min_max_normalization=true \
  min_max_margin=0 \
  method.num_train_steps=1000000 \
  method.num_diffusion_iters=50 \
  method.adaptive_lr=true \
  method.lr_schedule=cosine \
  method.use_ema=true \
  method.ema_decay=0.995 \
  method.ema_decay_schedule=constant \
  method.weight_decay=0.01 \
  method.objective.sampler=ddpm \
  method.backbone.type=transformer \
  method.backbone.sequence_length="${ACTION_HORIZON}" \
  method.backbone.d_model=256 \
  method.backbone.n_heads=4 \
  method.backbone.num_layers=8 \
  method.backbone.n_cond_layers=0 \
  method.backbone.dropout=0.3 \
  wandb.use=false \
  log_eval_video=false \
  hydra.run.dir="${RUN_DIR}"

if [[ -f "${RUN_DIR}/pretrain_eval.csv" ]]; then
  uv run python RoboMimic/export_final_eval_csv.py \
    --run-dir "${RUN_DIR}" \
    --output "${RUN_DIR}/final_eval.csv"

  cp "${RUN_DIR}/final_eval.csv" "${ROOT_DIR}/RoboMimic/toolhang_final_eval_latest.csv"
else
  echo "No pretrain_eval.csv found; skipping final eval export."
fi
