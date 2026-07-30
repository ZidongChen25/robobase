#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PYTHON:-$ROOT_DIR/.venv/bin/python}"
STAMP="${STAMP:-$(date +%Y%m%d%H%M%S)}"
RUN_ROOT="${RUN_ROOT:-$ROOT_DIR/exp_local/adroit_fm_transformer_${STAMP}_noeval}"

export WANDB_MODE="${WANDB_MODE:-offline}"
export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYTHONUNBUFFERED=1

mkdir -p "$RUN_ROOT/logs"

TASKS=(pen door hammer relocate)
GPUS=(0 1 2 3)
TRAIN_STEPS=(243500 488000 488000 488000)

COMMON_OVERRIDES=(
  method=flow_matching
  demos=.inf
  num_pretrain_steps=0
  num_pretrain_epochs=500
  eval_every_steps=0
  snapshot_every_n=0
  snapshot_every_epochs=50
  snapshot_save_start_step=0
  num_train_frames=0
  num_train_envs=0
  num_eval_episodes=0
  num_eval_envs=0
  batch_size=1024
  save_snapshot=true
  save_csv=true
  is_imitation_learning=true
  pixels=false
  frame_stack=1
  action_sequence=1
  execution_length=1
  temporal_ensemble=false
  use_standardization=false
  use_min_max_normalization=false
  norm_obs=false
  update_every_steps=1
  log_pretrain_every=1000
  log_eval_video=false
  wandb.use=false
  replay.nstep=1
  replay.num_workers=0
  replay.pin_memory=false
  replay.epoch_style_sampling=true
  replay.epoch_load_all_episodes=false
  replay.epoch_batch_chunk_size=32
  replay.max_cached_episodes=32
  replay.max_cached_episode_bytes=8589934592
  replay.discard_loaded_demos_after_replay=true
  backend.replay_prefetch_size=32
  backend.replay_device_prefetch=true
  backend.fused_update_steps=8
  backend.update_block_every_steps=8
  method.backbone.type=transformer
  method.backbone.sequence_length=1
  method.backbone.d_model=256
  method.backbone.n_heads=4
  method.backbone.num_layers=8
  method.backbone.n_cond_layers=0
  method.backbone.dropout=0.0
  method.num_flow_steps=2
  method.objective.num_flow_steps=2
  method.objective.sampler=euler
  method.objective.train_time_schedule=beta_0p5_0p5
  method.adaptive_lr=true
  +method.lr_schedule=cosine
  method.use_ema=true
  method.ema_decay=0.995
  method.weight_decay=0.01
)

for idx in "${!TASKS[@]}"; do
  task="${TASKS[$idx]}"
  gpu="${GPUS[$idx]}"
  train_steps="${TRAIN_STEPS[$idx]}"
  run_dir="$RUN_ROOT/$task"
  log_file="$RUN_ROOT/logs/${task}.log"
  pid_file="$RUN_ROOT/logs/${task}.pid"

  mkdir -p "$run_dir"
  echo "Launching $task on GPU $gpu -> $run_dir"
  (
    cd "$ROOT_DIR"
    nohup setsid "$PYTHON" train.py \
      "env=d4rl/$task" \
      "gpu_id=$gpu" \
      "method.num_train_steps=$train_steps" \
      "hydra.run.dir=$run_dir" \
      "${COMMON_OVERRIDES[@]}" \
      >"$log_file" 2>&1 < /dev/null &
    echo "$!" >"$pid_file"
  )
  sleep 6
  echo "Launched $task pid=$(cat "$pid_file")"

done

echo "Run root: $RUN_ROOT"
echo "Logs: $RUN_ROOT/logs"
