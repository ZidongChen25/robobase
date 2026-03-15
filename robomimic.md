# robomimic Diffusion Policy

This repo can train `robomimic` datasets with `method=diffusion` and log to Weights & Biases.

Before running:

```bash
cd /home/zc1525/robobase
source /home/zc1525/robobase/.venv/bin/activate
wandb login
```

Notes:

- `env.use_live_env=true` enables live `robosuite` eval during training.
- The commands below use the low-dimensional state datasets only.
- `pixels=false` means the policy uses state input, not image observations.
- For state-only live eval, keep `num_train_envs=1` and `log_eval_video=false`.

## Transport

```bash
python train.py \
  method=diffusion \
  env=robomimic \
  env.task_name=TwoArmTransport \
  env.dataset_path=third_party_datasets/robomimic/transport/ph/low_dim_v141.hdf5 \
  env.use_live_env=true \
  demos=.inf \
  is_imitation_learning=true \
  pixels=false \
  frame_stack=1 \
  action_sequence=16 \
  execution_length=1 \
  temporal_ensemble=true \
  use_min_max_normalization=true \
  min_max_margin=0 \
  norm_obs=true \
  num_pretrain_steps=200000 \
  num_train_frames=0 \
  num_train_envs=1 \
  num_eval_episodes=50 \
  eval_every_steps=10000 \
  batch_size=256 \
  replay.nstep=1 \
  log_eval_video=false \
  hydra.run.dir=./exp_local/robomimic_transport_dp_${now:%Y%m%d%H%M%S} \
  wandb.use=true \
  wandb.project=robomimic_dp \
  wandb.name=dp_transport_ph_state
```

## Tool Hang

```bash
python train.py \
  method=diffusion \
  env=robomimic \
  env.task_name=ToolHang \
  env.dataset_path=third_party_datasets/robomimic/tool_hang/ph/low_dim_v141.hdf5 \
  env.use_live_env=true \
  demos=.inf \
  is_imitation_learning=true \
  pixels=false \
  frame_stack=1 \
  action_sequence=20 \
  execution_length=1 \
  temporal_ensemble=true \
  use_min_max_normalization=true \
  min_max_margin=0 \
  norm_obs=true \
  num_pretrain_steps=100000 \
  num_train_frames=0 \
  num_train_envs=1 \
  num_eval_episodes=5 \
  eval_every_steps=10000 \
  batch_size=256 \
  replay.nstep=1 \
  log_eval_video=false \
  hydra.run.dir=./exp_local/robomimic_tool_hang_dp_${now:%Y%m%d%H%M%S} \
  wandb.use=true \
  wandb.project=robomimic_dp \
  wandb.name=dp_tool_hang_ph_state
```

## Optional

If you want to disable live eval and run pure offline training, change:

```bash
env.use_live_env=false num_eval_episodes=0
```
