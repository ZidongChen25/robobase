# Transport Offline BC Comparison

Use these two commands to compare real offline BC training for robomimic `Transport`
with `backend=torch` and `backend=jax` in Weights & Biases.

PyTorch:

```bash
PATH=/home/zc1525/robobase/.venv/bin:$PATH python train.py \
  launch=bc_state_robomimic \
  env=robomimic/transport \
  backend=torch \
  gpu_id=0 \
  num_pretrain_steps=200000 \
  num_train_envs=1 \
  num_eval_envs=10 \
  num_eval_episodes=50 \
  action_sequence=16 \
  execution_length=8 \
  eval_every_steps=25000 \
  log_pretrain_every=100 \
  env.use_live_env=true \
  replay.num_workers=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=25000 \
  hydra.run.dir=./exp_local/transport_bc_torch \
  wandb.use=true \
  wandb.entity=tsztungchen25-imperial-college-london \
  wandb.project=robobase_transport_backend_compare \
  wandb.name=transport_bc_torch_fix
```

JAX:

```bash
PATH=/home/zc1525/robobase/.venv/bin:$PATH python train.py \
  launch=bc_state_robomimic \
  env=robomimic/transport \
  backend=jax \
  gpu_id=1 \
  num_pretrain_steps=200000 \
  num_train_envs=1 \
  num_eval_envs=10 \
  num_eval_episodes=50 \
  action_sequence=16 \
  execution_length=8 \
  eval_every_steps=25000 \
  log_pretrain_every=100 \
  env.use_live_env=true \
  replay.num_workers=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=25000 \
  hydra.run.dir=./exp_local/transport_bc_jax \
  wandb.use=true \
  wandb.entity=tsztungchen25-imperial-college-london \
  wandb.project=robobase_transport_backend_compare \
  wandb.name=transport_bc_jax
```

Metrics to compare in W&B:

- training update speed: `pretrain/agent_batched_updates_per_second`
- training sample throughput: `pretrain/agent_updates_per_second`
- evaluation policy speed: `pretrain_eval/agent_act_steps_per_second`
- evaluation env speed: `pretrain_eval/env_steps_per_second`
- final eval result: `pretrain_eval/episode_reward`
- final eval success: `pretrain_eval/episode_success`
- training loss: `pretrain/actor_loss`

Notes:

- `tb.use` controls TensorBoard logging. It is `false` here because this repo rejects using TensorBoard and W&B at the same time.
- `gpu_id=<n>` now controls both the training GPU and the robosuite render GPU. The entrypoint maps that GPU to the single visible device for the process and sets `env.render_gpu_device_id=0` internally.
- `log_pretrain_every` controls how often offline BC training metrics are logged. `100` means one pretrain log every 100 update steps.
- The repo default `wandb.entity` is now `tsztungchen25-imperial-college-london`. Override it only if you want to log somewhere else.
- `launch=bc_state_robomimic` provides the generic robomimic state-BC defaults. The task then comes from `env=robomimic/transport`, and the implementation comes from `backend=torch|jax`.
- `env=robomimic/transport` already points at the sibling robomimic Transport dataset, so `env.dataset_path=...` does not need to be overridden here.
- `env.use_live_env=true` is only for real evaluation. Training is still fully offline imitation learning from the dataset. Without this flag, robomimic evaluation falls back to a placeholder env, so the evaluation-speed numbers are not meaningful.
- `replay.num_workers=0` avoids JAX multiprocessing issues during offline replay iteration.
- `num_pretrain_steps=200000` is the total number of offline BC update steps for the comparison.
- `action_sequence=16` sets the training action chunk length, and `execution_length=8` sets how many actions are executed per evaluation rollout step before replanning.
- `save_snapshot=true` and `snapshot_every_n=25000` save checkpoints during training.
- `hydra.run.dir=...` keeps the run directory short. Without it, Hydra uses the full override string in the output path, which can exceed the filesystem filename limit.

# ToolHang Image Offline JAX Replay Comparison

Use these two commands to compare real image-based offline BC training for robomimic `ToolHang`
with `backend=jax`, where the only intended difference is `replay.num_workers=0` vs
`replay.num_workers=4`.

JAX `replay.num_workers=0`:

```bash
PATH=/home/zc1525/robobase/.venv/bin:$PATH python train.py \
  launch=bc_pixel_robomimic \
  env=robomimic/tool_hang \
  backend=jax \
  gpu_id=0 \
  pixels=true \
  env.dataset_path=third_party_datasets/robomimic/tool_hang/ph/image_v141.hdf5 \
  num_pretrain_steps=100000 \
  num_train_envs=1 \
  num_eval_envs=10 \
  num_eval_episodes=50 \
  action_sequence=16 \
  execution_length=1 \
  eval_every_steps=25000 \
  log_pretrain_every=100 \
  env.use_live_env=true \
  replay.save_dir=/home/zc1525/robobase_backend/exp_local/toolhang_image_bc_jax_w0/replay \
  replay.persist=true \
  replay.reuse_saved=true \
  replay.num_workers=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=25000 \
  hydra.run.dir=./exp_local/toolhang_image_bc_jax_reuse_w0 \
  wandb.use=true \
  wandb.entity=tsztungchen25-imperial-college-london \
  wandb.project=robobase_toolhang_image_jax_replay_compare \
  wandb.name=toolhang_image_bc_jax_reuse_w0
```

JAX `replay.num_workers=4`:

```bash
PATH=/home/zc1525/robobase/.venv/bin:$PATH python train.py \
  launch=bc_pixel_robomimic \
  env=robomimic/tool_hang \
  backend=jax \
  gpu_id=1 \
  pixels=true \
  env.dataset_path=third_party_datasets/robomimic/tool_hang/ph/image_v141.hdf5 \
  num_pretrain_steps=200000 \
  num_train_envs=1 \
  num_eval_envs=10 \
  num_eval_episodes=50 \
  action_sequence=16 \
  execution_length=8 \
  eval_every_steps=25000 \
  log_pretrain_every=100 \
  env.use_live_env=true \
  replay.save_dir=/home/zc1525/robobase_backend/exp_local/toolhang_image_bc_jax_w0/replay \
  replay.persist=true \
  replay.reuse_saved=true \
  replay.num_workers=4 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=25000 \
  hydra.run.dir=./exp_local/toolhang_image_bc_jax_reuse_w4 \
  wandb.use=true \
  wandb.entity=tsztungchen25-imperial-college-london \
  wandb.project=robobase_toolhang_image_jax_replay_compare \
  wandb.name=toolhang_image_bc_jax_reuse_w4
```

Notes:

- `launch=bc_pixel_robomimic` provides the generic robomimic image-BC defaults. The task comes from `env=robomimic/tool_hang`, and the implementation comes from `backend=jax`.
- `env=robomimic/tool_hang` defaults to the low-dimensional dataset, so image training must override `env.dataset_path` to `third_party_datasets/robomimic/tool_hang/ph/image_v141.hdf5`.
- `pixels=true` switches BC onto the image observation path. In the current BC setup, JAX uses the frozen `resnet18` visual encoder plus the shared BC actor head.
- `launch=bc_pixel_robomimic` defaults to `action_sequence=1` and `execution_length=1`, so these commands explicitly override `action_sequence=16` while keeping `execution_length=1` fixed.
- These commands explicitly reuse the previous replay cache at `/home/zc1525/robobase_backend/exp_local/toolhang_image_bc_jax_w0/replay` via `replay.save_dir=...` and `replay.reuse_saved=true`, so they do not rebuild replay episodes from the HDF5 dataset if that cache is already present.
- The replay `.npz` files are still episode data stored as single-step transitions. That does not block `action_sequence=16`: RoboBase assembles and pads the 16-step action chunk at sample time, instead of storing 16-step chunks in the `.npz` files.
- The current reusable replay cache under `toolhang_image_bc_jax_w0/replay` currently contains 90 `.npz` files on this machine.
- `replay.num_workers=4` on JAX now enables host-side replay prefetching. It is not the same implementation as Torch's multi-process replay workers, so expect a smaller effect than the Torch `0 -> 4` jump.
- As with the transport comparison, `env.use_live_env=true` is only for meaningful real evaluation metrics in W&B; training itself is still offline imitation learning from the robomimic dataset.
- If image eval runs out of memory, reduce `num_eval_envs` first. The render envs are created only during evaluation and then closed, but 10 live image envs can still be heavy during the eval window.
