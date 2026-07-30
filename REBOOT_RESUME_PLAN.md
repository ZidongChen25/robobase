# 重启后恢复清单(2026-07-28 晚)

重启目的:修复 GPU1(遥测降级,`nvidia-smi -i 1` utilization 显示 [N/A])。
重启后先确认 `nvidia-smi` 六卡 utilization 都是数字。

所有训练都从各自 run 目录的 `snapshots/latest_snapshot.pkl` 自动续跑
(命令里指定原目录即可)。每条命令后台 nohup 运行,GPU 分配可按当时空闲调整。

## 1. 探索消融 U 臂(uniform,GPU0,断点 ~35k)
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 MUJOCO_EGL_DEVICE_ID=0 nohup .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_stage162_uniform env=bigym/move_plate seed=1 save_csv=true wandb.use=false \
  hydra.run.dir=exp_local/cqn_stage162_eps_ablation/move_plate_uniform_seed1_gpu0_20260728141948 \
  >> exp_local/cqn_stage162_eps_ablation/move_plate_uniform_seed1_gpu0_20260728141948.launch.log 2>&1 &
```

## 2. 探索消融 E 臂(edecay,GPU2,断点 ~35k)
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=2 MUJOCO_EGL_DEVICE_ID=2 nohup .venv/bin/python train_fast.py \
  launch=cqn_as_pixel_bigym_stage162_edecay env=bigym/move_plate seed=1 save_csv=true wandb.use=false \
  hydra.run.dir=exp_local/cqn_stage162_eps_ablation/move_plate_edecay_seed1_gpu2_20260728141948 \
  >> exp_local/cqn_stage162_eps_ablation/move_plate_edecay_seed1_gpu2_20260728141948.launch.log 2>&1 &
```

## 3. 官方+mask(GPU3,断点 ~80k 里程碑,20k 粒度)
```bash
MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=3 MUJOCO_EGL_DEVICE_ID=3 nohup .venv/bin/python train.py \
  launch=cqn_as_pixel_bigym_stage161_official_mask env=bigym/move_plate seed=1 save_csv=true \
  wandb.name="cqn_as_stage161_offmask_seed1_move_plate" \
  hydra.run.dir=exp_local/cqn_stage161_official_mask/move_plate_offmask_seed1_gpu3_20260728084320 \
  >> exp_local/cqn_stage161_official_mask/move_plate_offmask_seed1_gpu3_20260728084320.launch.log 2>&1 &
```

## 4. 探索消融 H 臂(double,原 GPU1→改 GPU4,断点 15k)
```bash
nohup bash scripts/resume_cqn_stage162_double.sh > exp_local/cqn_stage162_eps_ablation/stage162r5_master.log 2>&1 &
```
(脚本已指向 GPU4,含训练后探针+200-ep 链。)

## 5. 163c 官方+QC(GPU5,断点 ~5k;重启后 EGL 顺序应恢复正常,
   若 gpu.py 的 setdefault 保留,外部 EGL 变量可不再需要)
```bash
nohup bash scripts/run_cqn_stage163c_official_qc8.sh > exp_local/cqn_stage163_replan8/stage163c_master.log 2>&1 &
```
注意:此脚本每次生成新时间戳目录——若想续 5k 断点,改用指定原目录的
train_fast 命令(目录 `exp_local/cqn_stage163_replan8/move_plate_offqc8_seed1_gpu5_17*`);
5k 而已,直接重跑也行(脚本原样执行即可,旧目录作废删除)。

## 6. 训练完成后的补链(163c/U/E 三臂的探针+200-ep 在原 controller 里,
   controller 已被杀,需按 scripts/run_cqn_stage162_eps_ablation.sh 中
   run_arm 的 probe/eval 段手工补,或告诉 Claude 重建)

## 7. 其他待办
- 162d nonoise 臂:从未开始,重新排队(scripts/run_cqn_stage162_nonoise.sh,
  等 GPU3 的 offmask 完成)
- 执行方式成对表缺 5 格:官方 s3、只衰减 s1/s2、遮罩 s1/s2 的 replan-8
  200-ep(scripts/eval_cqn_as_snapshot_sweep.py --replan-interval 8)
- watcher 全部停用中;曲线由批扫回填
- GPU1 修好后解除隔离
