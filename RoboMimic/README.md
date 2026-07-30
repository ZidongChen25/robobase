# RoboMimic ToolHang Runs

This folder keeps RoboMimic-specific run scripts and exported results.

Default ToolHang diffusion run:

```bash
bash RoboMimic/run_toolhang_diffusion_transformer_state_gpu2.sh
```

The script maps the requested hyperparameters to RoboBase overrides:

- GPU: `gpu_id=2`
- Task: `env=robomimic/tool_hang`, state observations, live eval enabled
- Method: diffusion policy with `method.backbone.type=transformer`
- Batch size: `batch_size=256`
- Train steps: `num_pretrain_steps=500000`
- Action horizon: `action_sequence=20`
- Observation history: `frame_stack=2`
- Eval action execution step: `execution_length=8`
- Eval and checkpoint cadence: every `100000` pretrain steps
- Step 0 eval/checkpoint is skipped by the pretrain loop and `snapshot_save_start_step=100000`

Each run writes `pretrain_eval.csv` in its Hydra run directory. After training finishes, the wrapper exports the latest eval row to `final_eval.csv` in the same run directory and also updates `RoboMimic/toolhang_final_eval_latest.csv`.
