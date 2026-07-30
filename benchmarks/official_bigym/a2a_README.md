# Official A2A and FM-UNet on BiGym

This directory runs the unmodified PyTorch policies from the
[official A2A repository](https://github.com/JIAjindou/A2A_Flow_Matching) on
BiGym data and environments. It is an isolated reference benchmark: none of
these modules are imported by RoboBase's production JAX policies, and no Torch
dependency is added to the production `pyproject.toml` or `uv.lock`.

The checkout is pinned and validated at startup:

```text
third_party/A2A_Flow_Matching_official
a5792ecf4e7f8fa4d85fe66ea9a50618138f925c
```

The adapter does not edit files in the official checkout. The small
`a2a_official_entrypoint.py` shim only restores typing names that current
Diffusers stopped exporting; the official scheduler, model, loss, optimizer,
EMA, dataset, and checkpoint code still execute unchanged.

## Protocol

The matched A2A/FM-UNet BiGym protocol is:

| Setting | A2A | FM-UNet |
| --- | ---: | ---: |
| Model horizon | 16 | 16 |
| Observation/history steps | 8 | 8 |
| Predicted action steps | 8 | 8 |
| Default executed steps | 8 | 8 |
| Flow steps | 6 | 10 |
| Image input | head RGB, 256x256 | head RGB, 256x256 |
| State/action dimension | 16 | 16 |
| Batch size | 32 | 32 |
| Control frequency | 20 Hz | 20 Hz |

A2A's source is the latest eight measured `robot.qpos_actuated` states, matching
the official policy implementation. It is not a buffer of previously commanded
actions. The exporter reconstructs the same 16 actuated positions from cached
floating-base, limb-joint, and gripper feedback. Observations are paired with
the action at the same index (`pre_action` alignment).

The paper's main simulation table used 100 demonstrations and 30 epochs on five
RoboVerse tasks. Its visual-generalization experiment used 200 epochs. Neither
is a published BiGym result. The `paper` preset reproduces the reported 30-epoch
budget; `bigym-200` is the requested BiGym extension with checkpoints every 20
epochs. BiGym's 20 Hz rate comes from the local task/cache protocol, not a
frequency claim in the A2A paper.

## Export

Export the successful reset-aligned FlipCutlery demonstrations to the official
Zarr schema:

```bash
.venv/bin/python -m benchmarks.official_bigym.a2a_export_dataset \
  --cache-dir /home/zc1525/.bigym_reset_aligned/demonstrations/0.9.0/FlipCutlery/JointPositionActionMode_floating_pelvis_x_pelvis_y_pelvis_z_pelvis_rz_absolute/pixel/head-rgb-256x256_left_wrist-rgb-256x256_right_wrist-rgb-256x256/20hz \
  --output exp_local/official_bigym/flip_cutlery_head_qpos_success.zarr
```

The exporter writes `adapter_manifest.json` beside the Zarr arrays. Keep that
manifest with experiment artifacts so camera, timing, source files, and control
frequency remain auditable.

## Train

Run the requested 200-epoch A2A experiment:

```bash
.venv/bin/python -m benchmarks.official_bigym.a2a_train \
  --method a2a \
  --dataset exp_local/official_bigym/flip_cutlery_head_qpos_success.zarr \
  --output exp_local/official_bigym/a2a_bigym200_seed0_20260719 \
  --preset bigym-200 \
  --device cuda:0
```

Run the official FM-UNet comparator with the identical data and budget:

```bash
.venv/bin/python -m benchmarks.official_bigym.fm_unet_train \
  --dataset exp_local/official_bigym/flip_cutlery_head_qpos_success.zarr \
  --output exp_local/official_bigym/fm_unet_bigym200_seed0_20260719 \
  --preset bigym-200 \
  --device cuda:0
```

`bigym-200` means 200 epochs, batch size 32, seed 0, and checkpoints at epochs
20, 40, ..., 200. Use `--print-command` to validate the fully resolved Hydra
command without launching a process. CLI overrides remain available for
controlled ablations.

## Evaluate

Evaluate A2A checkpoints at epochs 100 and 200 with 50 fixed seeds:

```bash
for epoch in 100 200; do
  .venv/bin/python -m benchmarks.official_bigym.a2a_eval \
    --method a2a \
    --checkpoint exp_local/official_bigym/a2a_bigym200_seed0_20260719/checkpoints/${epoch}.ckpt \
    --output exp_local/official_bigym/a2a_bigym200_seed0_20260719/eval_${epoch}_50ep.json \
    --num-episodes 50 \
    --seed-start 0 \
    --max-steps 500 \
    --execution-length 8 \
    --clip-actions \
    --device cuda:0
done
```

Use the FM-UNet wrapper for the matched comparator:

```bash
for epoch in 100 200; do
  .venv/bin/python -m benchmarks.official_bigym.fm_unet_eval \
    --checkpoint exp_local/official_bigym/fm_unet_bigym200_seed0_20260719/checkpoints/${epoch}.ckpt \
    --output exp_local/official_bigym/fm_unet_bigym200_seed0_20260719/eval_${epoch}_50ep.json \
    --num-episodes 50 \
    --seed-start 0 \
    --max-steps 500 \
    --execution-length 8 \
    --clip-actions \
    --device cuda:0
done
```

Evaluation uses raw BiGym absolute joint-position actions, refreshes measured
state and camera history after every executed control step, and reports success,
policy latency, action differences, acceleration, jerk, and chunk-boundary
jumps. The formal comparison uses a 500-step cap and clips commands to the raw
environment action bounds, matching the safety boundary used by the existing
RoboBase evaluation path. The Torch policy PRNG is reset to each episode's
environment seed so stochastic FM sources are independently reproducible.
These choices are recorded in the result JSON.

## Smoke Validation

The adapter has been exercised end to end with one official optimizer step:

1. exported a real successful BiGym demonstration (220 frames);
2. trained and checkpointed official A2A and official FM-UNet;
3. restored the official runner checkpoint and EMA model;
4. executed the restored policy in raw BiGym;
5. passed focused state-order, temporal-alignment, Zarr-schema, launcher, and
   `deque` history tests.

These smoke runs validate integration only. Their untrained success values must
not be reported as method performance.
