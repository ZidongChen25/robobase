# Flow Matching Speed and Flow-Step Evaluation Report

Date: 2026-05-31

This note records the current Flow Matching training speed/MFU estimate and the
completed flow-step evaluation sweep. The training/eval runs summarized here are
the `action_sequence=1`, `batch_size=128`, trainable-ResNet18, language-conditioned
Flow Matching runs.

## Sources

- Training logs: `exp_local/bigym_*_fm_transformer_trainable_lang_200e_b128_20260530233*/pretrain.csv`
- Sandwich run log: `exp_local/bigym_sandwich_remove_fm_transformer_trainable_lang_200e_b128_20260531084011/pretrain.csv`
- Strict flow-step sweep: `exp_local/flow_step_sweep_actseq1_strict50_nonzero_20260531113018/flow_step_sweep_results.csv`
- Strict flow-step plot, clearer zoomed version:
  `exp_local/flow_step_sweep_actseq1_strict50_nonzero_20260531113018/flow_step_sweep_success_strict50_actseq1_zoom.svg`
- Original unzoomed plot:
  `exp_local/flow_step_sweep_actseq1_strict50_nonzero_20260531113018/flow_step_sweep_success.svg`
- Fast 1-episode sweep: `exp_local/flow_step_sweep_actseq1_fast1ep_20260531111105/flow_step_sweep_results.csv`
- RTX 5090 peak specs: NVIDIA RTX Blackwell GPU architecture PDF, Appendix A
  <https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf>

## Training Speed

`pretrain.csv` logs every 100 optimizer updates. The timing below uses adjacent
`total_time` deltas with `iteration` delta equal to 100, filtering out long
eval/checkpoint/JIT pauses (`delta < 30s`). One 100-step interval is therefore
100 optimizer updates, and each update consumes one batch of 128 examples.

| Task | 100 updates median (s) | Latest 100 updates (s) | Updates/s | Samples/s |
|---|---:|---:|---:|---:|
| `dishwasher_load_cups` | 10.500 | 9.927 | 9.52 | 1219 |
| `dishwasher_open` | 11.399 | 11.386 | 8.77 | 1123 |
| `flip_cup` | 9.983 | 10.062 | 10.02 | 1282 |
| `flip_cutlery` | 10.012 | 10.239 | 9.99 | 1278 |
| `move_plate` | 10.607 | 10.419 | 9.43 | 1207 |
| `put_cups` | 10.823 | 10.904 | 9.24 | 1183 |
| `sandwich_remove` | 9.983 | 10.061 | 10.02 | 1282 |
| **Overall median** | **10.415** | - | **9.60** | **1229** |

Distribution over filtered 100-update intervals:

| Percentile | 100-update time (s) |
|---|---:|
| p05 | 9.886 |
| p10 | 9.915 |
| p25 | 9.988 |
| p50 | 10.415 |
| p75 | 10.809 |
| p90 | 11.582 |
| p95 | 12.127 |

## FLOPs and MFU Estimate

Assumptions:

- FLOP convention: 1 multiply-add MAC = 2 FLOPs.
- The cached images are 256x256, but the JAX ResNet encoder resizes RGB inputs
  to 224x224 before the ResNet.
- There are 3 camera views per sample.
- Encoder is trainable, so the estimate uses forward + backward as about 3x
  the forward pass FLOPs.
- `num_flow_steps` is not multiplied into training FLOPs. Flow Matching training
  does one network evaluation per update; `num_flow_steps` affects eval/sampling.
- Transformer head cost is included, but with `action_sequence=1` it is tiny
  compared with the trainable ResNet encoder.

Estimated compute:

| Component | Estimate |
|---|---:|
| ResNet18 forward, 224x224, one image | 3.627 GFLOPs |
| ResNet18 train, 3 views, one sample | 32.644 GFLOPs |
| Transformer train, one sample | 0.066 GFLOPs |
| Total train compute, one sample | 32.710 GFLOPs |
| Total train compute, one update (`batch_size=128`) | 4.187 TFLOPs/update |
| Measured training throughput | 9.60 updates/s |
| Estimated delivered model compute | 40.2 TFLOP/s |

RTX 5090 peak comparison:

| Peak denominator | RTX 5090 theoretical peak | Current MFU | Gap to peak |
|---|---:|---:|---:|
| FP32 / dense TF32 | 104.8 TFLOP/s | 38.3% | 2.61x |
| TF32 sparse | 209.5 TFLOP/s | 19.2% | 5.21x |
| FP16 Tensor dense | 419 TFLOP/s | 9.6% | 10.4x |
| FP16 Tensor sparse | 838 TFLOP/s | 4.8% | 20.8x |

Recommended reporting number for these runs: about **40.2 TFLOP/s delivered**,
or **38.3% MFU vs RTX 5090 FP32/dense-TF32 peak**. This is an estimate from
model FLOPs, not a profiler trace; optimizer, dataloader, image resize, BN/ReLU,
and XLA fusion details can shift the exact number.

## Strict Flow-Step Evaluation

This is the main evaluation table. It uses the strict 50-episode sweep for
flow steps `2,4,6,8,15,20` and stitches in the previously completed 10-step
baseline. Tasks whose selected 10-step baseline success was 0 were skipped.
The clearer plot for this table is
`exp_local/flow_step_sweep_actseq1_strict50_nonzero_20260531113018/flow_step_sweep_success_strict50_actseq1_zoom.svg`.

Important caveat: these selected checkpoints are from the earlier `post_action`
alignment runs. The table is useful for comparing sampling flow steps for the
same checkpoint, but it should not be treated as final fixed-alignment policy
quality.

| Task | Ckpt | Timing | 2 | 4 | 6 | 8 | 10 | 15 | 20 | Best |
|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---|
| `move_plate` | 21600 | `post_action` | 0.10 | 0.10 | 0.10 | 0.10 | 0.12 | 0.10 | 0.12 | 10/20 (0.12) |
| `flip_cup` | 27400 | `post_action` | 0.06 | 0.04 | 0.10 | 0.12 | 0.16 | 0.22 | 0.12 | 15 (0.22) |
| `flip_cutlery` | 39000 | `post_action` | 0.22 | 0.26 | 0.22 | 0.22 | 0.18 | 0.20 | 0.30 | 20 (0.30) |
| `dishwasher_open` | 30000 | `post_action` | 0.02 | 0.00 | 0.00 | 0.04 | 0.04 | 0.04 | 0.00 | 8/10/15 (0.04) |
| `dishwasher_load_cups` | 63000 | `post_action` | 0.00 | 0.02 | 0.02 | 0.06 | 0.06 | 0.00 | 0.08 | 20 (0.08) |

Episode length from the same strict sweep:

| Task | 2 | 4 | 6 | 8 | 10 | 15 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `move_plate` | 3260.7 | 2850.7 | 3467.6 | 3201.9 | 3111.2 | 3405.2 | 3011.7 |
| `flip_cup` | 3405.6 | 2268.6 | 3357.7 | 3730.5 | 4206.0 | 4224.5 | 4316.0 |
| `flip_cutlery` | 9849.6 | 9378.8 | 9991.7 | 10154.9 | 10289.8 | 9958.2 | 8923.2 |
| `dishwasher_open` | 6944.7 | 6870.5 | 7059.8 | 7044.0 | 7186.1 | 7070.8 | 7361.9 |
| `dishwasher_load_cups` | 7061.6 | 6549.4 | 6225.0 | 6009.3 | 6179.7 | 6408.6 | 6225.5 |

Skipped zero-baseline tasks:

| Task | Ckpt | Baseline 10-step success | Timing |
|---|---:|---:|---|
| `put_cups` | 56500 | 0.00 | `post_action` |
| `sandwich_remove` | 16200 | 0.00 | `pre_action` |

## Fast 1-Episode Sweep Reference

This was only a quick smoke/trend pass and should not be used as the final
comparison, because one episode has high variance. It is kept here to explain
why the later strict sweep was needed.

| Task | Best flow step | Success |
|---|---:|---:|
| `move_plate` | 4 | 1.00 |
| `flip_cup` | 20 | 1.00 |
| `flip_cutlery` | 20 | 1.00 |
| `dishwasher_open` | 10 | 0.04 |
| `dishwasher_load_cups` | 10 | 0.06 |

## Takeaways

- Current single-GPU FM training speed is stable around 10.4 seconds per
  100 updates, or about 9.6 updates/s with batch size 128.
- The MFU estimate is about 38% against RTX 5090 FP32/dense-TF32 peak.
- In the strict flow-step sweep, larger sampling steps help `flip_cutlery` and
  `dishwasher_load_cups`; `flip_cup` peaks at 15; `move_plate` and
  `dishwasher_open` do not clearly improve beyond the existing 10-step baseline.
- Because most selected checkpoints are from `post_action` alignment runs, the
  low absolute success on dishwasher tasks is still consistent with the earlier
  alignment bug diagnosis.
