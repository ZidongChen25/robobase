# Pixel-policy JAX/PyTorch benchmark (2026-09-02)

> The later operator-parity audit supersedes the ACT/remat and cross-framework
> interpretation below.  See
> `reports/jax_pixel_policy_memory_speed_audit_20260902.md` and the `parity_v3`
> strict/TF32 JSONL files.  In the current two-run ABBA matrix, with equal
> differentiable parameter counts and matched active graphs, strict-FP32
> JAX/PyTorch is 1.10x/0.87x/0.91x for ACT batches 8/32/64 and 1.00x/1.02x for
> DP/FM batch 128.  When both sides permit TF32, those ratios become
> 1.28x/1.17x/1.21x and 1.40x/1.40x.  ACT without remat still cannot fit batch
> 128 in JAX.  Small-batch absolute throughput is host-load sensitive; the
> large-batch direction reproduced across both runs.

## 1. Previous-stage result

The benchmark was run on `swirl03` GPU5 (RTX 5090, 32,607 MiB), in fresh
processes, with three 256x256 RGB views, 63 low-dimensional inputs, language
conditioning, and a 20x16 action chunk.  Each reported value is the mean of
two runs with three warm-up updates and 20 measured updates.  ACT image
augmentation was disabled on both sides because the original PyTorch ACT path
has no corresponding augmentation; DP/FM use their configured
`image_augmentation_type=none`.

Strict FP32 means `JAX_DEFAULT_MATMUL_PRECISION=highest` for JAX and both
PyTorch matmul and cuDNN TF32 switches disabled.  JAX used
`XLA_PYTHON_CLIENT_PREALLOCATE=false` and
`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`.  The latter raises XLA's allocation
ceiling without preallocating the pool: with the default 75% ceiling, ACT
batch 128 failed while requesting a 9.45 GiB cuDNN workspace even though the
card still had physical headroom.

`samples/s` is `batch_size * updates / steady-state wall time` and is higher
when better.  Peak MiB is the process-scoped `nvidia-smi` high-water mark,
including CUDA context, allocator pool, and compilation/autotuning scratch;
it is the amount of card capacity unavailable to another job and is lower
when better.

### Optimization delta within JAX, strict FP32, batch 128

| Method | Baseline samples/s | Optimized samples/s | Speed delta | Baseline peak MiB | Optimized peak MiB | Memory delta |
|---|---:|---:|---:|---:|---:|---:|
| ACT | 419.4 | 424.8 | +1.3% | 31,050 | 31,050 | 0.0% |
| Diffusion | 847.0 | 866.9 | +2.3% | 25,116 | 17,948 | -28.5% |
| Flow Matching | 847.1 | 870.6 | +2.8% | 25,116 | 17,948 | -28.5% |

The baseline is detached commit `bdc28d0`; the optimized arm is the current
working-tree implementation on top of that commit.  Both arms used the same
worker, environment, GPU, shapes, precision, and measurement order.

### JAX versus PyTorch, strict FP32, same batch 128

| Method | JAX samples/s | PyTorch samples/s | JAX / PyTorch | JAX peak MiB | PyTorch peak MiB |
|---|---:|---:|---:|---:|---:|
| ACT | 424.8 | 585.9 | 0.725x | 31,050 | 13,892 |
| Diffusion | 866.9 | 873.6 | 0.992x | 17,948 | 11,465 |
| Flow Matching | 870.6 | 870.4 | 1.000x | 17,948 | 11,466 |

### JAX versus PyTorch, matched operational memory

| Method | JAX batch / peak MiB | JAX samples/s | PyTorch batch / peak MiB | PyTorch samples/s | JAX / PyTorch |
|---|---:|---:|---:|---:|---:|
| ACT | 128 / 31,050 | 424.8 | 304 / 31,824 | 558.6 | 0.760x |
| Diffusion | 128 / 17,948 | 866.9 | 208 / 18,044 | 902.0 | 0.961x |
| Flow Matching | 128 / 17,948 | 870.6 | 208 / 18,044 | 903.1 | 0.964x |

The matched peaks differ by 2.4% for ACT and 0.53% for DP/FM.  PyTorch uses
slightly more memory in every matched pair, so the table is conservative in
PyTorch's favour but is close enough to reject a 1.5x JAX advantage.

### Optional JAX bfloat16 ResNet compute, batch 128

Parameters, batch statistics, normalisation, and policy heads remain FP32;
only ResNet convolution compute and stored trunk activations use bfloat16.

| Method | samples/s | Peak MiB | versus JAX FP32 | versus PyTorch FP32, same batch |
|---|---:|---:|---:|---:|
| ACT | 1,045.6 | 16,934 | 2.46x | 1.78x |
| Diffusion | 2,313.8 | 8,738 | 2.67x | 2.65x |
| Flow Matching | 2,285.5 | 8,738 | 2.63x | 2.63x |

The last column is deliberately not a framework-isolated comparison: it
compares mixed-precision JAX with strict-FP32 PyTorch.  It measures the benefit
available from the new opt-in JAX path, not an inherent JAX/PyTorch speed gap.

Raw artifacts:

- `results/baseline_clean_swirl03_20260902.jsonl` and `.log`: unmodified JAX
  baseline, two runs per method.
- `results/clean_swirl03_20260902.jsonl` and `.log`: optimized JAX, PyTorch,
  same-memory probes, and optional BF16 runs.
- The optimized and baseline JAX measurements used separate worktrees at the
  recorded commits above.
- The PyTorch reference used an isolated PyTorch 2.10.0+cu128 environment.
  Set `TORCH_ROBOBASE_ROOT` and `CLEAN_DIFFUSER_ROOT` to the corresponding
  reference checkouts when reproducing it.

ACT uses the original PyTorch RoboBase `ActBCAgent`.  DP/FM use matched
PyTorch adapters built from torchvision ResNet18 and CleanDiffuser's
`ChiTransformer`; those rows are architecture/objective comparisons rather
than a claim about a preserved historical native implementation.

## 2. Interpretation

The scatter-free exact max-pool and deferred uint8 conversion materially
reduce DP/FM's operational peak memory (about 28.5%) and give a small but
repeatable FP32 throughput improvement (2-3%).  ACT's end-to-end FP32 peak is
instead dominated by cuDNN convolution autotuning/workspace, so the pooling
change does not move its `nvidia-smi` high-water mark and improves throughput
only 1.3%.

Strict FP32 does **not** establish a 1.5x JAX advantage.  ACT is slower and
larger; DP and FM are essentially tied at the same batch and 3-4% slower at a
matched operational peak.  The opt-in BF16 ResNet path is much faster and
smaller, but its policy-quality equivalence is unresolved.  These are
synthetic update-kernel measurements: they exclude environment collection,
replay loading, and checkpoint/evaluation time, and they say nothing about
policy success rate or validation-selected checkpoint quality.

## 3. Next-stage decision

The next hypothesis is that BF16 ResNet compute preserves policy quality while
retaining at least a 1.5x end-to-end training-throughput gain.  The required
test is a matched FP32/BF16 pair for each method with identical demos, task,
training seeds, optimizer settings, and evaluation seeds.  Select checkpoints
only on a fixed validation split, then evaluate the selected FP32 and BF16
checkpoints once on a sealed held-out split.  Report samples/s, process peak
MiB, full validation curves/AUC, validation-best checkpoint, per-seed held-out
success, and aggregate success.  Pass requires at least 1.5x measured training
throughput and a held-out BF16-minus-FP32 lower 95% confidence bound above a
predeclared -5 percentage-point non-inferiority margin.

No policy-training launch is made from this benchmark stage: the four current
latent-dynamics training jobs already occupy the repository's hard ceiling of
four concurrent training runs, and this benchmark does not define a new task,
training-seed pair, validation split, or sealed held-out split on the user's
behalf.

## 4. Execution

Implemented and exercised in the optimized arm:

- an exact 3x3/stride-2/pad-1 scatter-free max-pool with a custom VJP;
- uint8 replay batches retained until conversion can fuse inside JIT updates;
- shared ACT elastic-warp gathers and a no-augmentation fast path;
- opt-in `encoder_model.compute_dtype=bfloat16` without changing the legacy
  FP32 checkpoint parameter tree;
- benchmark workers that record process-scoped GPU memory and explicitly
  disable/enable both PyTorch TF32 paths;
- regression tests for pool forward/gradient equivalence, legacy checkpoint
  tree compatibility, uint8/fused update equivalence, and BF16 update paths.

Verification completed:

- 155 focused ACT/DP/FM tests passed under `JAX_PLATFORMS=cpu` (one expected
  headless GLFW warning about missing `DISPLAY`);
- all three optimized methods completed real RTX 5090 FP32 and BF16 updates;
- Python compilation and `git diff --check` passed;
- both benchmark JSONL files contain two completed measurements per formal
  row; GPU5 returned to 16 MiB and 0% utilization after the runs.
