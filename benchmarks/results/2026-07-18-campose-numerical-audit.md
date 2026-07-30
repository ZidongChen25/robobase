# CamPose JAX numerical audit

Date: 2026-07-18

## Reference

- Official repository: `ripl/CamPoseOpensource@e0647105a0f3946581d253c55dc9fd1218fa0974`
- JAX implementation: the current worktree at the time of this audit
- Dtype: FP32

Weights were mapped by operator semantics, including Torch convolution kernel
layout, rather than compared from unrelated random initializations.

## Results

| Path | Maximum absolute error | Mean absolute error |
|---|---:|---:|
| Plucker ray generation | 1.788e-7 | not recorded |
| ImageNet ResNet18 RGB trunk | 5.245e-6 | 4.759e-7 |
| ACT five-layer Plucker CNN | 2.980e-8 | 2.820e-9 |
| Complete DP early-fusion encoder | 9.537e-7 | 2.001e-7 |
| CamPose geometric crop | 0 | 0 |
| Conditional UNet forward | 6.109e-7 | 1.125e-7 |

The audited paths agree within `1e-5` in FP32. The checks covered the official
double pixel offset, OpenGL ray sign convention, C2W rotation, moment-direction
channel order, ACT late fusion, DP RGB+ray early fusion, GroupNorm ResNet18,
SpatialSoftmax, and Torch-compatible UNet padding/operators.

## Release limitation

This is an engineering audit record, not yet a reproducible CI gate. The next
step is to export versioned golden NPZ/safetensors fixtures from the pinned
official commit and run the same comparisons in a fresh JAX-only environment.
No Torch dependency will be added to the production package or its test job.

