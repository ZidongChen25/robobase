# Official Policy References on BiGym

This directory contains isolated adapters for running pinned upstream policy
implementations on the local FlipCutlery data and environment. It is a
benchmark boundary, not part of the `robobase` production runtime.

| Reference | Upstream core | Local boundary | Protocol |
| --- | --- | --- | --- |
| A2A | Official PyTorch A2A and FM-UNet | Zarr export, raw BiGym rollout | O8 / H16 / K8; A2A S6, FM S10 |
| Legato | Official JAX/Flax NNX Kinetix core | Frozen FM visual features, checkpointing, BiGym rollout | H8 / K4 / S5; delays 0 and 2 |

The main library remains JAX-only. A2A's public implementation is PyTorch, so
it is used only as an external oracle under `benchmarks/`; no Torch dependency
is added to `pyproject.toml`, `uv.lock`, or any `robobase` import path. Legato's
published Kinetix core is pure JAX and is imported directly from its pinned
checkout.

Pinned revisions:

```text
A2A_Flow_Matching: a5792ecf4e7f8fa4d85fe66ea9a50618138f925c
Legato-kinetix:    d302701268aa3a50ec7f07189cc3af3b31014f63
Kinetix submodule: cf7453ea103fa0b77348af1a39f689c658161613
```

See `a2a_README.md` for dataset export, official A2A/FM-UNet training, and
epoch-100/200 evaluation commands. See `legato_README.md` for frozen-feature
export, official vanilla/RTC/Legato training, and delay-aware evaluation.

The two comparisons answer different questions. A2A versus official FM-UNet is
a matched one-camera, 43-demonstration experiment. Legato versus RTC is a
matched frozen-encoder action-core experiment. Neither should be presented as
an architecture-only comparison against the existing three-camera H20 RoboBase
FM checkpoint without explicitly reporting those observation and horizon
differences.
