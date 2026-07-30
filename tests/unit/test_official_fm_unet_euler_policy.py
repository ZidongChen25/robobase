from __future__ import annotations

import torch

from benchmarks.official_bigym.fm_unet_euler_policy import (
    EulerFlowMatchingUnetImagePolicy,
)


class _CountingConstantVelocity(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(1.0))
        self.times = []

    def forward(self, sample, timestep, *, local_cond, global_cond):
        del local_cond, global_cond
        self.times.append(timestep.detach().clone())
        return torch.ones_like(sample) * self.scale


def _make_policy(steps: int):
    policy = EulerFlowMatchingUnetImagePolicy.__new__(
        EulerFlowMatchingUnetImagePolicy
    )
    torch.nn.Module.__init__(policy)
    policy.model = _CountingConstantVelocity()
    policy.num_inference_steps = steps
    return policy


def test_six_step_euler_uses_six_nfe_and_preserves_conditioning():
    policy = _make_policy(steps=6)
    condition_data = torch.zeros((2, 4, 3), dtype=torch.float32)
    condition_data[:, 0, 1] = torch.tensor([3.0, -2.0])
    condition_mask = torch.zeros_like(condition_data, dtype=torch.bool)
    condition_mask[:, 0, 1] = True

    sample_generator = torch.Generator().manual_seed(7)
    expected_generator = torch.Generator().manual_seed(7)
    expected = torch.randn(
        condition_data.shape,
        dtype=condition_data.dtype,
        generator=expected_generator,
    ) + 1.0
    expected[condition_mask] = condition_data[condition_mask]

    result = policy.conditional_sample(
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=torch.zeros((2, 5)),
        generator=sample_generator,
    )

    assert len(policy.model.times) == 6
    expected_times = torch.arange(6, dtype=torch.float32) / 6
    actual_times = torch.stack(policy.model.times)[:, 0]
    torch.testing.assert_close(actual_times, expected_times)
    torch.testing.assert_close(result, expected)
    assert result.shape == condition_data.shape
    assert result.dtype == condition_data.dtype
    torch.testing.assert_close(result[condition_mask], condition_data[condition_mask])
