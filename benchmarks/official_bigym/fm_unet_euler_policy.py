"""Explicit-Euler evaluation adapter for the official FM-UNet policy."""

from __future__ import annotations

import torch

from roboverse_learn.il.policies.fm.fm_unet_image_policy import (
    FlowMatchingUnetImagePolicy,
)


class EulerFlowMatchingUnetImagePolicy(FlowMatchingUnetImagePolicy):
    """Keep the official policy unchanged except for its ODE integration rule."""

    def conditional_sample(
        self,
        condition_data,
        condition_mask,
        local_cond=None,
        global_cond=None,
        generator=None,
        **kwargs,
    ):
        del kwargs
        trajectory = torch.randn(
            size=condition_data.shape,
            dtype=condition_data.dtype,
            device=condition_data.device,
            generator=generator,
        )

        time_steps = torch.linspace(0, 1.0, self.num_inference_steps + 1)
        for step in range(self.num_inference_steps):
            trajectory[condition_mask] = condition_data[condition_mask]
            t_start = time_steps[step].view(1).expand(trajectory.shape[0])
            t_start = t_start.to(self.device)
            dt = (time_steps[step + 1] - time_steps[step]).to(self.device)
            trajectory = trajectory + dt * self.model(
                trajectory,
                t_start,
                local_cond=local_cond,
                global_cond=global_cond,
            )

        trajectory[condition_mask] = condition_data[condition_mask]
        return trajectory


__all__ = ["EulerFlowMatchingUnetImagePolicy"]
