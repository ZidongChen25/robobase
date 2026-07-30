"""Gaussian-source ablation with the official A2A network architecture."""

from __future__ import annotations

from contextlib import contextmanager

import torch
import torch.nn as nn

from roboverse_learn.il.policies.a2a.a2a_policy import A2AImagePolicy


class GaussianSourceEncoder(nn.Module):
    """Preserve A2A source-encoder compute but replace its value with N(0, I)."""

    def __init__(self, reference_encoder: nn.Module):
        super().__init__()
        self.reference_encoder = reference_encoder

    def forward(self, actions, deterministic: bool = False):
        reference = self.reference_encoder(actions, deterministic=deterministic)
        return torch.randn_like(reference)


class GaussianLatentA2AImagePolicy(A2AImagePolicy):
    """Standard Gaussian FM in A2A's action-latent MLP architecture."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.history_action_encoder = GaussianSourceEncoder(
            self.history_action_encoder
        )

    @staticmethod
    @contextmanager
    def _preserve_rng():
        cpu_state = torch.random.get_rng_state()
        cuda_states = (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        )
        try:
            yield
        finally:
            torch.random.set_rng_state(cpu_state)
            if cuda_states is not None:
                torch.cuda.set_rng_state_all(cuda_states)

    @torch.no_grad()
    def get_latents_for_visualization(self, batch):
        with self._preserve_rng():
            return super().get_latents_for_visualization(batch)

    @torch.no_grad()
    def get_flow_trajectories(self, batch, num_steps=None, n_samples=5):
        with self._preserve_rng():
            return super().get_flow_trajectories(
                batch, num_steps=num_steps, n_samples=n_samples
            )


__all__ = ["GaussianLatentA2AImagePolicy", "GaussianSourceEncoder"]
