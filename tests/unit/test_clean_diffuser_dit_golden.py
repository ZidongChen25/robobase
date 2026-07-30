import hashlib
from pathlib import Path

from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict
import numpy as np
import pytest

from robobase.models.backbones.dit import JaxDiT1DBackbone


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1] / "fixtures/clean_diffuser_dit_fourier_v1.npz"
)
EXPORTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/export_clean_diffuser_dit_fixture.py"
)
SOURCE_COMMIT = "05f17fc9dbeae7c19a5e264632c9ae9aaac5994e"
PARAM_PREFIX = "param::"


def _scalar(fixture, name: str):
    return fixture[name].item()


def _variables_from_fixture(fixture):
    flat_params = {
        tuple(name.removeprefix(PARAM_PREFIX).split("/")): jnp.asarray(fixture[name])
        for name in fixture.files
        if name.startswith(PARAM_PREFIX)
    }
    return freeze({"params": unflatten_dict(flat_params)})


def test_clean_diffuser_dit_fourier_forward_and_input_vjp_golden():
    with np.load(FIXTURE_PATH, allow_pickle=False) as fixture:
        assert _scalar(fixture, "fixture_version") == 1
        assert _scalar(fixture, "source_commit") == SOURCE_COMMIT
        assert (
            _scalar(fixture, "exporter_sha256")
            == hashlib.sha256(EXPORTER_PATH.read_bytes()).hexdigest()
        )
        action_dim = int(_scalar(fixture, "action_dim"))
        horizon = int(_scalar(fixture, "horizon"))
        embed_dim = int(_scalar(fixture, "embed_dim"))
        model = JaxDiT1DBackbone(
            action_dim=action_dim,
            sequence_length=horizon,
            condition_dim=embed_dim,
            time_embed_dim=embed_dim,
            d_model=int(_scalar(fixture, "d_model")),
            n_heads=int(_scalar(fixture, "n_heads")),
            depth=int(_scalar(fixture, "depth")),
            dropout=0.0,
            timestep_embedding_type="fourier",
            operator_variant="torch",
            condition_adapter="direct",
            fourier_scale=float(_scalar(fixture, "fourier_scale")),
        )
        variables = _variables_from_fixture(fixture)
        parameter_count = sum(
            leaf.size for leaf in jax.tree.leaves(unfreeze(variables)["params"])
        )
        assert parameter_count == int(_scalar(fixture, "parameter_count"))

        actions = jnp.asarray(fixture["actions"])
        timesteps = jnp.asarray(fixture["timesteps"])
        condition = jnp.asarray(fixture["condition"])
        cotangent = jnp.asarray(fixture["cotangent"])
        expected_output = np.asarray(fixture["expected_output"])
        expected_vjps = (
            np.asarray(fixture["expected_action_vjp"]),
            np.asarray(fixture["expected_timestep_vjp"]),
            np.asarray(fixture["expected_condition_vjp"]),
        )

    def forward(action_input, timestep_input, condition_input):
        return model.apply(
            variables,
            action_input,
            timestep_input,
            condition_input,
            train=False,
        )

    # The fixture is exported by Torch on CPU. Avoid accelerator-default TF32
    # precision changing the cross-framework golden comparison.
    with jax.default_matmul_precision("highest"):
        output, pullback = jax.vjp(forward, actions, timesteps, condition)
        actual_vjps = pullback(cotangent)

    np.testing.assert_allclose(output, expected_output, rtol=2e-5, atol=2e-5)
    for actual, expected in zip(actual_vjps, expected_vjps, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=3e-5, atol=3e-5)
