import hashlib
from pathlib import Path

from flax.core import freeze, unfreeze
from flax.traverse_util import unflatten_dict
import numpy as np
import pytest

from robobase.models.backbones.unet1d import JaxConditionalUnet1D


jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")


FIXTURE_PATH = (
    Path(__file__).resolve().parents[1]
    / "fixtures/clean_diffuser_chiunet_global_v1.npz"
)
EXPORTER_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks/export_clean_diffuser_chiunet_fixture.py"
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


def test_clean_diffuser_global_chiunet_forward_and_input_vjp_golden():
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
        obs_steps = int(_scalar(fixture, "obs_steps"))
        obs_dim = int(_scalar(fixture, "obs_dim"))
        down_dims = tuple(int(value) for value in fixture["down_dims"])
        model = JaxConditionalUnet1D(
            action_dim=action_dim,
            sequence_length=horizon,
            feature_dim=obs_steps * obs_dim,
            diffusion_step_embed_dim=embed_dim,
            down_dims=down_dims,
            kernel_size=int(_scalar(fixture, "kernel_size")),
            n_groups=int(_scalar(fixture, "n_groups")),
            cond_predict_scale=bool(_scalar(fixture, "cond_predict_scale")),
            global_condition_embed_dim=embed_dim,
            timestep_embedding_type="clean_diffuser",
            operator_variant="torch",
        )
        variables = _variables_from_fixture(fixture)
        parameter_count = sum(
            leaf.size for leaf in jax.tree.leaves(unfreeze(variables)["params"])
        )
        assert parameter_count == int(_scalar(fixture, "parameter_count"))

        actions = jnp.asarray(fixture["actions"])
        timesteps = jnp.asarray(fixture["timesteps"])
        features = jnp.asarray(fixture["features"])
        cotangent = jnp.asarray(fixture["cotangent"])
        expected_output = np.asarray(fixture["expected_output"])
        expected_vjps = (
            np.asarray(fixture["expected_action_vjp"]),
            np.asarray(fixture["expected_timestep_vjp"]),
            np.asarray(fixture["expected_feature_vjp"]),
        )

    def forward(action_input, timestep_input, feature_input):
        return model.apply(
            variables,
            action_input,
            timestep_input,
            feature_input,
        )

    output, pullback = jax.vjp(forward, actions, timesteps, features)
    actual_vjps = pullback(cotangent)

    np.testing.assert_allclose(output, expected_output, rtol=1e-5, atol=1e-5)
    assert float(jnp.max(jnp.abs(output - expected_output))) <= 1e-5
    for actual, expected in zip(actual_vjps, expected_vjps, strict=True):
        np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-5)
        assert float(jnp.max(jnp.abs(actual - expected))) <= 1e-5
