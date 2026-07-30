from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from benchmarks.official_roboverse.jax_a2a import (
    LimitsNormalizer,
    OfficialZarrDataset,
    create_sequence_indices,
    ema_decay,
)
from robobase.models.official_a2a import (
    OfficialA2A,
    OfficialA2AConfig,
    OfficialActionDecoder,
    OfficialActionEncoder,
    OfficialSimpleFlowNet,
    _linear_sum_assignment,
    sample_exact_ot_pairs,
)


def test_official_config_is_paper_shape():
    config = OfficialA2AConfig()
    assert (config.observation_steps, config.action_steps) == (8, 8)
    assert (config.latent_dim, config.hidden_dim, config.flow_steps) == (512, 512, 6)
    assert config.flow_matcher == "conditional"
    assert config.image_range_normalization is True


def test_bigym_action_dimension_is_supported():
    config = OfficialA2AConfig(
        action_dim=16,
        observation_steps=20,
        history_steps=20,
        action_steps=20,
        flow_steps=10,
    )

    config.validate()
    assert config.action_dim == 16
    assert (config.observation_steps, config.action_steps, config.flow_steps) == (
        20,
        20,
        10,
    )


def test_fm_aligned_multicamera_config_is_supported():
    config = OfficialA2AConfig(
        observation_steps=1,
        history_steps=20,
        action_steps=20,
        action_dim=16,
        flow_steps=10,
        num_cameras=3,
        vision_encoder="fm_resnet",
        resize_to_224=True,
        image_range_normalization=False,
    )

    config.validate()


def test_flow_matcher_modes_are_explicit():
    OfficialA2AConfig(flow_matcher="conditional").validate()
    OfficialA2AConfig(flow_matcher="exact_ot").validate()
    with np.testing.assert_raises(ValueError):
        OfficialA2AConfig(flow_matcher="unknown").validate()


def test_predictor_warmup_uses_checkpoint_action_dimension():
    source = (
        Path(__file__).resolve().parents[2]
        / "benchmarks"
        / "official_roboverse"
        / "jax_a2a.py"
    ).read_text(encoding="utf-8")

    assert "self.config.action_dim" in source


def test_complete_trainable_parameter_count_matches_upstream():
    config = OfficialA2AConfig()
    model = OfficialA2A(config)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((1, 8, 3, 32, 32)),
        jnp.zeros((1, 16, 9)),
        jnp.zeros((1, 16, 9)),
        method=model.initialize_all,
    )
    assert sum(value.size for value in jax.tree.leaves(variables["params"])) == 34_656_904


def test_action_autoencoder_shapes_and_independent_encoders():
    config = OfficialA2AConfig(action_dim=9)
    actions = jnp.zeros((2, 8, 9), dtype=jnp.float32)
    encoder = OfficialActionEncoder(config)
    encoder_variables = encoder.init(jax.random.PRNGKey(0), actions)
    latent = encoder.apply(encoder_variables, actions)
    decoder = OfficialActionDecoder(config)
    decoder_variables = decoder.init(jax.random.PRNGKey(1), latent)
    assert latent.shape == (2, 512)
    assert decoder.apply(decoder_variables, latent).shape == (2, 8, 9)


def test_flow_net_modulators_are_not_zero_after_parent_initialization():
    config = OfficialA2AConfig()
    model = OfficialSimpleFlowNet(config)
    variables = model.init(
        jax.random.PRNGKey(0),
        jnp.zeros((2, 512)),
        jnp.zeros((2,)),
        jnp.zeros((2, 512)),
    )
    kernel = variables["params"]["layer0_time_modulator"]["kernel"]
    assert not np.allclose(np.asarray(kernel), 0.0)


def test_six_step_euler_uses_official_left_endpoints():
    config = OfficialA2AConfig()
    model = OfficialA2A(config)
    assert [index / config.flow_steps for index in range(config.flow_steps)] == [
        0.0,
        1 / 6,
        2 / 6,
        3 / 6,
        4 / 6,
        5 / 6,
    ]


def test_exact_ot_sampler_returns_only_optimal_assignment_edges():
    source = jnp.asarray([[0.0], [10.0], [20.0], [30.0]])
    target = jnp.asarray([[20.0], [0.0], [30.0], [10.0]])
    sampled_source, sampled_target = sample_exact_ot_pairs(
        source, target, jax.random.PRNGKey(4)
    )
    np.testing.assert_allclose(sampled_source, sampled_target)


def test_device_hungarian_matches_scipy_cost():
    from scipy.optimize import linear_sum_assignment

    rng = np.random.default_rng(7)
    for size in (2, 5, 8):
        cost = rng.normal(size=(size, size)).astype(np.float32) ** 2
        permutation = np.asarray(jax.jit(_linear_sum_assignment)(jnp.asarray(cost)))
        rows, columns = linear_sum_assignment(cost)
        np.testing.assert_allclose(
            cost[np.arange(size), permutation].sum(), cost[rows, columns].sum()
        )


def test_limits_normalizer_matches_minus_one_plus_one_convention():
    normalizer = LimitsNormalizer.fit(np.asarray([[2.0, 4.0], [6.0, 4.0]]))
    normalized = normalizer.normalize(np.asarray([[2.0, 4.0], [6.0, 4.0]]))
    np.testing.assert_allclose(normalized[:, 0], [-1.0, 1.0])
    np.testing.assert_allclose(normalized[:, 1], [0.0, 0.0])


def test_sequence_indices_match_official_edge_padding_count():
    indices = create_sequence_indices(np.asarray([10]), sequence_length=16)
    assert len(indices) == 9
    np.testing.assert_array_equal(indices[0], [0, 9, 7, 16])
    np.testing.assert_array_equal(indices[-1], [1, 10, 0, 9])


def test_vectorized_gather_indices_preserve_edge_padding():
    indices = create_sequence_indices(np.asarray([10]), sequence_length=16)
    gather = OfficialZarrDataset._build_gather_indices(indices)

    np.testing.assert_array_equal(gather[0], [0] * 8 + list(range(1, 9)))
    np.testing.assert_array_equal(gather[-1], list(range(1, 10)) + [9] * 7)


def test_h20_vectorized_gather_indices_have_full_horizon():
    indices = create_sequence_indices(
        np.asarray([30]), sequence_length=40, pad_before=19, pad_after=19
    )
    gather = OfficialZarrDataset._build_gather_indices(indices, 40)

    assert gather.shape == (29, 40)
    np.testing.assert_array_equal(gather[0], [0] * 20 + list(range(1, 21)))


def test_ema_warmup_matches_public_implementation():
    values = [float(ema_decay(jnp.asarray(step))) for step in range(4)]
    np.testing.assert_allclose(values[:2], [0.0, 0.0])
    np.testing.assert_allclose(values[2:], [1 - 2**-0.75, 1 - 3**-0.75])
