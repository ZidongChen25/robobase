import numpy as np
import pytest
from flax.errors import InvalidRngError
from gymnasium import spaces

from robobase.method.a2a import A2A
from robobase.method.flow_matching import (
    FlowMatchingBackboneSpec,
    FlowMatchingModelSpec,
    FlowSourceSpec,
)
from robobase.method.legato import Legato
from robobase.models.backbone import build_diffusion_backbone
from robobase.models.backbones.transformer import (
    JaxChiTransformerBackbone,
    _CleanDiffuserPositionalEmbedding,
    _TransformerMLP,
)

jax = pytest.importorskip("jax")
pytest.importorskip("optax")


def _make_agent(
    method_cls,
    flow_source,
    *,
    backbone_type="fully_connected",
    observation_steps=1,
    execution_length=2,
    action_execution_start=1,
):
    observation_space = spaces.Dict(
        {
            "low_dim_state": spaces.Box(
                low=-1.0,
                high=1.0,
                shape=(observation_steps, 3),
                dtype=np.float32,
            )
        }
    )
    action_space = spaces.Box(
        low=-1.0,
        high=1.0,
        shape=(4, 2),
        dtype=np.float32,
    )
    actor_sequence_length = 1 if flow_source.type.startswith("a2a") else 4
    model = FlowMatchingModelSpec(
        backbone=FlowMatchingBackboneSpec(
            type=backbone_type,
            sequence_length=actor_sequence_length,
            diffusion_step_embed_dim=8,
            hidden_dims=(16,),
            d_model=16,
            n_heads=2,
            num_layers=1,
        ),
        encoder_model=None,
        view_fusion_model=None,
    )
    agent = method_cls(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=4,
        num_flow_steps=2,
        model=model,
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=2,
        num_eval_envs=1,
        replay_alpha=0.6,
        replay_beta=0.4,
        frame_stack_on_channel=True,
        actor_grad_clip=None,
        jit=True,
        seed=0,
        use_ema=False,
        flow_source=flow_source,
        execution_length=execution_length,
        action_execution_start=action_execution_start,
    )
    agent.logging = True
    return agent


def _observations():
    return {"low_dim_state": np.zeros((2, 1, 3), dtype=np.float32)}


def _actions():
    return np.linspace(-0.5, 0.5, num=16, dtype=np.float32).reshape(2, 4, 2)


def _assert_finite_update(agent, batch):
    metrics = agent.update(iter([batch]), step=0)

    assert metrics
    assert isinstance(metrics["fm_loss"], float)
    assert all(np.isfinite(np.asarray(value)).all() for value in metrics.values())
    assert all(
        np.isfinite(np.asarray(leaf)).all()
        for leaf in jax.tree_util.tree_leaves(agent.params)
    )


def test_a2a_jit_update_act_appends_history_and_resets_selected_env():
    source = FlowSourceSpec(
        type="a2a",
        history_horizon=3,
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
    )
    agent = _make_agent(A2A, source)
    actions = _actions()
    history = actions[:, :3, ::-1].copy()
    batch = {
        **_observations(),
        "action": actions,
        "action_history": history,
        "action_history_pad_mask": np.array(
            [[False, False, False], [True, False, False]],
            dtype=np.bool_,
        ),
    }

    _assert_finite_update(agent, batch)
    first_actions = agent.act(_observations(), step=0, eval_mode=False)

    assert first_actions.shape == (2, 4, 2)
    assert np.isfinite(first_actions).all()
    expected_first_history = np.concatenate(
        [np.zeros((2, 2, 2), dtype=np.float32), first_actions[:, 1:3]],
        axis=1,
    )
    np.testing.assert_allclose(agent._train_action_history, expected_first_history)
    np.testing.assert_array_equal(
        agent._train_action_history_valid,
        np.array([[False, False, True, True], [False, False, True, True]]),
    )
    source_history, source_valid = agent._rollout_history(False, 2)
    np.testing.assert_allclose(source_history, expected_first_history[:, :3])
    np.testing.assert_array_equal(source_valid, [[False, False, True]] * 2)

    second_actions = agent.act(_observations(), step=1, eval_mode=False)
    expected_second_history = np.concatenate(
        [first_actions[:, 1:3], second_actions[:, 1:3]], axis=1
    )
    np.testing.assert_allclose(agent._train_action_history, expected_second_history)
    np.testing.assert_array_equal(agent._train_action_history_valid, True)

    agent.reset(step=2, agents_to_reset=[0])

    np.testing.assert_allclose(agent._train_action_history[0], 0.0)
    np.testing.assert_array_equal(agent._train_action_history_valid[0], False)
    np.testing.assert_allclose(
        agent._train_action_history[1], expected_second_history[1]
    )
    np.testing.assert_array_equal(agent._train_action_history_valid[1], True)


@pytest.mark.parametrize("missing_field", ["action_history", "action_history_pad_mask"])
@pytest.mark.parametrize("fused", [False, True])
def test_a2a_replay_requires_history_and_padding_mask(missing_field, fused):
    source = FlowSourceSpec(
        type="a2a",
        history_horizon=3,
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
    )
    agent = _make_agent(A2A, source)
    batch = {
        **_observations(),
        "action": _actions(),
        "action_history": _actions()[:, :3],
        "action_history_pad_mask": np.zeros((2, 3), dtype=np.bool_),
    }
    del batch[missing_field]

    with pytest.raises(ValueError, match=missing_field):
        if fused:
            agent.update_many(iter([batch, batch]), num_updates=2)
        else:
            agent.update(iter([batch]), step=0)


@pytest.mark.parametrize("reference_padding", ["last", "zero"])
def test_legato_jit_update_act_shifts_pads_and_resets_previous_chunk(
    reference_padding,
):
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=1,
        ramp_max_steps=1,
        schedule_profiles=("linear",),
        eval_delay_steps=1,
        eval_ramp_steps=1,
        eval_schedule_profile="linear",
        reference_padding=reference_padding,
    )
    agent = _make_agent(Legato, source, action_execution_start=0)
    batch = {**_observations(), "action": _actions()}

    _assert_finite_update(agent, batch)
    sampled_actions = agent.act(_observations(), step=0, eval_mode=False)

    assert sampled_actions.shape == (2, 4, 2)
    assert np.isfinite(sampled_actions).all()
    np.testing.assert_allclose(agent._train_previous_chunk, sampled_actions)
    np.testing.assert_array_equal(agent._train_previous_chunk_valid, True)

    reference, reference_valid = agent._legato_reference(
        eval_mode=False,
        batch_size=2,
    )
    if reference_padding == "last":
        padding = np.repeat(sampled_actions[:, -1:], repeats=2, axis=1)
        expected_valid = np.ones((2, 4), dtype=np.bool_)
    else:
        padding = np.zeros((2, 2, 2), dtype=np.float32)
        expected_valid = np.array(
            [[True, True, False, False], [True, True, False, False]],
            dtype=np.bool_,
        )
    expected_reference = np.concatenate(
        [sampled_actions[:, 2:], padding],
        axis=1,
    )
    np.testing.assert_allclose(reference, expected_reference)
    np.testing.assert_array_equal(reference_valid, expected_valid)
    assert float(agent._eval_legato_omega(reference_valid).sum()) > 0.0

    agent.reset(step=1, agents_to_reset=[0])
    reset_reference, reset_valid = agent._legato_reference(
        eval_mode=False,
        batch_size=2,
    )

    np.testing.assert_allclose(reset_reference[0], 0.0)
    np.testing.assert_array_equal(reset_valid[0], False)
    np.testing.assert_allclose(reset_reference[1], expected_reference[1])
    np.testing.assert_array_equal(reset_valid[1], expected_valid[1])


def test_legato_issues_old_delay_prefix_but_keeps_new_raw_chunk():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=0,
        ramp_max_steps=0,
        eval_delay_steps=1,
        eval_ramp_steps=0,
    )
    agent = _make_agent(Legato, source)
    generated = jax.numpy.zeros((2, 4, 2), dtype=jax.numpy.float32)
    reference = jax.numpy.ones_like(generated)
    reference_valid = jax.numpy.asarray(
        [[True, True, True, True], [True, False, True, True]]
    )

    issued = agent._legato_issued_chunk(generated, reference, reference_valid)
    agent._store_previous_chunk(generated, eval_mode=False)

    np.testing.assert_allclose(issued[0, 0], 0.0)
    np.testing.assert_allclose(issued[0, 1], 1.0)
    np.testing.assert_allclose(issued[0, 2:], 0.0)
    np.testing.assert_allclose(issued[1], 0.0)
    np.testing.assert_allclose(
        issued[:, 1:3],
        np.array(
            [
                [[1.0, 1.0], [0.0, 0.0]],
                [[0.0, 0.0], [0.0, 0.0]],
            ],
            dtype=np.float32,
        ),
    )
    np.testing.assert_allclose(agent._train_previous_chunk, generated)


def test_a2a_noise_eval_keys_are_seed_aligned_across_batching():
    source = FlowSourceSpec(
        type="a2a_noise",
        history_horizon=3,
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
        eval_history_noise_std=0.02,
    )
    agent = _make_agent(A2A, source)
    history = jax.numpy.zeros((2, 3, 2), dtype=jax.numpy.float32)
    pad_mask = jax.numpy.zeros((2, 3), dtype=jax.numpy.bool_)

    batched_keys = agent._aligned_eval_keys([7, 11])
    batched = agent._perturb_a2a_history(history, pad_mask, batched_keys, 0.02)
    agent.reset_aligned_eval_noise()
    serial = jax.numpy.concatenate(
        [
            agent._perturb_a2a_history(
                history[index : index + 1],
                pad_mask[index : index + 1],
                agent._aligned_eval_keys([seed]),
                0.02,
            )
            for index, seed in enumerate((7, 11))
        ],
        axis=0,
    )

    np.testing.assert_allclose(batched, serial)


def test_legato_rejects_schedule_bounds_that_exceed_horizon():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=2,
        delay_max_steps=2,
        ramp_min_steps=2,
        ramp_max_steps=2,
        eval_delay_steps=1,
        eval_ramp_steps=1,
    )

    with pytest.raises(ValueError, match="action_execution_start"):
        _make_agent(Legato, source)


def test_legato_rejects_guidance_without_previous_chunk_overlap():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=0,
        ramp_max_steps=0,
        eval_delay_steps=1,
        eval_ramp_steps=0,
    )

    with pytest.raises(ValueError, match="previous chunk overlap"):
        _make_agent(
            Legato,
            source,
            execution_length=4,
            action_execution_start=0,
        )


def test_legato_rejects_eval_guidance_outside_previous_chunk_overlap():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=0,
        delay_max_steps=0,
        ramp_min_steps=0,
        ramp_max_steps=0,
        eval_delay_steps=1,
        eval_ramp_steps=0,
    )

    with pytest.raises(ValueError, match="previous chunk overlap"):
        _make_agent(
            Legato,
            source,
            execution_length=4,
            action_execution_start=0,
        )


def test_legato_eval_schedule_starts_at_action_execution_offset():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=0,
        ramp_max_steps=0,
        schedule_profiles=("hard",),
        eval_delay_steps=1,
        eval_ramp_steps=0,
        eval_schedule_profile="hard",
    )
    agent = _make_agent(Legato, source)

    omega = agent._eval_legato_omega(jax.numpy.ones((2, 4), dtype=jax.numpy.bool_))

    np.testing.assert_array_equal(
        omega[..., 0],
        np.array([[0.0, 1.0, 0.0, 0.0]] * 2, dtype=np.float32),
    )


def test_legato_no_guidance_training_probability_covers_bootstrap_path():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=1,
        ramp_max_steps=1,
        no_guidance_probability=1.0,
        eval_delay_steps=1,
        eval_ramp_steps=1,
    )
    agent = _make_agent(Legato, source, action_execution_start=0)
    keys = jax.random.split(jax.random.PRNGKey(19), 5)

    omega = agent._sample_legato_schedule(*keys, batch_size=8)

    np.testing.assert_array_equal(omega, 0.0)


def test_legato_records_prefix_and_guided_overlap_error():
    source = FlowSourceSpec(
        type="legato",
        delay_min_steps=1,
        delay_max_steps=1,
        ramp_min_steps=1,
        ramp_max_steps=1,
        eval_delay_steps=1,
        eval_ramp_steps=1,
    )
    agent = _make_agent(Legato, source, action_execution_start=0)
    generated = jax.numpy.zeros((2, 4, 2), dtype=jax.numpy.float32)
    reference = jax.numpy.ones_like(generated)
    reference_valid = jax.numpy.asarray(
        [[True, True, False, False], [False, False, False, False]]
    )

    agent._record_legato_metrics(
        generated,
        reference,
        reference_valid,
        eval_mode=True,
    )

    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["legato_prefix_rmse"] == pytest.approx(1.0)
    assert diagnostics["legato_overlap_rmse"] == pytest.approx(1.0)


def test_a2a_transformer_automatically_attends_to_all_observation_tokens():
    source = FlowSourceSpec(
        type="a2a",
        history_horizon=3,
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
    )

    agent = _make_agent(
        A2A,
        source,
        backbone_type="transformer",
        observation_steps=2,
    )

    assert agent.model_spec.backbone.full_memory_attention is True
    assert agent.actor_model.full_memory_attention is True


def test_a2a_rejects_multiscale_unet_before_backbone_construction():
    source = FlowSourceSpec(
        type="a2a",
        history_horizon=3,
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
    )

    with pytest.raises(NotImplementedError, match="multi-scale UNet"):
        _make_agent(A2A, source, backbone_type="unet1d")


def test_transformer_full_memory_attention_uses_later_condition_tokens():
    kwargs = dict(
        action_dim=2,
        sequence_length=1,
        condition_dim=3,
        d_model=16,
        n_heads=2,
        num_layers=1,
        n_cond_layers=0,
        dropout=0.0,
    )
    full_model = JaxChiTransformerBackbone(
        **kwargs,
        full_memory_attention=True,
    )
    causal_model = JaxChiTransformerBackbone(
        **kwargs,
        full_memory_attention=False,
    )
    actions = jax.numpy.zeros((1, 1, 2), dtype=jax.numpy.float32)
    timesteps = jax.numpy.zeros((1,), dtype=jax.numpy.float32)
    condition = jax.numpy.zeros((1, 2, 3), dtype=jax.numpy.float32)
    changed_condition = condition.at[:, 1].set(100.0)
    params = full_model.init(
        jax.random.PRNGKey(3),
        actions,
        timesteps,
        condition,
    )

    causal_before = causal_model.apply(params, actions, timesteps, condition)
    causal_after = causal_model.apply(params, actions, timesteps, changed_condition)
    full_before = full_model.apply(params, actions, timesteps, condition)
    full_after = full_model.apply(params, actions, timesteps, changed_condition)

    np.testing.assert_array_equal(causal_before, causal_after)
    assert np.max(np.abs(np.asarray(full_after - full_before))) > 1e-6


def test_transformer_operator_variant_preserves_legacy_gelu_semantics():
    inputs = jax.numpy.asarray(
        [[[-3.0, -0.5, 0.5, 3.0]]],
        dtype=jax.numpy.float32,
    )
    legacy = _TransformerMLP(
        d_model=4,
        dropout=0.0,
        operator_variant="legacy",
    )
    torch_variant = _TransformerMLP(
        d_model=4,
        dropout=0.0,
        operator_variant="torch",
    )
    variables = legacy.init(jax.random.PRNGKey(29), inputs, train=False)
    params = variables["params"]
    hidden = (
        inputs @ params["Dense_0"]["kernel"]
        + params["Dense_0"]["bias"]
    )
    expected = (
        jax.nn.gelu(hidden, approximate=True)
        @ params["Dense_1"]["kernel"]
        + params["Dense_1"]["bias"]
    )

    legacy_output = legacy.apply(variables, inputs, train=False)
    torch_output = torch_variant.apply(variables, inputs, train=False)

    np.testing.assert_allclose(legacy_output, expected, rtol=0.0, atol=0.0)
    assert np.max(np.abs(np.asarray(legacy_output - torch_output))) > 0.0


def test_transformer_torch_positional_embedding_matches_reference_formula():
    timesteps = jax.numpy.asarray([0.0, 1.0, 17.0], dtype=jax.numpy.float32)
    module = _CleanDiffuserPositionalEmbedding(dim=8, operator_variant="torch")
    output = module.apply({}, timesteps)
    half_dim = 4
    frequencies = np.exp(
        np.arange(half_dim, dtype=np.float32)
        * -(np.log(10000.0) / (half_dim - 1))
    )
    angles = np.asarray(timesteps)[:, None] * frequencies[None, :]
    expected = np.concatenate([np.sin(angles), np.cos(angles)], axis=-1)

    np.testing.assert_allclose(output, expected, rtol=1e-6, atol=1e-7)


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"dropout": 1.0}, "dropout must lie"),
        ({"embedding_dropout": -0.1}, "embedding_dropout must lie"),
        ({"d_model": 10, "n_heads": 4}, "divide d_model"),
    ],
)
def test_transformer_rejects_invalid_static_configuration(overrides, message):
    values = dict(
        type="transformer",
        sequence_length=2,
        d_model=8,
        n_heads=2,
        num_layers=1,
        n_cond_layers=0,
    )
    values.update(overrides)
    spec = FlowMatchingBackboneSpec(**values)

    with pytest.raises(ValueError, match=message):
        build_diffusion_backbone(
            spec,
            action_dim=2,
            sequence_length=2,
            condition_dim=3,
        )


def test_transformer_embedding_dropout_is_torch_only():
    kwargs = dict(
        action_dim=2,
        sequence_length=2,
        condition_dim=3,
        d_model=8,
        n_heads=2,
        num_layers=0,
        n_cond_layers=0,
        dropout=0.0,
        embedding_dropout=0.5,
    )
    actions = jax.numpy.zeros((1, 2, 2), dtype=jax.numpy.float32)
    timesteps = jax.numpy.zeros((1,), dtype=jax.numpy.float32)
    condition = jax.numpy.zeros((1, 2, 3), dtype=jax.numpy.float32)
    legacy = JaxChiTransformerBackbone(**kwargs, operator_variant="legacy")
    torch_variant = JaxChiTransformerBackbone(**kwargs, operator_variant="torch")
    variables = legacy.init(
        jax.random.PRNGKey(30), actions, timesteps, condition, train=False
    )

    legacy.apply(variables, actions, timesteps, condition, train=True)
    with pytest.raises(InvalidRngError):
        torch_variant.apply(variables, actions, timesteps, condition, train=True)


def test_transformer_operator_variant_changes_layer_norm_epsilon():
    kwargs = dict(
        action_dim=2,
        sequence_length=2,
        condition_dim=3,
        d_model=8,
        n_heads=2,
        num_layers=1,
        n_cond_layers=0,
        dropout=0.0,
    )
    legacy = JaxChiTransformerBackbone(**kwargs, operator_variant="legacy")
    torch_variant = JaxChiTransformerBackbone(**kwargs, operator_variant="torch")
    actions = jax.numpy.full((1, 2, 2), 1.0e-4, dtype=jax.numpy.float32)
    timesteps = jax.numpy.zeros((1,), dtype=jax.numpy.float32)
    condition = jax.numpy.full((1, 2, 3), 1.0e-4, dtype=jax.numpy.float32)
    variables = legacy.init(
        jax.random.PRNGKey(31),
        actions,
        timesteps,
        condition,
    )

    legacy_output = legacy.apply(variables, actions, timesteps, condition)
    torch_output = torch_variant.apply(variables, actions, timesteps, condition)

    assert np.max(np.abs(np.asarray(legacy_output - torch_output))) > 0.0


@pytest.mark.parametrize("method_cls", [A2A, Legato])
def test_flow_extensions_fuse_multiple_updates_with_lax_scan(method_cls):
    if method_cls is A2A:
        source = FlowSourceSpec(
            type="a2a",
            history_horizon=3,
            latent_dim=8,
            hidden_dim=8,
            encoder_layers=1,
            decoder_layers=1,
            kernel_size=3,
        )
        batch = {
            **_observations(),
            "action": _actions(),
            "action_history": _actions()[:, :3],
            "action_history_pad_mask": np.zeros((2, 3), dtype=np.bool_),
        }
    else:
        source = FlowSourceSpec(
            type="legato",
            delay_min_steps=1,
            delay_max_steps=1,
            ramp_min_steps=1,
            ramp_max_steps=1,
            eval_delay_steps=1,
            eval_ramp_steps=1,
        )
        batch = {**_observations(), "action": _actions()}
    agent = _make_agent(
        method_cls,
        source,
        action_execution_start=0 if method_cls is Legato else 1,
    )

    metrics = agent.update_many(iter([batch, batch]), num_updates=2)

    assert np.isfinite(metrics["actor_loss"])
    assert np.isfinite(metrics["fm_loss"])


def test_a2a_feedback_rollout_cold_start_repeats_current_measured_state():
    source = FlowSourceSpec(
        type="a2a",
        history_horizon=3,
        history_padding="edge",
        history_source="executed_action_feedback",
        latent_dim=8,
        hidden_dim=8,
        encoder_layers=1,
        decoder_layers=1,
        kernel_size=3,
    )
    agent = _make_agent(A2A, source, action_execution_start=0)
    feedback = np.asarray([[1.0, 2.0]], dtype=np.float32)

    agent._append_feedback_history(
        {"executed_action_feedback": feedback}, eval_mode=True
    )
    history, valid = agent._rollout_history(eval_mode=True, batch_size=1)

    np.testing.assert_array_equal(
        np.asarray(history[0]), np.repeat(feedback, 3, axis=0)
    )
    np.testing.assert_array_equal(np.asarray(valid[0]), [True, True, True])
