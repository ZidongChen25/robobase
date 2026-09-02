"""Bit-equivalence tests for the memory-oriented pixel-path rewrites.

Covers the scatter-free max-pool backward, the shared-gather elastic warp,
and the uint8-stacked fused update path of ACT / Diffusion / Flow Matching.
"""

from __future__ import annotations

import flax.linen as nn
import jax
import jax.numpy as jnp
import jax.scipy.ndimage as jndi
import numpy as np
import pytest
from gymnasium import spaces

from robobase.models.pooling import max_pool_3x3_stride2_pad1


def _reference_pool(x):
    return nn.max_pool(
        x, window_shape=(3, 3), strides=(2, 2), padding=((1, 1), (1, 1))
    )


@pytest.mark.parametrize("shape", [(2, 9, 9, 3), (3, 16, 16, 4), (1, 7, 10, 2)])
@pytest.mark.parametrize("dtype", [jnp.float32, jnp.bfloat16])
def test_max_pool_backward_matches_select_and_scatter_bitwise(shape, dtype):
    key = jax.random.PRNGKey(0)
    x = jax.random.normal(key, shape, dtype)
    # ReLU-style ties at zero plus duplicated values inside windows.
    x = jnp.where(x > 0.3, x, 0.0)
    x = x.at[:, 1:3, 1:3, :].set(x[:, 1:2, 1:2, :])
    cotangent = jax.random.normal(jax.random.fold_in(key, 1), _reference_pool(x).shape)

    def loss(pool):
        return lambda value: jnp.sum(pool(value) * cotangent)

    out_new, grad_new = jax.value_and_grad(loss(max_pool_3x3_stride2_pad1))(x)
    out_ref, grad_ref = jax.value_and_grad(loss(_reference_pool))(x)
    np.testing.assert_array_equal(np.asarray(out_new), np.asarray(out_ref))
    np.testing.assert_array_equal(np.asarray(grad_new), np.asarray(grad_ref))
    np.testing.assert_array_equal(
        np.asarray(max_pool_3x3_stride2_pad1(x)), np.asarray(_reference_pool(x))
    )


def _reference_elastic_warp(flat, displacement):
    """The previous vmap(map_coordinates) formulation."""

    def warp_one(image):
        channels, height, width = image.shape
        y, x = jnp.meshgrid(
            jnp.arange(height, dtype=jnp.float32),
            jnp.arange(width, dtype=jnp.float32),
            indexing="ij",
        )
        y = jnp.clip(y + displacement[..., 0], 0.0, float(height - 1))
        x = jnp.clip(x + displacement[..., 1], 0.0, float(width - 1))
        c = jnp.broadcast_to(
            jnp.arange(channels, dtype=jnp.float32)[:, None, None], image.shape
        )
        return jndi.map_coordinates(
            image,
            (c, jnp.broadcast_to(y[None], image.shape), jnp.broadcast_to(x[None], image.shape)),
            order=1,
            mode="nearest",
        )

    return jax.vmap(warp_one)(flat)


def test_elastic_warp_matches_map_coordinates_bitwise():
    from robobase.method.act import _elastic_warp_batch

    key = jax.random.PRNGKey(3)
    image_key, disp_key = jax.random.split(key)
    flat = jax.random.uniform(image_key, (5, 3, 17, 23), minval=0.0, maxval=255.0)
    displacement = jax.random.uniform(disp_key, (17, 23, 2), minval=-6.0, maxval=6.0)
    expected = np.asarray(_reference_elastic_warp(flat, displacement))
    actual = np.asarray(jax.jit(_elastic_warp_batch)(flat, displacement))
    np.testing.assert_array_equal(actual, expected)


def test_act_legacy_augmentation_accepts_uint8_and_matches_float_input():
    from robobase.method.act import ACT

    key = jax.random.PRNGKey(7)
    rgb_u8 = np.random.default_rng(0).integers(0, 256, (2, 2, 3, 12, 12), np.uint8)
    out_u8 = np.asarray(ACT._augment_rgb_impl(None, jnp.asarray(rgb_u8), key))
    out_f32 = np.asarray(
        ACT._augment_rgb_impl(None, jnp.asarray(rgb_u8, dtype=jnp.float32), key)
    )
    np.testing.assert_array_equal(out_u8, out_f32)
    assert out_u8.dtype == np.float32


@pytest.mark.parametrize("use_film", [False, True])
def test_bfloat16_resnet_preserves_checkpoint_tree_and_runs(use_film):
    """Changing compute dtype must not rename legacy checkpoint parameters."""

    from robobase.models.encoder import JaxResNetEncoder

    kwargs = {
        "input_shape": (1, 3, 32, 32),
        "model": "resnet18",
        "jit": False,
        "pretrained": False,
        "resize_to_224": False,
        "use_film": use_film,
        "film_task_input_dim": 8,
        "film_task_hidden_dim": 16,
    }
    fp32 = JaxResNetEncoder(**kwargs, compute_dtype="float32")
    bf16 = JaxResNetEncoder(**kwargs, compute_dtype="bfloat16")

    assert jax.tree.structure(fp32.trainable_params) == jax.tree.structure(
        bf16.trainable_params
    )
    assert jax.tree.structure(fp32.batch_stats) == jax.tree.structure(
        bf16.batch_stats
    )

    rgb = jnp.arange(2 * 1 * 3 * 32 * 32, dtype=jnp.uint16).reshape(
        (2, 1, 3, 32, 32)
    ).astype(jnp.uint8)
    task_emb = jnp.ones((2, 8), dtype=jnp.float32) if use_film else None
    output = bf16.apply_trainable(
        fp32.trainable_params,
        rgb,
        task_emb=task_emb,
    )

    assert output.shape == (2, 1, 512)
    assert output.dtype == jnp.float32
    assert np.all(np.isfinite(np.asarray(output)))


def test_resnet_compute_dtype_rejects_unknown_value():
    from robobase.models.encoder import resolve_compute_dtype

    with pytest.raises(ValueError, match="compute_dtype"):
        resolve_compute_dtype("float16")


def _pixel_spaces():
    observation_space = spaces.Dict(
        {
            "rgb_head": spaces.Box(0, 255, (1, 3, 16, 16), np.uint8),
            "rgb_wrist": spaces.Box(0, 255, (1, 3, 16, 16), np.uint8),
            "low_dim_state": spaces.Box(-1.0, 1.0, (1, 4), np.float32),
        }
    )
    action_space = spaces.Box(-1.0, 1.0, (4, 2), np.float32)
    return observation_space, action_space


def _pixel_batches(count, batch_size=2):
    rng = np.random.default_rng(11)
    batches = []
    for _ in range(count):
        batches.append(
            {
                "rgb_head": rng.integers(0, 256, (batch_size, 1, 3, 16, 16), np.uint8),
                "rgb_wrist": rng.integers(0, 256, (batch_size, 1, 3, 16, 16), np.uint8),
                "low_dim_state": rng.standard_normal((batch_size, 1, 4)).astype(np.float32),
                "action": rng.uniform(-1, 1, (batch_size, 4, 2)).astype(np.float32),
                "action_pad_mask": np.zeros((batch_size, 4), dtype=bool),
            }
        )
    return batches


def _leaves(tree):
    return [np.asarray(leaf) for leaf in jax.tree_util.tree_leaves(tree)]


def _assert_trees_equal(a, b):
    for left, right in zip(_leaves(a), _leaves(b), strict=True):
        np.testing.assert_array_equal(left, right)


def _make_diffusion(observation_space, action_space, augmentation):
    from robobase.method.diffusion import Diffusion
    from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
    from robobase.method.diffusion import DiffusionModelSpec
    from robobase.models.backbone import DiffusionBackboneSpec

    backbone = DiffusionBackboneSpec(
        type="unet1d",
        sequence_length=4,
        diffusion_step_embed_dim=8,
        down_dims=(8, 16),
        kernel_size=3,
        n_groups=4,
    )
    encoder = BCEncoderModelSpec(
        type="resnet", model="resnet18", trainable=True, pretrained=False
    )
    fusion = BCViewFusionModelSpec(type="multicam_feature", mode="flatten")
    return Diffusion(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=10,
        num_diffusion_iters=4,
        model=DiffusionModelSpec(backbone, encoder, fusion),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.7,
        replay_beta=0.5,
        frame_stack_on_channel=True,
        jit=False,
        platform="cpu",
        seed=0,
        image_augmentation_type=augmentation,
    )


def _float_batches(batches):
    return [
        {
            key: (value.astype(np.float32) if key.startswith("rgb_") else value)
            for key, value in batch.items()
        }
        for batch in batches
    ]


@pytest.mark.parametrize("augmentation", ["none", "campose_crop"])
def test_diffusion_fused_update_keeps_uint8_and_matches_float_inputs(augmentation):
    """The deferred cast changes where the float conversion happens, not values."""

    observation_space, action_space = _pixel_spaces()
    batches = _pixel_batches(3)

    uint8_agent = _make_diffusion(observation_space, action_space, augmentation)
    float_agent = _make_diffusion(observation_space, action_space, augmentation)
    _assert_trees_equal(uint8_agent.params, float_agent.params)

    obs_inputs = uint8_agent._prepare_trainable_obs_inputs(batches[0])
    assert obs_inputs["rgb"].dtype == jnp.uint8
    assert float_agent._prepare_trainable_obs_inputs(_float_batches(batches)[0])[
        "rgb"
    ].dtype == jnp.float32

    before = _leaves(uint8_agent.params)
    uint8_agent.update_many(iter(batches), num_updates=3)
    float_agent.update_many(iter(_float_batches(batches)), num_updates=3)

    assert uint8_agent._update_step_count == float_agent._update_step_count == 3
    assert any(not np.array_equal(a, b) for a, b in zip(before, _leaves(uint8_agent.params)))
    _assert_trees_equal(uint8_agent.params, float_agent.params)
    _assert_trees_equal(uint8_agent.opt_state, float_agent.opt_state)
    np.testing.assert_array_equal(
        np.asarray(uint8_agent.rng_key), np.asarray(float_agent.rng_key)
    )


def _make_act(observation_space, action_space):
    from robobase.method.act import ACT, ACTActorModelSpec, ACTModelSpec
    from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec

    actor = ACTActorModelSpec(
        type="transformer",
        hidden_dim=16,
        enc_layers=1,
        dec_layers=1,
        dim_feedforward=32,
        dropout=0.1,
        nheads=2,
        num_queries=int(action_space.shape[0]),
        pre_norm=False,
        latent_dim=4,
        data_augmentation=True,
    )
    encoder = BCEncoderModelSpec(
        type="resnet", model="resnet18", trainable=True, pretrained=False
    )
    fusion = BCViewFusionModelSpec(type="multicam_feature", mode="flatten")
    return ACT(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=10,
        model=ACTModelSpec(actor, encoder, fusion),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.7,
        replay_beta=0.5,
        frame_stack_on_channel=True,
        jit=False,
        platform="cpu",
        seed=0,
    )


def test_act_fused_update_keeps_uint8_and_matches_float_inputs():
    observation_space, action_space = _pixel_spaces()
    batches = _pixel_batches(2)

    uint8_agent = _make_act(observation_space, action_space)
    float_agent = _make_act(observation_space, action_space)
    _assert_trees_equal(uint8_agent.params, float_agent.params)
    assert uint8_agent._prepare_trainable_obs_inputs(batches[0])["rgb"].dtype == jnp.uint8

    uint8_agent.update_many(iter(batches), num_updates=2)
    float_agent.update_many(iter(_float_batches(batches)), num_updates=2)
    _assert_trees_equal(uint8_agent.params, float_agent.params)

    # The single-step program augments in-program as well.
    uint8_agent.update(iter(batches), 0)
    float_agent.update(iter(_float_batches(batches)), 0)
    _assert_trees_equal(uint8_agent.params, float_agent.params)
    np.testing.assert_array_equal(
        np.asarray(uint8_agent.rng_key), np.asarray(float_agent.rng_key)
    )


def _make_flow_matching(observation_space, action_space, augmentation):
    from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec
    from robobase.method.flow_matching import (
        FlowMatching,
        FlowMatchingBackboneSpec,
        FlowMatchingModelSpec,
    )

    backbone = FlowMatchingBackboneSpec(
        type="fully_connected",
        sequence_length=4,
        diffusion_step_embed_dim=8,
        hidden_dims=(16,),
    )
    encoder = BCEncoderModelSpec(
        type="resnet", model="resnet18", trainable=True, pretrained=False
    )
    fusion = BCViewFusionModelSpec(type="multicam_feature", mode="flatten")
    return FlowMatching(
        lr=1e-3,
        adaptive_lr=False,
        num_train_steps=10,
        num_flow_steps=2,
        model=FlowMatchingModelSpec(backbone, encoder, fusion),
        observation_space=observation_space,
        action_space=action_space,
        num_train_envs=1,
        num_eval_envs=1,
        replay_alpha=0.7,
        replay_beta=0.5,
        frame_stack_on_channel=True,
        jit=False,
        platform="cpu",
        seed=0,
        image_augmentation_type=augmentation,
    )


@pytest.mark.parametrize("augmentation", ["none", "campose_crop"])
def test_flow_matching_fused_update_keeps_uint8_and_matches_float_inputs(
    augmentation,
):
    observation_space, action_space = _pixel_spaces()
    batches = _pixel_batches(3)

    uint8_agent = _make_flow_matching(
        observation_space, action_space, augmentation
    )
    float_agent = _make_flow_matching(
        observation_space, action_space, augmentation
    )
    _assert_trees_equal(uint8_agent.params, float_agent.params)
    assert uint8_agent._prepare_trainable_obs_inputs(batches[0])["rgb"].dtype == jnp.uint8

    uint8_agent.update_many(iter(batches), num_updates=3)
    float_agent.update_many(iter(_float_batches(batches)), num_updates=3)

    assert uint8_agent._update_step_count == float_agent._update_step_count == 3
    _assert_trees_equal(uint8_agent.params, float_agent.params)
    _assert_trees_equal(uint8_agent.opt_state, float_agent.opt_state)
    np.testing.assert_array_equal(
        np.asarray(uint8_agent.rng_key), np.asarray(float_agent.rng_key)
    )


@pytest.mark.parametrize("use_film", [False, True])
def test_resnet_bfloat16_compute_dtype_shares_variables_and_tracks_float32(use_film):
    from robobase.models.encoder import JaxResNetEncoder

    kwargs = dict(
        input_shape=(2, 3, 32, 32),
        model="resnet18",
        jit=False,
        pretrained=True,
        resize_to_224=False,
        use_film=use_film,
        film_task_input_dim=8,
        film_task_hidden_dim=16,
    )
    float_encoder = JaxResNetEncoder(compute_dtype="float32", **kwargs)
    bf16_encoder = JaxResNetEncoder(compute_dtype="bfloat16", **kwargs)
    # Same checkpoint layout: the dtype only changes the compute path.
    assert jax.tree_util.tree_structure(
        float_encoder.trainable_params
    ) == jax.tree_util.tree_structure(bf16_encoder.trainable_params)

    rgb = jnp.asarray(
        np.random.default_rng(5).integers(0, 256, (2, 2, 3, 32, 32), np.uint8)
    )
    task = {"task_emb": jnp.ones((2, 8))} if use_film else {}

    def spatial(encoder):
        out = encoder.apply_trainable_spatial(encoder.trainable_params, rgb, **task)
        return out[0] if isinstance(out, tuple) else out

    reference = spatial(float_encoder)
    narrowed = spatial(bf16_encoder)
    assert narrowed.dtype == jnp.float32
    assert narrowed.shape == reference.shape
    relative_error = float(
        jnp.linalg.norm(narrowed - reference) / jnp.linalg.norm(reference)
    )
    assert relative_error < 0.05, relative_error

    grads = jax.grad(lambda params: jnp.sum(spatial_with(bf16_encoder, params) ** 2))(
        bf16_encoder.trainable_params
    )
    leaves = jax.tree_util.tree_leaves(grads)
    assert all(leaf.dtype == jnp.float32 for leaf in leaves)
    assert all(bool(jnp.all(jnp.isfinite(leaf))) for leaf in leaves)


def spatial_with(encoder, params):
    rgb = jnp.asarray(
        np.random.default_rng(5).integers(0, 256, (2, 2, 3, 32, 32), np.uint8)
    )
    task = {"task_emb": jnp.ones((2, 8))} if encoder._use_film else {}
    out = encoder.apply_trainable_spatial(params, rgb, **task)
    return out[0] if isinstance(out, tuple) else out


def test_encoder_rejects_unknown_compute_dtype():
    from robobase.models.encoder import JaxResNetEncoder

    with pytest.raises(ValueError, match="compute_dtype"):
        JaxResNetEncoder(
            input_shape=(1, 3, 32, 32),
            model="resnet18",
            jit=False,
            pretrained=False,
            compute_dtype="float16",
        )


def test_act_xla_attention_backend_shares_parameters_and_matches_flax():
    observation_space, action_space = _pixel_spaces()
    batch = _pixel_batches(1)[0]

    def make(attention_impl):
        from robobase.method.act import ACT, ACTActorModelSpec, ACTModelSpec
        from robobase.method.bc import BCEncoderModelSpec, BCViewFusionModelSpec

        actor = ACTActorModelSpec(
            type="transformer",
            hidden_dim=16,
            enc_layers=1,
            dec_layers=1,
            dim_feedforward=32,
            dropout=0.1,
            nheads=2,
            num_queries=int(action_space.shape[0]),
            pre_norm=False,
            latent_dim=4,
            data_augmentation=False,
            attention_impl=attention_impl,
        )
        encoder = BCEncoderModelSpec(
            type="resnet", model="resnet18", trainable=True, pretrained=False
        )
        fusion = BCViewFusionModelSpec(type="multicam_feature", mode="flatten")
        return ACT(
            lr=1e-3,
            adaptive_lr=False,
            num_train_steps=10,
            model=ACTModelSpec(actor, encoder, fusion),
            observation_space=observation_space,
            action_space=action_space,
            num_train_envs=1,
            num_eval_envs=1,
            replay_alpha=0.7,
            replay_beta=0.5,
            frame_stack_on_channel=True,
            jit=False,
            platform="cpu",
            seed=0,
        )

    flax_agent = make("flax")
    xla_agent = make("xla")
    assert jax.tree_util.tree_structure(flax_agent.params) == jax.tree_util.tree_structure(
        xla_agent.params
    )
    _assert_trees_equal(flax_agent.params, xla_agent.params)
    reference = flax_agent.act(batch, 0, True)
    candidate = xla_agent.act(batch, 0, True)
    np.testing.assert_allclose(candidate, reference, rtol=1e-5, atol=1e-5)
    xla_agent.update(iter([batch]), 0)

    with pytest.raises(ValueError, match="attention_impl"):
        make("triton")
