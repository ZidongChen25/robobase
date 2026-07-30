import jax
import jax.numpy as jnp

from robobase.models.backbones.unet1d import JaxConditionalUnet1D


def test_unet_supports_legato_schedule_channel_without_changing_output_dim():
    model = JaxConditionalUnet1D(
        action_dim=3,
        input_action_dim=4,
        sequence_length=8,
        feature_dim=5,
        diffusion_step_embed_dim=16,
        down_dims=(16, 32),
        kernel_size=3,
        n_groups=4,
    )
    actions_and_schedule = jnp.zeros((2, 8, 4), dtype=jnp.float32)
    time = jnp.zeros((2,), dtype=jnp.float32)
    condition = jnp.zeros((2, 5), dtype=jnp.float32)

    variables = model.init(jax.random.PRNGKey(0), actions_and_schedule, time, condition)
    output = jax.jit(model.apply)(variables, actions_and_schedule, time, condition)

    assert output.shape == (2, 8, 3)
    assert jnp.isfinite(output).all()
