from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from benchmarks.official_bigym.legato_adapter import (
    DelayState,
    OfficialBigymPolicy,
    OfficialPolicyConfig,
    merge_delayed_execution,
    shift_generated_chunk,
)
from benchmarks.official_bigym.legato_checkpoint import (
    load_checkpoint,
    save_checkpoint,
    warm_start_legato_from_vanilla,
)
from benchmarks.official_bigym.legato_data import (
    FeatureEpisode,
    MinMaxActionTransform,
    build_window_dataset,
)
from benchmarks.official_bigym.legato_features import FrozenFMVisualFeatures
from benchmarks.official_bigym.legato_eval import (
    EpisodeAudit,
    WorkspaceOfficialPolicyAgent,
    _validate_episode_audit,
    evaluate_episode,
)
from benchmarks.official_bigym.legato_train import OfficialTrainer, TrainConfig
from benchmarks.official_bigym.legato_upstream import (
    UPSTREAM_COMMIT,
    checkout_commit,
)


def _config() -> OfficialPolicyConfig:
    return OfficialPolicyConfig(
        action_horizon=4,
        execute_horizon=2,
        inference_delay=1,
        num_flow_steps=2,
        channel_dim=16,
        channel_hidden_dim=32,
        token_hidden_dim=8,
        num_layers=1,
        warmup_max=2,
    )


def test_upstream_checkout_is_pinned():
    assert checkout_commit() == UPSTREAM_COMMIT


def test_bigym_windows_preserve_alignment_and_episode_boundaries():
    first = FeatureEpisode(
        np.arange(12, dtype=np.float32).reshape(6, 2),
        np.arange(18, dtype=np.float32).reshape(6, 3),
        "first",
    )
    second = FeatureEpisode(
        np.full((5, 2), 100, dtype=np.float32),
        np.full((5, 3), 200, dtype=np.float32),
        "second",
    )
    dataset = build_window_dataset([first, second], horizon=4)
    assert len(dataset) == 5
    np.testing.assert_array_equal(dataset.features[1], first.features[1])
    np.testing.assert_array_equal(dataset.action_chunks[1], first.actions[1:5])
    assert dataset.episode_index.tolist() == [0, 0, 0, 1, 1]


def test_action_transform_matches_tanh_minmax_round_trip():
    transform = MinMaxActionTransform(
        np.array([-2.0, 0.0], np.float32),
        np.array([2.0, 4.0], np.float32),
        margin=0.1,
    )
    raw = np.array([[-2.2, 0.0], [2.2, 4.4]], np.float32)
    normalized = transform.normalize(raw)
    np.testing.assert_allclose(normalized, [[-1, -1], [1, 1]], atol=1e-6)
    np.testing.assert_allclose(transform.denormalize(normalized), raw, atol=1e-6)


def test_delay_state_matches_official_evaluator_semantics():
    previous = jnp.arange(8, dtype=jnp.float32).reshape(1, 4, 2)
    generated = previous + 100
    executed = merge_delayed_execution(
        previous, generated, inference_delay=1, execute_horizon=2
    )
    np.testing.assert_array_equal(executed[0, 0], previous[0, 0])
    np.testing.assert_array_equal(executed[0, 1], generated[0, 1])
    shifted = shift_generated_chunk(generated, 2)
    np.testing.assert_array_equal(shifted[:, :2], generated[:, 2:])
    np.testing.assert_array_equal(shifted[:, 2:], 0)


@pytest.mark.parametrize("mode", ["vanilla", "rtc", "legato"])
def test_all_official_modes_have_one_rollout_protocol(mode):
    adapter = OfficialBigymPolicy(
        mode=mode,
        obs_dim=5,
        action_dim=3,
        config=_config(),
        seed=0,
    )
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    actions = jnp.zeros((2, 4, 3), dtype=jnp.float32)
    assert jnp.isfinite(adapter.training_loss(jax.random.key(1), features, actions))
    state = adapter.bootstrap(jax.random.key(2), features)
    prediction = adapter.predict(jax.random.key(3), features, state)
    assert prediction.execute_actions.shape == (2, 2, 3)
    assert prediction.generated_chunk.shape == (2, 4, 3)
    assert bool(jnp.isfinite(prediction.generated_chunk).all())


def test_visual_feature_boundary_requires_and_flattens_pixels():
    class FakeAgent:
        use_pixels = True
        _condition_as_local = False
        _trainable_encoder = False
        params = {}
        ema_params = {}

        def _prepare_obs_features(self, observations):
            return observations["encoded"], {}

        def _features_from_inputs(self, params, inputs):
            del params
            return inputs

    boundary = FrozenFMVisualFeatures(FakeAgent())
    encoded = boundary.encode({"encoded": jnp.ones((2, 3, 4))})
    assert encoded.shape == (2, 12)


def test_workspace_boundary_pads_official_horizon_to_fm_horizon():
    class FakeAgent:
        use_pixels = True
        _condition_as_local = False
        _trainable_encoder = False
        params = {}
        ema_params = {}

        def _prepare_obs_features(self, observations):
            return observations["encoded"], {}

        def _features_from_inputs(self, params, inputs):
            del params
            return inputs

    adapter = OfficialBigymPolicy(
        mode="vanilla", obs_dim=5, action_dim=3, config=_config(), seed=0
    )
    agent = WorkspaceOfficialPolicyAgent(
        adapter,
        FrozenFMVisualFeatures(FakeAgent()),
        output_horizon=8,
    )
    agent.reset(0, [0])
    action = agent.act({"encoded": np.zeros((1, 5), np.float32)}, 0, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0, 0, 0, 0, 0])}
    )
    assert action.shape == (1, 8, 3)
    np.testing.assert_array_equal(action[:, 2:], 0)
    diagnostics = agent.rollout_diagnostics()
    assert np.isfinite(diagnostics["normalized_action_first_difference"])
    assert np.isfinite(diagnostics["normalized_action_continuation_jump"])


def test_workspace_boundary_records_executed_clipped_actions():
    class FakeBoundary:
        def encode(self, observations):
            return jnp.asarray(observations["encoded"], dtype=jnp.float32)

    class FakeAdapter:
        action_dim = 1
        config = SimpleNamespace(
            action_horizon=4,
            execute_horizon=2,
            inference_delay=1,
        )

        def bootstrap(self, key, features):
            del key
            return SimpleNamespace(
                previous_chunk=jnp.zeros((features.shape[0], 4, 1)),
                valid=jnp.ones((features.shape[0],), dtype=jnp.bool_),
            )

        def predict(self, key, features, state):
            del key, features
            execute = jnp.asarray([[[-2.0], [2.0]]], dtype=jnp.float32)
            return SimpleNamespace(execute_actions=execute, next_state=state)

    agent = WorkspaceOfficialPolicyAgent(
        FakeAdapter(), FakeBoundary(), output_horizon=4
    )
    action = agent.act({"encoded": np.zeros((1, 3), np.float32)}, 0, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0])}
    )

    np.testing.assert_array_equal(action[0, :2, 0], [-1.0, 1.0])
    np.testing.assert_array_equal(action[0, 2:, 0], 0.0)
    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["policy_action_clip_fraction"] == 1.0
    assert diagnostics["normalized_action_first_difference"] == 2.0


def test_workspace_boundary_aligns_policy_rng_by_episode_seed():
    class FakeBoundary:
        def encode(self, observations):
            return jnp.asarray(observations["encoded"], dtype=jnp.float32)

    class FakeAdapter:
        action_dim = 1
        config = SimpleNamespace(
            action_horizon=4,
            execute_horizon=2,
            inference_delay=0,
        )

        def bootstrap(self, key, features):
            return DelayState(
                jax.random.normal(key, (features.shape[0], 4, 1)),
                jnp.ones((features.shape[0],), dtype=jnp.bool_),
            )

        def predict(self, key, features, state):
            del features
            execute = jax.random.normal(key, (1, 2, 1))
            return SimpleNamespace(execute_actions=execute, next_state=state)

    agent = WorkspaceOfficialPolicyAgent(
        FakeAdapter(), FakeBoundary(), output_horizon=4
    )
    observation = {"encoded": np.zeros((1, 3), np.float32)}
    agent.reset_aligned_eval_noise()
    agent.set_eval_env_running(True)

    agent.set_active_eval_seeds([7])
    agent.reset(0, [0])
    first = agent.act(observation, 0, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0])}
    )
    agent.act(observation, 1, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0])}
    )

    agent.set_active_eval_seeds([19])
    agent.reset(0, [0])
    agent.act(observation, 0, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0])}
    )

    agent.set_active_eval_seeds([7])
    agent.reset(0, [0])
    repeated = agent.act(observation, 0, True)

    np.testing.assert_array_equal(first, repeated)


def test_workspace_boundary_records_only_executed_action_prefix():
    class FakeBoundary:
        def encode(self, observations):
            return jnp.asarray(observations["encoded"], dtype=jnp.float32)

    class FakeAdapter:
        action_dim = 1
        config = SimpleNamespace(
            action_horizon=4,
            execute_horizon=4,
            inference_delay=2,
        )

        def bootstrap(self, key, features):
            del key
            return DelayState(
                jnp.zeros((features.shape[0], 4, 1)),
                jnp.ones((features.shape[0],), dtype=jnp.bool_),
            )

        def predict(self, key, features, state):
            del key, features
            execute = jnp.asarray([[[0.0], [1.0], [100.0], [-100.0]]])
            return SimpleNamespace(execute_actions=execute, next_state=state)

    agent = WorkspaceOfficialPolicyAgent(
        FakeAdapter(), FakeBoundary(), output_horizon=4
    )
    agent.act({"encoded": np.zeros((1, 3), np.float32)}, 0, True)
    agent.record_action_execution(
        {"action_sequence_mask": np.array([1, 1, 0, 0])}
    )

    diagnostics = agent.rollout_diagnostics()
    assert diagnostics["normalized_action_first_difference"] == 1.0
    assert np.isnan(diagnostics["normalized_action_continuation_jump"])
    assert diagnostics["policy_action_clip_fraction"] == 0.0


def test_episode_audit_records_seed_outcome_and_matches_aggregate():
    audit = EpisodeAudit()
    audit.record_step(
        seed=4,
        reward=0.25,
        executed_action_steps=4,
        terminated=False,
        truncated=False,
        info={},
    )
    audit.record_step(
        seed=4,
        reward=0.75,
        executed_action_steps=2,
        terminated=True,
        truncated=False,
        info={"task_success": 1},
    )

    assert audit.records == [
        {
            "episode_index": 0,
            "seed": 4,
            "success": True,
            "episode_return": 1.0,
            "executed_action_steps": 6,
            "terminated": True,
            "truncated": False,
        }
    ]
    _validate_episode_audit(
        audit.records,
        {"episode_reward": 1.0, "episode_length": 12.0, "episode_success": 1.0},
        num_episodes=1,
        seed_start=4,
        action_repeat=2,
    )


def test_one_step_checkpoint_restore_and_function_preserving_warm_start(tmp_path: Path):
    vanilla = OfficialBigymPolicy(
        mode="vanilla", obs_dim=5, action_dim=3, config=_config(), seed=0
    )
    legato = OfficialBigymPolicy(
        mode="legato", obs_dim=5, action_dim=3, config=_config(), seed=1
    )
    warm_start_legato_from_vanilla(vanilla, legato)
    features = jnp.zeros((2, 5), dtype=jnp.float32)
    key = jax.random.key(4)
    np.testing.assert_allclose(
        vanilla.policy.action(key, features, 2),
        legato.policy.action(key, features, 2),
        atol=0,
        rtol=0,
    )

    trainer = OfficialTrainer(legato, TrainConfig())
    metrics = trainer.step(
        jax.random.key(5),
        features,
        jnp.zeros((2, 4, 3), dtype=jnp.float32),
    )
    assert np.isfinite(metrics["loss"])
    assert metrics["learning_rate"] == pytest.approx(0.0)
    checkpoint = tmp_path / "legato.pkl"
    save_checkpoint(checkpoint, legato, step=1)

    restored = OfficialBigymPolicy(
        mode="legato", obs_dim=5, action_dim=3, config=_config(), seed=9
    )
    assert load_checkpoint(checkpoint, restored)["step"] == 1
    np.testing.assert_allclose(
        legato.policy.action(key, features, 2),
        restored.policy.action(key, features, 2),
        atol=0,
        rtol=0,
    )

    vanilla_checkpoint = tmp_path / "vanilla.pkl"
    save_checkpoint(vanilla_checkpoint, vanilla, step=0)
    rtc = OfficialBigymPolicy(
        mode="rtc", obs_dim=5, action_dim=3, config=_config(), seed=10
    )
    load_checkpoint(vanilla_checkpoint, rtc)
    np.testing.assert_allclose(
        vanilla.policy.action(key, features, 2),
        rtc.policy.action(key, features, 2),
        atol=0,
        rtol=0,
    )


def test_official_trainer_uses_upstream_warmup_schedule():
    adapter = OfficialBigymPolicy(
        mode="vanilla", obs_dim=5, action_dim=3, config=_config(), seed=0
    )
    trainer = OfficialTrainer(adapter, TrainConfig(lr_warmup_steps=1000))

    assert float(trainer.learning_rate_schedule(0)) == pytest.approx(0.0)
    assert float(trainer.learning_rate_schedule(500)) == pytest.approx(1.5e-4)
    assert float(trainer.learning_rate_schedule(1000)) == pytest.approx(3e-4)


def test_window_batches_drop_incomplete_tail_by_default():
    episode = FeatureEpisode(
        np.arange(14, dtype=np.float32).reshape(7, 2),
        np.arange(21, dtype=np.float32).reshape(7, 3),
    )
    dataset = build_window_dataset([episode], horizon=2)

    batches = list(dataset.batches(4, seed=0))

    assert len(dataset) == 6
    assert len(batches) == 1
    assert batches[0][0].shape[0] == 4


def test_minimal_one_episode_eval_path():
    class TinyEnv:
        def reset(self, *, seed):
            self.steps = 0
            return np.full(5, seed, dtype=np.float32), {}

        def step(self, action):
            assert np.asarray(action).shape == (3,)
            self.steps += 1
            done = self.steps == 2
            return (
                np.full(5, self.steps, dtype=np.float32),
                float(done),
                done,
                False,
                {"success": done},
            )

    adapter = OfficialBigymPolicy(
        mode="vanilla", obs_dim=5, action_dim=3, config=_config(), seed=0
    )
    result = evaluate_episode(
        TinyEnv(),
        adapter,
        lambda observation: observation,
        seed=0,
        max_steps=4,
    )
    assert result.success
    assert result.episode_length == 2
    assert result.episode_return == 1.0
