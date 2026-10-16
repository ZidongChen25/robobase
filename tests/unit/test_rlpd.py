"""RLPD correctness tests: math, sampling, replay and an independent reference."""

from __future__ import annotations

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import optax
import pytest
from gymnasium import spaces

from robobase.method.rlpd import (
    RLPD,
    RLPDSpec,
    actor_apply,
    critic_apply,
    numpy_actor_mode,
    pack_transitions,
    tanh_normal_sample,
)
from robobase.workspace_rlpd import FlatTransitionRecorder

OBS_DIM, ACT_DIM = 5, 3


def _agent(**overrides) -> RLPD:
    spec = RLPDSpec(
        hidden_dims=(16, 16),
        num_qs=4,
        num_min_qs=2,
        utd_ratio=3,
        online_batch_size=4,
        offline_batch_size=4,
        online_buffer_size=32,
        num_explore_steps=0,
    )
    spec = dataclasses.replace(spec, **overrides)
    obs_space = spaces.Dict(
        {"low_dim_state": spaces.Box(-np.inf, np.inf, (1, OBS_DIM), np.float32)}
    )
    act_space = spaces.Box(-1.0, 1.0, (1, ACT_DIM), np.float32)
    return RLPD(spec, obs_space, act_space, 1, 1, 0.7, 0.5, True, seed=3)


def _rows(rng, count, reward=0.0):
    return pack_transitions(
        rng.standard_normal((count, OBS_DIM)),
        rng.uniform(-0.9, 0.9, (count, ACT_DIM)),
        np.full(count, reward),
        rng.integers(0, 2, count),
        rng.standard_normal((count, OBS_DIM)),
    )


def test_tanh_normal_log_prob_matches_change_of_variables():
    key = jax.random.PRNGKey(0)
    mean = jnp.array([[0.3, -1.2, 2.0], [0.0, 0.5, -0.4]])
    log_std = jnp.array([[-0.5, 0.1, -1.0], [0.3, -2.0, 0.0]])
    action, log_prob = tanh_normal_sample(key, mean, log_std)
    noise = np.asarray(jax.random.normal(key, mean.shape), np.float64)
    std = np.exp(np.asarray(log_std, np.float64))
    pre = np.asarray(mean, np.float64) + std * noise
    expected = (
        -0.5 * noise**2
        - np.log(std)
        - 0.5 * np.log(2 * np.pi)
        - np.log(1.0 - np.tanh(pre) ** 2)
    ).sum(-1)
    np.testing.assert_allclose(np.asarray(action), np.tanh(pre), rtol=1e-5)
    np.testing.assert_allclose(np.asarray(log_prob), expected, rtol=1e-4, atol=1e-4)


def test_saturated_log_prob_is_finite():
    _, log_prob = tanh_normal_sample(
        jax.random.PRNGKey(1), jnp.full((1, 3), 30.0), jnp.full((1, 3), -20.0)
    )
    assert np.all(np.isfinite(np.asarray(log_prob)))


def test_critic_ensemble_matches_per_member_layernorm_mlp():
    agent = _agent()
    critic = agent.state["critic"]
    rng = np.random.default_rng(0)
    obs = rng.standard_normal((6, OBS_DIM)).astype(np.float32)
    act = rng.uniform(-1, 1, (6, ACT_DIM)).astype(np.float32)
    q = np.asarray(critic_apply(critic, obs, act))
    assert q.shape == (4, 6)
    params = jax.tree.map(np.asarray, critic)
    for member in range(4):
        x = np.concatenate([obs, act], -1).astype(np.float64)
        for layer in params["hidden"]:
            x = x @ layer["w"][member] + layer["b"][member]  # Dense
            mu = x.mean(-1, keepdims=True)
            var = x.var(-1, keepdims=True)
            x = (x - mu) / np.sqrt(var + 1e-6)  # LayerNorm
            x = x * layer["ln_scale"][member] + layer["ln_bias"][member]
            x = np.maximum(x, 0.0)  # ReLU
        out = x @ params["out"]["w"][member] + params["out"]["b"][member]
        np.testing.assert_allclose(q[member], out[:, 0], rtol=1e-4, atol=1e-5)


def test_members_are_initialised_independently():
    w = np.asarray(_agent().state["critic"]["hidden"][0]["w"])
    assert not np.allclose(w[0], w[1])


def test_symmetric_sampling_and_valid_online_range():
    agent = _agent(utd_ratio=50, online_batch_size=8, offline_batch_size=8)
    rng = np.random.default_rng(1)
    agent.attach_device_replay(_rows(rng, 20, reward=7.0))
    for _ in range(5):
        agent.observe(_rows(rng, 1, reward=-3.0)[0], update=False)
    reward_col = OBS_DIM + ACT_DIM
    rows = np.asarray(
        agent._sample_rows(jax.random.PRNGKey(0), agent.replay, agent.offline)
    )
    assert rows.shape == (50, 16, agent.width)
    # every minibatch: first half offline, second half online, never the
    # zero-filled online slots beyond the current size
    assert np.all(rows[:, :8, reward_col] == 7.0)
    assert np.all(rows[:, 8:, reward_col] == -3.0)


def test_ring_buffer_wraps():
    agent = _agent(online_buffer_size=4)
    agent.attach_device_replay(_rows(np.random.default_rng(2), 8))
    for i in range(6):
        row = np.full(agent.width, float(i), np.float32)
        agent.observe(row, update=False)
    assert int(agent.replay["ptr"]) == 2 and int(agent.replay["size"]) == 4
    np.testing.assert_array_equal(
        np.asarray(agent.replay["data"])[:, 0], [4.0, 5.0, 2.0, 3.0]
    )


def test_offline_actions_are_clipped_to_tanh_range():
    agent = _agent()
    rows = _rows(np.random.default_rng(3), 4)
    rows[:, OBS_DIM : OBS_DIM + ACT_DIM] = 1.0
    agent.attach_device_replay(rows)
    actions = np.asarray(agent.offline)[:, OBS_DIM : OBS_DIM + ACT_DIM]
    assert np.all(actions < 1.0) and np.allclose(actions, 1.0 - 1e-5)


def test_scan_unroll_does_not_change_the_update():
    rng = np.random.default_rng(4)
    batches = jnp.asarray(_rows(rng, 3 * 8).reshape(3, 8, -1))
    out = {}
    for unroll in (1, 3):
        agent = _agent(scan_unroll=unroll)
        out[unroll] = agent._update_core(agent.state, batches)
    for a, b in zip(jax.tree.leaves(out[1]), jax.tree.leaves(out[3])):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=2e-5, atol=2e-6)


def test_target_subsets_are_uniform_without_replacement():
    agent = _agent(num_qs=5, num_min_qs=2)
    rows = jnp.asarray(_rows(np.random.default_rng(9), 4000 * 2).reshape(4000, 2, -1))
    mask = np.asarray(
        agent._critic_inputs(agent.state["actor"], rows, jax.random.PRNGKey(0))[-1]
    )
    assert np.all(mask.sum(-1) == 2)
    np.testing.assert_allclose(mask.mean(0), 0.4, atol=0.03)


def _reference_update(agent: RLPD, state, batches):
    """Plain re-statement of ``SACLearner.update`` (per-member Python loops)."""

    spec = agent.spec
    gamma, tau, eps = spec.discount, spec.tau, 1e-6
    lo, hi = spec.log_std_min, spec.log_std_max

    def q_member(params, m, obs, act):
        x = jnp.concatenate([obs, act], -1)
        for layer in params["hidden"]:
            x = x @ layer["w"][m] + layer["b"][m]
            mu = x.mean(-1, keepdims=True)
            var = ((x - mu) ** 2).mean(-1, keepdims=True)
            x = (x - mu) / jnp.sqrt(var + eps) * layer["ln_scale"][m] + layer[
                "ln_bias"
            ][m]
            x = jax.nn.relu(x)
        return (x @ params["out"]["w"][m] + params["out"]["b"][m])[:, 0]

    actor, critic, target = state["actor"], state["critic"], state["target_critic"]
    critic_opt = state["critic_opt"]
    temp = jnp.exp(state["log_temp"])
    rng, critic_key, actor_key = jax.random.split(state["rng"], 3)
    # same random draws as RLPD._critic_inputs, consumed one minibatch at a time
    noise_key, member_key = jax.random.split(critic_key)
    utd, B = batches.shape[:2]
    noise = jax.random.normal(noise_key, (utd * B, ACT_DIM))
    uniform = jax.random.uniform(member_key, (utd, spec.num_qs))
    E = spec.num_qs
    for step, rows in enumerate(batches):
        o = rows[:, :OBS_DIM]
        a = rows[:, OBS_DIM : OBS_DIM + ACT_DIM]
        r = rows[:, OBS_DIM + ACT_DIM] * spec.reward_scale + spec.reward_bias
        m = rows[:, OBS_DIM + ACT_DIM + 1]
        o2 = rows[:, OBS_DIM + ACT_DIM + 2 :]
        mean, log_std = actor_apply(actor, o2, lo, hi)
        eps_ = noise[step * B : (step + 1) * B]
        pre = mean + jnp.exp(log_std) * eps_
        a2 = jnp.tanh(pre)
        logp2 = jnp.sum(
            -0.5 * eps_**2
            - log_std
            - 0.5 * jnp.log(2 * jnp.pi)
            - jnp.log(1.0 - jnp.tanh(pre) ** 2),
            -1,
        )
        members = np.argsort(np.asarray(uniform[step]))[: spec.num_min_qs]
        next_q = jnp.min(
            jnp.stack([q_member(target, int(i), o2, a2) for i in members]), 0
        )
        y = r + gamma * m * next_q
        if spec.backup_entropy:
            y = y - gamma * m * temp * logp2

        def loss(c):
            return jnp.mean(
                jnp.stack([(q_member(c, i, o, a) - y) ** 2 for i in range(E)])
            )

        grads = jax.grad(loss)(critic)
        upd, critic_opt = optax.adam(spec.critic_lr).update(grads, critic_opt, critic)
        critic = optax.apply_updates(critic, upd)
        target = jax.tree.map(lambda t, c: (1 - tau) * t + tau * c, target, critic)

    o = batches[-1][:, :OBS_DIM]

    def actor_loss(p):
        mean, log_std = actor_apply(p, o, lo, hi)
        act, logp = tanh_normal_sample(actor_key, mean, log_std)
        q = jnp.mean(jnp.stack([q_member(critic, i, o, act) for i in range(E)]), 0)
        return jnp.mean(logp * temp - q), -jnp.mean(logp)

    (_, entropy), g = jax.value_and_grad(actor_loss, has_aux=True)(actor)
    upd, actor_opt = optax.adam(spec.actor_lr).update(g, state["actor_opt"], actor)
    actor = optax.apply_updates(actor, upd)
    tg = jax.grad(lambda lt: jnp.exp(lt) * (entropy - agent.target_entropy))(
        state["log_temp"]
    )
    upd, temp_opt = optax.adam(spec.temp_lr).update(
        tg, state["temp_opt"], state["log_temp"]
    )
    return {
        "actor": actor,
        "critic": critic,
        "target_critic": target,
        "log_temp": optax.apply_updates(state["log_temp"], upd),
        "actor_opt": actor_opt,
        "critic_opt": critic_opt,
        "temp_opt": temp_opt,
        "rng": rng,
    }


@pytest.mark.parametrize(
    "overrides",
    [
        {},
        {"num_min_qs": 1, "backup_entropy": False, "reward_bias": -1.0},
        {"num_min_qs": 4},
    ],
)
def test_update_matches_independent_reference(overrides):
    agent = _agent(**overrides)
    rng = np.random.default_rng(5)
    batches = jnp.asarray(_rows(rng, 3 * 8).reshape(3, 8, -1))
    state = agent.state
    expected = _reference_update(agent, state, batches)
    actual, metrics = agent._update_core(state, batches)
    for path, (a, b) in zip(
        jax.tree_util.tree_flatten_with_path(expected)[0],
        zip(jax.tree.leaves(expected), jax.tree.leaves(actual)),
    ):
        np.testing.assert_allclose(
            np.asarray(b), np.asarray(a), rtol=1e-4, atol=1e-5, err_msg=str(path[0])
        )
    assert np.all(np.isfinite(np.asarray(metrics)))


def test_fused_step_trains_and_reports_metrics():
    agent = _agent()
    rng = np.random.default_rng(6)
    agent.attach_device_replay(_rows(rng, 64))
    before = np.asarray(agent.state["actor"]["mean"]["w"]).copy()
    for row in _rows(rng, 10):
        agent.observe(row, update=True)
    metrics = agent.last_metrics()
    assert metrics["online_size"] == 10.0
    assert np.isfinite(metrics["critic_loss"]) and metrics["temperature"] > 0
    assert not np.allclose(before, np.asarray(agent.state["actor"]["mean"]["w"]))


def test_pure_online_variant_needs_no_offline_data():
    agent = _agent(offline_batch_size=0)
    agent.attach_device_replay(None)
    rng = np.random.default_rng(7)
    for row in _rows(rng, 5):
        agent.observe(row, update=True)
    assert np.isfinite(agent.last_metrics()["critic_loss"])


def test_generic_replay_iter_path():
    agent = _agent()
    rng = np.random.default_rng(8)

    def batches():
        while True:
            yield {
                "low_dim_state": rng.standard_normal((8, 1, OBS_DIM)).astype(
                    np.float32
                ),
                "low_dim_state_tp1": rng.standard_normal((8, 1, OBS_DIM)).astype(
                    np.float32
                ),
                "action": rng.uniform(-1, 1, (8, 1, ACT_DIM)).astype(np.float32),
                "reward": rng.standard_normal(8).astype(np.float32),
                "terminal": rng.integers(0, 2, 8).astype(np.float32),
                "discount": np.full(8, 0.99, np.float32),
            }

    agent.logging = True
    metrics = agent.update(batches(), step=0)
    assert np.isfinite(metrics["critic_loss"])


def test_act_modes():
    agent = _agent(num_explore_steps=10)
    obs = {"low_dim_state": np.ones((1, 1, OBS_DIM), np.float32)}
    explore = agent.act(obs, step=0, eval_mode=False)
    assert explore.shape == (1, 1, ACT_DIM) and np.all(np.abs(explore) <= 1.0)
    sampled = agent.act(obs, step=100, eval_mode=False)
    assert sampled.shape == (1, 1, ACT_DIM)
    greedy = agent.act(obs, step=0, eval_mode=True)
    expected = numpy_actor_mode(
        agent.actor_params_numpy(), obs["low_dim_state"].reshape(1, -1)
    )
    np.testing.assert_allclose(greedy.reshape(1, -1), expected, rtol=1e-5, atol=1e-6)


def test_flat_transition_recorder_chains_next_observations():
    recorder = FlatTransitionRecorder()
    obs = [{"low_dim_state": np.full(OBS_DIM, float(i), np.float32)} for i in range(4)]
    for i in range(3):
        recorder.add(obs[i], np.full(ACT_DIM, 0.1 * i), float(i), i == 2, False, demo=1)
    recorder.add_final(obs[3])
    rows = recorder.rows()
    assert rows.shape == (3, 2 * OBS_DIM + ACT_DIM + 2)
    np.testing.assert_array_equal(rows[:, 0], [0, 1, 2])  # obs
    np.testing.assert_array_equal(rows[:, -1], [1, 2, 3])  # next_obs
    np.testing.assert_array_equal(rows[:, OBS_DIM + ACT_DIM + 1], [1, 1, 0])  # mask


def test_legacy_d4rl_rows_only_handles_gym_era_d4rl():
    from omegaconf import OmegaConf

    from robobase.workspace_rlpd import legacy_d4rl_rows

    assert legacy_d4rl_rows(OmegaConf.create({"env": {"env_name": "dmc"}})) is None
    minari = OmegaConf.create(
        {
            "env": {
                "env_name": "d4rl",
                "task_name": "antmaze_umaze",
                "dataset_format": "minari",
                "dataset_id": "D4RL/antmaze/umaze-v2",
            }
        }
    )
    assert legacy_d4rl_rows(minari) is None


def test_eval_worker_environment_is_restored():
    import os

    from robobase.workspace_rlpd import _EVAL_WORKER_ENV, _environment

    before = {key: os.environ.get(key) for key in _EVAL_WORKER_ENV}
    with _environment(**_EVAL_WORKER_ENV):
        assert os.environ["CUDA_VISIBLE_DEVICES"] == ""
        assert os.environ["JAX_PLATFORMS"] == "cpu"
    assert {key: os.environ.get(key) for key in _EVAL_WORKER_ENV} == before
