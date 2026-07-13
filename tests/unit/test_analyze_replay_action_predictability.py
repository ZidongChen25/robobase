import csv

import numpy as np

from scripts.analyze_replay_action_predictability import (
    ReplayEpisode,
    analyze_replay,
    cross_fit_offset,
    load_replay_episodes,
    main,
    make_group_folds,
    predictability_metrics,
)


def _write_uniform_replay(replay_dir, episodes):
    replay_dir.mkdir(parents=True, exist_ok=True)
    global_index = 0
    for episode_index, (states, actions) in enumerate(episodes):
        length = len(actions)
        stored_states = np.concatenate(
            [states, np.full_like(states[:1], 12345.0)], axis=0
        )
        stored_actions = np.concatenate(
            [actions, np.full_like(actions[:1], -12345.0)], axis=0
        )
        terminal = np.zeros(length + 1, dtype=np.int8)
        truncated = np.zeros(length + 1, dtype=np.int8)
        terminal[-1] = -1
        truncated[-2:] = (1, -1)
        path = replay_dir / (
            f"20260710T000000_{episode_index}_{length}_{global_index}.npz"
        )
        np.savez(
            path,
            low_dim_state=stored_states,
            action=stored_actions,
            terminal=terminal,
            truncated=truncated,
        )
        global_index += length


def _ar_episodes(
    *,
    seed=0,
    num_episodes=12,
    length=24,
    state_dim=3,
    action_dim=2,
):
    rng = np.random.default_rng(seed)
    episodes = []
    readout = rng.normal(size=(action_dim, state_dim))
    for episode_index in range(num_episodes):
        states = np.empty((length, state_dim), dtype=np.float64)
        states[0] = rng.normal(size=state_dim)
        for step in range(1, length):
            states[step] = 0.55 * states[step - 1] + rng.normal(
                scale=0.9,
                size=state_dim,
            )
        actions = states @ readout.T + rng.normal(
            scale=0.02,
            size=(length, action_dim),
        )
        episodes.append(
            ReplayEpisode(
                episode_id=f"episode-{episode_index}",
                states=states,
                actions=actions,
            )
        )
    return episodes


def test_loader_auto_excludes_uniform_replay_terminal_sentinel(tmp_path):
    replay_dir = tmp_path / "replay"
    states = np.arange(15, dtype=np.float64).reshape(5, 3)
    actions = np.arange(10, dtype=np.float64).reshape(5, 2)
    _write_uniform_replay(replay_dir, [(states, actions)])

    episodes = load_replay_episodes(replay_dir)

    assert len(episodes) == 1
    assert episodes[0].states.shape == (5, 3)
    assert episodes[0].actions.shape == (5, 2)
    np.testing.assert_array_equal(episodes[0].states, states)
    np.testing.assert_array_equal(episodes[0].actions, actions)


def test_loader_orders_uniform_replay_by_global_index(tmp_path):
    replay_dir = tmp_path / "replay"
    source = [
        (
            np.full((2, 1), episode_index, dtype=np.float64),
            np.full((2, 1), -episode_index, dtype=np.float64),
        )
        for episode_index in range(12)
    ]
    _write_uniform_replay(replay_dir, source)

    episodes = load_replay_episodes(replay_dir)

    assert [int(episode.states[0, 0]) for episode in episodes] == list(range(12))


def test_group_folds_are_disjoint_complete_and_reproducible():
    groups = np.arange(17)

    first = make_group_folds(groups, n_splits=5, seed=7)
    repeated = make_group_folds(groups, n_splits=5, seed=7)
    changed = make_group_folds(groups, n_splits=5, seed=8)

    np.testing.assert_array_equal(np.sort(np.concatenate(first)), groups)
    for left_index, left in enumerate(first):
        for right in first[left_index + 1 :]:
            assert not np.intersect1d(left, right).size
    assert all(np.array_equal(a, b) for a, b in zip(first, repeated))
    assert any(not np.array_equal(a, b) for a, b in zip(first, changed))


def test_cross_fit_is_episode_disjoint_and_future_offsets_get_harder():
    episodes = _ar_episodes()
    outer_folds = make_group_folds(np.arange(len(episodes)), 3, seed=11)
    results = [
        cross_fit_offset(
            episodes,
            offset,
            outer_group_folds=outer_folds,
            inner_splits=2,
            alphas=(0.01, 1.0, 100.0),
            cv_seed=19,
        )
        for offset in range(4)
    ]

    for result in results:
        for group in np.unique(result.groups):
            row_folds = np.unique(result.outer_fold_by_row[result.groups == group])
            assert len(row_folds) == 1
            assert group in outer_folds[int(row_folds[0])]
        assert len(result.selected_alphas) == 3

    profile = [
        predictability_metrics(result)["nmse_variance_weighted"]
        for result in results
    ]
    assert profile[0] < 0.01
    assert profile[-1] > profile[0] + 0.5


def test_cli_writes_reproducible_offset_csv(tmp_path):
    episodes = _ar_episodes(seed=5, num_episodes=8, length=12)
    replay_dir = tmp_path / "replay"
    _write_uniform_replay(
        replay_dir,
        [(episode.states, episode.actions) for episode in episodes],
    )
    first_csv = tmp_path / "first.csv"
    second_csv = tmp_path / "second.csv"
    common_args = [
        "--replay-dir",
        str(replay_dir),
        "--max-horizon",
        "3",
        "--outer-folds",
        "4",
        "--inner-folds",
        "2",
        "--ridge-alphas",
        "0.1",
        "10",
        "--cv-seed",
        "13",
        "--bootstrap-replicates",
        "64",
        "--bootstrap-seed",
        "17",
    ]

    assert main([*common_args, "--output-csv", str(first_csv)]) == 0
    assert main([*common_args, "--output-csv", str(second_csv)]) == 0

    assert first_csv.read_text() == second_csv.read_text()
    with first_csv.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["offset"]) for row in rows] == [0, 1, 2]
    assert [int(row["n_pairs"]) for row in rows] == [96, 88, 80]
    required = {
        "nmse_variance_weighted",
        "nmse_macro_action_dim",
        "crossfit_r2",
        "nmse_ci_low",
        "nmse_ci_high",
        "selected_alpha_by_outer_fold",
        "bootstrap_seed",
    }
    assert required <= set(rows[0])
    assert all(float(row["nmse_ci_low"]) <= float(row["nmse_ci_high"]) for row in rows)


def test_analyze_replay_bootstrap_is_seed_reproducible():
    episodes = _ar_episodes(seed=21, num_episodes=8, length=10)
    kwargs = dict(
        max_horizon=2,
        outer_splits=4,
        inner_splits=2,
        alphas=(0.1, 10.0),
        cv_seed=3,
        bootstrap_replicates=32,
        bootstrap_seed=4,
        confidence=0.9,
    )

    first_rows, first_bootstrap = analyze_replay(episodes, **kwargs)
    second_rows, second_bootstrap = analyze_replay(episodes, **kwargs)

    assert first_rows == second_rows
    np.testing.assert_array_equal(first_bootstrap, second_bootstrap)
