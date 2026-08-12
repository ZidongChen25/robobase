import json

from scripts.prune_experiment_storage import select_validation


def _write_sweep_eval(run_dir, step, success, seed_start=400, episodes=50):
    sweep_dir = run_dir / "sweep_evals"
    sweep_dir.mkdir(parents=True, exist_ok=True)
    (sweep_dir / f"eval_{step}.json").write_text(
        json.dumps(
            {
                "env_steps": step,
                "episode_success": success,
                "eval_episodes": episodes,
                "eval_seed_start": seed_start,
            }
        )
    )


def test_async_sweep_evals_provide_a_validation_curve(tmp_path):
    _write_sweep_eval(tmp_path, 10000, 0.2)
    _write_sweep_eval(tmp_path, 20000, 0.6)

    selected = select_validation(tmp_path)
    assert selected is not None
    path, rows = selected
    assert path == tmp_path / "sweep_evals"
    assert sorted(rows) == [(10000, 0.2), (20000, 0.6)]


def test_sealed_seed_sweep_evals_never_drive_selection(tmp_path):
    # eval_seed_start >= 800 is the held-out test protocol; selecting a
    # checkpoint on it would leak the sealed split.
    _write_sweep_eval(tmp_path, 10000, 0.2, seed_start=800)
    _write_sweep_eval(tmp_path, 20000, 0.6, seed_start=800)

    assert select_validation(tmp_path) is None


def test_single_sweep_eval_point_is_not_a_curve(tmp_path):
    _write_sweep_eval(tmp_path, 10000, 0.2)

    assert select_validation(tmp_path) is None


def test_in_run_validation_csv_outranks_sweep_evals(tmp_path):
    _write_sweep_eval(tmp_path, 10000, 0.2)
    _write_sweep_eval(tmp_path, 20000, 0.6)
    (tmp_path / "val50_seeds400.csv").write_text(
        "env_steps,episode_success,eval_episodes,eval_seed_start\n"
        "10000,0.3,50,400\n"
        "20000,0.7,50,400\n"
    )

    path, rows = select_validation(tmp_path)
    assert path.name == "val50_seeds400.csv"
    assert sorted(rows) == [(10000, 0.3), (20000, 0.7)]
