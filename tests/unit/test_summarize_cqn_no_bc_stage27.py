import csv

from scripts.summarize_cqn_no_bc_stage27 import _decision, summarize


def _arms(seed1_17k, seed1_20k, seed2_17k, seed2_20k):
    return {
        "seed1": {
            "new_curve": {
                "17500": seed1_17k,
                "20000": seed1_20k,
            }
        },
        "seed2": {
            "new_curve": {
                "17500": seed2_17k,
                "20000": seed2_20k,
            }
        },
    }


def test_stage27_mechanism_and_scale_pass_extends_to50k():
    decision, flags = _decision(
        {"seed1": 0.06, "seed2": 0.04},
        _arms(0.48, 0.52, 0.44, 0.46),
    )
    assert decision == "extend_reward_scale_seeds1_2_to50k"
    assert flags["mechanism_pass"]
    assert flags["scale_continuation"]


def test_stage27_rising_boundary_can_extend_without_mechanism_pass():
    decision, flags = _decision(
        {"seed1": -0.02, "seed2": -0.04},
        _arms(0.48, 0.52, 0.44, 0.46),
    )
    assert decision == "extend_reward_scale_seeds1_2_to50k_for_scale"
    assert not flags["mechanism_pass"]
    assert flags["scale_continuation"]


def test_stage27_declining_boundary_stops_without_full_budget_claim():
    decision, flags = _decision(
        {"seed1": -0.02, "seed2": -0.04},
        _arms(0.56, 0.50, 0.48, 0.46),
    )
    assert decision == "stop_reward_scale_variant_without_full_budget_claim"
    assert not flags["mechanism_pass"]
    assert not flags["scale_continuation"]


def _write_curve(run_dir, name, rows):
    run_dir.mkdir()
    with (run_dir / name).open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("env_steps", "episode_success"),
        )
        writer.writeheader()
        for step, success in rows.items():
            writer.writerow(
                {"env_steps": step, "episode_success": success}
            )


def test_stage27_summary_computes_boundary_from_fixed_curves(tmp_path):
    curves = {
        "baseline1": (
            {2500: 0.12, 5000: 0.24, 7500: 0.34, 10000: 0.48},
            {12500: 0.44, 15000: 0.60, 17500: 0.54, 20000: 0.46},
        ),
        "baseline2": (
            {2500: 0.04, 5000: 0.08, 7500: 0.32, 10000: 0.46},
            {12500: 0.46, 15000: 0.40, 17500: 0.44, 20000: 0.38},
        ),
        "treatment1": (
            {2500: 0.06, 5000: 0.18, 7500: 0.40, 10000: 0.46},
            {12500: 0.48, 15000: 0.48, 17500: 0.50, 20000: 0.52},
        ),
        "treatment2": (
            {2500: 0.02, 5000: 0.16, 7500: 0.36, 10000: 0.46},
            {12500: 0.46, 15000: 0.48, 17500: 0.48, 20000: 0.50},
        ),
    }
    paths = {}
    for name, (short, extended) in curves.items():
        path = tmp_path / name
        _write_curve(path, "val50_seeds400.csv", short)
        with (path / "val50_ext20k_seeds400.csv").open(
            "w",
            newline="",
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=("env_steps", "episode_success"),
            )
            writer.writeheader()
            for step, success in extended.items():
                writer.writerow(
                    {"env_steps": step, "episode_success": success}
                )
        paths[name] = path

    result = summarize(
        paths["baseline1"],
        paths["baseline2"],
        paths["treatment1"],
        paths["treatment2"],
    )

    assert result["decision_flags"]["good_20k_boundary"] == {
        "seed1": True,
        "seed2": True,
    }
    assert result["decision_flags"]["scale_continuation"]
    assert result["next_decision"] == (
        "extend_reward_scale_seeds1_2_to50k_for_scale"
    )
