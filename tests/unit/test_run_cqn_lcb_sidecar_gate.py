from scripts.run_cqn_lcb_sidecar_gate import margin_label
from scripts.summarize_cqn_lcb_calibration import summarize


def test_margin_label_is_stable_for_gate_artifacts():
    assert margin_label(0.01) == "margin_0p01"
    assert margin_label(0.04) == "margin_0p04"
    assert margin_label(0.1) == "margin_0p1"


def _eval(successes, *, overrides=False):
    return {
        "status": "ok",
        "episode_results": [
            {
                "seed": 100 + index,
                "episode_success": success,
                "episode_reward": success,
                "inference_count": 10,
                "applied_override_count": int(overrides),
            }
            for index, success in enumerate(successes)
        ],
    }


def test_confirmation_requires_paired_noninferiority_ci():
    bc = _eval([1.0, 0.0, 0.0, 0.0])
    candidate = _eval([0.0, 1.0, 1.0, 0.0], overrides=True)

    strict = summarize(
        bc,
        {"candidate": candidate},
        bootstrap_replicates=1_000,
        seed=4,
        stage="confirmation",
        noninferiority_margin=0.0,
    )
    relaxed = summarize(
        bc,
        {"candidate": candidate},
        bootstrap_replicates=1_000,
        seed=4,
        stage="confirmation",
        noninferiority_margin=1.0,
    )

    assert not strict["gate_passed"]
    assert relaxed["gate_passed"]
