import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / (
    "summarize_cqn_stage141_gate.py"
)


def _probe(sign, ci_low, ci_high):
    return {
        "status": "ok",
        "pairwise_sign_accuracy": sign,
        "mean_spearman": 0.1,
        "num_informative_states": 10,
        "num_informative_pairs": 40,
        "num_states": 24,
        "snapshot": "s.pkl",
        "state_bootstrap": {
            "pairwise_sign_accuracy_ci": [ci_low, ci_high],
            "mean_spearman_ci": [-0.1, 0.3],
        },
    }


def _write(tmp_path, seed, tag, payload):
    path = tmp_path / f"seed{seed}_w{tag}_branch_L0_scoreL2.json"
    path.write_text(json.dumps(payload))


def _run(tmp_path, out):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gate-dir",
            str(tmp_path),
            "--seeds",
            "1,2",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def test_stage141_gate_passes_when_both_seeds_meet_criteria(tmp_path):
    for seed in (1, 2):
        _write(tmp_path, seed, "0p0", _probe(0.52, 0.40, 0.66))
        _write(tmp_path, seed, "0p1", _probe(0.71, 0.55, 0.86))
    out = tmp_path / "gate.json"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["gate"] == "pass"


def test_stage141_gate_fails_on_single_seed_ci_or_control(tmp_path):
    # Seed 1 treatment CI lower bound below 0.5.
    _write(tmp_path, 1, "0p0", _probe(0.52, 0.40, 0.66))
    _write(tmp_path, 1, "0p1", _probe(0.71, 0.45, 0.86))
    # Seed 2 treatment fine on CI but below its control.
    _write(tmp_path, 2, "0p0", _probe(0.80, 0.60, 0.92))
    _write(tmp_path, 2, "0p1", _probe(0.70, 0.55, 0.86))
    out = tmp_path / "gate.json"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["gate"] == "fail"
    assert not result["per_seed"]["seed1"]["treatment_ci_lower_gt_0.5"]
    assert not result["per_seed"]["seed2"]["treatment_beats_control"]
