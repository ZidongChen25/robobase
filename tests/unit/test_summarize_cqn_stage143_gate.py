import json
import subprocess
import sys
from pathlib import Path

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / (
    "summarize_cqn_stage143_gate.py"
)


def _probe(records):
    return {"status": "ok", "records": records}


def _record(eval_seed, accuracy, pairs):
    return {
        "eval_seed": eval_seed,
        "pairwise_sign_accuracy": accuracy,
        "num_informative_pairs": pairs,
    }


def _write(tmp_path, seed, tag, records):
    path = tmp_path / f"seed{seed}_w{tag}_sibling_L0_rr.json"
    path.write_text(json.dumps(_probe(records)))


def _run(tmp_path, out, seeds="1,2"):
    return subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--gate-dir",
            str(tmp_path),
            "--seeds",
            seeds,
            "--replicates",
            "500",
            "--output",
            str(out),
        ],
        capture_output=True,
        text=True,
    )


def test_stage143_gate_passes_on_strong_consistent_treatment(tmp_path):
    for seed in (1, 2):
        _write(
            tmp_path,
            seed,
            "0p0",
            [_record(e, 0.5, 10) for e in range(400, 412)],
        )
        _write(
            tmp_path,
            seed,
            "0p1",
            [_record(e, 0.8, 10) for e in range(400, 412)],
        )
    out = tmp_path / "gate.json"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["gate"] == "pass"
    assert result["pooled_treatment_ci"][0] > 0.5


def test_stage143_gate_fails_when_one_seed_loses_to_control(tmp_path):
    _write(tmp_path, 1, "0p0", [_record(e, 0.5, 10) for e in range(400, 412)])
    _write(tmp_path, 1, "0p1", [_record(e, 0.8, 10) for e in range(400, 412)])
    _write(tmp_path, 2, "0p0", [_record(e, 0.7, 10) for e in range(400, 412)])
    _write(tmp_path, 2, "0p1", [_record(e, 0.6, 10) for e in range(400, 412)])
    out = tmp_path / "gate.json"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["gate"] == "fail"
    assert not result["per_seed"]["seed2"]["treatment_beats_control"]


def test_stage143_gate_fails_on_wide_ci_near_chance(tmp_path):
    # Treatment barely above control with tiny pair counts: CI spans 0.5.
    for seed in (1, 2):
        _write(tmp_path, seed, "0p0", [_record(400, 0.4, 4)])
        _write(tmp_path, seed, "0p1", [_record(400, 0.6, 4), _record(401, 0.4, 4)])
    out = tmp_path / "gate.json"
    proc = _run(tmp_path, out)
    assert proc.returncode == 0, proc.stderr
    result = json.loads(out.read_text())
    assert result["gate"] == "fail"
