import json

from scripts.summarize_cqn_flow_ranking_sweep import summarize


def _write_probe(path, step, samples, flip, snr):
    path.write_text(
        json.dumps(
            {
                "status": "ok",
                "snapshot": f"/tmp/{step}_snapshot.pkl",
                "probe_action_flow_samples": samples,
                "metrics": {
                    "per_level_bin_flip_rate": flip,
                    "per_level_rank_snr": snr,
                },
            }
        )
    )


def test_ranking_gate_requires_all_neighbor_snapshots(tmp_path):
    _write_probe(tmp_path / "7k_r16.json", 7000, 16, [0.05, 0.08], [2, 3])
    payload = summarize(
        tmp_path,
        required_snapshots=[7000, 8000],
        gate_action_flow_samples=16,
        max_flip_rate=0.1,
    )
    assert payload["gate"] == "incomplete"

    _write_probe(tmp_path / "8k_r16.json", 8000, 16, [0.02, 0.1], [3, 4])
    payload = summarize(
        tmp_path,
        required_snapshots=[7000, 8000],
        gate_action_flow_samples=16,
        max_flip_rate=0.1,
    )
    assert payload["gate"] == "pass"


def test_ranking_gate_fails_on_any_unstable_level(tmp_path):
    _write_probe(tmp_path / "7k_r16.json", 7000, 16, [0.05, 0.11], [2, 3])
    _write_probe(tmp_path / "8k_r16.json", 8000, 16, [0.02, 0.09], [3, 4])
    payload = summarize(
        tmp_path,
        required_snapshots=[7000, 8000],
        gate_action_flow_samples=16,
        max_flip_rate=0.1,
    )
    assert payload["gate"] == "fail"
