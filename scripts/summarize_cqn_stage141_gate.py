#!/usr/bin/env python3
"""Stage-141 gate verdict from matched control/treatment branch probes.

Reads the four branch-probe JSONs (2 training seeds x cv_rct_weight
{0.0, 0.1}) and applies the pre-registered criteria (cqn-flow.md 22.3):

1. treatment pairwise-sign state-bootstrap CI lower bound > 0.5;
2. treatment point estimate above its matched control arm, per seed.

Both must hold on every training seed for a pass.  This summarizer never
reads training curves or task success; it only aggregates frozen artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="1,2")
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_probe(path: Path) -> dict:
    data = json.loads(path.read_text())
    if data.get("status") != "ok":
        raise ValueError(f"{path} status is {data.get('status')!r}")
    return {
        "pairwise_sign_accuracy": float(data["pairwise_sign_accuracy"]),
        "pairwise_ci": [
            float(v)
            for v in data["state_bootstrap"]["pairwise_sign_accuracy_ci"]
        ],
        "mean_spearman": float(data["mean_spearman"]),
        "spearman_ci": [
            float(v) for v in data["state_bootstrap"]["mean_spearman_ci"]
        ],
        "informative_states": int(data["num_informative_states"]),
        "informative_pairs": int(data["num_informative_pairs"]),
        "num_states": int(data["num_states"]),
        "snapshot": data["snapshot"],
    }


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    per_seed: dict = {}
    all_pass = True
    for seed in seeds:
        arms = {}
        for weight_tag in ("0p0", "0p1"):
            path = (
                args.gate_dir
                / f"seed{seed}_w{weight_tag}_branch_L0_scoreL2.json"
            )
            arms["control" if weight_tag == "0p0" else "treatment"] = (
                load_probe(path)
            )
        treatment = arms["treatment"]
        control = arms["control"]
        ci_pass = treatment["pairwise_ci"][0] > 0.5
        beats_control = (
            treatment["pairwise_sign_accuracy"]
            > control["pairwise_sign_accuracy"]
        )
        seed_pass = ci_pass and beats_control
        all_pass = all_pass and seed_pass
        per_seed[f"seed{seed}"] = {
            "control": control,
            "treatment": treatment,
            "treatment_ci_lower_gt_0.5": ci_pass,
            "treatment_beats_control": beats_control,
            "seed_pass": seed_pass,
        }
    result = {
        "criteria": (
            "treatment pairwise-sign state-bootstrap CI lower > 0.5 AND "
            "treatment point estimate > matched control, on every seed"
        ),
        "per_seed": per_seed,
        "gate": "pass" if all_pass else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for seed, entry in per_seed.items():
        t, c = entry["treatment"], entry["control"]
        print(
            f"{seed}: control {c['pairwise_sign_accuracy']:.3f} "
            f"CI[{c['pairwise_ci'][0]:.3f},{c['pairwise_ci'][1]:.3f}] | "
            f"treatment {t['pairwise_sign_accuracy']:.3f} "
            f"CI[{t['pairwise_ci'][0]:.3f},{t['pairwise_ci'][1]:.3f}] | "
            f"pass={entry['seed_pass']}"
        )
    print(f"gate: {result['gate']}")


if __name__ == "__main__":
    main()
