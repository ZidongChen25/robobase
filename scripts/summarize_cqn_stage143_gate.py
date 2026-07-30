#!/usr/bin/env python3
"""Stage-143 confirmation verdict: sibling-protocol crossed bootstrap.

Pre-registered in cqn-flow.md 23.4 before seed-3 data existed:

* primary probe: sibling_horizon + round_robin + level-0 + H=4;
* pass requires (a) treatment > matched control on every training seed and
  (b) pooled treatment pairwise-sign CI lower bound > 0.5 under a
  training-seed x eval-seed crossed cluster bootstrap.

Reads only frozen probe JSONs; correct-pair counts are reconstructed from
per-state pairwise accuracy times pair count.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gate-dir", required=True, type=Path)
    parser.add_argument("--seeds", default="1,2,3")
    parser.add_argument("--suffix", default="sibling_L0_rr")
    parser.add_argument("--replicates", type=int, default=10000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_cells(path: Path) -> dict[int, list[tuple[int, int]]]:
    """Map eval seed -> [(correct_pairs, total_pairs)] per state."""
    data = json.loads(path.read_text())
    if data.get("status") != "ok":
        raise ValueError(f"{path} status is {data.get('status')!r}")
    cells: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for record in data["records"]:
        pairs = int(record["num_informative_pairs"])
        if pairs <= 0:
            continue
        accuracy = float(record["pairwise_sign_accuracy"])
        if math.isnan(accuracy):
            continue
        correct = int(round(accuracy * pairs))
        cells[int(record["eval_seed"])].append((correct, pairs))
    return dict(cells)


def pooled_accuracy(cell_sets: list[dict[int, list[tuple[int, int]]]]):
    correct = total = 0
    for cells in cell_sets:
        for states in cells.values():
            for c, n in states:
                correct += c
                total += n
    return (correct / total if total else float("nan")), total


def crossed_bootstrap(
    cell_sets: list[dict[int, list[tuple[int, int]]]],
    replicates: int,
    seed: int,
) -> tuple[float, float]:
    """Resample training seeds and eval seeds with replacement."""
    rng = np.random.default_rng(seed)
    eval_seeds = sorted({e for cells in cell_sets for e in cells})
    n_train = len(cell_sets)
    n_eval = len(eval_seeds)
    stats = []
    for _ in range(replicates):
        train_draw = rng.integers(0, n_train, size=n_train)
        eval_draw = rng.integers(0, n_eval, size=n_eval)
        correct = total = 0
        for t in train_draw:
            cells = cell_sets[t]
            for e in eval_draw:
                for c, n in cells.get(eval_seeds[e], ()):  # noqa: B905
                    correct += c
                    total += n
        if total:
            stats.append(correct / total)
    lower, upper = np.quantile(stats, [0.025, 0.975])
    return float(lower), float(upper)


def main() -> None:
    args = parse_args()
    seeds = [int(s) for s in str(args.seeds).split(",") if s]
    treatment_sets, per_seed = [], {}
    all_seeds_beat_control = True
    for seed in seeds:
        arms = {}
        for tag, arm in (("0p0", "control"), ("0p1", "treatment")):
            path = args.gate_dir / f"seed{seed}_w{tag}_{args.suffix}.json"
            arms[arm] = load_cells(path)
        control_acc, control_pairs = pooled_accuracy([arms["control"]])
        treat_acc, treat_pairs = pooled_accuracy([arms["treatment"]])
        beats = treat_acc > control_acc
        all_seeds_beat_control = all_seeds_beat_control and beats
        treatment_sets.append(arms["treatment"])
        per_seed[f"seed{seed}"] = {
            "control_accuracy": control_acc,
            "control_pairs": control_pairs,
            "treatment_accuracy": treat_acc,
            "treatment_pairs": treat_pairs,
            "treatment_beats_control": beats,
        }
    pooled_acc, pooled_pairs = pooled_accuracy(treatment_sets)
    ci_lower, ci_upper = crossed_bootstrap(
        treatment_sets, args.replicates, args.bootstrap_seed
    )
    ci_pass = ci_lower > 0.5
    result = {
        "criteria": (
            "treatment > matched control on every training seed AND pooled "
            "treatment crossed-bootstrap CI lower bound > 0.5"
        ),
        "per_seed": per_seed,
        "pooled_treatment_accuracy": pooled_acc,
        "pooled_treatment_pairs": pooled_pairs,
        "pooled_treatment_ci": [ci_lower, ci_upper],
        "all_seeds_beat_control": all_seeds_beat_control,
        "pooled_ci_lower_gt_0.5": ci_pass,
        "gate": "pass" if (all_seeds_beat_control and ci_pass) else "fail",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2))
    for seed, entry in per_seed.items():
        print(
            f"{seed}: control {entry['control_accuracy']:.3f} "
            f"({entry['control_pairs']}p) | treatment "
            f"{entry['treatment_accuracy']:.3f} "
            f"({entry['treatment_pairs']}p) | "
            f"beats={entry['treatment_beats_control']}"
        )
    print(
        f"pooled treatment: {pooled_acc:.3f} ({pooled_pairs}p) "
        f"CI[{ci_lower:.3f},{ci_upper:.3f}] -> gate {result['gate']}"
    )


if __name__ == "__main__":
    main()
