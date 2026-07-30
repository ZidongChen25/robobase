#!/usr/bin/env python3
"""Measure coarse-to-fine zoom-path input coverage in existing replay data.

Mechanism under test: CQN-family training drives every level>0 forward pass
with the *replay action's* coarse bins (``_critic_logits_per_level`` zooms via
``encode_action(replay_action)``).  If replay level-0 bins collapse to a
narrow set per (phase, dimension), then the network inputs needed to evaluate
counterfactual coarse bins at finer levels are simply absent from training.

This script is read-only and CPU-only.  For each run's replay directory it
reports, per action dimension and per task-phase stratum:

* level-0 bin histograms and normalized entropy, split by episode group
  (demo / online_success / online_failure) and structured-exploration flag;
* prefix coverage: distinct visited level-0 / level-01 / level-012 paths and
  the sample-count occupancy above thresholds;
* modal-subtree concentration: the fraction of level-1 training inputs whose
  midpoint comes from the per-(phase, dim) modal level-0 bin;
* observational outcome contrast: within each (phase, dim), per-level-0-bin
  ``mc_return`` means/counts, which bound the identifiable between-bin signal
  and feed the RCT power analysis.

The visited-prefix sets are also saved so that later probes can classify
arbitrary Q queries as on-path / off-path against this replay support.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GROUPS = ("demo", "online_success", "online_failure")
PHASES = ("early", "middle", "late")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=REPLAY_DIR",
        help="Labelled replay directory; repeatable.",
    )
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--support-output",
        type=Path,
        help="Optional npz path storing visited prefix sets per run for "
        "later on-path/off-path classification.",
    )
    parser.add_argument("--levels", type=int, default=3)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--action-low", type=float, default=-1.0)
    parser.add_argument("--action-high", type=float, default=1.0)
    parser.add_argument(
        "--offline-episode-count",
        type=int,
        default=60,
        help="Episodes inserted before online collection (demo episodes).",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=5,
        help="Samples required before a bin counts as usably covered.",
    )
    parser.add_argument(
        "--success-reward-threshold",
        type=float,
        default=0.25,
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount for reward-to-go; computed from rewards so runs "
        "without a stored mc_return element stay comparable.",
    )
    return parser.parse_args()


def encode_action_bins(
    actions: np.ndarray,
    low: float,
    high: float,
    levels: int,
    bins: int,
) -> np.ndarray:
    """Vectorised replica of robobase.method.cqn.encode_action.

    actions: [T, D] continuous.  Returns int bins [T, levels, D].
    """
    current_low = np.full_like(actions, low, dtype=np.float64)
    current_high = np.full_like(actions, high, dtype=np.float64)
    action = actions.astype(np.float64)
    indices = []
    for _ in range(levels):
        width = np.maximum((current_high - current_low) / bins, 1e-8)
        index = np.floor((action - current_low) / width).astype(np.int64)
        index = np.clip(index, 0, bins - 1)
        indices.append(index)
        new_low = current_low + width * index
        new_high = new_low + width
        current_low = np.maximum(low, new_low)
        current_high = np.minimum(high, new_high)
    return np.stack(indices, axis=1)


@dataclass
class Transition:
    group: str
    phase: str
    structured: bool
    mc_return: float
    bins: np.ndarray  # [levels, D]


def parse_episode_length(path: Path) -> int:
    return int(path.stem.split("_")[-2])


def discounted_reward_to_go(rewards: np.ndarray, gamma: float) -> np.ndarray:
    result = np.zeros(rewards.shape[0], dtype=np.float64)
    running = 0.0
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = float(rewards[index]) + gamma * running
        result[index] = running
    return result


def load_run(
    replay_dir: Path,
    *,
    levels: int,
    bins: int,
    low: float,
    high: float,
    offline_episode_count: int,
    success_threshold: float,
    gamma: float,
) -> tuple[list[Transition], dict]:
    transitions: list[Transition] = []
    episode_counts = defaultdict(int)
    for path in sorted(replay_dir.glob("*.npz")):
        episode_index = int(path.stem.split("_")[-3])
        length = parse_episode_length(path)
        with np.load(path) as data:
            # Arrays carry one trailing sentinel row; slice to true length.
            actions = np.asarray(data["action"][:length], np.float64)
            rewards = np.asarray(data["reward"][:length], np.float64)
            structured = (
                np.asarray(data["structured_explore"][:length], bool)
                if "structured_explore" in data.files
                else np.zeros(length, dtype=bool)
            )
        mc_return = discounted_reward_to_go(rewards, gamma)
        successful = bool((rewards > success_threshold).any())
        if episode_index < offline_episode_count:
            group = "demo"
        else:
            group = "online_success" if successful else "online_failure"
        episode_counts[group] += 1
        encoded = encode_action_bins(actions, low, high, levels, bins)
        for t in range(length):
            fraction = t / max(length - 1, 1)
            phase = PHASES[min(int(fraction * 3), 2)]
            transitions.append(
                Transition(
                    group=group,
                    phase=phase,
                    structured=bool(structured[t]),
                    mc_return=float(mc_return[t]),
                    bins=encoded[t],
                )
            )
    return transitions, dict(episode_counts)


def normalized_entropy(counts: np.ndarray) -> float:
    total = counts.sum()
    if total <= 0 or counts.size <= 1:
        return 0.0
    p = counts[counts > 0] / total
    return float(-(p * np.log(p)).sum() / np.log(counts.size))


def prefix_ids(bin_matrix: np.ndarray, bins: int, depth: int) -> np.ndarray:
    """bin_matrix: [N, levels, D] -> integer prefix id [N, D] at given depth."""
    ids = np.zeros(bin_matrix.shape[::2], dtype=np.int64)  # [N, D]
    for level in range(depth):
        ids = ids * bins + bin_matrix[:, level, :]
    return ids


def analyze_run(
    transitions: list[Transition],
    *,
    levels: int,
    bins: int,
    min_count: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    if not transitions:
        raise ValueError("replay produced no transitions")
    all_bins = np.stack([t.bins for t in transitions])  # [N, levels, D]
    dims = all_bins.shape[2]
    groups = np.array([t.group for t in transitions])
    phases = np.array([t.phase for t in transitions])
    structured = np.array([t.structured for t in transitions])
    mc = np.array([t.mc_return for t in transitions])

    result: dict = {"transitions": len(transitions)}

    # --- level-0 histograms & entropy, by group -------------------------
    by_group: dict = {}
    for group in GROUPS:
        mask = groups == group
        if not mask.any():
            continue
        level0 = all_bins[mask][:, 0, :]  # [n, D]
        histograms = np.stack(
            [np.bincount(level0[:, d], minlength=bins) for d in range(dims)]
        )
        entropies = [normalized_entropy(histograms[d]) for d in range(dims)]
        modal_share = histograms.max(axis=1) / np.maximum(
            histograms.sum(axis=1), 1
        )
        by_group[group] = {
            "count": int(mask.sum()),
            "level0_entropy_mean": float(np.mean(entropies)),
            "level0_entropy_per_dim": [round(e, 4) for e in entropies],
            "level0_modal_share_mean": float(modal_share.mean()),
            "level0_modal_share_per_dim": [
                round(float(s), 4) for s in modal_share
            ],
        }
    result["by_group"] = by_group

    # --- structured-exploration split (online only) ---------------------
    online_mask = groups != "demo"
    online_structured = {}
    for label, mask in (
        ("structured", online_mask & structured),
        ("unstructured", online_mask & ~structured),
    ):
        if not mask.any():
            online_structured[label] = {"count": 0}
            continue
        level0 = all_bins[mask][:, 0, :]
        entropies = [
            normalized_entropy(np.bincount(level0[:, d], minlength=bins))
            for d in range(dims)
        ]
        online_structured[label] = {
            "count": int(mask.sum()),
            "level0_entropy_mean": float(np.mean(entropies)),
        }
    result["online_by_structured_flag"] = online_structured

    # --- prefix coverage -------------------------------------------------
    coverage = {}
    for depth in range(1, levels + 1):
        ids = prefix_ids(all_bins, bins, depth)  # [N, D]
        space = bins**depth
        visited, usable = [], []
        for d in range(dims):
            counts = np.bincount(ids[:, d], minlength=space)
            visited.append(int((counts > 0).sum()))
            usable.append(int((counts >= min_count).sum()))
        coverage[f"depth{depth}"] = {
            "space": space,
            "visited_mean": float(np.mean(visited)),
            "usable_mean": float(np.mean(usable)),
            "visited_per_dim": visited,
            "usable_per_dim": usable,
        }
    result["prefix_coverage"] = coverage

    # --- phase-stratified level-0 stats & modal-subtree concentration ---
    phase_stats = {}
    for phase in PHASES:
        pmask = phases == phase
        if not pmask.any():
            continue
        level0 = all_bins[pmask][:, 0, :]
        entry = {"count": int(pmask.sum())}
        entropies, modal_shares, usable_bins = [], [], []
        contrasts = []
        for d in range(dims):
            counts = np.bincount(level0[:, d], minlength=bins)
            entropies.append(normalized_entropy(counts))
            modal_shares.append(
                float(counts.max() / max(counts.sum(), 1))
            )
            usable = counts >= min_count
            usable_bins.append(int(usable.sum()))
            # Observational between-bin mc_return contrast.
            if usable.sum() >= 2:
                means = [
                    float(mc[pmask][level0[:, d] == b].mean())
                    for b in range(bins)
                    if usable[b]
                ]
                contrasts.append(max(means) - min(means))
        entry["level0_entropy_mean"] = float(np.mean(entropies))
        entry["level0_modal_share_mean"] = float(np.mean(modal_shares))
        entry["level0_usable_bins_mean"] = float(np.mean(usable_bins))
        entry["dims_with_2plus_usable_bins"] = int(
            sum(1 for u in usable_bins if u >= 2)
        )
        entry["mc_return_bin_contrast_mean"] = (
            float(np.mean(contrasts)) if contrasts else None
        )
        entry["mc_return_std"] = float(mc[pmask].std())
        phase_stats[phase] = entry
    result["by_phase"] = phase_stats

    # Modal-subtree concentration: share of level-1 forward-pass inputs whose
    # zoom midpoint comes from the per-(phase, dim) modal level-0 bin.
    concentration = []
    for phase in PHASES:
        pmask = phases == phase
        if not pmask.any():
            continue
        level0 = all_bins[pmask][:, 0, :]
        for d in range(dims):
            counts = np.bincount(level0[:, d], minlength=bins)
            concentration.append(counts.max() / max(counts.sum(), 1))
    result["level1_input_modal_concentration_mean"] = float(
        np.mean(concentration)
    )

    # --- artifacts for on/off-path classification -----------------------
    support: dict[str, np.ndarray] = {}
    for depth in range(1, levels + 1):
        ids = prefix_ids(all_bins, bins, depth)
        support[f"depth{depth}_counts"] = np.stack(
            [
                np.bincount(ids[:, d], minlength=bins**depth)
                for d in range(dims)
            ]
        )
    return result, support


def main() -> None:
    args = parse_args()
    report: dict = {
        "config": {
            "levels": args.levels,
            "bins": args.bins,
            "action_low": args.action_low,
            "action_high": args.action_high,
            "offline_episode_count": args.offline_episode_count,
            "min_count": args.min_count,
        },
        "runs": {},
    }
    support_arrays: dict[str, np.ndarray] = {}
    for spec in args.run:
        label, _, replay = spec.partition("=")
        if not replay:
            raise SystemExit(f"--run must be LABEL=DIR, got: {spec}")
        replay_dir = Path(replay)
        transitions, episode_counts = load_run(
            replay_dir,
            levels=args.levels,
            bins=args.bins,
            low=args.action_low,
            high=args.action_high,
            offline_episode_count=args.offline_episode_count,
            success_threshold=args.success_reward_threshold,
            gamma=args.gamma,
        )
        run_result, support = analyze_run(
            transitions,
            levels=args.levels,
            bins=args.bins,
            min_count=args.min_count,
        )
        run_result["episode_counts"] = episode_counts
        run_result["replay_dir"] = str(replay_dir)
        report["runs"][label] = run_result
        for key, value in support.items():
            support_arrays[f"{label}/{key}"] = value
        print(f"[done] {label}: {run_result['transitions']} transitions")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(f"wrote {args.output}")
    if args.support_output:
        args.support_output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.support_output, **support_arrays)
        print(f"wrote {args.support_output}")


if __name__ == "__main__":
    main()
