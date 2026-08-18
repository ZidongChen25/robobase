#!/usr/bin/env python
"""T1: render the per-arm probe tables from reports/t1_probes/*.json."""

import argparse
import json
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--probe-dir", default="reports/t1_probes")
    p.add_argument("--step", default="", help="checkpoint step; default = last")
    p.add_argument("--out-md", default="")
    return p.parse_args()


def fmt(value, digits=3):
    if value is None:
        return "-"
    try:
        if value != value:  # nan
            return "nan"
        return f"{float(value):.{digits}f}"
    except (TypeError, ValueError):
        return str(value)


def main():
    args = parse_args()
    probe_dir = Path(args.probe_dir)
    files = sorted(probe_dir.glob("*.json"))
    if not files:
        raise SystemExit(f"no probe json in {probe_dir}")

    arms = {}
    for path in files:
        data = json.loads(path.read_text())
        steps = sorted(int(s) for s in data["checkpoints"])
        step = int(args.step) if args.step else steps[-1]
        arms[path.stem] = (data, step, steps)

    lines = []

    def table(title, header, rows):
        lines.append(f"### {title}\n")
        lines.append("| " + " | ".join(header) + " |")
        lines.append("|" + "|".join(["---"] * len(header)) + "|")
        for row in rows:
            lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    groups = ["demo", "heldout_success", "heldout_explore", "heldout_fail"]

    rows = []
    for name, (data, step, steps) in arms.items():
        entry = data["checkpoints"][str(step)]
        cells = [name, str(step)]
        for group in groups:
            value = entry.get(group)
            if value is None:
                cells.append("-")
                continue
            levels = value["argmax_agreement_per_level"]
            cells.append("/".join(fmt(x, 2) for x in levels))
        rows.append(cells)
    table(
        "(a) Per-level argmax agreement (L0/L1/L2), teacher-forced on the "
        "recorded action's coarse bins",
        ["arm", "step"] + groups,
        rows,
    )

    rows = []
    for name, (data, step, _) in arms.items():
        entry = data["checkpoints"][str(step)]
        for group in groups:
            value = entry.get(group)
            if value is None:
                continue
            rows.append(
                [
                    name,
                    group,
                    fmt(value["healthy_top_of_three"]),
                    fmt(value["healthy_beats_hijack"]),
                    fmt(value["healthy_beats_garbage"]),
                    fmt(value["q_healthy_mean"]),
                    fmt(value["q_hijack_mean"]),
                    fmt(value["q_garbage_mean"]),
                ]
            )
    table(
        "(b) Discrimination ranking: healthy vs hijacked vs garbage chunk",
        [
            "arm",
            "group",
            "healthy top of 3",
            "beats hijack",
            "beats garbage",
            "Q healthy",
            "Q hijack",
            "Q garbage",
        ],
        rows,
    )

    rows = []
    for name, (data, step, _) in arms.items():
        entry = data["checkpoints"][str(step)]
        cells = [name]
        for group in groups:
            value = entry.get(group)
            cells.append(
                "-"
                if value is None
                else "/".join(fmt(x, 3) for x in value["q_span_per_level"])
            )
        rows.append(cells)
    table(
        "(c) Q-span per level (max-min over the 5 sibling bins), L0/L1/L2",
        ["arm"] + groups,
        rows,
    )

    rows = []
    for name, (data, step, _) in arms.items():
        entry = data["checkpoints"][str(step)]
        value = entry.get("heldout_fail")
        if value is None:
            continue
        rows.append(
            [
                name,
                str(value["states"]),
                fmt(value["q_healthy_mean"]),
                fmt(value["q_nearest_demo_mean"]),
                fmt(value["q_minus_nearest_demo_mean"]),
                fmt(1.0 - value["healthy_beats_nearest_demo"]),
            ]
        )
    table(
        "(d) Failure-state verdict: executed failing chunk vs nearest-demo chunk",
        [
            "arm",
            "states",
            "Q(executed)",
            "Q(nearest demo)",
            "gap",
            "demo action rated better",
        ],
        rows,
    )

    rows = []
    for name, (data, step, _) in arms.items():
        entry = data["checkpoints"][str(step)]
        cells = [name, fmt(entry.get("success_fail_auc"))]
        for group in groups:
            value = entry.get(group)
            cells.append(
                "-"
                if value is None
                else f"{fmt(value['q_mc_pearson'])} / {fmt(value['q_minus_mc_mae'])}"
            )
        rows.append(cells)
    table(
        "(e) Return calibration (Pearson r / MAE of Q vs true discounted "
        "return-to-go) and success-vs-failure state AUC",
        ["arm", "succ/fail AUC"] + groups,
        rows,
    )

    text = "\n".join(lines)
    print(text)
    if args.out_md:
        Path(args.out_md).write_text(text + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
