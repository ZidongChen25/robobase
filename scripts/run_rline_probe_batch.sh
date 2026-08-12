#!/usr/bin/env bash
# R-line mechanism gate (cqn-rline.md): paired value-fidelity probes on the
# 100k checkpoints of all wave-1 arms plus both baseline seeds, in ONE batch
# with one shared data source and one probe seed (A21 instrument rule: never
# compare probe values across separately-run batches).
# Data source: the aborted baseline seed2 attempt keeps demo_replay (51
# successful demos) + 62 early online episodes; demo states are the shared
# probe currency across all critics.
# CPU is sufficient (~2-4 min per snapshot); pass a GPU uuid to speed up.
set -uo pipefail
cd "$(dirname "$0")/.."

STAMP="${1:-$(date +%Y%m%d%H%M%S)}"
OUT="exp_local/cqn_rline/probe_batch_${STAMP}"
DATA="exp_local/cqn_trunc_arms/official_basestate_move_plate/seed2_20260806093205"
mkdir -p "${OUT}"

probe() {
  local NAME=$1 DIR=$2 CKPT=$3
  [ -f "${CKPT}" ] || { echo "[probe] MISSING ${CKPT}"; return 1; }
  echo "[probe] ${NAME} ($(date +%H:%M:%S))"
  JAX_PLATFORMS=cpu .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${DIR}" --snapshot "${CKPT}" --data-run-dir "${DATA}" \
    --output "${OUT}/${NAME}.json" \
    --samples-per-group 48 --batch-size 16 --seed 7 \
    --offline-episode-count 60 --groups demo_success \
    > "${OUT}/${NAME}.log" 2>&1 || echo "[probe] ${NAME} FAILED"
}

B1=$(cat exp_local/cqn_trunc_arms/official_basestate_move_plate/seed1_latest.txt)
B2=$(cat exp_local/cqn_trunc_arms/official_basestate_move_plate/seed2_latest.txt)
probe baseline_s1 "${B1}" "${B1}/eval_checkpoints/100000_checkpoint.pkl"
probe baseline_s2 "${B2}" "${B2}/eval_checkpoints/100000_checkpoint.pkl"

for ARM in rfloor nstep3; do
  for SEED in 1 2; do
    D="exp_local/cqn_trunc_arms/${ARM}_move_plate/seed${SEED}_20260809rline1"
    CKPT="${D}/eval_checkpoints/100000_checkpoint.pkl"
    [ -f "${CKPT}" ] || CKPT="${D}/snapshots/100000_snapshot.pkl"
    probe "${ARM}_s${SEED}" "${D}" "${CKPT}"
  done
done

.venv/bin/python - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
rows = []
for f in sorted(out.glob("*.json")):
    d = json.load(open(f))
    s = d.get("summary", {}).get("all", {})
    rows.append((
        f.stem,
        s.get("value", {}).get("q_raw_return_spearman"),
        s.get("imitation", {}).get("replay_bin_top1_rate"),
        s.get("imitation", {}).get("greedy_bin_agreement"),
        s.get("collapse", {}).get("candidate_top2_gap"),
    ))
print(f"{'name':14s} {'spearman':>9s} {'top1':>7s} {'agree':>7s} {'top2gap':>8s}")
for name, sp, t1, ag, gap in rows:
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"{name:14s} {fmt(sp):>9s} {fmt(t1):>7s} {fmt(ag):>7s} {fmt(gap):>8s}")
EOF
echo "[probe] batch complete -> ${OUT}"
