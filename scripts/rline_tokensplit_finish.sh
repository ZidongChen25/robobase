#!/usr/bin/env bash
# Tokensplit completion orchestrator (cqn-rline.md wave 2): the live runners
# hold the pre-fix script inode and will die at the SKIP sed after val50.
# This waits for both train_complete sentinels, then runs the whole eval
# chain itself at 3 processes per card (user directive):
#   val50 seeds400 + seeds450 for both seeds, split across GPU2+GPU5,
#   then sealed 200ep @100k/101k, then the paired probe batch
#   (tokensplit x2 + baseline x2, one batch, shared data source).
set -uo pipefail
cd "$(dirname "$0")/.."

U2=GPU-80b9cc0d-df5c-be12-e848-042d37578544   # GPU2 -> EGL 2
U5=GPU-2f044e6a-9150-0e30-7d97-009bdd425b11   # GPU5 -> EGL 1
D1=exp_local/cqn_trunc_arms/tokensplit_move_plate/seed1_20260809rline2
D2=exp_local/cqn_trunc_arms/tokensplit_move_plate/seed2_20260809rline2

until [ -f "${D1}/train_complete" ] && [ -f "${D2}/train_complete" ]; do
  sleep 120
done
echo "[tsfinish] both tokensplit runs trained ($(date +%H:%M:%S))"
# Give the doomed runners a moment to run their own val50 then die at sed;
# kill them to avoid double-evaluating.
sleep 30
pkill -f "run_cqn_trunc_arm.sh tokensplit" 2>/dev/null || true

sweep() {
  local D=$1 UUID=$2 EGL=$3 START=$4 CSV=$5
  env CUDA_VISIBLE_DEVICES="${UUID}" MUJOCO_EGL_DEVICE_ID="${EGL}" \
      MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${D}" --num-eval-episodes 50 --eval-seed-start "${START}" \
    --num-eval-envs 25 --csv-name "${CSV}" \
    > "${D}/${CSV%.csv}.log" 2>&1
}

# 3 procs per card, 25 s stagger (EGL context race):
sweep "${D1}" "${U2}" 2 400 val50_seeds400.csv & sleep 25
sweep "${D1}" "${U2}" 2 450 val50_seeds450.csv & sleep 25
sweep "${D2}" "${U2}" 2 400 val50_seeds400.csv &
P_GPU2=$!
sweep "${D2}" "${U5}" 1 450 val50_seeds450.csv &
P_GPU5=$!
wait
echo "[tsfinish] val50 double-band sweeps done ($(date +%H:%M:%S))"

sealed() {
  local D=$1 UUID=$2 EGL=$3
  local SKIP
  SKIP="$(find "${D}/eval_checkpoints" -maxdepth 1 \( -type f -o -type l \) \
          | sed -n "s#^.*/\([0-9][0-9]*\)_checkpoint.pkl\$#\1#p" | sort -n -u \
          | awk '$0 != 100000 && $0 != 101000' | paste -sd, -)"
  env CUDA_VISIBLE_DEVICES="${UUID}" MUJOCO_EGL_DEVICE_ID="${EGL}" \
      MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false JAX_PLATFORMS=cuda \
    .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
    --run-dir "${D}" --num-eval-episodes 200 --eval-seed-start 800 \
    --num-eval-envs 25 --csv-name ep200_seeds800.csv --skip-steps "${SKIP}" \
    --finalize-artifacts --selection-csv "${D}/val50_seeds400.csv" \
    > "${D}/ep200.log" 2>&1
  touch "${D}/complete"
  grep '^\(100000\|101000\),' "${D}/ep200_seeds800.csv" | sed "s#^#[sealed ${D##*/}] #"
}
sealed "${D1}" "${U2}" 2 &
sealed "${D2}" "${U5}" 1 &
wait
echo "[tsfinish] sealed done ($(date +%H:%M:%S))"

OUT="exp_local/cqn_rline/probe_batch_tokensplit_$(date +%Y%m%d%H%M%S)"
DATA="exp_local/cqn_trunc_arms/official_basestate_move_plate/seed2_20260806093205"
mkdir -p "${OUT}"
probe() {
  local NAME=$1 DIR=$2 CKPT=$3
  JAX_PLATFORMS=cpu .venv/bin/python scripts/analyze_cqn_value_fidelity.py \
    --run-dir "${DIR}" --snapshot "${CKPT}" --data-run-dir "${DATA}" \
    --output "${OUT}/${NAME}.json" \
    --samples-per-group 48 --batch-size 16 --seed 7 \
    --offline-episode-count 60 --groups demo_success \
    > "${OUT}/${NAME}.log" 2>&1 || echo "[probe] ${NAME} FAILED"
}
B1=exp_local/cqn_trunc_arms/official_basestate_move_plate/seed1_20260806093205
B2=exp_local/cqn_trunc_arms/official_basestate_move_plate/seed2_20260806101521
probe baseline_s1 "${B1}" "${B1}/eval_checkpoints/100000_checkpoint.pkl"
probe baseline_s2 "${B2}" "${B2}/eval_checkpoints/100000_checkpoint.pkl"
probe tokensplit_s1 "${D1}" "${D1}/eval_checkpoints/100000_checkpoint.pkl"
probe tokensplit_s2 "${D2}" "${D2}/eval_checkpoints/100000_checkpoint.pkl"
.venv/bin/python - "$OUT" <<'EOF'
import json, sys, pathlib
out = pathlib.Path(sys.argv[1])
print(f"{'name':16s} {'spearman':>9s} {'top1':>7s} {'agree':>7s} {'top2gap':>8s}")
for f in sorted(out.glob("*.json")):
    d = json.load(open(f))
    s = d.get("summary", {}).get("all", {})
    fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
    print(f"{f.stem:16s} {fmt(s.get('value',{}).get('q_raw_return_spearman')):>9s}"
          f" {fmt(s.get('imitation',{}).get('replay_bin_top1_rate')):>7s}"
          f" {fmt(s.get('imitation',{}).get('greedy_bin_agreement')):>7s}"
          f" {fmt(s.get('collapse',{}).get('candidate_top2_gap')):>8s}")
EOF
echo "[tsfinish] all complete ($(date +%H:%M:%S)) -> ${OUT}"
