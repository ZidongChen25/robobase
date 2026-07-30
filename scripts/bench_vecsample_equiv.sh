#!/usr/bin/env bash
# Infra acceptance for the vectorized replay assembly + device-side demo merge
# (cqn-flow.md 48/48.1): 3000-frame same-seed A/B on the crown launch config.
#   A = ROBOBASE_SCALAR_SAMPLE=1 ROBOBASE_HOST_MERGE=1   (old prod path)
#   B = defaults                     (slice-copy sampling + device merge)
# ABBA order so neither arm systematically inherits page-cache/driver warm-up;
# metric = steady-state steps/s over env_steps 1000->3000 from train.csv
# (excludes JIT compile + startup), plus critic_loss trajectory diffs with the
# within-arm rerun drift (A1 vs A2, B1 vs B2) as the equivalence yardstick.
# User directive: infra benchmarks run on GPU5 (override: first arg).
set -uo pipefail
cd "$(dirname "$0")/.."

GPU="${1:-5}"
STAMP="$(date +%Y%m%d%H%M%S)"
OUT="exp_local/infra_bench_vecsample"
mkdir -p "${OUT}"

run_arm () {
  local TAG="$1"; shift
  local DIR="${OUT}/bench3k_${TAG}_gpu${GPU}_${STAMP}"
  echo "[bench] run ${TAG} on GPU${GPU} ($(date +%H:%M:%S))"
  env "$@" MUJOCO_GL=egl CUDA_VISIBLE_DEVICES="${GPU}" \
    MUJOCO_EGL_DEVICE_ID="${GPU}" \
    .venv/bin/python train_fast.py \
    launch=cqn_as_pixel_bigym_stage163b_qc_nstep8 \
    env=bigym/move_plate \
    seed=1 \
    num_train_frames=3000 \
    save_snapshot=false \
    save_csv=true \
    wandb.use=false \
    hydra.run.dir="${DIR}" \
    > "${DIR}.log" 2>&1 || { echo "[bench] run ${TAG} FAILED (see ${DIR}.log)"; exit 1; }
  echo "[bench] run ${TAG} done ($(date +%H:%M:%S))"
}

OLD=(ROBOBASE_SCALAR_SAMPLE=1 ROBOBASE_HOST_MERGE=1)
NEW=(ROBOBASE_PLACEHOLDER=)

run_arm old1 "${OLD[@]}"
run_arm new1 "${NEW[@]}"
run_arm new2 "${NEW[@]}"
run_arm old2 "${OLD[@]}"

.venv/bin/python - "$OUT" "$STAMP" "$GPU" <<'EOF'
import sys
from pathlib import Path
import numpy as np

out, stamp, gpu = sys.argv[1:4]
runs = {}
for tag in ("old1", "new1", "new2", "old2"):
    csv = Path(f"{out}/bench3k_{tag}_gpu{gpu}_{stamp}/train.csv")
    data = np.genfromtxt(csv, delimiter=",", names=True)
    steps, times = data["env_steps"], data["total_time"]
    window = (steps >= 1000) & (steps <= 3000)
    span_steps = steps[window][-1] - steps[window][0]
    span_time = times[window][-1] - times[window][0]
    runs[tag] = {
        "sps": span_steps / span_time,
        "loss": data["critic_loss"],
        "steps": steps,
    }
    print(f"{tag}: steady-state {runs[tag]['sps']:.2f} steps/s "
          f"(window {steps[window][0]:.0f}->{steps[window][-1]:.0f})")

old_sps = np.mean([runs["old1"]["sps"], runs["old2"]["sps"]])
new_sps = np.mean([runs["new1"]["sps"], runs["new2"]["sps"]])
print(f"\nsteady-state speedup: {new_sps / old_sps:.2f}x "
      f"(old {old_sps:.2f} -> new {new_sps:.2f} steps/s)")

def loss_drift(a, b):
    n = min(len(runs[a]["loss"]), len(runs[b]["loss"]))
    la, lb = runs[a]["loss"][:n], runs[b]["loss"][:n]
    rel = np.abs(la - lb) / np.maximum(np.abs(la), 1e-8)
    return np.median(rel) * 100, np.max(rel) * 100

for pair in (("old1", "old2"), ("new1", "new2"), ("old1", "new1"),
             ("old2", "new2")):
    med, mx = loss_drift(*pair)
    kind = "within-arm (rerun drift)" if pair[0][:3] == pair[1][:3] else "cross-arm"
    print(f"critic_loss drift {pair[0]} vs {pair[1]}: "
          f"median {med:.2f}%  max {mx:.2f}%   [{kind}]")

print("\nVerdict guide: cross-arm median drift should sit within the "
      "within-arm rerun drift; steady-state speedup is the wall-clock claim.")
EOF
