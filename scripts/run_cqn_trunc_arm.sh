#!/usr/bin/env bash
# Re-measure a value-line arm on TRUNCATED demos (truncate_demo_at_success is
# now the config default). One seed per invocation; two seeds share a card.
#
# Usage: run_cqn_trunc_arm.sh <ARM: combined|qc> <GPU_UUID> <EGL_ID> <SEED> [STAMP]
#
# Protocol:
#   - train to the official 101k budget, snapshot every 5k
#   - validation curve: 50 ep, seeds 400-449, every snapshot
#   - SEALED: 200 ep, seeds 800-999, at the 100k snapshot (the reporting point
#     the user fixed on 08-03) AND at 101k (the only point where the official
#     four-seed 64.625 reference exists, since its other snapshots were deleted)
#   - QC arm evaluates with --replan-interval 8 to match its training execution
set -euo pipefail
cd "$(dirname "$0")/.."

ARM="${1:?arm: combined|qc|mask|official}"
GPU_UUID="${2:?gpu uuid}"
EGL_ID="${3:?egl id}"
SEED="${4:?seed}"
STAMP="${5:-$(date +%Y%m%d%H%M%S)}"
TASK="${6:-move_plate}"
# Long-horizon tasks need replay_size_before_train >= effective episode
# length (move_two_plates 550, sandwich_remove 540); the official 500 aborts.
REPLAY_MIN="${7:-}"
PASSTHRU=("${@:8}")
PRE=(); [ -n "${REPLAY_MIN}" ] && PRE=(replay_size_before_train="${REPLAY_MIN}")

case "${ARM}" in
  combined)
    LAUNCH=cqn_as_pixel_bigym_stage158_explore_100k
    EXTRA=(); REPLAN=() ;;
  qc)
    LAUNCH=cqn_as_pixel_bigym_stage163c_official_qc8
    EXTRA=(); REPLAN=(--replan-interval 8) ;;
  mask)
    LAUNCH=cqn_as_pixel_bigym_stage161_official_mask
    EXTRA=(); REPLAN=() ;;
  official)
    LAUNCH=cqn_as_pixel_bigym_demo_driven
    EXTRA=(); REPLAN=() ;;
  rfloor)
    # R-line arm ALPHA: unseen_return_floor 0.1 + constant bc_lambda 0.0125
    # (research_paper.md 5 / cqn-rline.md wave 1).
    LAUNCH=cqn_as_pixel_bigym_rline_floor_lambda_gate
    EXTRA=(); REPLAN=() ;;
  nstep3)
    # R-line arm GAMMA: canonical + replay.nstep=3, standard execution
    # (research_paper.md 5 / cqn-rline.md wave 1).
    LAUNCH=cqn_as_pixel_bigym_rline_nstep3_gate
    EXTRA=(); REPLAN=() ;;
  tokensplit)
    # R-line wave 2: per-token horizon split (1-step tokens 1-2,
    # aux 4-step tokens 3-4), execution unchanged (cqn-rline.md wave 2).
    LAUNCH=cqn_as_pixel_bigym_rline_tokensplit_gate
    EXTRA=(); REPLAN=() ;;
  tokensplit_b8)
    # R-line wave 2b: margin-conservative split (tokens 1-8 one-step,
    # aux fraction 0.5) — brackets the aux-fraction axis vs b2.
    LAUNCH=cqn_as_pixel_bigym_rline_tokensplit_b8_gate
    EXTRA=(); REPLAN=() ;;
  cfaug)
    # R-line wave 3 D1: canonical + injected counterfactual failure
    # episodes (pre-populated <run>/replay, cqn-rline.md D1).
    LAUNCH=cqn_as_pixel_bigym_rline_cfaug_gate
    EXTRA=(); REPLAN=() ;;
  noens)
    # Train WITHOUT the agent-internal temporal ensemble (execute the newest
    # plan directly, replan every step). Replay then stores exactly the
    # actions the critic chose -- the action-semantics mismatch of
    # cqn-flow.md 20.4 disappears from the training data. Inference-side
    # smoothing is evaluated separately (ep200 with --replan-interval 1
    # turns the ensemble back on at eval).
    LAUNCH=cqn_as_pixel_bigym_demo_driven
    EXTRA=(method.temporal_ensemble=false); REPLAN=() ;;
  *) echo "unknown arm ${ARM}"; exit 2 ;;
esac

OUT="exp_local/cqn_trunc_arms/${ARM}_${TASK}"
RUN_DIR="${OUT}/seed${SEED}_${STAMP}"
mkdir -p "${OUT}"
echo "${RUN_DIR}" > "${OUT}/seed${SEED}_latest.txt"

E=(CUDA_VISIBLE_DEVICES="${GPU_UUID}" MUJOCO_EGL_DEVICE_ID="${EGL_ID}"
   MUJOCO_GL=egl XLA_PYTHON_CLIENT_PREALLOCATE=false
   JAX_PLATFORMS=cuda ROBOBASE_HOST_MERGE=1)

echo "[${ARM}/s${SEED}] train on ${GPU_UUID} egl=${EGL_ID} ($(date +%H:%M:%S))"
if ! env "${E[@]}" .venv/bin/python train_fast.py \
  launch="${LAUNCH}" \
  env=bigym/${TASK} \
  seed="${SEED}" \
  xla_mem_fraction=0.45 \
  eval_every_steps=1000000 \
  num_eval_episodes=0 \
  num_eval_envs=0 \
  log_eval_video=false \
  save_snapshot=true \
  snapshot_every_n=5000 \
  save_csv=true \
  wandb.use=false \
  "${EXTRA[@]}" "${PRE[@]}" "${PASSTHRU[@]}" \
  hydra.run.dir="${RUN_DIR}" \
  > "${RUN_DIR}.launch.log" 2>&1; then
  touch "${RUN_DIR}/failed"; echo "[${ARM}/s${SEED}] TRAIN FAILED"; exit 1
fi
touch "${RUN_DIR}/train_complete"
echo "[${ARM}/s${SEED}] train done ($(date +%H:%M:%S))"

env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --num-eval-episodes 50 --eval-seed-start 400 \
  --num-eval-envs 25 --csv-name val50_seeds400.csv "${REPLAN[@]}" \
  > "${RUN_DIR}/val50.log" 2>&1 || touch "${RUN_DIR}/val_failed"

KEEP="100000,101000"
CHECKPOINT_DIR="${RUN_DIR}/snapshots"
CHECKPOINT_SUFFIX="_snapshot.pkl"
FINALIZE=()
if compgen -G "${RUN_DIR}/eval_checkpoints/*_checkpoint.pkl" > /dev/null; then
  CHECKPOINT_DIR="${RUN_DIR}/eval_checkpoints"
  CHECKPOINT_SUFFIX="_checkpoint.pkl"
  FINALIZE=(--finalize-artifacts \
    --selection-csv val50_seeds400.csv)
fi
SKIP="$(find "${CHECKPOINT_DIR}" -maxdepth 1 \( -type f -o -type l \) \
        | sed -n "s#^.*/\([0-9][0-9]*\)${CHECKPOINT_SUFFIX}\$#\1#p" | sort -n -u \
        | awk '$0 != 100000 && $0 != 101000' | paste -sd, -)"
env "${E[@]}" .venv/bin/python scripts/eval_cqn_as_snapshot_sweep.py \
  --run-dir "${RUN_DIR}" --num-eval-episodes 200 --eval-seed-start 800 \
  --num-eval-envs 25 --csv-name ep200_seeds800.csv --skip-steps "${SKIP}" \
  "${REPLAN[@]}" "${FINALIZE[@]}" > "${RUN_DIR}/ep200.log" 2>&1 \
  || touch "${RUN_DIR}/sealed_failed"

touch "${RUN_DIR}/complete"
echo "[${ARM}/s${SEED}] sealed done ($(date +%H:%M:%S)) [kept ${KEEP}]"
grep '^\(100000\|101000\),' "${RUN_DIR}/ep200_seeds800.csv" 2>/dev/null || true
