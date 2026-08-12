#!/usr/bin/env bash
# External validity for the truncation result: official CQN-AS (arm =
# "official", i.e. NOTHING changed except the now-default
# truncate_demo_at_success + obs_std_floor_relative) on the two hard BiGym
# tasks, 2 seeds each, one task per card.
#
# Prior numbers on these tasks are all from the COMBINED arm (untruncated):
#   move_two_plates  10.0 / 19.5  -> 14.8   (hardest)
#   sandwich_remove  52.5 / 67.5  -> 60.0
set -uo pipefail
cd "$(dirname "$0")/.."
{
  echo "[hard] 20260804095718 start $(date +%H:%M:%S)"
  bash scripts/run_cqn_trunc_arm.sh official GPU-ce804993-c33e-3d10-5676-5bae093a7d96 5 1 20260804095718 move_two_plates & sleep 120
  bash scripts/run_cqn_trunc_arm.sh official GPU-ce804993-c33e-3d10-5676-5bae093a7d96 5 2 20260804095718 move_two_plates & sleep 120
  bash scripts/run_cqn_trunc_arm.sh official GPU-03f1431f-36c0-b258-6ca1-05007175e3eb 3 1 20260804095718 sandwich_remove & sleep 120
  bash scripts/run_cqn_trunc_arm.sh official GPU-03f1431f-36c0-b258-6ca1-05007175e3eb 3 2 20260804095718 sandwich_remove &
  wait
  echo "[hard] all exited $(date +%H:%M:%S)"
} > exp_local/cqn_trunc_arms/hard_tasks_20260804095718.log 2>&1
