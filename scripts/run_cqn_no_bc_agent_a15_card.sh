#!/usr/bin/env bash
# Per-card A15 orchestrator: two arms, trainings in parallel (staggered),
# then evals in parallel after both trainings exit.
# Usage: run_cqn_no_bc_agent_a15_card.sh GPU ARM1 LAUNCH1 OV1 ARM2 LAUNCH2 OV2
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="$1"; A1N="$2"; L1="$3"; O1="$4"; A2N="$5"; L2="$6"; O2="$7"
S=scripts/run_cqn_no_bc_agent_a15_screen.sh
if [[ -n "$O1" ]]; then bash "$S" "$GPU" "$A1N" train "$L1" "$O1" & else bash "$S" "$GPU" "$A1N" train "$L1" & fi
p1=$!
sleep 120
if [[ -n "$O2" ]]; then bash "$S" "$GPU" "$A2N" train "$L2" "$O2" & else bash "$S" "$GPU" "$A2N" train "$L2" & fi
p2=$!
st=0; wait "$p1" || st=$?; st2=0; wait "$p2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "train failed $st/$st2" > "exp_local/cqn_no_bc/a15_${A1N}_${A2N}_failed"; exit 1; fi
if [[ -n "$O1" ]]; then bash "$S" "$GPU" "$A1N" eval "$L1" "$O1" & else bash "$S" "$GPU" "$A1N" eval "$L1" & fi
e1=$!
if [[ -n "$O2" ]]; then bash "$S" "$GPU" "$A2N" eval "$L2" "$O2" & else bash "$S" "$GPU" "$A2N" eval "$L2" & fi
e2=$!
st=0; wait "$e1" || st=$?; st2=0; wait "$e2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "eval failed $st/$st2" > "exp_local/cqn_no_bc/a15_${A1N}_${A2N}_failed"; exit 1; fi
touch "exp_local/cqn_no_bc/a15_${A1N}_${A2N}_complete"
