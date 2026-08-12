#!/usr/bin/env bash
# A16 card wave: two (arm, seed) specs on one card; trainings parallel,
# evals after both. Usage: GPU ARM1 L1 OV1 S1 ARM2 L2 OV2 S2
set -euo pipefail
cd "$(dirname "$0")/.."
GPU="$1"; A1N="$2"; L1="$3"; O1="$4"; S1="$5"; A2N="$6"; L2="$7"; O2="$8"; S2="$9"
S=scripts/run_cqn_no_bc_agent_a15_screen.sh
run () { local arm="$1" l="$2" ov="$3" seed="$4" phase="$5"; if [[ -n "$ov" ]]; then A15_SEED="$seed" bash "$S" "$GPU" "$arm" "$phase" "$l" "$ov"; else A15_SEED="$seed" bash "$S" "$GPU" "$arm" "$phase" "$l"; fi; }
run "$A1N" "$L1" "$O1" "$S1" train & p1=$!
sleep 120
run "$A2N" "$L2" "$O2" "$S2" train & p2=$!
st=0; wait "$p1" || st=$?; st2=0; wait "$p2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "$st/$st2" > "exp_local/cqn_no_bc/a16w_${A1N}${S1}_${A2N}${S2}_failed"; exit 1; fi
run "$A1N" "$L1" "$O1" "$S1" eval & e1=$!
run "$A2N" "$L2" "$O2" "$S2" eval & e2=$!
st=0; wait "$e1" || st=$?; st2=0; wait "$e2" || st2=$?
if [[ "$st" -ne 0 || "$st2" -ne 0 ]]; then echo "eval $st/$st2" > "exp_local/cqn_no_bc/a16w_${A1N}${S1}_${A2N}${S2}_failed"; exit 1; fi
touch "exp_local/cqn_no_bc/a16w_${A1N}${S1}_${A2N}${S2}_complete"
