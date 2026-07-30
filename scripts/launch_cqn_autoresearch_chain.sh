#!/usr/bin/env bash
set -euo pipefail

# Launch the event-driven CQN autoresearch dependency chain.  This launcher
# never terminates a live controller; callers must first verify that any
# superseded controllers are idle and stop only those process groups.

cd /home/zc1525/robobase_jaxflat

launch_controller() {
  local control=$1
  local script=$2
  local stamp
  local script_snapshot
  stamp=$(date +%Y%m%dT%H%M%S.%N)
  mkdir -p "$control"

  if ! test -f "$script"; then
    printf 'controller script does not exist: %s\n' "$script" >&2
    return 1
  fi

  if test -s "$control/controller.pid"; then
    local old_pid
    old_pid=$(sed -n '1p' "$control/controller.pid")
    if kill -0 "$old_pid" 2>/dev/null; then
      printf 'controller is already live: %s pid=%s\n' \
        "$control" "$old_pid" >&2
      return 1
    fi
  fi

  for artifact in controller.pid controller.log complete failed; do
    if test -e "$control/$artifact"; then
      mv \
      "$control/$artifact" \
      "$control/${artifact}.superseded.${stamp}"
    fi
  done

  # A controller may wait for hours before it reaches later stages.  Execute an
  # immutable launch-time copy so edits to the source script cannot alter a
  # live research chain halfway through the run.
  script_snapshot="$control/controller_script.${stamp}.sh"
  cp -- "$script" "$script_snapshot"
  chmod a-w -- "$script_snapshot"
  sha256sum -- "$script_snapshot" > "$script_snapshot.sha256"

  setsid -f env \
    CQN_AUTORESEARCH_CONTROL="$control" \
    CQN_AUTORESEARCH_SCRIPT="$script_snapshot" \
    bash -lc '
      set -euo pipefail
      cd /home/zc1525/robobase_jaxflat
      control=$CQN_AUTORESEARCH_CONTROL
      script=$CQN_AUTORESEARCH_SCRIPT
      printf "%s\n" "$BASHPID" > "$control/controller.pid"
      trap '\''touch "$control/failed"'\'' ERR
      bash "$script"
      touch "$control/complete"
    ' < /dev/null > "$control/controller.log" 2>&1

  local attempt
  for attempt in $(seq 1 100); do
    if test -s "$control/controller.pid"; then
      local new_pid
      new_pid=$(sed -n '1p' "$control/controller.pid")
      if kill -0 "$new_pid" 2>/dev/null; then
        printf '%s pid=%s\n' "$control" "$new_pid"
        return 0
      fi
    fi
    sleep 0.05
  done
  printf 'controller did not materialize: %s\n' "$control" >&2
  return 1
}

launch_controller \
  exp_local/cqn_flow_high_utd/stage74_80_fidelity_recovery_master \
  scripts/run_cqn_stage74_80_fidelity_recovery.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage82_83_td_target_mechanism_master \
  scripts/run_cqn_stage82_83_td_target_mechanism.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage84_87_route_a_master \
  scripts/run_cqn_stage84_87_route_a.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage88_90_td_target_final_master \
  scripts/run_cqn_stage88_90_td_target_final.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage91_final_summary_master \
  scripts/run_cqn_stage91_final_summary.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage92_93_bc_policy_target_master \
  scripts/run_cqn_stage92_93_bc_policy_target.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage94_97_bc_policy_final_master \
  scripts/run_cqn_stage94_97_bc_policy_final.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage98_101_h1_cf_fqe_master \
  scripts/run_cqn_stage98_101_h1_cf_fqe.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage102_105_floq_full_interaction_master \
  scripts/run_cqn_stage102_105_floq_full_interaction.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage106_fallback_aggregate_master \
  scripts/run_cqn_stage106_fallback_aggregate.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage107_113_value_flows_confidence_master \
  scripts/run_cqn_stage107_113_value_flows_confidence.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage114_118_flowcritic_truncated_master \
  scripts/run_cqn_stage114_118_flowcritic_truncated.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage119_flow_utilization_master \
  scripts/run_cqn_stage119_flow_utilization.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage120_126_action_centered_master \
  scripts/run_cqn_stage120_126_action_centered_route_a.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage127_133_evor_flowtd_master \
  scripts/run_cqn_stage127_133_evor_flowtd.sh
launch_controller \
  exp_local/cqn_flow_high_utd/stage134_unbiased_dimension_confirmation_master \
  scripts/run_cqn_stage134_unbiased_dimension_confirmation.sh

launch_controller \
  exp_local/cqn_flow_high_utd/stage135_qr_flowiqn_smoke_master \
  scripts/run_cqn_stage135_qr_flowiqn_smoke.sh
