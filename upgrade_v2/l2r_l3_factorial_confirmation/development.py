from __future__ import annotations

from pathlib import Path

from upgrade_v2.l2r_loss_observability_fusion.fusion import run_fusion
from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_jsonl, write_json
from upgrade_v2.l2r_recovery_supervisor import RecoverySupervisorV2

from .forensics import CLP3, build_forensics


def _proposal_count(path: Path) -> int:
    rows = read_jsonl(path / "candidate_input/live_observations.jsonl")
    return sum(row["selected_action"] == "recover_object" for row in run_fusion(rows))


def _supervisor_scenarios() -> dict[str, bool]:
    supervisor = RecoverySupervisorV2()

    def run(executions, goals, losses):
        return supervisor.run(
            initial_action="recover_object",
            execute_recovery=lambda action, cycle: executions[cycle - 1],
            verify_goal=lambda cycle: goals[cycle - 1],
            next_loss_action=lambda cycle: losses[cycle - 1],
            rearm_after_verified_hold=lambda cycle: None,
        )

    relocation = run([{"success": False, "failure_stage": "RELOCATION_ERROR"}, {"success": True}],
                     [False, True], [None, None])
    goal = run([{"success": True}, {"success": True}], [False, True], [None, None])
    secondary = run([{"success": True}, {"success": True}], [False, True], ["recover_object", None])
    bounded = run([{"success": False, "failure_stage": "RELOCATION_ERROR"}] * 3,
                  [False] * 3, [None] * 3)
    return {
        "relocation_retry_succeeds": relocation["success"] and relocation["outer_cycles"] == 2,
        "goal_verification_retry_succeeds": goal["success"] and goal["outer_cycles"] == 2,
        "secondary_loss_rearmed": secondary["success"] and secondary["rearmed_loss_episodes"] == 2,
        "permanent_failure_bounded": not bounded["success"] and bounded["outer_cycles"] == 3,
    }


def validate_development(r24_root: Path, output: Path) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    forensics = build_forensics(r24_root, output / "forensics")
    signal_paths = sorted(r24_root.glob(f"*__R24C[24]_*__{CLP3}"))
    negative_paths = sorted(r24_root.glob(f"*__R24C[35]_*__{CLP3}"))
    signal_counts = {path.name: _proposal_count(path) for path in signal_paths}
    negative_counts = {path.name: _proposal_count(path) for path in negative_paths}
    supervisor = _supervisor_scenarios()
    gates = {
        "r24_signal_not_observed_8": len(signal_paths) == 8,
        "fusion_proposes_on_all_8": len(signal_paths) == 8 and all(value == 1 for value in signal_counts.values()),
        "partial_and_release_false_proposals_0": len(negative_paths) == 8 and all(value == 0 for value in negative_counts.values()),
        "signal_not_observed_candidate_errors_0": forensics["signal_not_observed_candidate_errors"] == 0,
        "forensic_observation_execution_separation": forensics["observation_and_execution_separated"],
        **supervisor,
    }
    result = {
        "schema": "l2rar2_r25_development_gate_v1",
        "status": "PASS" if all(gates.values()) else "FAIL",
        "gates": gates,
        "fusion_signal_episode_counts": signal_counts,
        "fusion_negative_episode_counts": negative_counts,
        "supervisor_scenarios": supervisor,
        "r24_used_as_development_only": True,
        "confirmation_parameters_frozen_after_this_gate": True,
    }
    write_json(output / "development_gate.json", result)
    return result
