"""Final L2R decision and L3-or-stop handoff."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .confirm import decide_status
from .io import now_iso, read_csv, read_json, sha256_file, write_json


ALLOWED_DECISIONS = {
    "GO_L3_REWARD_GROUNDING_SINGLE_VIEW", "GO_L3_REWARD_GROUNDING_ACTIVE_MULTIVIEW",
    "L2R_PARTIAL_KEEP_COARSE_GRAPH", "STOP_VISUAL_REFINEMENT", "EXECUTION_BLOCKED",
}


def decide_final(l1v_decision: Path, source_lock: Path, predicate_metrics: Path, selection_lock: Path, confirmation: Path, paired_effects: Path, per_scenario: Path, output: Path, report: Path, final_root: Path) -> dict[str, Any]:
    l1v = read_json(l1v_decision); source = read_json(source_lock); selection = read_json(selection_lock)
    decision, reasons = decide_status(selection_lock, confirmation, paired_effects, per_scenario)
    if decision not in ALLOWED_DECISIONS:
        raise AssertionError(decision)
    predicate_rows = {row["baseline"]: row for row in read_csv(predicate_metrics)}
    predicate = predicate_rows["B3_single_view_temporal_contact_action"]
    confirmation_rows = {row["graph_id"]: row for row in read_csv(confirmation)}
    selected_metrics = confirmation_rows[selection["selected_graph_id"]]
    effects = {row["comparison"]: row for row in read_csv(paired_effects)}
    refinement_supported = decision.startswith("GO_L3_")
    retained_graph_id = selection["selected_graph_id"] if refinement_supported else "G1_predicate_bound"
    selected_graph = Path(selection["selected_graph_path"])
    graphs_dir = final_root / "graphs"; graphs_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(selected_graph, graphs_dir / selected_graph.name)
    payload = {
        "schema": "pathgraph_l2r_l3_or_stop_handoff_v1", "status": decision, "created_at": now_iso(),
        "reasons": reasons, "source": {"l1v_decision": l1v["status"], "coarse_graph_source": "V1", "source_candidates": source["candidate_count"], "source_lock_sha256": sha256_file(source_lock)},
        "dynamic_benchmark": {"kind": "dynamic MuJoCo primitive tabletop refinement benchmark", "active_objects": True, "real_robot": False},
        "predicate_interface": {"metrics": predicate, "threshold_lock_path": selection["predicate_thresholds_path"], "threshold_lock_sha256": selection["predicate_thresholds_sha256"]},
        "graph": {
            "development_selected_graph_id": selection["selected_graph_id"],
            "development_selected_graph_sha256": selection["selected_graph_sha256"],
            "retained_graph_id": retained_graph_id,
            "refinement_claim_supported": refinement_supported,
            "accepted_edits": selection["accepted_edits"],
            "fresh_confirmation_metrics": selected_metrics,
        },
        "l3_interface": {
            "entry_allowed": decision.startswith("GO_L3_"),
            "allowed_if_go": ["learn q/D/remaining cost on the frozen graph", "construct potential reward from frozen state transitions", "compare coarse, predicate-bound, and refined graph rewards"],
            "not_allowed": ["start policy training before reward validation", "claim physical robot success", "claim policy gain", "claim new-task generalization"],
            "online_observation_schema": ["front_rgb", "action_history", "gripper_command", "contact_sensor"],
        },
        "claims": {"dynamic_mujoco_primitive_evidence": True, "reward_gain": False, "policy_gain": False, "physical_robot": False, "new_task_generalization": False},
        "execution": {"training_jobs": 0, "api_calls": 0, "api_key_read": False},
    }
    payload["graph"]["selected_graph_uses_development_edits"] = selection["selected_graph_id"] in {"G2_evidence_refined", "G3_active_second_view"}
    payload["graph"]["accepted_development_edits"] = selection["accepted_edits"]
    write_json(output, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# L2R Final Report\n\n"
        f"- Decision: `{decision}`\n- L1V visual contribution: preserved from the explicit V1 rerating; not re-estimated here.\n"
        f"- Predicate binding contribution: B3 observable interface was frozen on development data and reused unchanged on fresh families.\n"
        f"- Structural edit contribution: {selection['accepted_edits']} development edits were accepted; `{selection['selected_graph_id']}` was selected on development and evaluated on fresh families. Refinement support is `{str(refinement_supported).lower()}`, so the retained graph is `{retained_graph_id}`.\n"
        f"- Active second-view contribution: {'selected' if selection['selected_graph_id'] == 'G3_active_second_view' else 'not selected'}.\n"
        f"- Predicate metrics: goal F1={predicate['goal_f1']}, stable-hold F1={predicate['stable_hold_f1']}, failure F1={predicate['failure_f1']}, recovery F1={predicate['recovery_f1']}, unknown={predicate['unknown_rate']}.\n"
        f"- Fresh selected metrics: branch accuracy={selected_metrics['branch_accuracy']}, goal precision={selected_metrics['goal_precision']}, failure recall={selected_metrics['failure_recall']}/{selected_metrics['failure_denominator']}, recovery recall={selected_metrics['recovery_recall']}/{selected_metrics['recovery_denominator']}, coverage={selected_metrics['graph_completion_coverage']}.\n"
        f"- Stop reasons: {'; '.join(reasons)}.\n"
        "- Dynamic evidence: MuJoCo primitive tabletop only.\n- Unsupported: reward gain, policy gain, physical robot success, and new-task generalization.\n",
        encoding="utf-8",
    )
    return payload
