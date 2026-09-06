"""Deterministic G0/G1/G2 graph construction with a bounded predicate DSL."""
from __future__ import annotations

from typing import Any

from .io import read_json, read_jsonl, write_csv, write_json, write_jsonl


def proposals(graph_path, diagnosis_path, continuations_path, output, report) -> dict[str, Any]:
    graph = read_json(graph_path); cases = read_jsonl(diagnosis_path) if diagnosis_path.is_file() else []
    result = []
    role_conflict_nodes = {"C04", "C05", "C08", "C10"}
    for node in graph.get("nodes", []):
        if node.get("id") in role_conflict_nodes and node.get("role") == "failure_terminal":
            cid = int(node["id"][1:])
            def counterexample(case):
                pair = case.get("transition_pair", [])
                return len(pair) == 2 and pair[1] == cid and not ({"terminal_failure", "horizon"} & set(case.get("gold_semantics", [])))
            fit = [x for x in cases if x.get("split") == "dev_fit" and counterexample(x)]
            route = [x for x in cases if x.get("split") == "dev_route" and counterexample(x)]
            result.append({"proposal_id": f"P_ROLE_{node['id']}", "operation": "narrow_role_condition", "target_handle": node["id"], "condition": {"any_of": [{"field": "terminal_failure_event", "comparison": "==", "constant": True}, {"field": "horizon", "comparison": "==", "constant": True}]}, "support_case_count": len(fit), "support_family_count": len({x["root_family_id"] for x in fit}), "validation_case_count": len(route), "validation_family_count": len({x["root_family_id"] for x in route}), "counterexample_ids": [x["case_id"] for x in fit[:5]], "status": "hypothesized"})
    write_jsonl(output, result[:12]); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Edit proposals\n\n- proposals are deterministic and evidence-linked; no LLM output is trusted directly.\n", encoding="utf-8")
    return {"status": "PASS", "proposal_count": len(result[:12])}


def select_edits(graph_path, proposals_path, output_graph, edit_log, comparison) -> dict[str, Any]:
    graph = read_json(graph_path); proposals = read_jsonl(proposals_path) if proposals_path.is_file() else []
    accepted = []
    # The first implementation deliberately accepts only edits with concrete
    # diagnostic support. In sparse data this correctly yields zero edits.
    for item in proposals:
        if int(item.get("support_family_count", 0)) >= 3 and int(item.get("validation_family_count", 0)) >= 3 and len(accepted) < 6:
            accepted.append(item)
    result = dict(graph); result["schema"] = "u4b_g2_evidence_edited_v1"; result["accepted_edits"] = accepted; result["accepted_edit_count"] = len(accepted)
    accepted_targets = {item["target_handle"]: item for item in accepted}
    for node in result.get("nodes", []):
        if node.get("id") in accepted_targets:
            node["original_role"] = node.get("role")
            node["role"] = "mixed"
            node["role_condition"] = accepted_targets[node["id"]]["condition"]
            node["role_validation_status"] = "contradicted_universal_terminal_claim"
    write_json(output_graph, result)
    write_csv(edit_log, [{"proposal_id": x.get("proposal_id"), "operation": x.get("operation"), "target_handle": x.get("target_handle"), "decision": "accepted", "reason": ">=3 dev_fit and >=3 dev_route families contradict universal terminal role", "support_family_count": x.get("support_family_count"), "validation_family_count": x.get("validation_family_count")} for x in accepted] + [{"proposal_id": x.get("proposal_id"), "operation": x.get("operation"), "target_handle": x.get("target_handle"), "decision": "rejected", "reason": "insufficient split-separated family support", "support_family_count": x.get("support_family_count"), "validation_family_count": x.get("validation_family_count")} for x in proposals if x not in accepted])
    write_csv(comparison, [{"graph_id": gid, "split": "dev_route", "transition_coverage_macro": None, "typed_occurrence_coverage": None, "unknown_rate": None, "status": "not_estimable_without_confirmation"} for gid in ("G0_raw_topology", "G1_semantic_only", "G2_evidence_edited")])
    return {"status": "PASS", "accepted_edit_count": len(accepted)}


def freeze(graph_paths, input_selection, route, claims, edit_log, family_lock, output, report) -> dict[str, Any]:
    import hashlib
    files = {}
    for path in graph_paths + [input_selection, route, claims, edit_log, family_lock]:
        if path and path.is_file(): files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
    lock = {"schema": "u4b_final_pipeline_lock_v1", "graphs": [str(x) for x in graph_paths], "input_hashes": files, "confirmation_locked": True, "confirm_anchor_rule": "first_eligible_per_claim_family", "query_order_locked": True, "edit_budget": 6, "fallback_rule": "retain_unknown_and_disclose", "metric_version": "u4b_metrics_v1"}
    write_json(output, lock); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Final pipeline lock\n\n- confirmation is now locked; no graph, mapper, boundary or semantic rule changes are allowed.\n", encoding="utf-8")
    return {"status": "PASS", "confirmation_locked": True, "input_count": len(files)}
