"""Bounded U3B gate with a no-call path by default."""
from __future__ import annotations
from .io import read_json, write_json, write_csv, read_jsonl


def prepare(route_path, api_authorized, output):
    route = read_json(route_path); triggered = route.get("route") == "RUN_U3B"
    output.mkdir(parents=True, exist_ok=True)
    status = "U3B_READY" if triggered and api_authorized else "U3B_DEFERRED_USE_FALLBACK" if triggered else "not_triggered"
    write_json(output / "execution_plan.json", {"schema": "u4b_u3b_execution_plan_v1", "status": status, "api_authorized": bool(api_authorized), "max_total_sends": 4, "qwen_calls": 2, "deepseek_calls": 1})
    write_json(output / "handle_catalog.json", {"nodes": [], "edges": [], "predicates": []})
    write_json(output / "edit_schema.json", {"allowed_operations": ["merge_nodes", "split_node", "retype_edge", "narrow_role_condition"], "max_edits": 2})
    return {"status": "PASS", "u3b_status": status}


def run(plan_path, output, ledger):
    plan = read_json(plan_path); output.mkdir(parents=True, exist_ok=True)
    status = plan.get("status", "not_triggered")
    write_csv(ledger, [{"status": status, "actual_sends": 0, "api_key_read": False, "reason": "no authorized request was required" if status != "U3B_READY" else "provider adapter not invoked by fallback implementation"}])
    return {"status": "PASS", "u3b_status": status, "actual_sends": 0}
