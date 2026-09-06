"""Guarded U2-R interface. It never trains unless the route explicitly asks."""
from __future__ import annotations
from .io import read_json, write_csv, write_json


def prepare(route_path, output, ledger):
    route = read_json(route_path)
    triggered = route.get("route") == "RUN_U2R"
    output.mkdir(parents=True, exist_ok=True)
    write_json(output / "repair_plan.json", {"schema": "u4b_u2r_plan_v1", "triggered": triggered, "repair_type": "rule_only" if triggered else "not_triggered", "max_jobs": 3, "max_steps": 1200, "max_clips": 40, "max_unique_frames": 480})
    write_csv(ledger, [{"route": route.get("route"), "triggered": triggered, "new_clips": 0, "new_unique_frames": 0, "status": "not_triggered" if not triggered else "prepared_rule_only"}])
    return {"status": "PASS", "triggered": triggered}


def select_or_fallback(repair_root, output):
    plan = read_json(repair_root / "repair_plan.json")
    result = {"schema": "u4b_selected_boundary_v1", "source": "original_boundary", "repair_type": plan.get("repair_type"), "status": "U2R_NOT_TRIGGERED" if not plan.get("triggered") else "U2R_RULE_ONLY_PENDING_COMPARISON"}
    write_json(output, result); return result
