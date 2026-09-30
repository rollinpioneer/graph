"""First-decision reference bank under shared B_PLAN continuation only."""
from __future__ import annotations

import csv
import math
from pathlib import Path

import numpy as np

from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
from cp_disr.contracts import nominal_overlay, precondition_value
from cp_disr.facts import Truth
from .family_b_pilot import read, write, _registered
from .family_b_representation import _snapshot

PADS = ("pad_u", "pad_v")


def _plan_options(snapshot, reference):
    planner = BPlanPlanner(SearchConfig(
        depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0,
        reference_skill_seconds=reference))
    result = planner.plan(snapshot.facts, snapshot.template, 60.)
    if result.status != "PLAN_FOUND" or not result.plan:
        return result, []
    by_id = {c.id: c for c in snapshot.template.contracts}
    initial = dict(snapshot.facts.values)
    choices = []
    for pad in PADS:
        first = f"a:PLACE_BUFFER:carrier:{pad}:v1"
        plan = (first,) + tuple(result.plan[1:])
        values = dict(initial)
        valid = True
        changes = {}
        for i, aid in enumerate(plan):
            contract = by_id[aid]
            if precondition_value(contract, values) is not Truth.TRUE:
                valid = False
                break
            after = dict(nominal_overlay(contract, values))
            changes[i] = {fid for fid in values if after[fid] != values[fid]}
            values = after
        if valid and all(values.get(g.fact_id) is Truth.TRUE for g in snapshot.template.goals):
            choices.append({"candidate_id": first, "plan": plan,
                            "cost": len(plan)*reference, "changed": changes})
    return result, choices


def _q_r(choice, edges, unmet):
    plan = choice["plan"]
    score = 0
    for source, target, kind, anchor in edges:
        if source not in plan:
            continue
        index = plan.index(source)
        if anchor not in choice["changed"].get(index, set()):
            continue
        if kind == "SOFT_SUPPORTS" and target in plan[index+1:]:
            score += 1
        elif kind == "SOFT_RELEVANT_TO_GOAL" and target in unmet:
            score += 1
    return score/max(1, len(edges))


def _geometry(public):
    xyz = public.get("public_object_xyz") or {}
    layout = public.get("public_layout") or {}
    values = public["fact_values"]
    pending = [obj for obj in ("obj_b", "obj_c")
               if values.get(f"p:Inside:{obj}:receiver") != "TRUE"]
    if not pending or any(xyz.get(obj) is None for obj in pending):
        return None
    show = np.array([0.02, -0.08])
    costs = {}
    for pad in PADS:
        p = np.asarray(layout[pad][:2], dtype=float)
        first = float(np.linalg.norm(show-p))
        if len(pending) == 1:
            future = float(np.linalg.norm(p-np.asarray(xyz[pending[0]][:2])))
        else:
            b, c = (np.asarray(xyz[obj][:2], dtype=float) for obj in pending)
            receiver = np.asarray(layout["receiver"][:2], dtype=float)
            future = min(
                np.linalg.norm(p-b)+np.linalg.norm(receiver-c),
                np.linalg.norm(p-c)+np.linalg.norm(receiver-b))
        costs[pad] = first+future
    winner = min(PADS, key=lambda pad: (costs[pad], pad))
    return f"a:PLACE_BUFFER:carrier:{winner}:v1", costs


def _relations(out, layout, repeat):
    path = Path(out) / "provider/relation_request_ledger.json"
    if not path.is_file():
        return None
    target = f"family_b_{layout}_repeat{repeat}"
    entries = read(path)["entries"]
    row = next((e for e in entries if e["scene_id"] == target), None)
    if row is None or row.get("state") != "SUCCESS":
        reused_path = Path(out) / "provider/reused_initial_sources.json"
        reused = read(reused_path) if reused_path.is_file() else []
        alias = next((r for r in reused if r["scene_id"] == target), None)
        row = next((e for e in entries if e.get("cache_key") == alias["cache_key"]), None) if alias else None
    if row is None or row.get("state") != "SUCCESS":
        return None
    from cp_disr.vlm_cache_pipeline import verify_audit_cache
    _, edges = verify_audit_cache(row["cache_path"])
    accepted = read(Path(row["cache_path"]) / "accepted_relations.json")
    return [(r["source_ref"], r["target_ref"], r["type"], r["effect_fact_ref"])
            for r in accepted]


def evaluate(root, cfg, out):
    root, out = Path(root), Path(out)
    reference = float(read(out / "bindings/binding_manifest.json")["reference_skill_seconds"])
    branches = _registered(out)
    action_ledger = out / "provider/action_request_ledger.json"
    action_rows = read(action_ledger)["entries"] if action_ledger.is_file() else []
    rows = []
    seen = set()
    for b in branches:
        key = (b["layout"], b["context"], b["repeat"])
        if key in seen or b["candidate"] != "pad_u":
            continue
        seen.add(key)
        cap = out / "captures" / b["branch_id"] / "boundary"
        if not (cap / "public.json").is_file() or not (cap / "snapshot_encoding.json").is_file():
            continue
        public = read(cap / "public.json")
        snapshot = _snapshot(root, cfg, out, b, ())
        plan_result, options = _plan_options(snapshot, reference)
        option_ids = {o["candidate_id"] for o in options}
        canonical = plan_result.plan[0] if plan_result.status == "PLAN_FOUND" and plan_result.plan else None
        geometry = _geometry(public)
        geom_choice = geometry[0] if geometry else None
        relations = _relations(out, b["layout"], b["repeat"])
        r_choice = None
        r_status = "NATURAL_R_UNAVAILABLE"
        if relations is not None and options:
            unmet = {g.fact_id for g in snapshot.template.goals
                     if snapshot.facts.values[g.fact_id] is not Truth.TRUE}
            scores = {o["candidate_id"]: _q_r(o, relations, unmet) for o in options}
            near = [o for o in options if o["cost"] <= min(x["cost"] for x in options)+0.1*reference]
            r_choice = min(near, key=lambda o: (-scores[o["candidate_id"]], o["cost"], o["candidate_id"]))["candidate_id"]
            r_status = ("HEURISTIC_NONDISCRIMINATIVE" if len(set(scores.values())) == 1 else "SCORED")
        action_row = next((r for r in action_rows if r["scene_id"] == f'{b["layout"]}_{b["context"]}_repeat0'), None)
        vlm = action_row.get("candidate_id") if b["repeat"] == 0 and action_row and action_row["state"] == "SUCCESS" else None
        paired = {x["candidate"]: x for x in branches
                  if (x["layout"], x["context"], x["repeat"]) == key}
        outcomes = {}
        for pad, branch in paired.items():
            result_path = out / "physical/branch_results" / f'{branch["branch_id"]}.json'
            outcomes[f"a:PLACE_BUFFER:carrier:{pad}:v1"] = read(result_path) if result_path.is_file() else None
        success = {cid: r["time_to_task_success"] for cid, r in outcomes.items()
                   if r and r.get("task_success") and r.get("time_to_task_success") is not None}
        comparable = len(success) == 2
        best = min(success.values()) if comparable else None
        for method, choice, status in (
            ("B_PLAN", canonical, plan_result.status),
            ("PUBLIC_GEOMETRY", geom_choice, "MEASURED" if geometry else "PUBLIC_POSITION_UNKNOWN"),
            ("B_PLAN+R", r_choice, r_status),
            ("B_PLAN+R*", None, "PARTIAL_ORACLE_RELATIONS_NOT_ADJUDICATED"),
            ("VLM_ACTION", vlm, "MEASURED" if vlm else "NOT_EVALUATED_OR_INVALID"),
        ):
            result = outcomes.get(choice) if choice else None
            rows.append({
                "layout": b["layout"], "context": b["context"], "repeat": b["repeat"],
                "method": method, "candidate_id": choice, "method_status": status,
                "nominal_legal_options": len(options),
                "nominal_plan_length": len(plan_result.plan),
                "nominal_cost_equal": len({o["cost"] for o in options}) == 1 if options else None,
                "paired_success_complete": comparable,
                "chosen_task_success": result.get("task_success") if result else None,
                "chosen_time_sim_seconds": result.get("time_to_task_success") if result else None,
                "regret_sim_seconds": (
                    result["time_to_task_success"]-best
                    if comparable and result and result.get("task_success") else None),
                "evaluation_scope": "FIRST_DECISION_UNDER_SHARED_CONTINUATION",
            })
    path = out / "baselines/first_decision_reference_bank.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["method"])
        writer.writeheader()
        writer.writerows(rows)
    result = {"status": "MEASURED" if rows else "NOT_MEASURABLE",
              "rows": len(rows), "contexts": len(seen),
              "closed_loop_performance_claim": False}
    write(out / "baselines/summary.json", result)
    return result
