"""CP-DISR-C1-MECH-CONFIRM-V1 result analysis (offline, pure python over the decision-level jsonl files).

File layout (runs/final_master/c1_route_b/mech_confirm_v1/fresh_confirm/): ``<LABEL>_s<seed>.jsonl`` for the neural methods and ``B_PLAN.jsonl``.
Every number here is computed from the logged decisions; the classification uses only the frozen rules in c1_classification.
"""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from . import c1_classification as K

CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
LABELS = {"B2": "B2", "ABS": "B2-ABS", "E": "B1-K+E", "NC": "B1-K+NC", "QMARK": "B1-K+QMARK", "B_PLAN": "B_PLAN"}
NEURAL = ("B2", "ABS", "E", "NC", "QMARK")
CELLS = K.CELLS
TRAIN_DEFAULT = ("a:PLACE:target:container:v1", "a:PLACE_BUFFER:second_object:buffer:v1")


def load_dir(fresh_dir):
    out = {}
    for path in sorted(Path(fresh_dir).glob("*.jsonl")):
        stem = path.stem
        label, seed = (stem.rsplit("_s", 1)[0], int(stem.rsplit("_s", 1)[1])) if "_s" in stem and stem.rsplit("_s", 1)[1].isdigit() else (stem, None)
        if label not in LABELS:
            continue
        out[(label, seed)] = [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    return out


def expected_case_ids(test_rows):
    return [r["case_id"] for r in test_rows]


def completeness(runs, case_ids):
    problems = []
    for key, eps in runs.items():
        ids = [e["case_id"] for e in eps]
        if sorted(ids) != sorted(case_ids):
            problems.append({"run": list(key), "problem": "case ids differ from the frozen final set", "n": len(ids)})
        if len(set(ids)) != len(ids):
            problems.append({"run": list(key), "problem": "a case was evaluated more than once"})
    return problems


def cell_rates(eps):
    by = defaultdict(lambda: [0, 0])
    for e in eps:
        by[e["cell"]][1] += 1
        by[e["cell"]][0] += int(bool(e["success"]))
    return {c: (by[c][0] / by[c][1] if by[c][1] else None) for c in CELLS}, {c: tuple(by[c]) for c in CELLS}


def total_rate(eps):
    return sum(int(bool(e["success"])) for e in eps) / len(eps) if eps else None


def case_rows(runs):
    rows = []
    for (label, seed), eps in sorted(runs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        for e in eps:
            fd = e.get("first_divergence") or {}
            rows.append({"method": LABELS[label], "label": label, "seed": "" if seed is None else seed, "case_id": e["case_id"], "cell": e["cell"], "success": int(bool(e["success"])), "reason": e["reason"],
                         "steps": e["steps"], "elapsed_seconds": round(e["elapsed_seconds"], 3), "n_decisions": e["n_decisions"], "first_divergence_index": fd.get("index", ""),
                         "first_divergence_type": fd.get("type", ""), "first_divergence_action": fd.get("selected", ""), "first_divergence_completed_goals": len([a for a, v in (fd.get("completed_goals_before") or {}).items() if v]) if fd else "",
                         "repeated_action_count": e["repeated_action_count"], "destroyed_completed_goal_count": e["destroyed_completed_goal_count"],
                         "ended_no_candidate_safe_termination": int(e["ended_with_no_candidate_safe_termination"]), "reference_unavailable_decisions": e["reference_unavailable_decisions"]})
    return rows


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def cell_rows(runs):
    rows = []
    for (label, seed), eps in sorted(runs.items(), key=lambda kv: (kv[0][0], -1 if kv[0][1] is None else kv[0][1])):
        rates, counts = cell_rates(eps)
        for c in CELLS:
            rows.append({"method": LABELS[label], "label": label, "seed": "" if seed is None else seed, "cell": c, "n": counts[c][1], "success_n": counts[c][0], "rate": rates[c]})
        rows.append({"method": LABELS[label], "label": label, "seed": "" if seed is None else seed, "cell": "ALL", "n": len(eps), "success_n": sum(int(bool(e["success"])) for e in eps), "rate": total_rate(eps)})
    return rows


# ----------------------------------------------------------------------------- mechanism trace
def failure_category(e):
    if e["success"]:
        return "SUCCESS"
    fd = e.get("first_divergence")
    if e["destroyed_completed_goal_count"] > 0:
        return "DESTROYED_A_COMPLETED_GOAL"
    if fd is None:
        if e["ended_with_no_candidate_safe_termination"]:
            return "OBSERVATION_LOSS_LIMITED_NO_CANDIDATE_WITHOUT_DIVERGENCE"
        return "NO_DIVERGENCE_FROM_PUBLIC_OPTIMAL"
    done = [a for a, v in (fd.get("completed_goals_before") or {}).items() if v]
    return "FIRST_DIVERGENCE_AFTER_A_COMPLETED_GOAL" if done else "FIRST_DIVERGENCE_BEFORE_ANY_GOAL"


def mechanism(eps):
    n = len(eps)
    fails = [e for e in eps if not e["success"]]
    decisions = [d for e in eps for d in e["decisions"] if not d.get("no_transition")]
    placements = Counter(d["placement_binding"] for d in decisions if d.get("placement_binding"))
    div_types = Counter((e["first_divergence"] or {}).get("type") for e in eps if e.get("first_divergence"))
    div_idx = [e["first_divergence"]["index"] for e in eps if e.get("first_divergence")]
    train_default_div = sum(1 for e in eps if any(d.get("divergence_type") == "PLACES_TRAIN_DEFAULT_BINDING" for d in e["decisions"]))
    return {"episodes": n, "failures": len(fails), "episodes_with_first_divergence": len(div_idx), "first_divergence_rate": len(div_idx) / n if n else None,
            "first_divergence_index_counts": dict(Counter(div_idx)), "first_divergence_type_counts": {str(k): v for k, v in div_types.items()},
            "training_default_binding_error_episode_rate": train_default_div / n if n else None, "training_default_binding_error_episodes": train_default_div,
            "repeated_action_episode_rate": sum(1 for e in eps if e["repeated_action_count"] > 0) / n if n else None,
            "destroyed_completed_goal_episode_rate": sum(1 for e in eps if e["destroyed_completed_goal_count"] > 0) / n if n else None,
            "reference_unavailable_decision_share": (sum(1 for d in decisions if d["reference_status"] != "AVAILABLE") / len(decisions)) if decisions else None,
            "no_candidate_safe_termination_episodes": sum(1 for e in eps if e["ended_with_no_candidate_safe_termination"]),
            "no_candidate_after_an_earlier_divergence": sum(1 for e in eps if e["no_candidate_after_earlier_divergence"]),
            "failure_categories": dict(Counter(failure_category(e) for e in fails)), "failure_reasons": dict(Counter(str(e["reason"]).split(":")[0] for e in fails)),
            "placement_binding_counts_selected": dict(placements)}


def single_vs_dual(eps):
    r, _ = cell_rates(eps)
    d = lambda a, b: None if r[a] is None or r[b] is None else r[a] - r[b]  # noqa: E731
    return {"BUF_T_minus_BUF_T+BUF_S": d("BUF_T", "BUF_T+BUF_S"), "IN_S_minus_IN_S+BUF_T": d("IN_S", "IN_S+BUF_T"), "cell_rates": r}


# ----------------------------------------------------------------------------- planner / efficiency
def planner_summary(eps):
    calls = [d["planner"] for e in eps for d in e["decisions"] if d.get("planner")]
    statuses = Counter(c["status"] for c in calls)
    cpu = sorted(c["cpu_seconds"] for c in calls)
    pct = lambda p: cpu[min(len(cpu) - 1, int(math.ceil(p * len(cpu))) - 1)] if cpu else None  # noqa: E731
    succ = [e for e in eps if e["success"]]
    plan_changes = 0
    for e in eps:
        prev = None
        for d in e["decisions"]:
            pl = (d.get("planner") or {}).get("plan") or []
            if prev is not None and pl and len(prev) > 1 and pl[0] != prev[1]:
                plan_changes += 1
            prev = pl or prev
    ended = Counter(str(e["reason"]).split(":")[0] for e in eps if not e["success"])
    return {"episodes": len(eps), "success_n": len(succ), "success_rate": len(succ) / len(eps) if eps else None, "cell_rates": cell_rates(eps)[0], "planning_calls": len(calls), "replanning_calls_after_first": len(calls) - len(eps),
            "plan_changes_vs_previous_plan": plan_changes, "search_status_counts": dict(statuses), "no_plan_or_timeout_episodes": sum(1 for e in eps if any((d.get("planner") or {}).get("status") in ("NO_PLAN", "SEARCH_TIMEOUT") for d in e["decisions"])),
            "expanded_nodes_mean": sum(c["expanded_nodes"] for c in calls) / len(calls) if calls else None, "expanded_nodes_max": max((c["expanded_nodes"] for c in calls), default=None),
            "generated_nodes_mean": sum(c["generated_nodes"] for c in calls) / len(calls) if calls else None, "generated_nodes_max": max((c["generated_nodes"] for c in calls), default=None),
            "cpu_seconds_mean": sum(cpu) / len(cpu) if cpu else None, "cpu_seconds_p95": pct(0.95), "cpu_seconds_max": max(cpu, default=None),
            "skills_per_successful_episode_mean": sum(e["steps"] for e in succ) / len(succ) if succ else None, "sim_seconds_per_successful_episode_mean": sum(e["elapsed_seconds"] for e in succ) / len(succ) if succ else None,
            "failure_reasons": dict(ended), "limits": {"depth_limit": 8, "max_nodes": 4096, "cpu_time_limit_seconds": 2.0}}


def efficiency_neural(eps):
    ds = [d for e in eps for d in e["decisions"] if "encoder_forward_calls" in d]
    succ = [e for e in eps if e["success"]]
    return {"decisions_with_forward": len(ds), "encoder_forward_calls_per_decision_mean": sum(d["encoder_forward_calls"] for d in ds) / len(ds) if ds else None,
            "forward_seconds_per_decision_mean": sum(d["forward_seconds"] for d in ds) / len(ds) if ds else None,
            "skills_per_successful_episode_mean": sum(e["steps"] for e in succ) / len(succ) if succ else None,
            "sim_seconds_per_successful_episode_mean": sum(e["elapsed_seconds"] for e in succ) / len(succ) if succ else None}


def planner_interpretation(planner, neural_totals):
    best = max(neural_totals.values()) if neural_totals else None
    nop = planner["no_plan_or_timeout_episodes"] / planner["episodes"] if planner["episodes"] else 0.0
    if planner["success_rate"] >= 0.9 and (planner["cpu_seconds_p95"] or 0) <= 0.5:
        return {"class": "HIGH_SUCCESS_LOW_COST", "reading": "contract search solves this suite cheaply: the paper cannot say learned methods are needed because search fails here; route B has to rest on amortised decisions, learned representation, or conditions beyond this suite."}
    if nop >= 0.25 and best is not None and best >= 0.7:
        return {"class": "PLANNER_FRAGILE_TO_PUBLIC_UNKNOWN", "reading": "the planner often has NO_PLAN / SEARCH_TIMEOUT under public UNKNOWN facts while a learned method succeeds: a follow-up hypothesis (tolerance to incomplete public state), not upgraded here."}
    if best is not None and planner["success_rate"] > best + 0.10:
        return {"class": "PLANNER_OUTPERFORMS_ALL_LEARNED", "reading": "route B can still study representation, but the paper value and application setting need re-examination."}
    return {"class": "MIXED", "reading": "no single frozen planner pattern; reported descriptively."}
