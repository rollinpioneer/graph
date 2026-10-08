"""Tables of card C1-DEPOTS-SEARCH-MATCH-V1 (plan sections 10 and 14), generated from the per-case search records only. Nothing here runs a model or a planner.

Budgets: node milestones 1,000 / 10,000 / 100,000 expansions are snapshots of ONE run; every run is also bounded by 300 s wall clock and the memory limit. ``status_at(record, M)``:
  SOLVED                       solved at or before expansion M (and, by construction, inside the wall limit)
  UNSOLVED_AT_NODE_BUDGET      the run reached M expansions without a solution
  TIMEOUT_BEFORE_NODE_LIMIT / MEMORY_BEFORE_NODE_LIMIT / OPEN_EXHAUSTED   the run ended before M expansions
  ADAPTER_ERROR / INVALID_PLAN / MODEL_NONFINITE   technical problems (never read as a low score)
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .. import report as OLDREP
from .analysis import wcsv

SCORERS = ("V_DENSE", "V_MG", "V_REL", "H_WL", "H_ADD", "H_COUNT")
SETS = ("struct", "joint", "ipc")
TECH = ("ADAPTER_ERROR", "INVALID_PLAN", "MODEL_NONFINITE")
RES = ("TIMEOUT", "MEMORY_LIMIT", "NODE_LIMIT")
TIME_POINTS = (1, 5, 10, 30, 60, 120, 300)
PAIRS = (("V_DENSE", "H_WL", "PRIMARY"), ("V_DENSE", "H_ADD", "reference"), ("V_DENSE", "H_COUNT", "reference"), ("V_DENSE", "V_MG", "auxiliary"), ("V_DENSE", "V_REL", "auxiliary"), ("V_REL", "V_MG", "auxiliary"),
         ("H_WL", "H_ADD", "reference"), ("H_WL", "H_COUNT", "reference"), ("H_ADD", "H_COUNT", "reference"))


def sign_p(a, b):
    n = a + b
    if n == 0:
        return 1.0
    k = min(a, b)
    return min(1.0, 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n))


def load_records(rr):
    recs = []
    for p in sorted((Path(rr) / "search").glob("*.jsonl")):
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return recs


def status_at(rec, M, max_exp=100_000):
    if rec["status"] in TECH:
        return rec["status"]
    if rec["solved"] and rec["solved_at_expansion"] is not None and rec["solved_at_expansion"] <= M:
        return "SOLVED"
    snaps = rec.get("snapshots") or {}
    if str(M) in snaps or M in snaps:
        return "UNSOLVED_AT_NODE_BUDGET"
    st = rec["status"]
    if st == "SOLVED":
        return "UNSOLVED_AT_NODE_BUDGET"                    # solved later than M (solved_at_expansion > M): not solved under the M-node budget
    if st == "TIMEOUT":
        return "TIMEOUT_BEFORE_NODE_LIMIT"
    if st == "MEMORY_LIMIT":
        return "MEMORY_BEFORE_NODE_LIMIT"
    return st


def best_observed(old, recs, cases_topology):
    """Best valid plan length per problem over ALL historical plans and the plans of this card (validated when produced)."""
    best = {}
    for k, v in cases_topology.items():
        if v.get("best_observed_valid_length") not in (None, ""):
            best[k] = int(v["best_observed_valid_length"])
    for r in recs:
        if r["solved"] and r["plan_length"] is not None:
            k = (r["set"], r["case_id"])
            best[k] = min(best.get(k, 10 ** 9), r["plan_length"])
    return best


def load_csv_index(p, keys=("set", "case_id")):
    out = {}
    if Path(p).is_file():
        with open(p, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                out[tuple(r[k] for k in keys)] = r
    return out


def build(rr, old, cfg):
    rr = Path(rr)
    res = rr / "results"
    recs = load_records(rr)
    topo = load_csv_index(res / "topology.csv")
    refs = load_csv_index(res / "reference_length_sensitivity.csv")
    prob = OLDREP.problem_table(old.root)
    best = best_observed(old, recs, refs)
    outs = []
    # ---- per case
    rows = []
    for r in recs:
        k = (r["set"], r["case_id"])
        m = prob[r["set"]][r["case_id"]]
        t = topo.get(k, {})
        em = r.get("evaluator_metrics") or {}
        proven = refs.get(k, {}).get("proven_optimal_length")
        proven = int(proven) if proven not in (None, "") else None
        row = {"scorer": r["scorer"], "set": r["set"], "case_id": r["case_id"], "status": r["status"], "solved": r["solved"], "plan_length": r["plan_length"], "plan_valid": r["plan_valid"], "expanded": r.get("expanded"),
               "solved_at_expansion": r.get("solved_at_expansion"), "generated": r.get("generated"), "duplicates": r.get("duplicates"), "path_updates": r.get("path_updates"), "h_root": r.get("h_root"), "best_h": r.get("best_h"),
               "evaluated_states": em.get("states_scored"), "eval_batches": r.get("eval_batches"), "forward_calls": em.get("forward_calls"), "graph_encodings": em.get("graph_encodings"), "wall_total_s": round(r["wall_total"], 3),
               "t_parse_ground_s": (r.get("timings") or {}).get("parse_ground"), "t_prepare_s": (r.get("timings") or {}).get("prepare"), "search_s": r.get("search_seconds"), "scoring_s": r.get("eval_seconds"),
               "inference_s": em.get("inference_seconds"), "translate_s": em.get("translate_seconds"), "rss_peak_growth_gib": round((r.get("rss_peak_growth_bytes") or 0) / 2 ** 30, 3),
               "gpu_peak_gib": round((r.get("gpu_peak_bytes") or 0) / 2 ** 30, 3) if r.get("gpu_peak_bytes") else None, "timeout_before_node_limit": r.get("timeout_before_node_limit"), "error": r.get("error"),
               "proven_optimal_length": proven, "best_observed_valid_length": best.get((r["set"], r["case_id"])), "registered_reference_length": m["L_ref"], "registered_reference_kind": m["L_kind"],
               "ratio_to_optimal": (r["plan_length"] / proven) if (r["solved"] and proven) else None,
               "ratio_to_best_observed": (r["plan_length"] / best[k]) if (r["solved"] and k in best) else None}
        for c in ("n_crates", "n_goal_towers", "max_goal_height", "needs_transport", "n_trucks", "n_places", "n_ground_actions", "destruction_label", "cell"):
            row[c] = t.get(c)
        rows.append(row)
    outs.append(wcsv(res / "search_by_case.csv", rows))
    # ---- budget curve (milestones of one run)
    by = {(r["scorer"], r["set"], r["case_id"]): r for r in recs}
    curve = []
    for r in recs:
        for M in cfg["search"]["milestones"]:
            st = status_at(r, M, cfg["search"]["max_expansions"])
            curve.append({"scorer": r["scorer"], "set": r["set"], "case_id": r["case_id"], "node_budget": M, "status_at_budget": st, "solved": st == "SOLVED", "plan_length": r["plan_length"] if st == "SOLVED" else None,
                          "expansions_at_solution": r.get("solved_at_expansion") if st == "SOLVED" else None, "elapsed_at_milestone_s": ((r.get("snapshots") or {}).get(str(M)) or {}).get("elapsed")})
    outs.append(wcsv(res / "search_budget_curve.csv", curve))
    # ---- time curve
    tc = []
    for sc in SCORERS:
        for s in SETS:
            rs = [r for r in recs if r["scorer"] == sc and r["set"] == s]
            if rs:
                row = {"scorer": sc, "set": s, "n": len(rs)}
                for T_ in TIME_POINTS:
                    row["solved_within_%ds" % T_] = sum(1 for r in rs if r["solved"] and r["wall_total"] <= T_)
                tc.append(row)
    outs.append(wcsv(res / "search_time_curve.csv", tc))
    # ---- summary per scorer / set / budget
    summ = []
    for sc in SCORERS:
        for s in SETS:
            rs = [r for r in recs if r["scorer"] == sc and r["set"] == s]
            if not rs:
                continue
            for M in cfg["search"]["milestones"]:
                sts = [status_at(r, M) for r in rs]
                solved = [r for r, st in zip(rs, sts) if st == "SOLVED"]
                lens = [r["plan_length"] for r in solved]
                opt = [(r["plan_length"] / int(refs[(s, r["case_id"])]["proven_optimal_length"])) for r in solved if refs.get((s, r["case_id"]), {}).get("proven_optimal_length") not in (None, "")]
                summ.append({"scorer": sc, "set": s, "node_budget": M, "n": len(rs), "solved": len(solved), "coverage": round(len(solved) / len(rs), 4), "mean_plan_length": round(statistics.mean(lens), 3) if lens else None,
                             "mean_ratio_to_proven_optimum": round(statistics.mean(opt), 4) if opt else None, "mean_expansions_at_solution": round(statistics.mean(r["solved_at_expansion"] for r in solved), 2) if solved else None,
                             "statuses": dict(Counter(sts))})
    outs.append(wcsv(res / "search_summary.csv", summ))
    # ---- paired comparisons
    pairs = []
    for a, b, role in PAIRS:
        for s in SETS:
            for M in cfg["search"]["milestones"]:
                ids = sorted(c for (sc, ss, c) in by if sc == a and ss == s and (b, s, c) in by)
                if not ids:
                    continue
                sa = {c: status_at(by[(a, s, c)], M) for c in ids}
                sb = {c: status_at(by[(b, s, c)], M) for c in ids}
                both = [c for c in ids if sa[c] == "SOLVED" and sb[c] == "SOLVED"]
                oa = [c for c in ids if sa[c] == "SOLVED" and sb[c] != "SOLVED"]
                ob = [c for c in ids if sb[c] == "SOLVED" and sa[c] != "SOLVED"]
                dl = [by[(a, s, c)]["plan_length"] - by[(b, s, c)]["plan_length"] for c in both]
                de = [by[(a, s, c)]["solved_at_expansion"] - by[(b, s, c)]["solved_at_expansion"] for c in both]
                shorter, longer = sum(1 for x in dl if x < 0), sum(1 for x in dl if x > 0)
                fewer, more = sum(1 for x in de if x < 0), sum(1 for x in de if x > 0)
                pairs.append({"first": a, "second": b, "role": role, "set": s, "node_budget": M, "n": len(ids), "first_solved": sum(v == "SOLVED" for v in sa.values()), "second_solved": sum(v == "SOLVED" for v in sb.values()),
                              "both": len(both), "only_first": len(oa), "only_second": len(ob), "neither": len(ids) - len(both) - len(oa) - len(ob), "coverage_sign_p": round(sign_p(len(oa), len(ob)), 6),
                              "common_solved": len(both), "first_shorter": shorter, "first_longer": longer, "same_length": len(dl) - shorter - longer, "sum_length_diff": sum(dl) if dl else None,
                              "median_length_diff": statistics.median(dl) if dl else None, "length_sign_p": round(sign_p(shorter, longer), 6), "first_fewer_expansions": fewer, "first_more_expansions": more,
                              "median_expansion_diff": statistics.median(de) if de else None, "expansion_sign_p": round(sign_p(fewer, more), 6),
                              "first_technical": sum(1 for v in sa.values() if v in TECH), "second_technical": sum(1 for v in sb.values() if v in TECH)})
    outs.append(wcsv(res / "paired_comparisons.csv", pairs))
    # ---- compute
    comp = []
    for sc in SCORERS:
        for s in SETS:
            rs = [r for r in recs if r["scorer"] == sc and r["set"] == s]
            if not rs:
                continue
            em = [(r.get("evaluator_metrics") or {}) for r in rs]
            comp.append({"scorer": sc, "set": s, "runs": len(rs), "wall_total_s": round(sum(r["wall_total"] for r in rs), 1), "expanded": sum(r.get("expanded") or 0 for r in rs), "generated": sum(r.get("generated") or 0 for r in rs),
                         "evaluated_states": sum(e.get("states_scored", 0) for e in em), "forward_calls": sum(e.get("forward_calls", 0) for e in em), "graph_encodings": sum(e.get("graph_encodings", 0) for e in em),
                         "scoring_s": round(sum(r.get("eval_seconds") or 0 for r in rs), 1), "inference_s": round(sum(e.get("inference_seconds", 0) for e in em), 1), "translate_s": round(sum(e.get("translate_seconds", 0) for e in em), 1),
                         "parse_ground_s": round(sum((r.get("timings") or {}).get("parse_ground", 0) for r in rs), 1), "peak_rss_growth_gib": round(max((r.get("rss_peak_growth_bytes") or 0) for r in rs) / 2 ** 30, 2),
                         "peak_gpu_gib": round(max((r.get("gpu_peak_bytes") or 0) for r in rs) / 2 ** 30, 2) if any(r.get("gpu_peak_bytes") for r in rs) else None,
                         "model_load_s": rs[0].get("model_load_seconds"), "statuses": dict(Counter(r["status"] for r in rs)), "technical": sum(1 for r in rs if r["status"] in TECH),
                         "wl_negative_values": sum(e.get("negative_values", 0) for e in em) if sc == "H_WL" else None, "h_add_infinite_values": sum(e.get("infinite_values", 0) for e in em) if sc == "H_ADD" else None})
    outs.append(wcsv(res / "compute.csv", comp))
    # ---- strata (final budget and the primary milestone)
    st_rows = []
    dims = ("n_crates", "n_goal_towers", "max_goal_height", "needs_transport", "destruction_label", "cell", "n_trucks", "n_places")
    for M in (cfg["search"]["primary_milestone"], cfg["search"]["max_expansions"]):
        groups = defaultdict(list)
        for r in recs:
            t = topo.get((r["set"], r["case_id"]), {})
            groups[(r["scorer"], r["set"], "ALL", "ALL")].append(r)
            for d in dims:
                v = t.get(d)
                if v not in (None, ""):
                    groups[(r["scorer"], r["set"], d, str(v))].append(r)
        for (sc, s, d, v), rs in sorted(groups.items()):
            solved = [r for r in rs if status_at(r, M) == "SOLVED"]
            st_rows.append({"node_budget": M, "scorer": sc, "set": s, "dimension": d, "stratum": v, "n": len(rs), "solved": len(solved), "coverage": round(len(solved) / len(rs), 4),
                            "mean_plan_length": round(statistics.mean(r["plan_length"] for r in solved), 2) if solved else None})
    outs.append(wcsv(res / "search_by_structure.csv", st_rows))
    # ---- IPC plan lengths against the best observed plan (fixed old penalty not used here)
    ip = []
    for r in recs:
        if r["set"] == "ipc" and r["solved"]:
            k = ("ipc", r["case_id"])
            ip.append({"scorer": r["scorer"], "case_id": r["case_id"], "plan_length": r["plan_length"], "best_observed_valid_length": best.get(k), "ratio_to_best_observed": round(r["plan_length"] / best[k], 4) if k in best else None,
                       "proven_optimal_length": refs.get(k, {}).get("proven_optimal_length") or None})
    outs.append(wcsv(res / "ipc_plan_quality.csv", ip))
    outs.append(numbers_md(rr, summ, pairs, comp, cfg))
    return outs


def numbers_md(rr, summ, pairs, comp, cfg):
    rr = Path(rr)
    L = ["# C1-DEPOTS-SEARCH-MATCH-V1: generated numbers (interpretation is in final_summary.md / claim_boundary.md)", ""]
    for s in SETS:
        L += ["## %s: coverage / mean plan length of solved problems (shared engine)" % s, "", "| scorer | nodes <= 1,000 | nodes <= 10,000 | nodes <= 100,000 (300 s) | mean expansions at solution (100k) | statuses (100k) |", "|---|---|---|---|---|---|"]
        for sc in SCORERS:
            cells = {r["node_budget"]: r for r in summ if r["scorer"] == sc and r["set"] == s}
            if cells:
                g = lambda M: "%d / %d (%s)" % (cells[M]["solved"], cells[M]["n"], cells[M]["mean_plan_length"])
                last = cells[cfg["search"]["max_expansions"]]
                L.append("| %s | %s | %s | %s | %s | %s |" % (sc, g(1000), g(10000), g(100000), last["mean_expansions_at_solution"], last["statuses"]))
        L.append("")
    L += ["## paired comparisons at the primary milestone (%d nodes)" % cfg["search"]["primary_milestone"], "", "| first | second | set | solved first / second | only first / only second (p) | common | first shorter / longer / same (p) | first fewer / more expansions (p) |", "|---|---|---|---|---|---|---|---|"]
    for p in pairs:
        if p["node_budget"] == cfg["search"]["primary_milestone"]:
            L.append("| %s | %s | %s | %d / %d | %d / %d (%.3g) | %d | %d / %d / %d (%.3g) | %d / %d (%.3g) |" % (p["first"], p["second"], p["set"], p["first_solved"], p["second_solved"], p["only_first"], p["only_second"], p["coverage_sign_p"], p["common_solved"],
                                                                                                             p["first_shorter"], p["first_longer"], p["same_length"], p["length_sign_p"], p["first_fewer_expansions"], p["first_more_expansions"], p["expansion_sign_p"]))
    p = rr / "results" / "final_summary_numbers.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p
