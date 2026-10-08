"""Tables of the public-Depots card (plan section 10 and 11.3). Everything is recomputed from the per-problem records on disk; nothing here runs a model or a planner.

Methods: internal MG-G / DENSE-G / REL-G x {one_step, two_step, gated}; external LAMA (first plan; anytime best), WL-GOOSE. A problem counts as solved by a planner when a plan that replays to the goal under our semantics
was returned inside the uniform wall-clock budget; its cost is the plan length. Internal controllers are bounded by the step cap 2*L_ref+4 (L_ref: exact optimum > A*+LM-cut optimum > best known LAMA plan); a failed
problem of any method is charged cap+1 in the failure-penalised cost.
"""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

MODES = ("mg", "dense", "rel")
NAME = {"mg": "MG-G", "dense": "DENSE-G", "rel": "REL-G"}
EXECS = ("one_step", "two_step", "gated")
SETS = ("struct", "joint", "ipc")
TIME_POINTS = (10, 30, 60, 120, 300)


def rj(p):
    return json.loads(Path(p).read_text())


def jl(p):
    return [json.loads(x) for x in Path(p).read_text().splitlines() if x.strip()]


def wcsv(p, rows):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(p, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in r.items()})


def sign_test_p(better, worse):
    n = better + worse
    if n == 0:
        return 1.0
    k = min(better, worse)
    p = 2.0 * sum(math.comb(n, i) for i in range(k + 1)) / (2.0 ** n)
    return min(1.0, p)


def quantile(xs, q):
    ys = sorted(xs)
    if not ys:
        return None
    k = (len(ys) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return ys[lo] + (ys[hi] - ys[lo]) * (k - lo)


# ------------------------------------------------------------------------------------------------ problems
def problem_table(rr):
    """{set: {case_id: meta}} with L_ref, cap, strata."""
    from . import stages as ST
    rr = Path(rr)
    man, exact = rj(rr / "data" / "manifest.json"), rj(rr / "data" / "exact.json")
    out = {}
    for s in SETS:
        refs = rj(rr / "data" / ("refs_%s.json" % s)) if (rr / "data" / ("refs_%s.json" % s)).is_file() else {}
        out[s] = {}
        for c in man[s]:
            cid = c["case_id"]
            ex = exact.get(cid)
            if ex and ex.get("status") == "OK":
                L, kind = ex["optimal_length"], "exact"
            elif cid in refs and refs[cid].get("L_ref") is not None:
                L, kind = refs[cid]["L_ref"], refs[cid]["L_kind"]
            else:
                L, kind = None, "no_reference"
            m = {"set": s, "case_id": cid, "L_ref": L, "L_kind": kind, "cap": (2 * L + 4) if L is not None else 304}
            if "cell" in c:
                m["cell"] = c["cell"]
            a = c.get("analysis")
            if a:
                m.update({"n_crates": a["n_crates"], "k_goal_towers": a["k_goal_towers"], "max_goal_height": a["max_goal_height"], "needs_transport": a["needs_transport"]})
            else:
                text = Path(c["file"]).read_text().lower()
                m["n_crates"] = len([t for t in text.replace("(", " ").split("\n") if t.strip().startswith("crate ")])
            m["destruction_label"] = ("TRUE" if ex["requires_goal_destruction"] else "FALSE") if (ex and ex.get("status") == "OK" and ex.get("requires_goal_destruction") is not None) else "UNKNOWN"
            out[s][cid] = m
    return out


# ------------------------------------------------------------------------------------------------ per-method records
def internal_records(rr):
    out = {}
    for mode in MODES:
        for ex in EXECS:
            for s in SETS:
                p = Path(rr) / "eval" / ("%s__%s__%s.jsonl" % (mode, ex, s))
                if p.is_file():
                    out[("%s/%s" % (NAME[mode], ex), s)] = {e["case_id"]: e for e in jl(p)}
    return out


def external_records(rr, probs):
    out = {}
    for s in SETS:
        p = Path(rr) / "data" / ("refs_%s.json" % s)
        if p.is_file():
            refs = rj(p)
            lama_any, lama_first = {}, {}
            for cid, r in refs.items():
                a, f = r["lama"], r["lama_first"]
                t_first = a["lengths_by_time"][0][0] if a["lengths_by_time"] else a["wall_seconds"]
                lama_any[cid] = {"success": a["best_length"] is not None, "steps": a["best_length"], "wall_seconds": t_first, "total_wall_seconds": a["wall_seconds"], "reason": "OK" if a["best_length"] is not None else "NO_PLAN"}
                ft = f["lengths_by_time"][0][0] if f["lengths_by_time"] else f["wall_seconds"]
                lama_first[cid] = {"success": f["best_length"] is not None, "steps": f["best_length"], "wall_seconds": ft, "reason": "OK" if f["best_length"] is not None else "NO_PLAN"}
            out[("LAMA-first", s)] = lama_first
            out[("LAMA-anytime", s)] = lama_any
        g = Path(rr) / "eval" / ("goose__%s.json" % s)
        if g.is_file():
            out[("WL-GOOSE", s)] = {r["case_id"]: {"success": r["solved"], "steps": r["plan_length"], "wall_seconds": r["wall_seconds"], "reason": "OK" if r["solved"] else ("TIMEOUT" if r.get("timed_out") else "NO_PLAN"),
                                                     "expanded": r.get("expanded"), "peak_kb": r.get("peak_kb")} for r in rj(g)["results"]}
    return out


def cost_of(rec, meta):
    """(success, cost, penalised cost) of one record under the shared cap."""
    if rec["success"] and rec["steps"] is not None:
        return True, rec["steps"], rec["steps"]
    return False, None, meta["cap"] + 1


def all_records(rr):
    probs = problem_table(rr)
    recs = {**internal_records(rr), **external_records(rr, probs)}
    return probs, recs


# ------------------------------------------------------------------------------------------------ tables
def by_problem(probs, recs):
    rows = []
    for (method, s), d in recs.items():
        for cid, meta in probs[s].items():
            r = d.get(cid)
            if r is None:
                continue
            ok, c, pen = cost_of(r, meta)
            row = {"set": s, "method": method, "case_id": cid, "success": ok, "cost": c, "penalised_cost": pen, "L_ref": meta["L_ref"], "L_kind": meta["L_kind"], "cap": meta["cap"], "reason": r.get("reason"),
                   "wall_seconds": r.get("wall_seconds"), "ratio_to_ref": (c / meta["L_ref"]) if (ok and meta["L_ref"]) else None, "penalised_ratio": pen / meta["L_ref"] if meta["L_ref"] else None}
            for k in ("cell", "n_crates", "k_goal_towers", "max_goal_height", "needs_transport", "destruction_label"):
                row[k] = meta.get(k)
            row["interventions"] = r.get("interventions")
            decs = r.get("decisions")
            if decs and "selected_is_optimal" in decs[0]:                                            # exact label of the visited states exists (struct, 6-crate joint)
                fe = next((x for x in decs if not x["selected_is_optimal"]), None)
                row.update({"decision_perfect": r.get("decision_perfect"), "first_error_step": fe["decision_index"] if fe else None, "first_error_family": fe["selected"].split(":")[1] if fe else None})
            cn = r.get("counters")
            if cn:
                row.update({"leaves": cn.get("leaves"), "model_calls": cn.get("model_calls"), "gate_expansions": cn.get("gate_expansions")})
            rows.append(row)
    return rows


def summarise_rows(rows):
    n = len(rows)
    succ = [r for r in rows if r["success"]]
    pr = [r["penalised_ratio"] for r in rows if r["penalised_ratio"] is not None]
    return {"n": n, "success": len(succ), "completion": round(len(succ) / n, 4) if n else None, "mean_penalised_ratio": round(statistics.mean(pr), 4) if pr else None,
            "mean_ratio_success": round(statistics.mean(r["ratio_to_ref"] for r in succ if r["ratio_to_ref"] is not None), 4) if any(r["ratio_to_ref"] is not None for r in succ) else None,
            "total_cost_success": sum(r["cost"] for r in succ), "excess_over_ref_success": sum(r["cost"] - r["L_ref"] for r in succ if r["L_ref"] is not None),
            "mean_wall_seconds": round(statistics.mean(r["wall_seconds"] for r in rows if r["wall_seconds"] is not None), 3) if any(r["wall_seconds"] is not None for r in rows) else None,
            "failures": dict(Counter(r["reason"] for r in rows if not r["success"]))}


def by_structure(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["set"], "ALL", "ALL")].append(r)
        for dim in ("cell", "n_crates", "k_goal_towers", "max_goal_height", "needs_transport", "destruction_label"):
            if r.get(dim) is not None:
                groups[(r["method"], r["set"], dim, str(r[dim]))].append(r)
    for (m, s, dim, val), rs in sorted(groups.items()):
        out.append({"method": m, "set": s, "dimension": dim, "stratum": val, **summarise_rows(rs)})
    return out


def completion_by_time(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["method"], r["set"])].append(r)
    for (m, s), rs in sorted(groups.items()):
        row = {"method": m, "set": s, "n": len(rs)}
        for t in TIME_POINTS:
            row["solved_within_%ds" % t] = sum(1 for r in rs if r["success"] and r["wall_seconds"] is not None and r["wall_seconds"] <= t)
        out.append(row)
    return out


def main_pairs():
    pairs = []
    for ex in EXECS:
        pairs += [("REL-G/%s" % ex, "DENSE-G/%s" % ex, "PRIMARY" if ex != "gated" else "secondary"), ("REL-G/%s" % ex, "MG-G/%s" % ex, "secondary"), ("DENSE-G/%s" % ex, "MG-G/%s" % ex, "secondary")]
    for m in ("MG-G", "DENSE-G", "REL-G"):
        pairs += [("%s/two_step" % m, "%s/one_step" % m, "secondary"), ("%s/gated" % m, "%s/two_step" % m, "secondary"), ("%s/gated" % m, "%s/one_step" % m, "secondary")]
    for ext in ("WL-GOOSE", "LAMA-first", "LAMA-anytime"):
        for ex in ("one_step", "two_step"):
            pairs.append(("REL-G/%s" % ex, ext, "system"))
    pairs += [("WL-GOOSE", "LAMA-anytime", "system"), ("WL-GOOSE", "LAMA-first", "system")]
    return pairs


def paired_costs(rows, probs):
    by = defaultdict(dict)
    for r in rows:
        by[(r["method"], r["set"])][r["case_id"]] = r
    out = []
    for a, b, status in main_pairs():
        for s in SETS + ("ALL",):
            ss = SETS if s == "ALL" else (s,)
            ids = [(t, c) for t in ss for c in by.get((a, t), {}) if c in by.get((b, t), {})]
            if not ids:
                continue
            for dim, val in (("ALL", "ALL"),) + tuple(sorted({("destruction_label", by[(a, t)][c]["destruction_label"]) for t, c in ids})):
                sel = [(t, c) for t, c in ids if dim == "ALL" or by[(a, t)][c].get(dim) == val]
                if not sel:
                    continue
                ra, rb = [by[(a, t)][c] for t, c in sel], [by[(b, t)][c] for t, c in sel]
                both = [(x, y) for x, y in zip(ra, rb) if x["success"] and y["success"]]
                diffs = [x["cost"] - y["cost"] for x, y in both]
                better, worse, same = sum(d < 0 for d in diffs), sum(d > 0 for d in diffs), sum(d == 0 for d in diffs)
                only_a = sum(1 for x, y in zip(ra, rb) if x["success"] and not y["success"])
                only_b = sum(1 for x, y in zip(ra, rb) if y["success"] and not x["success"])
                pa = [x["penalised_cost"] for x in ra]
                pb = [y["penalised_cost"] for y in rb]
                out.append({"first": a, "second": b, "role": status, "set": s, "stratum_dim": dim, "stratum": val, "n_problems": len(sel), "first_success": sum(x["success"] for x in ra), "second_success": sum(y["success"] for y in rb),
                            "only_first": only_a, "only_second": only_b, "common_success_n": len(both), "first_better": better, "first_worse": worse, "same_cost": same,
                            "mean_cost_diff_first_minus_second": round(statistics.mean(diffs), 3) if diffs else None, "median_diff": statistics.median(diffs) if diffs else None, "p10_diff": quantile(diffs, 0.1), "p90_diff": quantile(diffs, 0.9),
                            "min_diff": min(diffs) if diffs else None, "max_diff": max(diffs) if diffs else None, "sign_test_p_two_sided": round(sign_test_p(better, worse), 6), "completion_sign_test_p": round(sign_test_p(only_a, only_b), 6),
                            "mean_penalised_cost_first": round(statistics.mean(pa), 3), "mean_penalised_cost_second": round(statistics.mean(pb), 3),
                            "mean_penalised_cost_diff": round(statistics.mean(x - y for x, y in zip(pa, pb)), 3)})
    return out


def compute_costs(rr, recs):
    rows = []
    for (m, s), d in sorted(recs.items()):
        rs = list(d.values())
        row = {"method": m, "set": s, "episodes": len(rs), "wall_seconds_total": round(sum(r.get("total_wall_seconds", r.get("wall_seconds")) or 0.0 for r in rs), 2), "wall_seconds_max": round(max((r.get("total_wall_seconds", r.get("wall_seconds")) or 0.0) for r in rs), 2) if rs else None}
        if rs and "counters" in rs[0]:
            cn = Counter()
            for r in rs:
                cn.update({k: v for k, v in r["counters"].items() if isinstance(v, (int, float))})
            steps = sum(r["steps"] for r in rs)
            row.update({"decisions": steps, "network_logit_passes": steps, "leaf_network_batches": cn.get("model_calls", 0), "expansions": cn.get("expansions", 0), "roots": cn.get("roots", 0), "leaves": cn.get("leaves", 0),
                        "unique_leaves": cn.get("unique_leaves", 0), "gate_checks": cn.get("gate_checks", 0), "gate_expansions": cn.get("gate_expansions", 0), "trap_unfolds": cn.get("trap_unfolds", 0),
                        "interventions": sum(r.get("interventions", 0) for r in rs), "label_seconds_excluded": round(sum(r.get("label_seconds", 0.0) for r in rs), 2)})
            p = Path(rr) / "receipts" / ("eval_%s_%s_%s.json" % (next(k for k, v in NAME.items() if v == m.split("/")[0]), m.split("/")[1], s))
            if p.is_file():
                row["peak_gpu_bytes"] = rj(p).get("peak_gpu_bytes")
        elif m == "WL-GOOSE":
            row.update({"expanded": sum(r.get("expanded") or 0 for r in rs), "peak_kb_max": max((r.get("peak_kb") or 0) for r in rs) if rs else None})
        rows.append(row)
    return rows


def training_costs(rr):
    rows = []
    for m in MODES:
        p = Path(rr) / "runs" / m / "training_accounting.json"
        sel = Path(rr) / "runs" / m / "selection.json"
        if p.is_file():
            a = rj(p)
            rows.append({"model": NAME[m], "optimizer_steps": a["optimizer_steps"], "decision_samples_shown": a["decision_samples_shown"], "wall_seconds": a["wall_seconds"], "attempts": a["attempts"], "nan_events": a["nan_events"],
                         "trainable_parameters": a["parameters_trainable"], "new_module_parameters": a["parameters_new_module"], "new_module_param_change_l2": a["new_module_param_change_l2"], "trajectories": a["trajectories"],
                         "selected_update": rj(sel)["selected_update"] if sel.is_file() else None})
    for t in ("typed", "ipc"):
        p = Path(rr) / "receipts" / ("goose_fit_%s.json" % t)
        if p.is_file():
            r = rj(p)
            rows.append({"model": "WL-GOOSE (%s encoding)" % t, "wall_seconds": r.get("wall_seconds"), "training_problems": r.get("training_problems"), "config": r.get("config")})
    ex = rj(Path(rr) / "data" / "exact.json")
    rows.append({"model": "exact labels (shared pool)", "problems": len(ex), "wall_seconds": round(sum(v.get("seconds", 0.0) for v in ex.values()), 1), "too_big": sum(1 for v in ex.values() if v["status"] != "OK")})
    for s in SETS:
        p = Path(rr) / "data" / ("refs_%s.json" % s)
        if p.is_file():
            refs = rj(p)
            rows.append({"model": "FD references (%s)" % s, "problems": len(refs), "wall_seconds": round(sum(r["lama"]["wall_seconds"] + r["lama_first"]["wall_seconds"] + r["lmcut"]["wall_seconds"] for r in refs.values()), 1),
                         "lmcut_solved": sum(1 for r in refs.values() if r["lmcut"]["solved"])})
    return rows


def raw_c3_final(recs):
    """Per internal method and set: how often each pipeline stage changes the choice and, where the exact label of the visited state exists, the transitions of optimality."""
    rows = []
    for (m, s), d in sorted(recs.items()):
        decs = [x for e in d.values() for x in e.get("decisions", [])] if d and "decisions" in next(iter(d.values())) else []
        if not decs:
            continue
        lab = [x for x in decs if "selected_is_optimal" in x]
        row = {"method": m, "set": s, "decisions": len(decs), "c3_changed_raw": sum(1 for x in decs if x.get("c3") and x["c3"] != x["raw"]), "final_differs_from_c3": sum(1 for x in decs if x.get("c3") and x["selected"] != x["c3"]),
               "final_differs_from_raw": sum(1 for x in decs if x["selected"] != x["raw"]), "labelled_decisions": len(lab)}
        if lab:
            row.update({"raw_optimal": sum(x["raw_is_optimal"] for x in lab), "c3_optimal": sum(bool(x.get("c3_is_optimal")) for x in lab), "final_optimal": sum(x["selected_is_optimal"] for x in lab),
                        "raw_opt_to_final_not": sum(1 for x in lab if x["raw_is_optimal"] and not x["selected_is_optimal"]),
                        "raw_opt_to_final_not_raw_survives_c3": sum(1 for x in lab if x["raw_is_optimal"] and not x["selected_is_optimal"] and x.get("raw_survives_c3")),
                        "raw_not_to_final_opt": sum(1 for x in lab if not x["raw_is_optimal"] and x["selected_is_optimal"]),
                        "c3_opt_to_final_not": sum(1 for x in lab if x.get("c3_is_optimal") and not x["selected_is_optimal"]), "c3_not_to_final_opt": sum(1 for x in lab if x.get("c3_is_optimal") is False and x["selected_is_optimal"])})
        gates = Counter(x.get("gate") for x in decs if x.get("gate"))
        if gates:
            row["gate_reasons"] = dict(gates)
        fam = Counter(x["selected"].split(":")[1] for x in decs)
        row["final_action_families"] = dict(fam)
        wrong = Counter(x["selected"].split(":")[1] for x in lab if not x["selected_is_optimal"])
        if lab:
            row["non_optimal_final_by_family"] = dict(wrong)
        rows.append(row)
    return rows


def build(rr):
    rr = Path(rr)
    res = rr / "results"
    probs, recs = all_records(rr)
    rows = by_problem(probs, recs)
    wcsv(res / "by_problem.csv", rows)
    wcsv(res / "by_structure.csv", by_structure(rows))
    wcsv(res / "paired_costs.csv", paired_costs(rows, probs))
    wcsv(res / "completion_by_time.csv", completion_by_time(rows))
    wcsv(res / "compute_costs.csv", compute_costs(rr, recs))
    wcsv(res / "training_and_label_costs.csv", training_costs(rr))
    wcsv(res / "raw_c3_final_actions.csv", raw_c3_final(recs))
    return rows, probs, recs


def numbers_md(rr, rows):
    """Plain numbers only (no interpretation): completion, cost and the registered primary paired comparisons."""
    rr = Path(rr)
    lines = ["# C1-PUBLIC-DEPOTS-REL-V1: auto-generated numbers (interpretation is in final_summary.md / claim_boundary.md)", ""]
    by = defaultdict(list)
    for r in rows:
        by[(r["set"], r["method"])].append(r)
    order = ["%s/%s" % (NAME[m], e) for m in MODES for e in EXECS] + ["LAMA-first", "LAMA-anytime", "WL-GOOSE"]
    for s in SETS:
        lines += ["## %s" % s, "", "| method | solved / n | failure-penalised cost ratio | cost ratio (solved) | total cost (solved) | excess over reference (solved) | mean wall s |", "|---|---|---|---|---|---|---|"]
        for m in order:
            rs = by.get((s, m))
            if rs:
                x = summarise_rows(rs)
                lines.append("| %s | %d / %d | %s | %s | %s | %s | %s |" % (m, x["success"], x["n"], x["mean_penalised_ratio"], x["mean_ratio_success"], x["total_cost_success"], x["excess_over_ref_success"], x["mean_wall_seconds"]))
        lines.append("")
    lines += ["Wall-clock columns: internal controllers = policy time (scoring-only label time excluded); WL-GOOSE = whole planner run; LAMA-first = time of its first plan; LAMA-anytime = time of ITS first plan inside the 300 s run (the full run lasts up to 300 s, see compute_costs.csv). The machine and its GPUs were shared with other users, so wall times are indicative only.", ""]
    pc = rr / "results" / "paired_costs.csv"
    if pc.is_file():
        lines += ["## primary and secondary paired comparisons (set ALL = struct + joint + ipc; per set in paired_costs.csv)", "", "| first | second | set | n | first solved | second solved | only first | only second | common | first cheaper | first costlier | same | mean diff (common) | sign p |", "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        with open(pc, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["stratum_dim"] == "ALL" and r["role"] in ("PRIMARY", "secondary") and r["set"] in ("struct", "joint", "ipc"):
                    lines.append("| %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (r["first"], r["second"], r["set"], r["n_problems"], r["first_success"], r["second_success"], r["only_first"], r["only_second"],
                                                                                                                   r["common_success_n"], r["first_better"], r["first_worse"], r["same_cost"], r["mean_cost_diff_first_minus_second"], r["sign_test_p_two_sided"]))
    return "\n".join(lines) + "\n"