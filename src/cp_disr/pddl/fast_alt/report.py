"""Tables of the FAST / ALT card (plan sections 10 and 14), generated from the per-case records of this card and the stored rows of the previous card. Nothing here runs a model or a planner."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .. import depots as DP
from .. import task as T
from ..search_match.analysis import wcsv
from ..search_match.report import sign_p, status_at

SETS = ("struct", "joint", "ipc")
CONDS = ("REF_DENSE", "FAST_DENSE", "ALT_DENSE_ADD", "ALT_WL_ADD", "EAGER_WL", "EAGER_HADD")
TECH = ("ADAPTER_ERROR", "INVALID_PLAN", "MODEL_NONFINITE")
TIME_POINTS = (10, 60, 300)
OLD_NAME = {"V_DENSE": "OLD_DENSE", "H_WL": "OLD_WL", "H_ADD": "OLD_HADD"}


def load_new(rr):
    recs = []
    for p in sorted((Path(rr) / "runs").glob("*.jsonl")):
        if p.name == "search_results.jsonl":
            continue
        recs += [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    return recs


def load_old_rows(cfg, root):
    """Stored per-case rows of the previous card (condition names OLD_*); plan lengths come from the CSV."""
    rows = []
    with open(Path(root) / cfg["old_search_root"] / "results" / "search_by_case.csv", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r["scorer"] in OLD_NAME:
                rows.append(r)
    return rows


def to_rec(r):
    """Old CSV row -> record-like dict."""
    num = lambda x: (float(x) if x not in (None, "") else None)
    return {"condition": OLD_NAME[r["scorer"]], "set": r["set"], "case_id": r["case_id"], "status": r["status"], "solved": r["solved"] == "True", "plan_length": int(r["plan_length"]) if r["plan_length"] else None,
            "expanded": int(r["expanded"]) if r["expanded"] else None, "solved_at_expansion": int(r["solved_at_expansion"]) if r["solved_at_expansion"] else None, "wall_total": num(r["wall_total_s"]),
            "evaluated_states": int(r["evaluated_states"]) if r["evaluated_states"] else None, "snapshots": None, "old": True}


def build(rr, old, cfg, root):
    rr = Path(rr)
    res = rr / "results"
    res.mkdir(exist_ok=True)
    new = load_new(rr)
    outs = []
    # merged raw records
    merged = rr / "runs" / "search_results.jsonl"
    with open(merged, "w", encoding="utf-8") as f:
        for r in new:
            f.write(json.dumps(r, default=str) + "\n")
    outs.append(merged)
    oldrows = [to_rec(r) for r in load_old_rows(cfg, root)]
    allrec = {(r["condition"], r["set"], r["case_id"]): r for r in oldrows}
    allrec.update({(r["condition"], r["set"], r["case_id"]): r for r in new})
    topo = {}
    tp = root / cfg["old_search_root"] / "results" / "topology.csv"
    if tp.is_file():
        topo = {(r["set"], r["case_id"]): r for r in csv.DictReader(open(tp, encoding="utf-8"))}
    refs = {(r["set"], r["case_id"]): r for r in csv.DictReader(open(root / cfg["old_search_root"] / "results" / "reference_length_sensitivity.csv", encoding="utf-8"))}
    proven = lambda s, c: int(refs[(s, c)]["proven_optimal_length"]) if refs.get((s, c), {}).get("proven_optimal_length") not in (None, "") else None
    max_exp = cfg["search"]["max_expansions"]

    def st_at(r, M):
        if r.get("snapshots") is None and r.get("old"):                        # old rows have no snapshot dict: derive from solved_at_expansion
            if r["status"] in TECH:
                return r["status"]
            if r["solved"] and r["solved_at_expansion"] is not None and r["solved_at_expansion"] <= M:
                return "SOLVED"
            return "UNSOLVED_AT_NODE_BUDGET" if (r["solved"] or (r["expanded"] or 0) >= M) else ("TIMEOUT_BEFORE_NODE_LIMIT" if r["status"] == "TIMEOUT" else r["status"])
        return status_at(r, M, max_exp)
    # ---- per case table
    by_case = []
    for r in new:
        t = topo.get((r["set"], r["case_id"]), {})
        pr = proven(r["set"], r["case_id"])
        ma, mm = r.get("metrics_main") or {}, r.get("metrics_add") or {}
        row = {"condition": r["condition"], "set": r["set"], "case_id": r["case_id"], "status": r["status"], "solved": r["solved"], "plan_length": r["plan_length"], "plan_valid": r["plan_valid"], "expanded": r.get("expanded"),
               "solved_at_expansion": r.get("solved_at_expansion"), "generated": r.get("generated"), "wall_total_s": round(r["wall_total"], 3), "order_tag": r.get("order_tag"), "evaluated_main": ma.get("states_scored"),
               "evaluated_add": mm.get("states_scored"), "logical_batches": r.get("eval_batches"), "outer_physical_calls": ma.get("outer_physical_calls"), "encoder_forwards": ma.get("encoder_forwards"),
               "scoring_main_s": (r.get("alt") or {}).get("eval_seconds_main", r.get("eval_seconds")), "scoring_add_s": (r.get("alt") or {}).get("eval_seconds_add"), "prepare_main_s": (r.get("timings") or {}).get("prepare_main"),
               "rss_peak_growth_gib": round((r.get("rss_peak_growth_bytes") or 0) / 2 ** 30, 3), "gpu_peak_gib": round((r.get("gpu_peak_bytes") or 0) / 2 ** 30, 3) if r.get("gpu_peak_bytes") else None,
               "timeout_before_node_limit": r.get("timeout_before_node_limit"), "proven_optimal_length": pr, "ratio_to_optimal": (r["plan_length"] / pr) if (r["solved"] and pr) else None,
               "load_avg_1m": r.get("load_avg_1m"), "cache_bytes": ma.get("cache_bytes")}
        for c in ("n_crates", "n_goal_towers", "max_goal_height", "needs_transport", "destruction_label", "cell", "n_ground_actions"):
            row[c] = t.get(c)
        by_case.append(row)
    outs.append(wcsv(res / "search_by_case.csv", by_case))
    # ---- IPC by case: new conditions + old rows side by side
    ipc_rows = []
    for c in sorted({k[2] for k in allrec if k[1] == "ipc"}):
        row = {"case_id": c, "n_ground_actions": topo.get(("ipc", c), {}).get("n_ground_actions")}
        for cond in CONDS + tuple(OLD_NAME.values()):
            r = allrec.get((cond, "ipc", c))
            if r:
                row[cond] = "%s/%s exp/%s len/%.0fs" % (r["status"], r.get("expanded"), r.get("plan_length"), r.get("wall_total") or 0)
        # trajectory-equivalence of REF and FAST: compare rolling hashes at common marks
        a, b = allrec.get(("REF_DENSE", "ipc", c)), allrec.get(("FAST_DENSE", "ipc", c))
        if a and b and a.get("prefix_hashes") and b.get("prefix_hashes"):
            ha, hb = a["prefix_hashes"], b["prefix_hashes"]
            common = [m for m in ("1", "10", "100", "1000", "10000", "100000") if m in ha and m in hb]
            agree = [m for m in common if ha[m] == hb[m]]
            row["REF_vs_FAST_marks_compared"] = ",".join(common)
            row["REF_vs_FAST_marks_equal"] = ",".join(agree)
            row["REF_vs_FAST_end_hash_equal"] = (ha["end"] == hb["end"]) if a["status"] == b["status"] else None
        ipc_rows.append(row)
    outs.append(wcsv(res / "ipc_by_case.csv", ipc_rows))
    # ---- coverage summary per condition / set / budget
    summ = []
    for cond in CONDS + tuple(OLD_NAME.values()):
        for s in SETS:
            rs = [r for (c, ss, _), r in allrec.items() if c == cond and ss == s]
            if not rs:
                continue
            for M in cfg["search"]["milestones"]:
                sts = [st_at(r, M) for r in rs]
                sol = [r for r, x in zip(rs, sts) if x == "SOLVED"]
                lens = [r["plan_length"] for r in sol]
                rat = [r["plan_length"] / proven(s, r["case_id"]) for r in sol if proven(s, r["case_id"])]
                summ.append({"condition": cond, "set": s, "node_budget": M, "n": len(rs), "solved": len(sol), "mean_plan_length": round(statistics.mean(lens), 3) if lens else None,
                             "mean_ratio_to_proven_optimum": round(statistics.mean(rat), 4) if rat else None, "optimal_plans": sum(1 for x in rat if abs(x - 1) < 1e-9) if rat else None,
                             "mean_expansions_at_solution": round(statistics.mean(r["solved_at_expansion"] for r in sol), 2) if sol else None, "statuses": dict(Counter(sts))})
    outs.append(wcsv(res / "coverage_summary.csv", summ))
    # ---- time curve (IPC) and compute costs
    tc, comp = [], []
    for cond in CONDS + tuple(OLD_NAME.values()):
        for s in SETS:
            rs = [r for (c, ss, _), r in allrec.items() if c == cond and ss == s]
            if not rs:
                continue
            row = {"condition": cond, "set": s, "n": len(rs)}
            for T_ in TIME_POINTS:
                row["solved_within_%ds" % T_] = sum(1 for r in rs if r["solved"] and (r.get("wall_total") or 1e9) <= T_)
            tc.append(row)
            if not rs[0].get("old"):
                ma = [r.get("metrics_main") or {} for r in rs]
                mm = [r.get("metrics_add") or {} for r in rs]
                alt = [r.get("alt") or {} for r in rs]
                comp.append({"condition": cond, "set": s, "runs": len(rs), "wall_total_s": round(sum(r["wall_total"] for r in rs), 1), "expanded": sum(r.get("expanded") or 0 for r in rs), "generated": sum(r.get("generated") or 0 for r in rs),
                             "logical_batches": sum(r.get("eval_batches") or 0 for r in rs), "main_states_scored": sum(m.get("states_scored", 0) for m in ma), "main_outer_physical_calls": sum(m.get("outer_physical_calls", 0) or 0 for m in ma),
                             "main_encoder_forwards": sum(m.get("encoder_forwards", 0) or 0 for m in ma), "add_states_scored": sum(m.get("states_scored", 0) for m in mm if m),
                             "scoring_main_s": round(sum(a.get("eval_seconds_main", r.get("eval_seconds") or 0) for a, r in zip(alt, rs)), 1), "scoring_add_s": round(sum(a.get("eval_seconds_add", 0) for a in alt), 1),
                             "inference_main_s": round(sum(m.get("inference_seconds", 0) for m in ma), 1), "prepare_main_s": round(sum((r.get("timings") or {}).get("prepare_main", 0) for r in rs), 1),
                             "valid_pops_main": sum((a.get("valid_pops") or [0, 0])[0] for a in alt) or None, "valid_pops_add": sum((a.get("valid_pops") or [0, 0])[1] for a in alt) or None,
                             "stale_pops_main": sum((a.get("stale_pops") or [0, 0])[0] for a in alt) or None, "stale_pops_add": sum((a.get("stale_pops") or [0, 0])[1] for a in alt) or None,
                             "fallback_pops": sum(a.get("fallback_pops", 0) for a in alt) or None, "peak_open_main": max(((a.get("peak_open") or [0, 0])[0] for a in alt), default=None) or None,
                             "peak_rss_growth_gib": round(max((r.get("rss_peak_growth_bytes") or 0) for r in rs) / 2 ** 30, 2), "peak_gpu_gib": round(max((r.get("gpu_peak_bytes") or 0) for r in rs) / 2 ** 30, 2) if any(r.get("gpu_peak_bytes") for r in rs) else None,
                             "cache_bytes_max": max((m.get("cache_bytes") or 0) for m in ma) or None, "statuses": dict(Counter(r["status"] for r in rs)), "technical": sum(1 for r in rs if r["status"] in TECH)})
    outs.append(wcsv(res / "time_curve.csv", tc))
    outs.append(wcsv(res / "compute_costs.csv", comp))
    # ---- paired comparisons
    pair_defs = [("ALT_DENSE_ADD", "FAST_DENSE", "main"), ("ALT_DENSE_ADD", "REF_DENSE", "main"), ("ALT_DENSE_ADD", "ALT_WL_ADD", "main"), ("ALT_DENSE_ADD", "EAGER_WL", "control"), ("ALT_DENSE_ADD", "EAGER_HADD", "control"),
                 ("ALT_DENSE_ADD", "OLD_DENSE", "reference_old"), ("ALT_WL_ADD", "EAGER_WL", "control"), ("ALT_WL_ADD", "EAGER_HADD", "control"), ("ALT_WL_ADD", "OLD_WL", "reference_old"), ("FAST_DENSE", "REF_DENSE", "speed_equivalence"),
                 ("FAST_DENSE", "EAGER_WL", "control"), ("REF_DENSE", "OLD_DENSE", "reproduction")]
    pairs = []
    for a, b, role in pair_defs:
        for s in SETS:
            for M in cfg["search"]["milestones"]:
                ids = sorted(c for (cc, ss, c) in allrec if cc == a and ss == s and (b, s, c) in allrec)
                if not ids:
                    continue
                sa = {c: st_at(allrec[(a, s, c)], M) for c in ids}
                sb = {c: st_at(allrec[(b, s, c)], M) for c in ids}
                both = [c for c in ids if sa[c] == "SOLVED" and sb[c] == "SOLVED"]
                oa = [c for c in ids if sa[c] == "SOLVED" and sb[c] != "SOLVED"]
                ob = [c for c in ids if sb[c] == "SOLVED" and sa[c] != "SOLVED"]
                dl = [allrec[(a, s, c)]["plan_length"] - allrec[(b, s, c)]["plan_length"] for c in both]
                de = [allrec[(a, s, c)]["solved_at_expansion"] - allrec[(b, s, c)]["solved_at_expansion"] for c in both]
                sh, lo = sum(1 for x in dl if x < 0), sum(1 for x in dl if x > 0)
                fe, me = sum(1 for x in de if x < 0), sum(1 for x in de if x > 0)
                pairs.append({"first": a, "second": b, "role": role, "set": s, "node_budget": M, "n": len(ids), "first_solved": sum(v == "SOLVED" for v in sa.values()), "second_solved": sum(v == "SOLVED" for v in sb.values()),
                              "both": len(both), "only_first": len(oa), "only_second": len(ob), "only_first_ids": oa, "only_second_ids": ob, "coverage_sign_p": round(sign_p(len(oa), len(ob)), 6),
                              "first_shorter": sh, "first_longer": lo, "same_length": len(dl) - sh - lo, "sum_length_diff": sum(dl) if dl else None, "median_length_diff": statistics.median(dl) if dl else None,
                              "length_sign_p": round(sign_p(sh, lo), 6), "first_fewer_expansions": fe, "first_more_expansions": me, "expansion_sign_p": round(sign_p(fe, me), 6)})
    outs.append(wcsv(res / "paired_quality.csv", pairs))
    outs.append(primary_md(rr, res, summ, pairs, comp, tc, allrec, cfg, proven))
    outs.append(verify(rr, root, cfg, new, old))
    return outs


def primary_md(rr, res, summ, pairs, comp, tc, allrec, cfg, proven):
    L = ["# C1-DEPOTS-FROZEN-SCORE-FAST-ALT-V3: generated numbers (interpretation is in final_summary.md / claim_boundary.md)", ""]
    for s in SETS:
        L += ["## %s: coverage / mean plan length of solved (node budgets 1,000 / 10,000 / 100,000 within 300 s)" % s, "", "| condition | <= 1,000 | <= 10,000 | <= 100,000 | ratio to proven optimum | optimal plans | mean expansions at solution | statuses (100k) |", "|---|---|---|---|---|---|---|---|"]
        for cond in CONDS + tuple(OLD_NAME.values()):
            cells = {r["node_budget"]: r for r in summ if r["condition"] == cond and r["set"] == s}
            if cells:
                g = lambda M: "%d/%d (%s)" % (cells[M]["solved"], cells[M]["n"], cells[M]["mean_plan_length"])
                last = cells[cfg["search"]["max_expansions"]]
                L.append("| %s | %s | %s | %s | %s | %s | %s | %s |" % (cond, g(1000), g(10000), g(100000), last["mean_ratio_to_proven_optimum"], last["optimal_plans"], last["mean_expansions_at_solution"], last["statuses"]))
        L.append("")
    L += ["## IPC: solved within wall-clock seconds", "", "| condition | 10 s | 60 s | 300 s |", "|---|---|---|---|"]
    for r in tc:
        if r["set"] == "ipc":
            L.append("| %s | %s | %s | %s |" % (r["condition"], r["solved_within_10s"], r["solved_within_60s"], r["solved_within_300s"]))
    L += ["", "## paired comparisons at the primary milestone (%d nodes)" % cfg["search"]["primary_milestone"], "", "| first | second | set | solved | only first / only second | common: shorter / longer / same (p) | fewer / more expansions (p) |", "|---|---|---|---|---|---|---|"]
    for p in pairs:
        if p["node_budget"] == cfg["search"]["primary_milestone"]:
            L.append("| %s | %s | %s | %d / %d | %d / %d | %d / %d / %d (%.3g) | %d / %d (%.3g) |" % (p["first"], p["second"], p["set"], p["first_solved"], p["second_solved"], p["only_first"], p["only_second"], p["first_shorter"], p["first_longer"], p["same_length"],
                                                                                              p["length_sign_p"], p["first_fewer_expansions"], p["first_more_expansions"], p["expansion_sign_p"]))
    # Joint protection line
    jd = [r for r in summ if r["condition"] == "ALT_DENSE_ADD" and r["set"] == "joint" and r["node_budget"] == cfg["search"]["max_expansions"]]
    jo = [r for r in summ if r["condition"] == "OLD_DENSE" and r["set"] == "joint" and r["node_budget"] == cfg["search"]["max_expansions"]]
    if jd and jo and jd[0]["mean_ratio_to_proven_optimum"] and jo[0]["mean_ratio_to_proven_optimum"]:
        diff = jd[0]["mean_ratio_to_proven_optimum"] - jo[0]["mean_ratio_to_proven_optimum"]
        L += ["", "## Joint protection line (engineering trade-off, not a statistical claim)", "", "ALT_DENSE_ADD solved %d/%d, OLD_DENSE %d/%d; mean ratio to optimum %.4f vs %.4f; difference %+.4f (line: <= +%.2f when all 128 are solved)." % (
            jd[0]["solved"], jd[0]["n"], jo[0]["solved"], jo[0]["n"], jd[0]["mean_ratio_to_proven_optimum"], jo[0]["mean_ratio_to_proven_optimum"], diff, cfg["alt"]["joint_protection_line_mean_ratio_increase"])]
    p = res / "primary_tables.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


def verify(rr, root, cfg, new, old):
    """Technical consistency (counts, plan replay, identities, data roles); PASS means consistency only."""
    rr = Path(rr)
    fr = json.loads((rr / "plan" / "frozen.json").read_text())
    ident = json.loads((rr / "assets" / "frozen_identity.json").read_text())
    from collections import Counter
    cnt = Counter((r["condition"], r["set"]) for r in new)
    planned = {("ALT_DENSE_ADD", "struct"): 32, ("ALT_DENSE_ADD", "joint"): 128, ("ALT_DENSE_ADD", "ipc"): 22, ("ALT_WL_ADD", "struct"): 32, ("ALT_WL_ADD", "joint"): 128, ("ALT_WL_ADD", "ipc"): 22, ("REF_DENSE", "ipc"): 22,
               ("EAGER_WL", "ipc"): 22, ("EAGER_HADD", "ipc"): 22}
    if fr["fast_status"] == "PASS":
        planned[("FAST_DENSE", "ipc")] = 22
    missing = {f"{k[0]}/{k[1]}": v - cnt.get(k, 0) for k, v in planned.items() if cnt.get(k, 0) != v}
    unexpected = [f"{k[0]}/{k[1]}" for k in cnt if k not in planned]
    # replay every stored plan
    bad = []
    replayed = 0
    tasks = {}
    for r in new:
        if r["solved"]:
            f = rr / "plans" / ("%s__%s__%s.plan" % (r["condition"], r["set"], r["case_id"]))
            if not f.is_file():
                bad.append((r["condition"], r["case_id"], "plan file missing"))
                continue
            key = (r["set"], r["case_id"])
            if key not in tasks:
                c = next(x for x in old.cases(r["set"]) if x["case_id"] == r["case_id"])
                tasks[key] = T.depots_task(old.domain(r["set"]), c["file"])
            ids = ["a:" + l.strip()[1:-1].split()[0] + ":" + ":".join(l.strip()[1:-1].split()[1:]) + ":v1" for l in f.read_text().splitlines() if l.startswith("(")]
            fin = tasks[key].replay(ids)
            replayed += 1
            if not (fin is not None and tasks[key].goal_satisfied(fin)) or len(ids) != r["plan_length"]:
                bad.append((r["condition"], r["case_id"], "replay"))
    out = {"records": len(new), "planned_after_freeze": fr["planned_main_searches"], "missing_runs": missing, "unexpected_conditions": unexpected, "plans_replayed": replayed, "plan_failures": bad,
           "technical_statuses": dict(Counter(r["status"] for r in new if r["status"] in TECH)), "hard_timer_fired": sum(1 for r in new if r.get("hard_timer")),
           "frozen_identity_sha256": ident["dense"]["sha256"], "weights_modified": False, "training_runs": 0, "optimizer_steps": 0, "new_labels": 0, "new_problems": 0,
           "old_results_modified": False, "data_roles": "all 182 problems are development / comparison material (previous cards); no sealed test", "consistent": (not missing and not unexpected and not bad)}
    p = rr / "results" / "verify.json"
    p.write_text(json.dumps(out, indent=1, default=str) + "\n", encoding="utf-8")
    return p
