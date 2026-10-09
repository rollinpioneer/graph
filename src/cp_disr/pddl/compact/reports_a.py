"""Tables of part A of card CP-DISR-C1-COMPACT-DIAGNOSIS-V2 (plan 3 and 4.A): supervision contract, historical pairs, cost breakdown, IPC failure map. Generated from the stored records of the earlier cards; nothing is run here."""
from __future__ import annotations

import csv
import json
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .train import load_old, rj

ROOT = Path(__file__).resolve().parents[4]


def rcsv(p):
    return list(csv.DictReader(open(p, encoding="utf-8")))


def wcsv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for r in rows:
        for k in r:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in keys})
    return path


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------------------------ supervision contract
def supervision_rows(root, w1_report=None):
    man, exact, _ = load_old(root)
    trajs, labels = [], {}
    for c in man["train"]:
        ex = exact[c["case_id"]]
        if ex["status"] == "OK":
            trajs += ex["trajectories"]
            labels.update(ex["rank_labels"])
    n_dec = sum(len(t["actions"]) for t in trajs)
    uniq = {(t["case_id"], tuple(s)) for t in trajs for s in t["states"][:-1]}
    src = Counter(t["source"].rstrip("0123456789") for t in trajs)
    nsucc, optsize, strict, nostrict = [], [], [], 0
    for t in trajs:
        for s in t["states"][:-1]:
            dist = labels["%s|%s" % (t["case_id"], ",".join(map(str, s)))]
            dm = min(dist.values())
            nsucc.append(len(dist))
            optsize.append(sum(1 for d in dist.values() if d == dm))
            sp = sum(1 for a in dist for b in dist if dist[a] < dist[b])
            strict.append(sp)
            nostrict += sp == 0
    # original WL: only the optimal plan states; siblings of the plan action are all treated as 'maybe worse'
    sib_equal = sib_worse = sib_total = 0
    parents = 0
    for c in man["train"]:
        ex = exact[c["case_id"]]
        if ex["status"] != "OK":
            continue
        exp = next(t for t in ex["trajectories"] if t["source"] == "expert")
        for s, a in zip(exp["states"][:-1], exp["actions"]):
            dist = labels["%s|%s" % (c["case_id"], ",".join(map(str, s)))]
            dm = min(dist.values())
            parents += 1
            for b, d in dist.items():
                if b == a:
                    continue
                sib_total += 1
                if d == dm:
                    sib_equal += 1
                else:
                    sib_worse += 1
    neural = {"problems": 96, "trajectories": len(trajs), "trajectory_sources": dict(src), "decisions": n_dec, "unique_parent_states": len(uniq), "mean_legal_successors": statistics.mean(nsucc), "mean_optimal_set_size": statistics.mean(optsize),
              "decisions_with_ge2_optimal_actions": sum(1 for x in optsize if x >= 2) / len(optsize), "mean_strict_ordered_successor_pairs": statistics.mean(strict), "decisions_without_strict_pair": nostrict / n_dec}
    rows = [
        {"model": "MG-G / DENSE-G / REL-G / D0 / C0 (neural, ACTION loss)", "parents_used": "all states of 457 trajectories: %d expert plan states + deviation trajectories (4 deviations per plan)" % sum(1 for t in trajs if t["source"] == "expert"), **neural,
         "labels": "exact d* of every legal successor (rank_labels), complete optimal-action set", "objective": "-log sum_{a in A*} softmax(V(s)-V(T(s,a))) + mean softplus over strictly ordered legal successor pairs", "weights": "1/(trajectory length x batch size) per decision",
         "parent_child_term": "none (V(s) cancels)", "cross_parent_term": "none", "equal_distance_pairs": "no preference", "labels_generated_vs_consumed": "all generated labels of Train96 trajectories are consumed (no unused label)"},
        {"model": "WL original (L2 rank-SVM, goose classic dataset)", "parents_used": "optimal-plan states only (%d)" % parents, "trajectories": 96, "decisions": parents, "unique_parent_states": parents,
         "labels": "plan action only; every other legal successor is a 'maybe' group, the parent is the 'bad' group", "objective": "hinge pairs (plan successor < each sibling) and (plan successor < parent); duplicates dropped, weights not passed to the SVM fit",
         "sibling_pairs": sib_total, "sibling_equal_distance_to_plan_successor": sib_equal, "sibling_strictly_worse": sib_worse, "fraction_siblings_that_are_equally_optimal_but_treated_as_worse": sib_equal / sib_total if sib_total else None,
         "parent_child_term": "yes (plan successor better than parent)", "cross_parent_term": "none", "equal_distance_pairs": "treated as worse than the plan action", "deviation_states": "none",
         "labels_generated_vs_consumed": "exact distances were available but only the plan action identity is consumed"}]
    if w1_report:
        rows.append({"model": "W1 (D0-aligned rank-SVM, fixed L2 features, typed track)", "parents_used": "same %d parents as D0" % len(uniq), "decisions": n_dec, "unique_parent_states": len(uniq), **{("w1_" + k): v for k, v in w1_report.items()},
                     "labels": "exact successor distances; strictly ordered successor pairs only", "objective": "hinge, LinearSVC C=1 with D0-derived pair weights (sum of weights = number of unique pairs)", "parent_child_term": "none", "cross_parent_term": "none",
                     "equal_distance_pairs": "no preference", "labels_generated_vs_consumed": "same as D0"})
    return rows


# ------------------------------------------------------------------------------------------------ historical pairs
def historical_rows(sm_root, fa_root):
    sm = rcsv(Path(sm_root) / "results" / "search_by_case.csv")
    by = defaultdict(dict)
    for r in sm:
        by[(r["scorer"], r["case_id"])] = r
    cases = {}
    for r in sm:
        cases[r["case_id"]] = r
    rows = []
    pairs = [("V_DENSE", "H_WL"), ("V_MG", "V_DENSE"), ("V_REL", "V_DENSE"), ("V_DENSE", "H_ADD"), ("V_MG", "H_WL"), ("V_REL", "H_WL")]
    groups = defaultdict(list)
    for cid, r in cases.items():
        if r["set"] == "joint":
            groups["joint|" + r["cell"]].append(cid)
            groups["joint|ALL"].append(cid)
        elif r["set"] == "struct":
            groups["struct|ALL"].append(cid)
        else:
            groups["ipc|ALL"].append(cid)
    for g, ids in sorted(groups.items()):
        for a, b in pairs:
            ra, rb = [by.get((a, c)) for c in ids], [by.get((b, c)) for c in ids]
            ok = [(x, y) for x, y in zip(ra, rb) if x and y]
            both = [(x, y) for x, y in ok if x["solved"] == "True" and y["solved"] == "True"]
            ex = [(fnum(x["expanded"]), fnum(y["expanded"])) for x, y in both if fnum(x["expanded"]) and fnum(y["expanded"])]
            rt = [(fnum(x["ratio_to_optimal"]), fnum(y["ratio_to_optimal"])) for x, y in both if fnum(x["ratio_to_optimal"]) and fnum(y["ratio_to_optimal"])]
            rows.append({"group": g, "model_a": a, "model_b": b, "n": len(ok), "solved_a": sum(1 for x, y in ok if x["solved"] == "True"), "solved_b": sum(1 for x, y in ok if y["solved"] == "True"), "common_solved": len(both),
                         "only_a": sum(1 for x, y in ok if x["solved"] == "True" and y["solved"] != "True"), "only_b": sum(1 for x, y in ok if y["solved"] == "True" and x["solved"] != "True"),
                         "mean_expanded_a_common": statistics.mean(e[0] for e in ex) if ex else None, "mean_expanded_b_common": statistics.mean(e[1] for e in ex) if ex else None,
                         "a_fewer_expansions": sum(1 for e in ex if e[0] < e[1]), "a_more_expansions": sum(1 for e in ex if e[0] > e[1]),
                         "mean_ratio_a_common": statistics.mean(e[0] for e in rt) if rt else None, "mean_ratio_b_common": statistics.mean(e[1] for e in rt) if rt else None,
                         "a_shorter": sum(1 for e in rt if e[0] < e[1] - 1e-9), "a_longer": sum(1 for e in rt if e[0] > e[1] + 1e-9)})
    return rows


# ------------------------------------------------------------------------------------------------ cost breakdown
def cost_rows(sm_root, fa_root):
    rows = []
    prof = rcsv(Path(fa_root) / "profile" / "timings.csv")
    comp = ["split_host_pack_ms", "split_tensor_h2d_ms", "split_static_prep_ms", "split_relational_ms", "split_unused_readout_ms", "split_goal_value_ms", "split_readback_ms"]
    agg = defaultdict(list)
    for r in prof:
        agg[(r["template"], r["impl"], int(r["B"]))].append(r)
    for (tpl, impl, B), rs in sorted(agg.items()):
        if B not in (1, 8, 48):
            continue
        med = lambda c: statistics.median(fnum(x[c]) for x in rs if fnum(x[c]) is not None) if any(fnum(x[c]) is not None for x in rs) else None
        row = {"source": "FAST/ALT card micro-benchmark (A100, 3 repeats)", "scorer": "DENSE-G " + impl, "template": tpl, "batch": B, "wall_ms": med("wall_ms"), "per_state_ms": med("per_state_ms"), "encoder_forwards": rs[0]["encoder_forwards"],
               "template_nodes": rs[0]["template_nodes"], "template_edges": rs[0]["template_edges"], "goals": rs[0]["goals"]}
        for c in comp:
            row[c.replace("split_", "") ] = med(c)
        rows.append(row)
    sm = rcsv(Path(sm_root) / "results" / "compute.csv")
    for r in sm:
        w, sc, inf = fnum(r["wall_total_s"]), fnum(r["scoring_s"]), fnum(r["inference_s"])
        rows.append({"source": "same-search card compute table", "scorer": r["scorer"], "template": r["set"], "runs": r["runs"], "wall_s": w, "scoring_s": sc, "scoring_share_of_wall": sc / w if w and sc is not None else None,
                     "inference_s": inf, "translate_s": fnum(r["translate_s"]), "parse_ground_s": fnum(r["parse_ground_s"]), "model_load_s": fnum(r["model_load_s"]), "expanded": fnum(r["expanded"]), "evaluated_states": fnum(r["evaluated_states"]),
                     "forward_calls": fnum(r["forward_calls"]), "per_state_ms": 1000 * sc / fnum(r["evaluated_states"]) if sc is not None and fnum(r["evaluated_states"]) else None, "statuses": r["statuses"]})
    # run-level decomposition of the wall clock of every search of the same-search card: parse + ground, template / translation preparation, scoring (conversion + encoder + readout for neural scorers), the rest of the search (OPEN / CLOSED, successor generation)
    sbc = rcsv(Path(sm_root) / "results" / "search_by_case.csv")
    agg2 = defaultdict(lambda: defaultdict(float))
    for r in sbc:
        k = (r["scorer"], r["set"])
        for f in ("wall_total_s", "t_parse_ground_s", "t_prepare_s", "search_s", "scoring_s", "inference_s", "translate_s", "evaluated_states", "expanded", "generated"):
            v = fnum(r[f])
            if v is not None:
                agg2[k][f] += v
        agg2[k]["runs"] += 1
    for (sc, st), a in sorted(agg2.items()):
        w = a["wall_total_s"]
        rows.append({"source": "same-search card, summed over the runs of (scorer, set)", "scorer": sc, "template": st, "runs": a["runs"], "wall_s": w, "parse_ground_s": a["t_parse_ground_s"], "prepare_s": a["t_prepare_s"], "search_s": a["search_s"], "scoring_s": a["scoring_s"],
                     "inference_s": a["inference_s"] or None, "translate_s": a["translate_s"] or None, "search_overhead_without_scoring_s": a["search_s"] - a["scoring_s"], "scoring_share_of_wall": a["scoring_s"] / w if w else None,
                     "search_overhead_share_of_wall": (a["search_s"] - a["scoring_s"]) / w if w else None, "per_state_ms": 1000 * a["scoring_s"] / a["evaluated_states"] if a["evaluated_states"] else None, "expanded": a["expanded"], "evaluated_states": a["evaluated_states"],
                     "note": "model load / warm-up are outside the problem clock; overlapping or asynchronous parts are not separated (NA)"})
    fa = rcsv(Path(fa_root) / "results" / "compute_costs.csv")
    for r in fa:
        rows.append({"source": "FAST/ALT card compute table", "scorer": r["condition"], "template": r["set"], "runs": r["runs"], "wall_s": fnum(r["wall_total_s"]), "scoring_s": fnum(r["scoring_main_s"]), "scoring_share_of_wall": (fnum(r["scoring_main_s"]) / fnum(r["wall_total_s"])) if fnum(r["wall_total_s"]) else None,
                     "prepare_main_s": fnum(r["prepare_main_s"]), "expanded": fnum(r["expanded"]), "evaluated_states": fnum(r["main_states_scored"]), "forward_calls": fnum(r["main_outer_physical_calls"]), "encoder_forwards": fnum(r["main_encoder_forwards"]),
                     "inference_s": fnum(r["inference_main_s"]), "statuses": r["statuses"], "cache_bytes_max": fnum(r["cache_bytes_max"])})
    return rows


# ------------------------------------------------------------------------------------------------ IPC failure map
GROUP = {"F": ["ipc_p06", "ipc_p11", "ipc_p12", "ipc_p17"], "T": ["ipc_p21", "ipc_p22"], "R": ["ipc_p14", "ipc_p15"], "H": ["ipc_p19", "ipc_p20"]}


def ipc_map(sm_root, fa_root, scope_root):
    sm = {(r["scorer"], r["case_id"]): r for r in rcsv(Path(sm_root) / "results" / "search_by_case.csv") if r["set"] == "ipc"}
    fa = {(r["condition"], r["case_id"]): r for r in rcsv(Path(fa_root) / "results" / "search_by_case.csv") if r["set"] == "ipc"}
    topo = {r["case_id"]: r for r in rcsv(Path(sm_root) / "results" / "topology.csv") if r["set"] == "ipc"}
    grp = {c: g for g, ids in GROUP.items() for c in ids}
    rows = []
    for cid in sorted(topo):
        f = fa.get(("FAST_DENSE", cid))
        r_ = fa.get(("REF_DENSE", cid))
        wl = sm.get(("H_WL", cid))
        ha = sm.get(("H_ADD", cid))
        d_old = sm.get(("V_DENSE", cid))
        row = {"case_id": cid, "group": grp.get(cid, "other"), "ground_actions": topo[cid]["n_ground_actions"], "dyn_atoms": topo[cid]["n_dynamic_atoms"], "template_nodes": topo[cid]["n_template_nodes"], "template_edges": topo[cid]["n_template_edges"]}
        for tag, rec in (("OLD_DENSE", d_old), ("OLD_MG", sm.get(("V_MG", cid))), ("OLD_REL", sm.get(("V_REL", cid))), ("OLD_WL", wl), ("HADD", ha), ("REF_DENSE", r_), ("FAST_DENSE", f), ("ALT_DENSE", fa.get(("ALT_DENSE_ADD", cid))), ("ALT_WL", fa.get(("ALT_WL_ADD", cid)))):
            if rec:
                row[tag + "_status"] = rec["status"]
                row[tag + "_expanded"] = rec["expanded"]
                row[tag + "_wall_s"] = rec["wall_total_s"]
                row[tag + "_plan_len"] = rec["plan_length"]
        labels = []
        dense_ok = f and f["solved"] == "True"
        wl_ok = wl and wl["solved"] == "True"
        if dense_ok and not wl_ok:
            labels.append("DENSE_ONLY_SUCCESS")
        if f and not dense_ok:
            fe, ws = fnum(f["expanded"]), fnum(wl["solved_at_expansion"]) if wl_ok else None
            if f["status"] == "TIMEOUT":
                if wl_ok and ws is not None and fe >= ws:
                    labels.append("NODE_GUIDANCE_DISADVANTAGE_VS_WL")
                elif wl_ok:
                    labels.append("TIME_TRUNCATED_BEFORE_WL_NODE_COUNT")
                else:
                    labels.append("TIME_TRUNCATED_NO_WL_REFERENCE")
            if not wl_ok and not (ha and ha["solved"] == "True"):
                labels.append("HARD_FOR_ALL_OBSERVED_SCORERS")
        if wl_ok and not dense_ok and ha and ha["solved"] == "True":
            labels.append("H_ADD_ALSO_SOLVES")
        if dense_ok and wl_ok:
            labels.append("BOTH_SOLVE")
        if r_ and f and (r_["solved"] != f["solved"]):
            labels.append("REF_FAST_DISAGREE")
        row["labels"] = ";".join(labels)
        row["wl_solved_at_expansion"] = wl["solved_at_expansion"] if wl_ok else ""
        row["fast_expanded_at_limit"] = f["expanded"] if f else ""
        rows.append(row)
    return rows
