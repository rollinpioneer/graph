"""Evaluation tables of the new models (D0, C0, W1, T1) against the old ones on the shared panel (Joint32, IPC22), the Attention-function readout and the compute / training accounting. Generated from records only."""
from __future__ import annotations

import csv
import json
import math
import statistics
from collections import Counter, defaultdict
from pathlib import Path

from .reports_a import fnum, rcsv, wcsv
from .train import load_old, rj

OLD_NAMES = {"V_DENSE": "OLD_DENSE", "V_MG": "OLD_MG", "V_REL": "OLD_REL", "H_WL": "OLD_WL", "H_ADD": "OLD_HADD"}
NEW = ("D0", "C0", "T1", "W1")


def load_new(rr):
    recs = []
    for p in sorted((Path(rr) / "evaluation" / "runs").glob("*.jsonl")):
        if p.name.startswith("DENSE_WL_variants"):
            continue
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    return recs


def panel_rows(rr, root, sm_root):
    """One row per (arm, set, case) for the new arms (this card) and the old arms (earlier card, same problems)."""
    panels = rj(Path(rr) / "registration" / "panels.json")
    man, exact, _ = load_old(root)
    joint = {c["case_id"]: c for c in man["joint"]}
    sm = rcsv(Path(sm_root) / "results" / "search_by_case.csv")
    ref = {}
    sens = {(r["set"], r["case_id"]): r for r in rcsv(Path(sm_root) / "results" / "reference_length_sensitivity.csv")}
    for r in sm:
        if r["set"] in ("joint", "ipc"):
            ref[(r["set"], r["case_id"])] = (fnum(r["proven_optimal_length"]), fnum(r["best_observed_valid_length"]))
    rows = []

    def add(arm, setname, cid, st, solved, plan, exp, solved_at, wall, load=None, evals=None, batches=None, extra=None):
        key = ("joint" if setname == "joint32" else "ipc", cid)
        po, bb = ref.get(key, (None, None))
        row = {"arm": arm, "set": setname, "case_id": cid, "cell": joint[cid]["cell"] if cid in joint else "", "status": st, "solved": bool(solved), "plan_length": plan, "expanded": exp, "solved_at_expansion": solved_at, "wall_s": wall,
               "proven_optimal": po, "best_observed": bb, "ratio_to_optimal": plan / po if (solved and plan and po) else None, "ratio_to_best_observed": plan / bb if (solved and plan and bb) else None,
               "k_towers": joint[cid]["analysis"]["k_goal_towers"] if cid in joint else "", "max_goal_height": joint[cid]["analysis"]["max_goal_height"] if cid in joint else "", "n_crates": joint[cid]["analysis"]["n_crates"] if cid in joint else "",
               "needs_transport": joint[cid]["analysis"]["needs_transport"] if cid in joint else "", "states_scored": evals, "load_avg_1m": load}
        row.update(extra or {})
        rows.append(row)
    for d in load_new(rr):
        add(d["scorer"], d["set"], d["case_id"], d["status"], d["solved"], d.get("plan_length"), d.get("expanded"), d.get("solved_at_expansion"), round(d["wall_total"], 3), d.get("load_avg_1m"), (d.get("evaluator_metrics") or {}).get("states_scored"),
            extra={"timeout_before_node_limit": d.get("timeout_before_node_limit"), "arm_order": d.get("arm_order"), "gpu_peak_gib": round((d.get("gpu_peak_bytes") or 0) / 2 ** 30, 2) or None, "rss_peak_growth_gib": round((d.get("rss_peak_growth_bytes") or 0) / 2 ** 30, 3)})
    keep = set(panels["joint32"])
    for r in sm:
        if r["scorer"] not in OLD_NAMES:
            continue
        if r["set"] == "joint" and r["case_id"] in keep:
            add(OLD_NAMES[r["scorer"]], "joint32", r["case_id"], r["status"], r["solved"] == "True", fnum(r["plan_length"]), fnum(r["expanded"]), fnum(r["solved_at_expansion"]), fnum(r["wall_total_s"]), None, fnum(r["evaluated_states"]),
                extra={"timeout_before_node_limit": r["timeout_before_node_limit"]})
        elif r["set"] == "ipc":
            add(OLD_NAMES[r["scorer"]], "ipc22", r["case_id"], r["status"], r["solved"] == "True", fnum(r["plan_length"]), fnum(r["expanded"]), fnum(r["solved_at_expansion"]), fnum(r["wall_total_s"]), None, fnum(r["evaluated_states"]),
                extra={"timeout_before_node_limit": r["timeout_before_node_limit"]})
    return rows


def summary(rows):
    out = []
    groups = defaultdict(list)
    for r in rows:
        groups[(r["arm"], r["set"], "ALL")].append(r)
        if r["set"] == "joint32":
            groups[(r["arm"], r["set"], r["cell"])].append(r)
    for (arm, st, cell), rs in sorted(groups.items()):
        sol = [r for r in rs if r["solved"]]
        ex = [r["expanded"] for r in sol if r["expanded"] is not None]
        rb = [r["ratio_to_best_observed"] for r in sol if r["ratio_to_best_observed"]]
        ro = [r["ratio_to_optimal"] for r in sol if r["ratio_to_optimal"]]
        out.append({"arm": arm, "set": st, "group": cell, "n": len(rs), "solved": len(sol), "mean_expanded": statistics.mean(ex) if ex else None, "median_expanded": statistics.median(ex) if ex else None,
                    "geo_mean_expanded": math.exp(statistics.mean(math.log(max(e, 1)) for e in ex)) if ex else None, "mean_ratio_to_best_observed": statistics.mean(rb) if rb else None, "n_ratio_best": len(rb),
                    "mean_ratio_to_optimal": statistics.mean(ro) if ro else None, "n_ratio_opt": len(ro), "mean_wall_s": statistics.mean(r["wall_s"] for r in rs if r["wall_s"] is not None) if rs else None,
                    "statuses": json.dumps(dict(Counter(r["status"] for r in rs)))})
    return out


def paired(rows, pairs):
    idx = {(r["arm"], r["set"], r["case_id"]): r for r in rows}
    out = []
    for a, b in pairs:
        for st in ("joint32", "ipc22"):
            ids = sorted({k[2] for k in idx if k[1] == st and k[0] == a} & {k[2] for k in idx if k[1] == st and k[0] == b})
            if not ids:
                continue
            both = [i for i in ids if idx[(a, st, i)]["solved"] and idx[(b, st, i)]["solved"]]
            only_a = [i for i in ids if idx[(a, st, i)]["solved"] and not idx[(b, st, i)]["solved"]]
            only_b = [i for i in ids if idx[(b, st, i)]["solved"] and not idx[(a, st, i)]["solved"]]
            pl = [(idx[(a, st, i)]["plan_length"], idx[(b, st, i)]["plan_length"]) for i in both if idx[(a, st, i)]["plan_length"] and idx[(b, st, i)]["plan_length"]]
            ex = [(idx[(a, st, i)]["expanded"], idx[(b, st, i)]["expanded"]) for i in both if idx[(a, st, i)]["expanded"] and idx[(b, st, i)]["expanded"]]
            lr = [math.log(y / x) for x, y in ex]
            out.append({"a": a, "b": b, "set": st, "n": len(ids), "solved_a": sum(idx[(a, st, i)]["solved"] for i in ids), "solved_b": sum(idx[(b, st, i)]["solved"] for i in ids), "common": len(both), "only_a": len(only_a), "only_b": len(only_b),
                        "only_a_ids": ";".join(only_a), "only_b_ids": ";".join(only_b), "a_shorter": sum(1 for x, y in pl if x < y), "a_longer": sum(1 for x, y in pl if x > y), "same_length": sum(1 for x, y in pl if x == y),
                        "mean_len_a": statistics.mean(x for x, y in pl) if pl else None, "mean_len_b": statistics.mean(y for x, y in pl) if pl else None, "a_fewer_expansions": sum(1 for x, y in ex if x < y), "a_more_expansions": sum(1 for x, y in ex if x > y),
                        "geo_mean_expansion_ratio_b_over_a": math.exp(statistics.mean(lr)) if lr else None})
    return out


def verify(rr, root, cfg):
    """Technical consistency (plan 12): identities, caps, white list, data roles, plan replays. PASS only means the records are internally consistent."""
    from .. import depots as DP, stages as ST
    from .train import sha_file
    rr, root = Path(rr), Path(root)
    ident = rj(rr / "assets" / "asset_identity.json")
    out = {"checks": {}}
    ck = out["checks"]
    ck["old_checkpoints_unchanged"] = {m: sha_file(v["path"]) == v["file_sha256"] for m, v in ident["checkpoints"].items()}
    ck["old_inputs_unchanged"] = {"exact_json": sha_file(root / "data" / "exact.json") == ident["exact_json_sha256"], "manifest": sha_file(root / "data" / "manifest.json") == ident["manifest_sha256"], "lock": sha_file(root / "runs" / "lock.json") == ident["lock_json_sha256"],
                                  "wl_models": {k: sha_file(v["model"]) == v["model_sha256"] for k, v in ident["wl_models"].items()}}
    tr = {}
    for arm in ("D0", "C0", "T1"):
        p = rr / "training" / arm / "training_accounting.json"
        if p.is_file():
            a = rj(p)
            tr[arm] = {"accepted_updates": a["optimizer_steps"], "complete": a["optimizer_steps"] == 4100, "nan_events": a["nan_events"], "attempts": a["attempts"], "schedule_matches_ledger": a.get("schedule_decisions_match_ledger"),
                       "checkpoint_files_present": all(Path(v["path"]).is_file() for v in a["checkpoints"].values())}
    ck["neural_trainings"] = tr
    ck["neural_trainings_count"] = len(tr)
    ck["accepted_updates_total"] = sum(v["accepted_updates"] for v in tr.values())
    ck["training_dirs_whitelist"] = sorted(p.name for p in (rr / "training").iterdir() if p.is_dir())
    fitp = rr / "training" / "W1" / "fit_accounting.json"
    ck["wl_fits"] = rj(fitp)["fit"].get("fits") if fitp.is_file() else 0
    # plan replay
    man, _e, _ = load_old(root)
    files = {c["case_id"]: c["file"] for fam in ("joint", "ipc", "dev") for c in man[fam]}
    for r in rj(rr / "diagnostics" / "problem_transform_manifest.json")["rows"]:
        if r.get("file"):
            files[r["problem_id"]] = r["file"]
    bad, replayed, recs = [], 0, []
    for p in sorted((rr / "evaluation" / "runs").glob("*.jsonl")):
        for line in p.read_text().splitlines():
            if line.strip():
                recs.append(json.loads(line))
    tasks = {}
    for d in recs:
        if not d.get("solved"):
            continue
        f = rr / "evaluation" / "plans" / ("%s__%s__%s.plan" % (d["scorer"], d["set"], d["case_id"]))
        if not f.is_file():
            bad.append((d["scorer"], d["case_id"], "plan file missing"))
            continue
        dom = DP.DOMAIN_IPC if d["set"] == "ipc22" else DP.DOMAIN_TYPED
        key = (str(dom), files[d["case_id"]])
        if key not in tasks:
            tasks[key] = ST.get_task(dom, files[d["case_id"]])
        task = tasks[key]
        ids = ["a:" + ln.strip()[1:-1].split()[0] + ":" + ":".join(ln.strip()[1:-1].split()[1:]) + ":v1" for ln in f.read_text().splitlines() if ln.startswith("(")]
        fin = task.replay(ids)
        replayed += 1
        if not (fin is not None and task.goal_satisfied(fin)) or len(ids) != d["plan_length"]:
            bad.append((d["scorer"], d["case_id"], "replay"))
    ck["plans_replayed"], ck["plan_failures"] = replayed, bad
    st = Counter((d["scorer"], d["set"], d["status"]) for d in recs)
    ck["status_counts"] = {"%s|%s|%s" % k: v for k, v in sorted(st.items())}
    ck["technical_statuses"] = sum(v for k, v in st.items() if k[2] in ("ADAPTER_ERROR", "INVALID_PLAN", "MODEL_NONFINITE"))
    cnt = Counter((d["scorer"], d["set"]) for d in recs)
    ck["records_per_condition"] = {"%s|%s" % k: v for k, v in sorted(cnt.items())}
    ck["duplicate_records"] = len(recs) - len({(d["scorer"], d["set"], d["case_id"]) for d in recs})
    ck["pt_in_run_root_outside_training"] = [str(p) for p in rr.rglob("*.pt") if "training" not in p.parts]
    out["consistent"] = bool(all(ck["old_checkpoints_unchanged"].values()) and all(v for v in ck["old_inputs_unchanged"].values() if isinstance(v, bool)) and all(ck["old_inputs_unchanged"]["wl_models"].values())
                             and not bad and ck["duplicate_records"] == 0 and ck["technical_statuses"] == 0 and ck["neural_trainings_count"] <= 3 and ck["accepted_updates_total"] <= 12300 and ck["wl_fits"] == 1 and not ck["pt_in_run_root_outside_training"])
    return out


def ipc_map_new(rows, old_map_rows):
    """Wide per-problem table: every neural arm (old and new) against WL; label per arm: SOLVED, NODE_GUIDANCE_DISADVANTAGE_VS_WL (timed out after at least as many expansions as WL needed), TIME_TRUNCATED (fewer), NO_WL_REFERENCE."""
    idx = {(r["arm"], r["case_id"]): r for r in rows if r["set"] == "ipc22"}
    arms = ["OLD_DENSE", "OLD_MG", "OLD_REL", "D0", "D0R", "C0", "T1", "OLD_WL", "OLD_HADD"]
    group = {r["case_id"]: r["group"] for r in old_map_rows}
    out = []
    for cid in sorted({k[1] for k in idx}):
        row = {"case_id": cid, "group": group.get(cid, "other")}
        wl = idx.get(("OLD_WL", cid))
        for a in arms:
            r = idx.get((a, cid))
            if not r:
                continue
            row[a + "_status"] = r["status"]
            row[a + "_expanded"] = r["expanded"]
            row[a + "_wall_s"] = r["wall_s"]
            row[a + "_plan_len"] = r["plan_length"]
            if a in ("OLD_DENSE", "OLD_MG", "OLD_REL", "D0", "D0R", "C0", "T1"):
                if r["solved"]:
                    lab = "SOLVED"
                elif wl and wl["solved"]:
                    lab = "NODE_GUIDANCE_DISADVANTAGE_VS_WL" if r["status"] == "TIMEOUT" and r["expanded"] and wl["solved_at_expansion"] and r["expanded"] >= wl["solved_at_expansion"] else ("TIME_TRUNCATED" if r["status"] == "TIMEOUT" else r["status"])
                else:
                    lab = "NO_WL_REFERENCE_" + r["status"]
                row[a + "_label"] = lab
        out.append(row)
    return out


def attention_readout(rows):
    """Plan 3.2: D0 vs C0 by Joint cell. Advantage = geometric mean over the cell's common-solved problems of expanded(C0) / expanded(D0) (> 1: D0 needs fewer expansions) and mean plan-length difference C0 - D0.
    Pre-registered prediction: the advantage is larger in the two-tower cells (k2) than in the one-tower cells (k1) at the same n and goal height in at least 3 of 4 pairs, and D0 does not lose coverage."""
    idx = {(r["arm"], r["case_id"]): r for r in rows if r["set"] == "joint32"}
    cells = defaultdict(list)
    for (arm, cid), r in idx.items():
        if arm == "D0":
            cells[r["cell"]].append(cid)
    res = {}
    for cell, ids in sorted(cells.items()):
        both = [i for i in ids if idx.get(("D0", i)) and idx.get(("C0", i)) and idx[("D0", i)]["solved"] and idx[("C0", i)]["solved"]]
        lr = [math.log(idx[("C0", i)]["expanded"] / idx[("D0", i)]["expanded"]) for i in both if idx[("D0", i)]["expanded"] and idx[("C0", i)]["expanded"]]
        dl = [idx[("C0", i)]["plan_length"] - idx[("D0", i)]["plan_length"] for i in both]
        res[cell] = {"n": len(ids), "common": len(both), "advantage_expansions_geo": math.exp(statistics.mean(lr)) if lr else None, "mean_plan_len_C0_minus_D0": statistics.mean(dl) if dl else None,
                     "d0_solved": sum(1 for i in ids if idx[("D0", i)]["solved"]), "c0_solved": sum(1 for i in ids if idx.get(("C0", i)) and idx[("C0", i)]["solved"])}
    pairs = []
    for n in ("n6", "n8"):
        for h in ("h2", "h4"):
            k1, k2 = res.get("%sk1%s" % (n, h)), res.get("%sk2%s" % (n, h))
            if k1 and k2 and k1["advantage_expansions_geo"] and k2["advantage_expansions_geo"]:
                pairs.append({"n": n, "h": h, "adv_k1": k1["advantage_expansions_geo"], "adv_k2": k2["advantage_expansions_geo"], "k2_larger": k2["advantage_expansions_geo"] > k1["advantage_expansions_geo"]})
    return {"cells": res, "pairs": pairs, "prediction_pairs_supporting": sum(1 for p in pairs if p["k2_larger"]), "pairs_total": len(pairs),
            "coverage_not_lost": sum(c["d0_solved"] for c in res.values()) >= sum(c["c0_solved"] for c in res.values())}
