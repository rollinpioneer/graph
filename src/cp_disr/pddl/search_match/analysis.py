"""Derived tables of card C1-DEPOTS-SEARCH-MATCH-V1 that need no search: the problem topology table (plan section 9) and the reference-length sensitivity table (plan section 8). Historical result files are only read."""
from __future__ import annotations

import csv
import hashlib
import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .. import data as DT
from .. import report as OLDREP
from .. import task as T
from ..parse import parse_problem

SETS = ("struct", "joint", "ipc")
UNARY_TYPES = ("crate", "pallet", "hoist", "truck", "depot", "distributor")


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
    return Path(p)


def object_counts(text):
    """Object counts per role from the problem file itself (typed objects, or the unary type facts of the untyped IPC encoding)."""
    p = parse_problem(text)
    cnt = Counter()
    if any(t != "object" for t in p.objects.values()):
        for o, t in p.objects.items():
            cnt[t] += 1
    else:
        for pred, args in p.init:
            if pred in UNARY_TYPES and len(args) == 1:
                cnt[pred] += 1
    return cnt, p


def goal_structure(p):
    """Explicit goal chains from the goal ``on`` atoms (partial goals allowed): number of towers with >= 2 crates, longest chain, and whether any goal crate must change place."""
    sup = {}
    for pred, args in p.goal:
        if pred == "on":
            sup[args[0]] = args[1]
    init_at = {x[1][0]: x[1][1] for x in p.init if x[0] == "at" and len(x[1]) == 2}
    init_on = {x[1][0]: x[1][1] for x in p.init if x[0] == "on" and len(x[1]) == 2}

    def place_of(o, depth=0):
        if o in init_at:
            return init_at[o]
        if o in init_on and depth < 50:
            return place_of(init_on[o], depth + 1)
        return None

    def chain(c, depth=0):
        return 1 + (chain(sup[c], depth + 1) if sup.get(c) in sup and depth < 50 else 0)
    lengths = {c: chain(c) for c in sup}
    bases = Counter()
    for c in sup:
        x = c
        d = 0
        while sup.get(x) in sup and d < 50:
            x = sup[x]
            d += 1
        bases[sup[x]] += 1
    towers = sum(1 for b, n in bases.items() if n >= 2)
    transport = False
    for c in sup:
        x = c
        d = 0
        while sup.get(x) in sup and d < 50:
            x = sup[x]
            d += 1
        base_place, crate_place = place_of(sup[x]), place_of(c)
        if base_place is not None and crate_place is not None and base_place != crate_place:
            transport = True
    return {"n_goal_atoms": len(p.goal), "n_goal_on_atoms": len(sup), "n_goal_towers_ge2": towers, "max_explicit_goal_chain": max(lengths.values()) if lengths else 0, "needs_transport_by_goal": transport}


def topology(old, out_csv):
    probs = OLDREP.problem_table(old.root)
    rows = []
    for s in SETS:
        domain = old.domain(s)
        for c in sorted(old.cases(s), key=lambda x: x["case_id"]):
            text = Path(c["file"]).read_text()
            cnt, p = object_counts(text)
            task = T.depots_task(domain, c["file"])
            tpl = task.template()
            meta = probs[s][c["case_id"]]
            gs = goal_structure(p)
            a = c.get("analysis") or {}
            rows.append({"set": s, "case_id": c["case_id"], "cell": c.get("cell"), "n_crates": cnt["crate"], "n_pallets": cnt["pallet"], "n_hoists": cnt["hoist"], "n_trucks": cnt["truck"], "n_depots": (cnt["depot"] or 1) if s != "ipc" else None,
                         "n_distributors": cnt["distributor"] if s != "ipc" else None,                                   # the IPC encoding has places only (depot / distributor are not distinguished) "n_places": sum(1 for o in task.obj_type.values() if o == "place"), "n_goal_atoms": gs["n_goal_atoms"], "n_goal_towers": a.get("k_goal_towers", gs["n_goal_towers_ge2"]),
                         "max_goal_height": a.get("max_goal_height", gs["max_explicit_goal_chain"]), "max_explicit_goal_chain": gs["max_explicit_goal_chain"], "needs_transport": a.get("needs_transport", gs["needs_transport_by_goal"]),
                         "max_init_height": a.get("max_init_height"), "n_ground_actions": len(task.actions), "n_dynamic_atoms": len(task.dyn_atoms), "n_template_nodes": len(tpl.nodes), "n_template_edges": len(tpl.edges),
                         "reference_length": meta["L_ref"], "reference_kind": meta["L_kind"], "destruction_label": meta["destruction_label"], "goal_is_complete_layout": bool(a.get("goal_complete", False)),
                         "file_sha256": c["sha256"]})
    return wcsv(out_csv, rows)


# ------------------------------------------------------------------------------------------------ reference lengths
def plan_ids_from_text(text):
    return ["a:" + l.strip()[1:-1].split()[0] + ":" + ":".join(l.strip()[1:-1].split()[1:]) + ":v1" for l in text.splitlines() if l.startswith("(")]


def valid_length(task, ids):
    fin = task.replay(ids)
    return len(ids) if (fin is not None and task.goal_satisfied(fin)) else None


def plan_hash(ids):
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]


def neural_successes(old):
    """{(set, case_id): [(label, action ids)]} of every successful episode of the nine neural rows of the previous card (read once)."""
    out = defaultdict(list)
    for s in SETS:
        for mode in ("mg", "dense", "rel"):
            for ex in ("one_step", "two_step", "gated"):
                p = old.root / "eval" / ("%s__%s__%s.jsonl" % (mode, ex, s))
                if p.is_file():
                    for line in p.read_text().splitlines():
                        if line.strip():
                            d = json.loads(line)
                            if d["success"]:
                                out[(s, d["case_id"])].append(("neural:%s/%s" % (mode, ex), [x["selected"] for x in d["decisions"]]))
    return out


def historical_candidates(old, s, cid, neural):
    """Every historical plan of the problem: (method, action ids); validated by the caller under the project semantics."""
    out = []
    root = old.root
    fd = root / "data" / "fd" / s / cid
    if fd.is_dir():
        for f in sorted(fd.glob("*.plan*")):
            ids = plan_ids_from_text(f.read_text())
            if ids:
                out.append(("lmcut" if "lmcut" in f.name else "fd:" + f.name.split(".")[0], ids))
    gp = root / "goose" / ("eval_" + s) / cid / "sas_plan"
    if gp.is_file():
        ids = plan_ids_from_text(gp.read_text())
        if ids:
            out.append(("wl_goose", ids))
    out += neural.get((s, cid), [])
    return out

def reference_sensitivity(old, out_csv):
    probs = OLDREP.problem_table(old.root)
    exact = json.loads((old.root / "data" / "exact_summary.json").read_text()) if (old.root / "data" / "exact_summary.json").is_file() else {}
    rows, improved, contradictions = [], 0, []
    neural = neural_successes(old)
    for s in SETS:
        refs = json.loads((old.root / "data" / ("refs_%s.json" % s)).read_text())
        domain = old.domain(s)
        for c in sorted(old.cases(s), key=lambda x: x["case_id"]):
            cid = c["case_id"]
            meta = probs[s][cid]
            task = T.depots_task(domain, c["file"])
            proven = None
            if exact.get(cid, {}).get("status") == "OK":
                proven = exact[cid]["optimal_length"]
            elif refs[cid]["lmcut"]["solved"] and refs[cid]["lmcut"]["best_length"] is not None:
                proven = refs[cid]["lmcut"]["best_length"]
            cands = []
            hist = historical_candidates(old, s, cid, neural)
            for method, ids in hist:
                L = valid_length(task, ids)
                if L is not None:
                    cands.append((L, method, ids))
            cands.sort(key=lambda x: (x[0], x[1]))
            best = cands[0] if cands else None
            neural_succ = [len(ids) for m, ids in hist if m.startswith("neural")]
            cf_cap = (2 * best[0] + 4) if best else None
            exceeding = sum(1 for L in neural_succ if cf_cap is not None and L > cf_cap)
            row = {"set": s, "case_id": cid, "registered_reference_length": meta["L_ref"], "registered_reference_kind": meta["L_kind"], "registered_step_cap": meta["cap"], "proven_optimal_length": proven,
                   "best_observed_valid_length": best[0] if best else None, "source_method": best[1] if best else None, "source_plan_hash": plan_hash(best[2]) if best else None, "valid_historical_plans": len(cands),
                   "counterfactual_step_cap_from_best": cf_cap, "old_neural_successes": len(neural_succ), "old_neural_successes_exceeding_counterfactual_cap": exceeding, "old_success_preserved": exceeding == 0,
                   "registered_reference_improved": bool(best and meta["L_ref"] is not None and best[0] < meta["L_ref"]),
                   "contradiction_with_proof": bool(best and proven is not None and best[0] < proven)}
            if row["registered_reference_improved"]:
                improved += 1
            if row["contradiction_with_proof"]:
                contradictions.append(cid)
            rows.append(row)
    p = wcsv(out_csv, rows)
    ipc = [r for r in rows if r["set"] == "ipc" and r["registered_reference_improved"]]
    return p, {"problems": len(rows), "registered_reference_improved": improved, "contradictions_with_proofs": contradictions, "ipc_improved": [(r["case_id"], r["registered_reference_length"], r["best_observed_valid_length"]) for r in ipc]}
