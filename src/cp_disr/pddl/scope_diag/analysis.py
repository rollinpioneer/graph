"""Certificates (R1 / R2 / R3), per-problem tables and the pre-registered scientific label of the search-scope diagnostic (plan sections 16 and 17). Reads only files written by earlier stages; nothing here
runs a model, a planner or a search. All thresholds come from the registered configuration."""
from __future__ import annotations

import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .core import DENSE, WL, gz_read, hx, ix, jload, tol

INF = float("inf")
LABELS = ("CROSS_SCOPE_SIGNAL", "LOCAL_WITNESS_PRESENT_NO_CROSS_LOCALIZATION", "INCONCLUSIVE_CERTIFICATES", "REFERENCE_ANCHOR_MISSING", "NUMERICAL_TIE_DOMINATED", "NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX")


def num(x):
    """JSON bound -> float | None (unknown). "INF" -> inf."""
    if x is None:
        return None
    if x == "INF":
        return INF
    return float(x)


def cert_remaining(U_better, L_worse):
    """-> 'CERT' if U(better) < L(worse) (strict, both known), 'UNKNOWN' if a bound is missing, else 'NO'. Equality is not a strict order; overlapping intervals are not equal."""
    if U_better is None or L_worse is None:
        return "UNKNOWN"
    return "CERT" if U_better < L_worse else "NO"


def order_of(v_better, v_worse):
    """Order of a (better, worse) pair under a scorer: CORRECT (better lower), INVERTED (better higher), TIE (within tolerance)."""
    if abs(v_better - v_worse) <= tol(v_better, v_worse):
        return "TIE"
    return "CORRECT" if v_better < v_worse else "INVERTED"


class Inputs:
    def __init__(self, rr):
        self.rr = Path(rr)
        self.manifest = jload(self.rr / "registration" / "case_manifest.json")
        self.refs = jload(self.rr / "references" / "plans.json")
        self.cfg = jload(self.rr / "registration" / "config.json")
        self.bounds = defaultdict(dict)                       # case -> hex -> row
        bp = self.rr / "labels" / "bounds.jsonl"
        if bp.is_file():
            for line in bp.read_text().splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.bounds[d["case_id"]][d["state_hex"]] = d
        self.scores = {}
        for c in self.manifest["cases"]:
            for sc in (DENSE, WL):
                p = self.rr / "states" / ("scores_%s_%s.json" % (sc, c["case_id"]))
                self.scores[(c["case_id"], sc)] = jload(p) if p.is_file() else {}
        self.local = {}
        for c in self.manifest["cases"]:
            p = self.rr / "states" / ("local_%s.json" % c["case_id"])
            self.local[c["case_id"]] = jload(p) if p.is_file() else []
        self.traces = {}
        for c in self.manifest["cases"]:
            for dr in (DENSE, WL):
                p = self.rr / "traces" / c["case_id"] / (dr + ".jsonl.gz")
                if p.is_file():
                    self.traces[(c["case_id"], dr)] = gz_read(p)
        self.ref_u = {}
        for cid, r in self.refs.items():
            L = r["length"]
            u = {}
            for k, h in enumerate(r["path_states_hex"]):
                u[h] = L - k
            self.ref_u[cid] = u

    # ---- bound access
    def lower(self, cid, h):
        row = self.bounds[cid].get(h)
        return num(row.get("lower")) if row else None

    def upper(self, cid, h):
        u = self.ref_u.get(cid, {}).get(h)
        row = self.bounds[cid].get(h)
        ub = num(row.get("upper")) if row else None
        cands = [x for x in (u, ub) if x is not None]
        return min(cands) if cands else None

    def upper_source(self, cid, h):
        row = self.bounds[cid].get(h)
        if h in self.ref_u.get(cid, {}):
            ub = num(row.get("upper")) if row and row.get("upper") is not None else None
            if ub is None or ub >= self.ref_u[cid][h]:
                return "VERIFIED_SUFFIX"
        return row.get("upper_source", "UNAVAILABLE") if row else "UNAVAILABLE"

    def lower_source(self, cid, h):
        row = self.bounds[cid].get(h)
        return row.get("lower_source", "UNAVAILABLE") if row else "UNAVAILABLE"

    def value(self, cid, scorer, h):
        return self.scores[(cid, scorer)].get(h)


# ------------------------------------------------------------------------------------------------ R3
def shared_parent(a, b):
    if not a["parents"] or not b["parents"]:
        return None
    return bool(set(a["parents"]) & set(b["parents"]))


def r3_rows(inp, cid, driver):
    """One row per (snapshot event, candidate): the candidate is a still-OPEN state next to the state about to be popped."""
    lines = inp.traces.get((cid, driver))
    rows = []
    if not lines:
        return rows
    for d in lines:
        if d.get("type") != "snapshot":
            continue
        s = d["popped"]
        Ls = inp.lower(cid, s["state"])
        Us_s = inp.upper(cid, s["state"])
        for cand in ([d["anchor"]] if d["anchor"] else []) + d["competitors"]:
            t = cand
            Ut = inp.upper(cid, t["state"])
            Lt = inp.lower(cid, t["state"])
            cr = cert_remaining(Ut, Ls)
            ct = "UNKNOWN"
            if cr != "UNKNOWN" and Ut is not None and Ls is not None:
                ct = "CERT" if (t["g"] + Ut) < (s["g"] + Ls) else "NO"
            sp = shared_parent(s, t)
            if sp is True:
                pclass = "SIBLING_IN_OPEN"
            elif sp is None:
                pclass = "GROUP_UNKNOWN"
            else:
                pclass = "CROSS_GROUP"
            dense_t = t["h"] if driver == DENSE else inp.value(cid, DENSE, t["state"])
            dense_s = s["h"] if driver == DENSE else inp.value(cid, DENSE, s["state"])
            wl_t = t["h"] if driver == WL else inp.value(cid, WL, t["state"])
            wl_s = s["h"] if driver == WL else inp.value(cid, WL, s["state"])
            drv_order = order_of(t["h"], s["h"])
            rows.append({"case_id": cid, "trace_id": driver, "event_id": d["event"], "pair_class": pclass, "role": t["role"], "better_hash": t["state"], "worse_hash": s["state"], "better_parent": ";".join(t["parents"][:4]),
                         "worse_parent": ";".join(s["parents"][:4]), "shared_recorded_parent": sp, "L_better": Lt, "U_better": Ut, "L_worse": Ls, "U_worse": Us_s, "g_better": t["g"], "g_worse": s["g"],
                         "certificate_remaining": cr, "certificate_total": ct, "dense_v_better": dense_t, "dense_v_worse": dense_s, "wl_h_better": wl_t, "wl_h_worse": wl_s,
                         "dense_order": order_of(dense_t, dense_s) if dense_t is not None and dense_s is not None else None, "wl_order": order_of(wl_t, wl_s) if wl_t is not None and wl_s is not None else None,
                         "driver_order": drv_order, "actual_popped": s["state"], "anchor_active": d["anchor"] is not None, "lower_backend": inp.lower_source(cid, s["state"]),
                         "upper_source": inp.upper_source(cid, t["state"]), "reference_plan_sha": inp.refs[cid]["sha256"], "ref_k": t.get("ref_k")})
    return rows


def r3_summary(inp, cid, driver, rows):
    lines = inp.traces.get((cid, driver)) or []
    snaps = [d for d in lines if d.get("type") == "snapshot"]
    summ = next((d for d in lines if d.get("type") == "summary"), {})
    by_event = defaultdict(list)
    for r in rows:
        by_event[r["event_id"]].append(r)
    cert = [r for r in rows if r["certificate_remaining"] == "CERT"]
    cross = [r for r in cert if r["pair_class"] == "CROSS_GROUP"]
    cross_inv = [r for r in cross if r["driver_order"] == "INVERTED"]
    cross_tie = [r for r in cross if r["driver_order"] == "TIE"]
    bounded_events = {e for e, rs in by_event.items() if any(r["certificate_remaining"] != "UNKNOWN" for r in rs)}
    other = WL if driver == DENSE else DENSE
    oth_key = "wl_order" if driver == DENSE else "dense_order"
    return {"case_id": cid, "trace": driver, "status": summ.get("status"), "expanded": summ.get("expanded"), "wall_total": summ.get("wall_total"), "snapshots_planned": len(summ.get("schedule", [])), "snapshots_taken": len(snaps),
            "anchor_active_events": sum(1 for d in snaps if d["anchor"]), "no_active_anchor_events": sum(1 for d in snaps if not d["anchor"]), "candidate_pairs": len(rows), "bounded_events": len(bounded_events),
            "pairs_with_both_bounds": sum(1 for r in rows if r["certificate_remaining"] != "UNKNOWN"), "pairs_unknown_bound": sum(1 for r in rows if r["certificate_remaining"] == "UNKNOWN"),
            "certified_remaining": len(cert), "certified_strict_total": sum(1 for r in cert if r["certificate_total"] == "CERT"), "certified_cross_group": len(cross), "certified_sibling": sum(1 for r in cert if r["pair_class"] == "SIBLING_IN_OPEN"),
            "certified_group_unknown": sum(1 for r in cert if r["pair_class"] == "GROUP_UNKNOWN"), "cross_inverted_under_driver": len(cross_inv), "cross_tie_under_driver": len(cross_tie),
            "cross_events": len({r["event_id"] for r in cross}), "cross_inverted_events": len({r["event_id"] for r in cross_inv}), "cross_delayed_states": len({r["better_hash"] for r in cross}),
            "cross_inverted_delayed_states": len({r["better_hash"] for r in cross_inv}), "cross_popped_states": len({r["worse_hash"] for r in cross}), "cross_distinct_state_pairs": len({(r["better_hash"], r["worse_hash"]) for r in cross}),
            "cross_witness_other_scorer_inverted": sum(1 for r in cross if r[oth_key] == "INVERTED"), "cross_witness_other_scorer_tie": sum(1 for r in cross if r[oth_key] == "TIE"),
            "cross_witness_other_scorer_correct": sum(1 for r in cross if r[oth_key] == "CORRECT"), "other_scorer": other}


# ------------------------------------------------------------------------------------------------ R1 / R2
def local_rows(inp, cid):
    """Per reference position: R1 (parent vs plan child), R2 (decision of each scorer among the labelled successors) and the additional all-pairs inversions."""
    out = []
    refU = inp.ref_u[cid]
    for pos in inp.local[cid]:
        p, c = pos["parent"], pos["child"]
        if pos.get("goal_successor"):
            out.append({"case_id": cid, "k": pos["k"], "status": "GOAL_AT_GENERATION"})
            continue
        succ = pos["succ"]
        rec = {"case_id": cid, "k": pos["k"], "status": "OK", "parent": p, "child": c, "n_successors": len(succ)}
        Lp, Uc = inp.lower(cid, p), inp.upper(cid, c)
        rec["R1_cert"] = cert_remaining(Uc, Lp)
        for sc in (DENSE, WL):
            vp, vc = inp.value(cid, sc, p), inp.value(cid, sc, c)
            rec["R1_%s_order" % sc] = order_of(vc, vp) if rec["R1_cert"] == "CERT" and vp is not None and vc is not None else None
        # R2: first choice of each scorer over ALL successors (value, canonical order), then relation to the labelled successors
        for sc in (DENSE, WL):
            vals = [(inp.value(cid, sc, h), fa, h) for h, fa, _aids in succ]
            if any(v[0] is None for v in vals):
                rec["R2_%s" % sc] = "NO_VALUES"
                continue
            best = min(vals, key=lambda x: (x[0], x[1]))
            y = best[2]
            Ly = inp.lower(cid, y)
            betters = []
            for v, fa, h in vals:
                if h == y:
                    continue
                Ub = inp.upper(cid, h)
                if cert_remaining(Ub, Ly) == "CERT":
                    betters.append((h, v))
            margin = [abs(v - best[0]) > tol(v, best[0]) for h, v in betters]
            rec["R2_%s_choice" % sc] = y
            rec["R2_%s_choice_L" % sc] = Ly
            rec["R2_%s_provably_better_available" % sc] = len(betters)
            rec["R2_%s_error" % sc] = "DECISION_ERROR" if any(margin) else ("TIE_CHOICE" if betters else ("OK" if Ly is not None else "UNKNOWN_L"))
            rec["R2_%s_choice_is_plan_child" % sc] = (y == c)
        # additional all-pairs inversions among labelled successors (not decision level)
        pairs = Counter()
        hs = [h for h, _fa, _a in succ]
        for hb in hs:
            Ub = inp.upper(cid, hb)
            if Ub is None:
                continue
            for hw in hs:
                if hw == hb:
                    continue
                if cert_remaining(Ub, inp.lower(cid, hw)) != "CERT":
                    continue
                for sc in (DENSE, WL):
                    vb, vw = inp.value(cid, sc, hb), inp.value(cid, sc, hw)
                    if vb is not None and vw is not None:
                        pairs["%s_%s" % (sc, order_of(vb, vw))] += 1
        rec["allpairs"] = dict(pairs)
        out.append(rec)
    return out


def local_summary(inp, cid, rows):
    ok = [r for r in rows if r["status"] == "OK"]
    s = {"case_id": cid, "positions": len(inp.local[cid]), "positions_ok": len(ok), "goal_at_generation": sum(1 for r in rows if r["status"] == "GOAL_AT_GENERATION"),
         "R1_certified": sum(1 for r in ok if r["R1_cert"] == "CERT"), "R1_unknown": sum(1 for r in ok if r["R1_cert"] == "UNKNOWN")}
    for sc in (DENSE, WL):
        s["R1_%s_inverted" % sc] = sum(1 for r in ok if r.get("R1_%s_order" % sc) == "INVERTED")
        s["R1_%s_tie" % sc] = sum(1 for r in ok if r.get("R1_%s_order" % sc) == "TIE")
        s["R1_%s_correct" % sc] = sum(1 for r in ok if r.get("R1_%s_order" % sc) == "CORRECT")
        s["R2_%s_decision_errors" % sc] = sum(1 for r in ok if r.get("R2_%s_error" % sc) == "DECISION_ERROR")
        s["R2_%s_tie_choices" % sc] = sum(1 for r in ok if r.get("R2_%s_error" % sc) == "TIE_CHOICE")
        s["R2_%s_ok" % sc] = sum(1 for r in ok if r.get("R2_%s_error" % sc) == "OK")
        s["R2_%s_unknown" % sc] = sum(1 for r in ok if r.get("R2_%s_error" % sc) in ("UNKNOWN_L", "NO_VALUES"))
        s["R2_%s_error_parents" % sc] = len({r["parent"] for r in ok if r.get("R2_%s_error" % sc) == "DECISION_ERROR"})
        s["R2_%s_choice_is_plan_child" % sc] = sum(1 for r in ok if r.get("R2_%s_choice_is_plan_child" % sc))
        for kind in ("INVERTED", "TIE", "CORRECT"):
            s["allpairs_%s_%s" % (sc, kind.lower())] = sum(r.get("allpairs", {}).get("%s_%s" % (sc, kind), 0) for r in ok)
    return s


# ------------------------------------------------------------------------------------------------ decision
def decide(cfg, groupF, r3_dense, local_dense, budget_exhausted):
    """Pre-registered rule (plan 17, numeric details frozen in the configuration)."""
    D = cfg["decision"]
    ev = {}
    for cid in groupF:
        a = r3_dense.get(cid, {})
        loc = local_dense.get(cid, {})
        ev[cid] = {"cross_inverted_events": a.get("cross_inverted_events", 0), "cross_inverted_delayed_states": a.get("cross_inverted_delayed_states", 0), "R2_error_parents": loc.get("R2_dense_error_parents", 0),
                   "anchor_active_events": a.get("anchor_active_events", 0), "bounded_events": a.get("bounded_events", 0), "certified_cross_group": a.get("certified_cross_group", 0), "cross_tie": a.get("cross_tie_under_driver", 0),
                   "cross_inv": a.get("cross_inverted_under_driver", 0), "certified_remaining": a.get("certified_remaining", 0)}
    sig = [c for c, e in ev.items() if e["cross_inverted_events"] >= D["signal"]["min_cross_events_per_problem"] and e["cross_inverted_delayed_states"] >= D["signal"]["min_delayed_states_per_problem"]]
    loc = [c for c, e in ev.items() if e["R2_error_parents"] >= D["local"]["min_error_parents_per_problem"]]
    anchor_missing = [c for c, e in ev.items() if e["anchor_active_events"] < D["coverage"]["min_active_anchor_events"]]
    cert_insuff = [c for c, e in ev.items() if e["bounded_events"] < D["coverage"]["min_bounded_events"]]
    tie_dom = []
    for c, e in ev.items():
        tot = e["cross_tie"] + e["cross_inv"]
        if tot > 0 and e["cross_tie"] / tot >= D["tie_dominated_fraction"]:
            tie_dom.append(c)
    cross_ok = len(sig) >= D["signal"]["min_problems"]
    local_ok = len(loc) >= D["local"]["min_problems"]
    if cross_ok:
        label = "CROSS_SCOPE_SIGNAL"
    elif local_ok:
        label = "LOCAL_WITNESS_PRESENT_NO_CROSS_LOCALIZATION"
    elif len(tie_dom) >= D["coverage"]["majority"]:
        label = "NUMERICAL_TIE_DOMINATED"
    elif len(anchor_missing) >= D["coverage"]["majority"]:
        label = "REFERENCE_ANCHOR_MISSING"
    elif len(cert_insuff) >= D["coverage"]["majority"] or budget_exhausted:
        label = "INCONCLUSIVE_CERTIFICATES"
    else:
        label = "NO_LOCALIZED_SIGNAL_IN_OBSERVED_PREFIX"
    return {"scientific_label": label, "mixed_local_and_cross": bool(cross_ok and local_ok), "signal_problems": sig, "local_problems": loc, "anchor_missing_problems": anchor_missing, "certificate_insufficient_problems": cert_insuff,
            "tie_dominated_problems": tie_dom, "budget_exhausted": budget_exhausted, "per_problem": ev}


def write_csv(path, rows, fields=None):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = []
        for r in rows:
            for k in r:
                if k not in fields:
                    fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})
    return path


CERT_FIELDS = ["case_id", "trace_id", "event_id", "pair_class", "better_hash", "worse_hash", "better_parent", "worse_parent", "shared_recorded_parent", "L_better", "U_better", "L_worse", "U_worse", "g_better", "g_worse",
               "certificate_remaining", "certificate_total", "dense_v_better", "dense_v_worse", "wl_h_better", "wl_h_worse", "dense_order", "wl_order", "actual_popped", "anchor_active", "lower_backend", "upper_source",
               "reference_plan_sha", "role", "driver_order", "ref_k"]


def analyze(rr):
    inp = Inputs(rr)
    cfg = inp.cfg
    cases = inp.manifest["cases"]
    groupF = [c["case_id"] for c in cases if c["group"] == "F"]
    all_rows, summaries, locs, loc_rows = [], [], {}, []
    r3s = {}
    for c in cases:
        cid = c["case_id"]
        for dr in (DENSE, WL):
            rows = r3_rows(inp, cid, dr)
            all_rows += rows
            s = r3_summary(inp, cid, dr, rows)
            s["group"] = c["group"]
            summaries.append(s)
            r3s[(cid, dr)] = s
        lr = local_rows(inp, cid)
        loc_rows += lr
        locs[cid] = local_summary(inp, cid, lr)
        locs[cid]["group"] = c["group"]
    res = Path(rr) / "results"
    certified = [r for r in all_rows if r["certificate_remaining"] == "CERT"]
    write_csv(Path(rr) / "labels" / "certified_pairs.csv", certified, CERT_FIELDS)
    write_csv(res / "candidate_pairs_all.csv", all_rows, CERT_FIELDS)
    write_csv(res / "r3_by_trace.csv", summaries)
    write_csv(res / "local_by_problem.csv", locs.values())
    flat = []
    for r in loc_rows:
        d = dict(r)
        d["allpairs"] = json.dumps(r.get("allpairs", {}))
        flat.append(d)
    write_csv(res / "local_positions.csv", flat)
    budget_exhausted = any(row.get("status") == "BUDGET_UNLABELLED" for cid in inp.bounds for row in inp.bounds[cid].values() if row.get("category", "").startswith("R3_POPPED") and cid in groupF)
    dec = decide(cfg, groupF, {c: r3s[(c, DENSE)] for c in groupF}, {c: locs[c] for c in groupF}, budget_exhausted)
    # per-problem table
    pp = []
    for c in cases:
        cid = c["case_id"]
        row = {"case_id": cid, "group": c["group"], "set": c["set"], "reference_length": inp.refs[cid]["length"], "reference_optimality": inp.refs[cid]["optimality"]}
        for dr in (DENSE, WL):
            s = r3s[(cid, dr)]
            for k in ("status", "expanded", "snapshots_taken", "anchor_active_events", "bounded_events", "certified_remaining", "certified_cross_group", "certified_sibling", "certified_group_unknown", "cross_inverted_under_driver",
                      "cross_tie_under_driver", "cross_inverted_events", "cross_inverted_delayed_states", "cross_distinct_state_pairs", "pairs_unknown_bound"):
                row["%s_%s" % (dr, k)] = s[k]
        for k, v in locs[cid].items():
            if k not in ("case_id", "group"):
                row["local_" + k] = v
        pp.append(row)
    write_csv(res / "per_problem.csv", pp)
    out = {"decision": dec, "r3": {"%s|%s" % k: v for k, v in r3s.items()}, "local": locs, "totals": {"certified_pairs": len(certified), "candidate_pairs": len(all_rows)},
           "config_decision": cfg["decision"]}
    (res / "diagnostic_summary.json").write_text(json.dumps(out, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return out
