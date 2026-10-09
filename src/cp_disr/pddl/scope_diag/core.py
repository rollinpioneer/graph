"""Stage functions of the search-scope diagnostic: cases, reference information, observed traces, local (parent -> successors) structure, offline scoring of the sampled states and distance labels."""
from __future__ import annotations

import gzip
import hashlib
import json
import os
import resource
import time
from pathlib import Path

from .. import depots as DP
from .. import task as T
from ..search_match.budget import Budget, HardTimeout
from ..search_match.engine import SuccessorIndex
from .observed import gbfs_observed, snapshot_schedule
from .references import path_states

DENSE, WL = "dense", "wl"
TOL_ABS, TOL_REL = 1e-5, 1e-6


def tol(a, b):
    return TOL_ABS + TOL_REL * max(abs(a), abs(b))


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def hx(state):
    return format(state, "x")


def ix(h):
    return int(h, 16)


def sid(domain_sha, problem_sha, state):
    return hashlib.sha256(("%s|%s|%x" % (domain_sha, problem_sha, state)).encode()).hexdigest()[:20]


def jdump(path, obj):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def jload(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def gz_write(path, lines):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for d in lines:
            f.write(json.dumps(d, sort_keys=True, default=str) + "\n")


def gz_read(path):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


# ------------------------------------------------------------------------------------------------ cases
def load_task(case):
    dom = DP.DOMAIN_IPC if case["set"] == "ipc" else DP.DOMAIN_TYPED
    return T.depots_task(dom, case["file"]), dom


def case_budget(cfg, case):
    c = cfg["traces"]["ipc" if case["set"] == "ipc" else "control"]
    return c["max_expansions"], c["wall_seconds"], c["memory_gib"]


def local_positions(L, n):
    """Parent positions k in 0..L-2 (the child w_{k+1} is not the goal state), at most n, equidistant by position, de-duplicated."""
    top = L - 2
    if top < 0:
        return []
    if top + 1 <= n:
        return list(range(top + 1))
    return sorted({int(round(i * top / (n - 1))) for i in range(n)})


def local_structure(task, index, seq, positions):
    """Deterministic (no scorer): for each reference position the legal successors in canonical action order, unique states with all their action ids."""
    out = []
    goal = task.goal_mask
    for k in positions:
        s = seq[k]
        succ, order = {}, []
        goal_succ = False
        for a in index.applicable(s):
            t = (s & ~a.del_mask) | a.add_mask
            if task.solvable_by_relaxation and (t & goal) == goal:
                goal_succ = True
            if t not in succ:
                succ[t] = {"first_action": a.index, "aids": []}
                order.append(t)
            succ[t]["aids"].append(a.aid)
        out.append({"k": k, "parent": s, "child": seq[k + 1], "goal_successor": goal_succ, "succ": [(t, succ[t]["first_action"], succ[t]["aids"]) for t in order]})
    return out


# ------------------------------------------------------------------------------------------------ traces
def snap_json(snap):
    def r(x):
        if x is None:
            return None
        d = dict(x)
        d["state"] = hx(d["state"])
        d["parents"] = [hx(p) for p in d["parents"]]
        return d
    return {"type": "snapshot", "event": snap["event"], "open": snap["open"], "closed": snap["closed"], "expanded": snap["expanded"], "elapsed": round(snap["elapsed"], 4), "popped": r(snap["popped"]), "anchor": r(snap["anchor"]),
            "competitors": [r(c) for c in snap["competitors"]], "ref_deepest_generated": snap["ref_deepest_generated"], "ref_deepest_closed": snap["ref_deepest_closed"]}


def run_trace(case, driver, evaluator, ref_info, cfg, out_path):
    """One budgeted observed prefix of ``driver`` on ``case``. Returns the summary record (always written, also for resource stops / failures)."""
    max_exp, wall, mem = case_budget(cfg, case)
    task, dom = load_task(case)
    index = SuccessorIndex(task)
    ids = ref_info["ids"]
    seq, ref = path_states(task, ids)
    sched = snapshot_schedule(max_exp, cfg["traces"]["snapshots"])
    budget = Budget(wall, max_exp, int(mem * 2 ** 30), cfg["traces"].get("grace_seconds", 20.0))
    rec = {"type": "summary", "case_id": case["case_id"], "driver": driver, "status": None, "max_expansions": max_exp, "wall_seconds": wall, "schedule": sched}
    snaps, extra = [], {}
    try:
        budget.arm_hard_timer()
        t0 = time.perf_counter()
        evaluator.prepare(task, {"domain": str(dom), "problem": case["file"]}, budget)
        rec["prepare_seconds"] = round(time.perf_counter() - t0, 3)
        res, snaps, extra = gbfs_observed(task, index, evaluator, budget, snapshot_at=sched, ref=ref, case_id=case["case_id"], n_comp=cfg["traces"]["competitors"])
        rec.update({k: getattr(res, k) for k in ("status", "expanded", "generated", "duplicates", "path_updates", "solved_at_expansion", "open_size", "closed_size", "best_h", "h_root", "search_seconds", "eval_seconds")})
        rec["evaluated_calls"] = res.evaluated_calls
        if res.status == "SOLVED":
            rec["plan_length"] = len(res.plan)
    except HardTimeout:
        rec["status"] = "TIMEOUT"
        rec["hard_timer"] = True
    except Exception as e:                                        # adapter failure: technical, recorded
        rec["status"] = "ADAPTER_ERROR"
        rec["error"] = repr(e)[-400:]
    finally:
        Budget.disarm_hard_timer()
    rec["wall_total"] = round(budget.elapsed(), 3)
    rec["snapshot_seconds"] = round(extra.get("snapshot_seconds", 0.0), 4)
    rec["order_hash"] = extra.get("order_hash")
    rec["snapshots_taken"] = len(snaps)
    rec["rss_peak_growth_bytes"] = budget.peak_rss_growth
    try:
        rec["evaluator_metrics"] = evaluator.metrics()
    except Exception:
        rec["evaluator_metrics"] = {}
    header = {"type": "header", "case_id": case["case_id"], "driver": driver, "domain_sha256": sha_file(dom), "problem_sha256": case["sha256"], "reference_plan_sha256": ref_info["sha256"], "reference_length": len(ids),
              "schedule": sched, "n_dyn_atoms": len(task.dyn_atoms)}
    gz_write(out_path, [header] + [snap_json(s) for s in snaps] + [rec])
    try:
        evaluator.close()
    except Exception:
        pass
    return rec


def trace_states(trace_lines):
    """All sampled states (hex) of one trace file."""
    out = set()
    for d in trace_lines:
        if d.get("type") != "snapshot":
            continue
        for key in ("popped", "anchor"):
            if d.get(key):
                out.add(d[key]["state"])
        for c in d["competitors"]:
            out.add(c["state"])
    return out


# ------------------------------------------------------------------------------------------------ offline scoring
def score_states(evaluator, task, case, states_hex, wall=900.0):
    """Offline values of a set of states by one scorer (batches sorted by state, the evaluator chunks them itself)."""
    dom = DP.DOMAIN_IPC if case["set"] == "ipc" else DP.DOMAIN_TYPED
    b = Budget(wall, 10 ** 9, 16 * 2 ** 30, 60.0)
    evaluator.prepare(task, {"domain": str(dom), "problem": case["file"]}, b)
    order = sorted(states_hex)
    vals = evaluator.evaluate([ix(h) for h in order], b)
    try:
        evaluator.close()
    except Exception:
        pass
    return dict(zip(order, vals)), b.elapsed()


def cpu_now():
    r = resource.getrusage(resource.RUSAGE_SELF)
    c = resource.getrusage(resource.RUSAGE_CHILDREN)
    return r.ru_utime + r.ru_stime + c.ru_utime + c.ru_stime
