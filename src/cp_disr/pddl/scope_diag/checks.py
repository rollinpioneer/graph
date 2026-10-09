"""Checks with real assets (plan sections 14.2 and 18): mapping of project states to the translated task, LM-cut admissibility, repeat consistency, observer equivalence with the real scorers."""
from __future__ import annotations

import random
import time

from .. import task as T
from ..search_match.budget import Budget
from ..search_match.engine import SuccessorIndex, gbfs
from . import bounds as BD
from .core import load_task
from .observed import gbfs_observed


def sample_states(task, index, n, seed, oracle_plan_ids=None):
    """Fixed rule: the states of the exact optimal plan plus states of seeded random walks (distinct, reachable)."""
    rng = random.Random(seed)
    seen, order = set(), []

    def add(s):
        if s not in seen:
            seen.add(s)
            order.append(s)
    s = task.init_mask
    add(s)
    if oracle_plan_ids:
        for aid in oracle_plan_ids:
            s = task.apply(s, task.action_by_id[aid])
            add(s)
    tries = 0
    while len(order) < n and tries < 200:
        tries += 1
        s = task.init_mask
        for _ in range(rng.randint(3, 40)):
            acts = index.applicable(s)
            if not acts:
                break
            s = task.apply(s, rng.choice(acts))
            add(s)
            if len(order) >= n:
                break
    return order[:n]


def check_control(case, n_states=24, seed=0, timeout=60):
    """LM-cut on translated control task: init mapping, 0 <= L <= d* (exact oracle), goal value 0, repeat consistency."""
    task, dom = load_task(case)
    index = SuccessorIndex(task)
    bridge = BD.SasBridge(task, dom, case["file"])
    out = {"case_id": case["case_id"], "metric": bridge.metric, "init_mapping_equal": bridge.values(task.init_mask) == bridge.sas["init"], "cpu_translate": round(bridge.cpu_translate, 3)}
    oracle = T.ExactOracle(task)
    try:
        plan = oracle.plan()
        states = sample_states(task, index, n_states, seed, plan)
        viol, rows, cpu = [], 0, 0.0
        goal_zero = None
        for st in states:
            d, _ = oracle.query(st)
            v, status, c = BD.run_lmcut(bridge.sas_text(bridge.values(st)), timeout)
            cpu += c
            rows += 1
            if v is None or v == BD.INF or v > d or v < 0:
                viol.append({"state": format(st, "x"), "d": d, "lmcut": v, "status": status})
            if d == 0:
                goal_zero = (v == 0) if goal_zero is None else (goal_zero and v == 0)
        rep = []
        for st in states[:5]:
            a = BD.run_lmcut(bridge.sas_text(bridge.values(st)), timeout)[0]
            b = BD.run_lmcut(bridge.sas_text(bridge.values(st)), timeout)[0]
            rep.append(a == b)
        out.update({"states": rows, "violations": viol, "admissible": not viol, "goal_state_value_zero": goal_zero, "repeat_consistent": all(rep), "cpu_seconds": round(cpu + bridge.cpu_translate, 3),
                    "optimal_length": len(plan)})
    finally:
        oracle.close()
        bridge.close()
    return out


def check_reference_suffix(case, ids, n_positions=12, timeout=60):
    """Real problem: for states on the verified reference path, L(w_k) <= remaining plan length (a valid upper bound on d*): a necessary condition of the whole mapping + LM-cut chain."""
    task, dom = load_task(case)
    bridge = BD.SasBridge(task, dom, case["file"])
    try:
        s = task.init_mask
        seq = [s]
        for aid in ids:
            s = task.apply(s, task.action_by_id[aid])
            seq.append(s)
        L = len(ids)
        ks = sorted({int(round(i * L / (n_positions - 1))) for i in range(n_positions)} | {0, L})
        viol, vals, cpu = [], [], bridge.cpu_translate
        init_equal = bridge.values(task.init_mask) == bridge.sas["init"]
        for k in ks:
            v, status, c = BD.run_lmcut(bridge.sas_text(bridge.values(seq[k])), timeout)
            cpu += c
            vals.append((k, v))
            if v is None or v > L - k:
                viol.append({"k": k, "lmcut": v, "suffix": L - k})
        return {"case_id": case["case_id"], "metric": bridge.metric, "init_mapping_equal": init_equal, "positions": len(ks), "violations": viol, "ok": not viol and init_equal, "cpu_seconds": round(cpu, 3),
                "values": vals[:14]}
    finally:
        bridge.close()


class Recorder:
    """Wraps an evaluator and records every (batch, values) call; the observed run replays these values so that GPU summation nondeterminism cannot be mistaken for an observer effect."""

    def __init__(self, inner):
        self.inner, self.calls = inner, []

    def prepare(self, task, ctx, budget):
        self.inner.prepare(task, ctx, budget)

    def evaluate(self, states, budget):
        v = self.inner.evaluate(states, budget)
        self.calls.append((list(states), list(v)))
        return v

    def metrics(self):
        return self.inner.metrics()

    def close(self):
        self.inner.close()


class Replayer:
    def __init__(self, calls):
        self.calls, self.i, self.mismatch = calls, 0, 0

    def prepare(self, task, ctx, budget):
        self.i = 0

    def evaluate(self, states, budget):
        if self.i >= len(self.calls) or self.calls[self.i][0] != list(states):
            self.mismatch += 1
            return [0.0] * len(states)
        v = self.calls[self.i][1]
        self.i += 1
        return list(v)

    def metrics(self):
        return {}

    def close(self):
        pass


def check_observer_real(case, evaluator, label, expansions=16):
    """Plain engine vs observed engine on a short prefix: identical pop sequence, plan and counters. The real scorer runs once in the plain search; the observed search replays its recorded values."""
    task, dom = load_task(case)
    index = SuccessorIndex(task)
    res = {}
    rec = Recorder(evaluator)
    order = []
    b = Budget(60, expansions, 8 * 2 ** 30, 20.0)
    t0 = time.perf_counter()
    rec.prepare(task, {"domain": str(dom), "problem": case["file"]}, b)
    r = gbfs(task, index, rec, b, milestones=(), observer=lambda s, n: order.append(s))
    res["plain"] = {"order": order, "plan": r.plan, "expanded": r.expanded, "generated": r.generated, "duplicates": r.duplicates, "path_updates": r.path_updates, "status": r.status, "best_h": r.best_h, "wall": time.perf_counter() - t0}
    rec.close()
    rep = Replayer(rec.calls)
    order2 = []
    b2 = Budget(60, expansions, 8 * 2 ** 30, 20.0)
    t0 = time.perf_counter()
    rep.prepare(task, {}, b2)
    r2, snaps, extra = gbfs_observed(task, index, rep, b2, snapshot_at=(1, 2, 5, 16), ref={}, case_id=case["case_id"], n_comp=3, observer=lambda s, n: order2.append(s))
    res["observed"] = {"order": order2, "plan": r2.plan, "expanded": r2.expanded, "generated": r2.generated, "duplicates": r2.duplicates, "path_updates": r2.path_updates, "status": r2.status, "best_h": r2.best_h,
                       "wall": time.perf_counter() - t0}
    same = all(res["plain"][k] == res["observed"][k] for k in ("order", "plan", "expanded", "generated", "duplicates", "path_updates", "status", "best_h")) and rep.mismatch == 0
    return {"scorer": label, "case_id": case["case_id"], "identical": same, "replay_mismatches": rep.mismatch, "expanded": res["plain"]["expanded"], "snapshots": len(snaps), "wall_plain": round(res["plain"]["wall"], 3),
            "wall_observed": round(res["observed"]["wall"], 3)}
