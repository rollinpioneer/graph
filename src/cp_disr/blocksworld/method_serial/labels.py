"""Planner-derived training labels of the method suite (offline supervision only; never a model input).

* next-event labels: for a decision state s and U0 = final goal atoms currently false, Y*(s) = {(g, a)}: a is an optimal first action and g is an atom of U0 made true at the FIRST step
  along some optimal continuation that begins with a (event semantics FIRST_CURRENTLY_UNMET_FINAL_GOAL_ATOM; no change to the first-On variant).
* calibration endpoints: the recorded trajectory states 2 and 4 steps ahead, kept only when the segment is optimal (exact d* drop equals the step count).
"""
from __future__ import annotations

import json
import time
from collections import Counter

from .. import goal_progress as GP
from .. import planner as P
from .. import state as S


def event_label(problem, state, solver):
    """Dict label for one (problem, state); {'complete': False} when the planner limit hits any branch."""
    goal = problem.goal
    truth = S.goal_atom_truth(problem, state)
    u0 = sorted(a for a, v in truth.items() if not v)
    try:
        _L, acts = solver.optimal_actions(state, goal)
    except P.PlannerLimit:
        return {"complete": False, "U0": u0}
    memo = {}

    def first_events(s):
        tr = S.goal_atom_truth(problem, s)
        hit = frozenset(a for a in u0 if tr[a])
        if hit:
            return hit
        if s in memo:
            return memo[s]
        _l, a2 = solver.optimal_actions(s, goal)
        out = frozenset().union(*[first_events(S.apply(s, b)) for b in a2]) if a2 else frozenset()
        memo[s] = out
        return out
    try:
        pairs, a_ids = [], []
        for a in acts:
            aid = S.action_id(problem.names, a)
            a_ids.append(aid)
            for g in sorted(first_events(S.apply(state, a))):
                pairs.append([g, aid])
    except P.PlannerLimit:
        return {"complete": False, "U0": u0}
    return {"complete": True, "U0": u0, "A": sorted(a_ids), "Y": sorted(pairs)}


def _event_worker(args):
    case, states = args
    solver = P.Solver()
    return case["case_id"], {",".join(map(str, s)): event_label(case["problem"], tuple(s), solver) for s in states}


def build_event_labels(cases, d_train, procs=48):
    import multiprocessing as mp
    by_case = {}
    for t in d_train:
        by_case.setdefault(t["case_id"], set()).update(tuple(s) for s in t["states"][:-1])
    jobs = [({"case_id": c.case_id, "problem": c.problem}, sorted(by_case[c.case_id])) for c in cases if c.case_id in by_case]
    t0 = time.time()
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as pool:
        res = pool.map(_event_worker, jobs, chunksize=1)
    labels = {}
    for cid, d in res:
        for sk, lab in d.items():
            labels["%s|%s" % (cid, sk)] = lab
    return labels, round(time.time() - t0, 1)


def event_information(labels, d_train):
    """Unique-state and trajectory-weighted statistics of the label information (plan 7.2 items 1-10)."""
    def stats(keys_with_w):
        agg = Counter()
        sizes = {"U0": Counter(), "A": Counter(), "Q": Counter(), "Y": Counter()}
        for key, w in keys_with_w:
            lab = labels[key]
            agg["total"] += w
            if not lab["complete"]:
                agg["unknown"] += w
                continue
            agg["complete"] += w
            u0, A = set(lab["U0"]), set(lab["A"])
            Y = {(g, a) for g, a in lab["Y"]}
            Q = {g for g, _a in Y}
            assert {a for _g, a in Y} == A, "projection_action(Y) must equal A*"
            cart = {(g, a) for g in Q for a in A}
            sizes["U0"][len(u0)] += w
            sizes["A"][len(A)] += w
            sizes["Q"][len(Q)] += w
            sizes["Y"][len(Y)] += w
            agg["event_information_Q_ne_U0"] += w * (Q != u0)
            agg["pair_information_Y_ne_QxA"] += w * (Y != cart)
            ea = {a: frozenset(g for g, a2 in Y if a2 == a) for a in A}
            agg["different_events_for_different_actions"] += w * (len(set(ea.values())) > 1)
            agg["singleton_event_rectangle"] += w * (len(Q) == 1 and Y == cart)
            agg["Y_eq_QxA"] += w * (Y == cart)
            agg["Q_eq_U0"] += w * (Q == u0)
            agg["density_sum"] += w * (len(Y) / max(1, len(cart)))
            kinds = {("OnTable" if ":OnTable:" in g else "On") for g in Q}
            agg["events_only_OnTable"] += w * (kinds == {"OnTable"})
            agg["events_only_On"] += w * (kinds == {"On"})
            agg["events_mixed"] += w * (kinds == {"On", "OnTable"})
            pairs_kind = [("OnTable" if ":OnTable:" in g else "On") for g, _a in Y]
            agg["pairs_total"] += w * len(Y)
            agg["pairs_OnTable"] += w * sum(k == "OnTable" for k in pairs_kind)
        out = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in agg.items()}
        c = max(1e-9, agg["complete"])
        out["fractions_of_complete"] = {k: round(agg[k] / c, 4) for k in ("event_information_Q_ne_U0", "pair_information_Y_ne_QxA", "different_events_for_different_actions", "singleton_event_rectangle", "Y_eq_QxA", "Q_eq_U0",
                                                                          "events_only_OnTable", "events_only_On", "events_mixed")}
        out["mean_pair_density_Y_over_QxA"] = round(agg["density_sum"] / c, 4)
        out["fraction_of_legal_event_pairs_that_are_OnTable"] = round(agg["pairs_OnTable"] / max(1e-9, agg["pairs_total"]), 4)
        out["size_distributions"] = {k: {str(a): (round(b, 3) if isinstance(b, float) else b) for a, b in sorted(v.items())} for k, v in sizes.items()}
        return out
    uniq = [(k, 1) for k in labels]
    traj = []
    for t in d_train:
        for s in t["states"][:-1]:
            traj.append(("%s|%s" % (t["case_id"], ",".join(map(str, s))), 1))
    return {"unique_goal_state_pairs": stats(uniq), "trajectory_weighted_decisions": stats(traj)}


def _cal_worker(args):
    case, trajs = args
    solver = P.Solver()
    problem = case["problem"]
    out = {}
    for t in trajs:
        states = [tuple(s) for s in t["states"]]
        for i in range(len(t["actions"])):
            d0 = t["remaining"][i]
            eps = []
            for k in (2, 4):
                if i + k < len(states):
                    s2 = states[i + k]
                    d2 = solver.cost_to_go(s2, problem.goal)
                    if d2 is not None and d0 - d2 == k:
                        eps.append({"k": k, "state": list(s2), "d": d2})
            out["%s|%d" % (t["tid"], i)] = {"d0": d0, "endpoints": eps}
    return out


def build_calibration_targets(cases, d_train, procs=48):
    import multiprocessing as mp
    by_case = {}
    for t in d_train:
        by_case.setdefault(t["case_id"], []).append(t)
    jobs = [({"case_id": c.case_id, "problem": c.problem}, by_case[c.case_id]) for c in cases if c.case_id in by_case]
    with mp.get_context("fork").Pool(min(procs, len(jobs))) as pool:
        res = pool.map(_cal_worker, jobs, chunksize=1)
    out = {}
    for d in res:
        out.update(d)
    return out


def calibration_scale(targets, rank_labels, d_train):
    """c = median of the non-zero exact distances in the endpoint pools (>= 1)."""
    import statistics
    ds = []
    for t in d_train:
        for i, s in enumerate(t["states"][:-1]):
            tg = targets["%s|%d" % (t["tid"], i)]
            ds.append(tg["d0"])
            ds += list(rank_labels["%s|%s" % (t["case_id"], ",".join(map(str, s)))].values())
            ds += [e["d"] for e in tg["endpoints"]]
    nz = [d for d in ds if d > 0]
    return max(1.0, float(statistics.median(nz))) if nz else 1.0
