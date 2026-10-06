"""CP-DISR-C1-BW-A04P, G0: paired goal / time forward check on the frozen A02 B2 checkpoint (runbook 6). No environment episode, no new score, no training.

Each sample is a first non-optimal decision of B2 in the (consumed) A03 A2 / B failures. The pre-decision GRU hidden is restored by replaying the frozen network over the logged prefix (the replay must
reproduce the logged probabilities to 1e-5). Four input conditions share that exact history and hidden: G00 original, G10 other goal towers flattened, G01 reference time input, G11 both.
"""
from __future__ import annotations

import dataclasses
import statistics

import torch

from . import state as S
from .environment import BwEpisode, Case
from .imitation import snapshot_at

CONDITIONS = ("G00", "G10", "G01", "G11")
MIN_VALID = 12
FLIP_FRACTION = 0.25
IMPROVE_FRACTION = 0.60
MEDIAN_DELTA = 1.0
REPLAY_TOL = 1e-5


def parse(aid):
    p = aid.split(":")
    return p[1], p[2:-1]


def goal_components(goal):
    """Blocks of every goal tower (chains of the goal's support relation). Returns a list of sorted block-index tuples."""
    n = len(goal)
    above = {}
    for b, below in enumerate(goal):
        if below >= 0:
            above[below] = b
    seen, comps = set(), []
    for b in range(n):
        if b in seen or goal[b] >= 0:
            continue
        chain, cur = [b], b                       # b rests on the table: walk up the tower
        while cur in above:
            cur = above[cur]
            chain.append(cur)
        seen.update(chain)
        comps.append(tuple(sorted(chain)))
    return comps


def flatten_other_towers(goal, x):
    """Keep every relation of the tower that contains block ``x``; every block of the other non-trivial towers is made OnTable in the goal (atom count and blocks unchanged)."""
    keep = next(c for c in goal_components(goal) if x in c)
    return tuple(g if i in keep else S.TABLE for i, g in enumerate(goal))


def tau_from_d0(d0, cases_by_id):
    """Median step_index / step_cap over D0 expert steps where the hand holds a block and the next optimal action stacks it onto its goal support (fixed once, from training data only)."""
    taus = []
    for t in d0:
        problem = cases_by_id[t["case_id"]].problem
        idx = {n: i for i, n in enumerate(problem.names)}
        for j, (s, a) in enumerate(zip(t["states"][:-1], t["actions"])):
            sch, args = parse(a)
            if sch == "STACK" and S.held(tuple(s)) != -1 and problem.goal[idx[args[0]]] == idx[args[1]]:
                taus.append(j / t["step_cap"])
    return (statistics.median(taus) if taus else None), len(taus)


def with_time(snap, tau):
    row = list(snap.base_input)
    row[0], row[1] = tau, 1.0 - tau
    return dataclasses.replace(snap, base_input=tuple(row))


@torch.no_grad()
def replay_to_first_error(policy, case, decisions):
    """Replay the logged B2 trajectory up to its first non-optimal decision. Returns (snapshot, hidden_in, index, max_abs_prob_error, ok)."""
    ep = BwEpisode(case)
    hidden = policy.initial_hidden()
    worst = 0.0
    for j, d in enumerate(decisions):
        snap = ep.snapshot()
        out = policy(snap, hidden)
        probs = out.distribution.probs
        for aid, p in d["probs"].items():
            worst = max(worst, abs(float(probs[snap.candidate_ids.index(aid)]) - p))
        legal_ok = sorted(d["probs"]) == sorted(i for i, m in zip(snap.candidate_ids, snap.mask) if m)
        arg_ok = out.select(True)[0] == d["selected"]
        if d["selected"] not in d["optimal_actions"]:
            return snap, hidden, j, worst, bool(worst <= REPLAY_TOL and legal_ok and arg_ok)
        if not (legal_ok and arg_ok) or worst > REPLAY_TOL:
            return snap, hidden, j, worst, False
        hidden = out.hidden
        ep.step(d["selected"])
    return None, None, None, worst, False


def _cond_row(out, snap, a_star, a_bad, orig_opt, new_opt, gstack):
    ids = snap.candidate_ids
    logits = out.logits
    probs = out.distribution.probs
    legal = [i for i, m in enumerate(snap.mask) if m]
    best = max(float(logits[i]) for i in legal)
    arg = min((i for i in legal if float(logits[i]) == best), key=lambda i: ids[i])
    row = {"margin": float(logits[ids.index(a_star)] - logits[ids.index(a_bad)]), "argmax": ids[arg], "argmax_in_original_optimal_set": ids[arg] in orig_opt, "argmax_is_goal_stack": ids[arg] == gstack,
           "optimal_action_probability_mass": float(sum(probs[ids.index(a)] for a in orig_opt))}
    if new_opt is not None:
        row["optimal_action_probability_mass_new_goal"] = float(sum(probs[ids.index(a)] for a in new_opt))
    return row


def probe_sample(policy, case, decisions, tau, solver):
    """One sample: classification, replay check, 4-condition forward. Returns (record dict, list of condition rows)."""
    problem = case.problem
    idx = {n: i for i, n in enumerate(problem.names)}
    snap, hidden, j, err, ok = replay_to_first_error(policy, case, decisions)
    rec = {"case_id": case.case_id, "decision_index": j, "replay_max_abs_prob_error": err, "status": None}
    if snap is None:
        rec["status"] = "NO_FIRST_ERROR"
        return rec, []
    d = decisions[j]
    sch, args = parse(d["selected"])
    rec["selected"] = d["selected"]
    if sch != "PUT_DOWN":
        rec["status"] = "EXCLUDED_NOT_PUT_DOWN"
        return rec, []
    x = idx[args[0]]
    gstack = next((o for o in d["optimal_actions"] if parse(o)[0] == "STACK" and idx[parse(o)[1][0]] == x and problem.goal[x] == idx[parse(o)[1][1]]), None)
    if gstack is None:
        rec["status"] = "EXCLUDED_NO_GOAL_STACK_IN_OPTIMAL_SET"
        return rec, []
    comps = goal_components(problem.goal)
    others = [c for c in comps if x not in c and len(c) >= 2]
    if not others:
        rec["status"] = "EXCLUDED_NO_OTHER_NONTRIVIAL_TOWER"
        return rec, []
    if not ok:
        rec["status"] = "REPLAY_MISMATCH"
        return rec, []
    new_goal = flatten_other_towers(problem.goal, x)
    state = tuple(d["state"])
    p2 = S.Problem(problem.names, problem.colors, problem.init, new_goal)
    case2 = Case(case.case_id + "#G10", "probe", p2, case.optimal_length, case.step_cap)
    snap10 = snapshot_at(case2, state, j)
    assert snap10.candidate_ids == snap.candidate_ids and snap10.mask == snap.mask, "intervention changed the candidate set"
    L2, opt2 = solver.optimal_actions(state, new_goal)
    new_opt = [S.action_id(problem.names, a) for a in opt2] if L2 is not None else None
    a_bad = d["selected"]
    rec.update({"goal_stack_action": gstack, "block_x": problem.names[x], "n_other_nontrivial_towers": len(others), "planner_new_goal_optimal": new_opt,
                "local_order_preserved": bool(new_opt is not None and gstack in new_opt and a_bad not in new_opt)})
    rec["status"] = "VALID" if rec["local_order_preserved"] else "LOCAL_ORDER_CHANGED_BY_INTERVENTION"
    rows = []
    snaps = {"G00": snap, "G10": snap10, "G01": with_time(snap, tau) if tau is not None else None, "G11": with_time(snap10, tau) if tau is not None else None}
    for cond, sn in snaps.items():
        if sn is None:
            continue
        out = policy(sn, hidden)
        rows.append({"case_id": case.case_id, "condition": cond, **_cond_row(out, sn, gstack, a_bad, d["optimal_actions"], new_opt if cond in ("G10", "G11") else None, gstack)})
    return rec, rows


def median(xs):
    return statistics.median(xs) if xs else None


def summarize_probe(records, rows):
    """Apply the registered section-8 rules (thresholds fixed before any probe output existed)."""
    valid = {r["case_id"] for r in records if r["status"] == "VALID"}
    by = {(r["case_id"], r["condition"]): r for r in rows}
    out = {"n_candidates": len(records), "status_counts": {}, "n_valid": len(valid), "conditions": {}}
    for r in records:
        out["status_counts"][r["status"]] = out["status_counts"].get(r["status"], 0) + 1
    supports = {}
    for cond in ("G10", "G01", "G11"):
        ids = [c for c in valid if (c, cond) in by and (c, "G00") in by]
        deltas = [by[(c, cond)]["margin"] - by[(c, "G00")]["margin"] for c in ids]
        flips = [c for c in ids if (not by[(c, "G00")]["argmax_is_goal_stack"]) and by[(c, cond)]["argmax_is_goal_stack"]]
        improved = [dl for dl in deltas if dl > 0]
        n = len(ids)
        sup = bool(n >= MIN_VALID and (len(flips) / n >= FLIP_FRACTION or (len(improved) / n >= IMPROVE_FRACTION and median(deltas) >= MEDIAN_DELTA)))
        supports[cond] = sup
        out["conditions"][cond] = {"n": n, "flip_to_goal_stack": len(flips), "flip_fraction": len(flips) / n if n else None, "margin_improved": len(improved), "margin_improved_fraction": len(improved) / n if n else None,
                                   "median_margin_change": median(deltas), "mean_mass_original_set_change": (statistics.mean(by[(c, cond)]["optimal_action_probability_mass"] - by[(c, "G00")]["optimal_action_probability_mass"] for c in ids) if ids else None),
                                   "supported": sup}
    if len(valid) < MIN_VALID:
        state = "INSUFFICIENT_PROBE"
    elif supports["G10"]:
        state = "GOAL_PROGRESS_READY"
    elif supports["G01"]:
        state = "TIME_INPUT_FIRST"
    else:
        state = "PROBE_INCONCLUSIVE"
    out["state"] = state
    out["rules"] = {"min_valid": MIN_VALID, "flip_fraction_min": FLIP_FRACTION, "margin_improved_fraction_min": IMPROVE_FRACTION, "median_margin_change_min": MEDIAN_DELTA}
    return out
