"""CP-DISR-C1-BW-A04P (Runbook CP_DISR_C1_Audit_Based_Goal_Progress_Runbook_v1, phase 1): four zero-training controllers on the FROZEN A02 B2 checkpoint.

C0 original B2 argmax; C1 state-action attempt counts; C3 visited-successor exclusion; G1 direct small-goal progress rule. All four use the same legal set, environment, step cap (2L*+4) and B2 file;
the hidden state is advanced by each controller's own trajectory. The planner is used only for offline labels (it never chooses an action); C3 / G1 read only the public one-step nominal successor.
"""
from __future__ import annotations

from collections import defaultdict

import torch

from . import metrics as M
from . import state as S
from .environment import BwEpisode

CONTROLLERS = ("C0", "C1", "C3", "G1")


def argmax_tie(logits, ids, idxs):
    """Original B2 tie-break restricted to ``idxs``: maximal logit, then the smallest candidate id (same rule as ``PolicyOutput.select``)."""
    best = max(float(logits[i]) for i in idxs)
    return min((i for i in idxs if float(logits[i]) == best), key=lambda i: ids[i])


class Memory:
    """Per-episode memory (rebuilt for every case; never shared across cases)."""

    def __init__(self, init_state):
        self.visits = defaultdict(int)          # (state, action id) -> executed count (C1)
        self.visited = {init_state}             # states reached so far, including the initial state (C3)


def choose(controller, ep, snap, logits, mem):
    """Return (selected index or None, info). ``None`` means the controller has no admissible action (C3 only)."""
    ids = snap.candidate_ids
    legal = [i for i, m in enumerate(snap.mask) if m]
    raw = argmax_tie(logits, ids, legal)
    info = {"raw": ids[raw], "intervened": False, "trigger": None}
    if controller == "C0":
        return raw, info
    state = ep.state
    if controller == "C1":
        if mem.visits[(state, ids[raw])] == 0:
            return raw, info
        n_min = min(mem.visits[(state, ids[i])] for i in legal)
        cand = [i for i in legal if mem.visits[(state, ids[i])] == n_min]
        sel = argmax_tie(logits, ids, cand)
        info.update({"intervened": sel != raw, "trigger": "RAW_ALREADY_TRIED"})
        return sel, info
    if controller == "C3":
        cand = [i for i in legal if S.apply(state, ep._action_of[ids[i]]) not in mem.visited]
        if not cand:
            info["trigger"] = "NO_UNVISITED_SUCCESSOR"
            return None, info
        sel = argmax_tie(logits, ids, cand)
        info.update({"intervened": sel != raw, "trigger": "VISITED_SUCCESSOR" if sel != raw else None})
        return sel, info
    if controller == "G1":
        before = {k for k, v in S.goal_atom_truth(ep.problem, state).items() if v}
        cand = []
        for i in legal:
            after = {k for k, v in S.goal_atom_truth(ep.problem, S.apply(state, ep._action_of[ids[i]])).items() if v}
            if before < after:                  # strict superset: nothing already satisfied is destroyed and at least one atom is added
                cand.append(i)
        if not cand:
            return raw, info
        sel = argmax_tie(logits, ids, cand)
        info.update({"intervened": sel != raw, "trigger": "GOAL_PROGRESS_SET" if sel != raw else None})
        return sel, info
    raise ValueError(controller)


@torch.no_grad()
def run_episode(policy, case, solver, controller, record_decisions=True):
    ep = BwEpisode(case)
    problem = case.problem
    hidden = policy.initial_hidden()
    mem = Memory(ep.state)
    decisions, trace, reason = [], [], None
    while not ep.done:
        snap = ep.snapshot()
        out = policy(snap, hidden)
        hidden = out.hidden
        L_before, opt = solver.optimal_actions(ep.state, problem.goal)
        opt_ids = [S.action_id(problem.names, a) for a in opt]
        ids = snap.candidate_ids
        sel_idx, info = choose(controller, ep, snap, out.logits, mem)
        if sel_idx is None:
            reason = "NO_UNVISITED_SUCCESSOR"
            break
        selected = ids[sel_idx]
        legal_ids = [ids[i] for i, m in enumerate(snap.mask) if m]
        probs = out.distribution.probs
        before = S.goal_atom_truth(problem, ep.state)
        state = ep.state
        ep.step(selected)
        after = S.goal_atom_truth(problem, ep.state)
        trace.append((state, selected))
        mem.visits[(state, selected)] += 1
        mem.visited.add(ep.state)
        if record_decisions:
            decisions.append({"decision_index": len(decisions), "state": list(state), "raw": info["raw"], "selected": selected, "intervened": info["intervened"], "trigger": info["trigger"],
                              "optimal_actions": opt_ids, "selected_is_optimal": selected in opt_ids, "raw_is_optimal": info["raw"] in opt_ids, "optimal_remaining": L_before,
                              "probs": {i: round(float(probs[ids.index(i)]), 6) for i in legal_ids}, "n_legal": len(legal_ids),
                              "destroyed_satisfied_goal": M.goal_destroyed(before, after)})
    flags = [d["selected_is_optimal"] for d in decisions]
    success = bool(ep.success) and reason is None
    first_int = next((d["decision_index"] for d in decisions if d["intervened"]), None)
    res = {"case_id": case.case_id, "controller": controller, "n_blocks": problem.n, "success": success, "reason": reason or ep.reason, "steps": len(decisions), "optimal_length": case.optimal_length,
           "step_cap": case.step_cap, "decision_perfect": bool(success and all(flags)), "first_divergence": M.first_divergence(flags), "cycle": M.repeated_state_action_cycle(trace),
           "excess_steps": M.excess_steps(success, len(decisions), case.optimal_length), "interventions": sum(d["intervened"] for d in decisions), "first_intervention": first_int,
           "destroyed_satisfied_goal_count": sum(len(d["destroyed_satisfied_goal"]) for d in decisions)}
    if record_decisions:
        res["decisions"] = decisions
    return res


# ------------------------------------------------------------------------------------------------ first-error anatomy (shared by the A03-log table and the pilot)
def _parse(aid):
    p = aid.split(":")
    return p[1], p[2:-1]


def first_error_fields(decisions, problem):
    """Per-episode first-error record. ``decisions`` carry selected / optimal_actions / probs / state."""
    names = {n: i for i, n in enumerate(problem.names)}
    out = {"has_first_error": False, "first_error_action": None, "optimal_action_types": None, "first_error_is_put_down_instead_of_goal_stack": None, "best_optimal_rank": None,
           "first_repeated_action_is_optimal": None, "first_error_index": None}
    for j, d in enumerate(decisions):
        if d["selected"] not in d["optimal_actions"]:
            sel_schema, sel_args = _parse(d["selected"])
            goal_stack = False
            if sel_schema == "PUT_DOWN":
                x = names[sel_args[0]]
                for o in d["optimal_actions"]:
                    schema, args = _parse(o)
                    if schema == "STACK" and names[args[0]] == x and problem.goal[x] == names[args[1]]:
                        goal_stack = True
            order = sorted(d["probs"], key=lambda a: (-d["probs"][a], a))
            ranks = [order.index(o) + 1 for o in d["optimal_actions"] if o in order]
            out.update({"has_first_error": True, "first_error_action": d["selected"], "optimal_action_types": "+".join(sorted({_parse(o)[0] for o in d["optimal_actions"]})),
                        "first_error_is_put_down_instead_of_goal_stack": goal_stack, "best_optimal_rank": min(ranks) if ranks else None, "first_error_index": j})
            break
    seen = set()
    for d in decisions:
        key = (tuple(d["state"]), d["selected"])
        if key in seen:
            out["first_repeated_action_is_optimal"] = d["selected"] in d["optimal_actions"]
            break
        seen.add(key)
    return out
