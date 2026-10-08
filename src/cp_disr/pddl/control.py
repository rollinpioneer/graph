"""Execution controllers shared by Blocksworld and grounded-PDDL tasks: one-step + C3, fixed two-step contract look-ahead, selective (gated) look-ahead, and a generic episode runner.

``ops`` adapts a task: ``ops.legal_ids(state)``, ``ops.apply_id(state, aid)``, ``ops.is_goal(state)``, ``ops.unmet(state)`` (number of unmet final goal atoms), ``ops.leaf_values(template, states)``
(model value of arbitrary states, lower = closer to the goal). Hypothetical states never enter the real visited set; no label, planner value or optimal action is ever consulted by a controller.
"""
from __future__ import annotations

import time
from types import SimpleNamespace

import torch

from ..blocksworld import a04p_controls as AC


class Counters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.decisions = self.roots = self.leaves = self.unique_leaves = self.model_calls = self.expansions = self.gate_checks = self.gate_expansions = self.trap_unfolds = self.terminal_decisions = self.fallback_decisions = 0
        self.seconds = 0.0

    def as_dict(self):
        return dict(self.__dict__)


class Memory:
    def __init__(self, init_state):
        self.visited = {init_state}


def _legal_idx(snap):
    return [i for i, m in enumerate(snap.mask) if m]


def available(ops, state, snap, visited):
    ids = snap.candidate_ids
    return [i for i in _legal_idx(snap) if ops.apply_id(state, ids[i]) not in visited]


def choose_one_step(ops, ep, snap, logits, mem):
    ids = snap.candidate_ids
    legal = _legal_idx(snap)
    raw = AC.argmax_tie(logits, ids, legal)
    info = {"raw": ids[raw], "c3": None, "intervened": False, "trigger": None, "available": 0}
    avail = available(ops, ep.state, snap, mem.visited)
    info["available"] = len(avail)
    if not avail:
        info["trigger"] = "NO_UNVISITED_SUCCESSOR"
        return None, info
    sel = AC.argmax_tie(logits, ids, avail)
    info.update({"c3": ids[sel], "intervened": sel != raw, "trigger": "VISITED_SUCCESSOR" if sel != raw else None})
    return sel, info


def two_step_root(ops, model, template, task_for_leaves, state, root_ids, visited, counters=None, leaf="model", steps=None):
    """Fixed two-step contract look-ahead over the C3 root set. Returns (root action id | None, info)."""
    t0 = time.perf_counter()
    terminals, leaves = [], {}
    for aid in root_ids:
        sa = ops.apply_id(state, aid)
        if ops.is_goal(sa):
            terminals.append((1, aid))
            continue
        ls = []
        for b in ops.legal_ids(sa):
            sab = ops.apply_id(sa, b)
            if sab in visited or sab == state or sab == sa:
                continue
            if ops.is_goal(sab):
                terminals.append((2, aid))
            else:
                ls.append(sab)
        leaves[aid] = ls
    info = {"terminal": bool(terminals), "n_roots": len(root_ids), "n_leaves": sum(len(v) for v in leaves.values())}
    sel = None
    if terminals:
        sel = min(terminals)[1]
    elif any(leaves.values()):
        uniq = sorted({s for v in leaves.values() for s in v})
        if leaf == "model":
            vals = ops.leaf_values(template, uniq, steps)
            score = {s: float(v) for s, v in zip(uniq, vals)}
        else:
            score = {s: float(leaf(s)) for s in uniq}
        best = {aid: min(score[s] for s in v) for aid, v in leaves.items() if v}
        sel = min(best, key=lambda a: (best[a], a))
        info["unique_leaves"] = len(uniq)
    if counters is not None:
        counters.decisions += 1
        counters.expansions += 1
        counters.roots += len(root_ids)
        counters.leaves += info["n_leaves"]
        counters.unique_leaves += info.get("unique_leaves", 0)
        counters.model_calls += 1 if (leaf == "model" and info.get("unique_leaves")) else 0
        counters.terminal_decisions += bool(terminals)
        counters.seconds += time.perf_counter() - t0
    return sel, info


def choose_two_step(ops, model, ep, snap, logits, mem, counters=None, leaf="model", steps=None):
    ids = snap.candidate_ids
    legal = _legal_idx(snap)
    raw = AC.argmax_tie(logits, ids, legal)
    info = {"raw": ids[raw], "c3": None, "intervened": False, "trigger": None, "available": 0, "gate": "always"}
    avail = available(ops, ep.state, snap, mem.visited)
    info["available"] = len(avail)
    if not avail:
        info["trigger"] = "NO_UNVISITED_SUCCESSOR"
        return None, info
    c3 = AC.argmax_tie(logits, ids, avail)
    info["c3"] = ids[c3]
    aid, linfo = two_step_root(ops, model, snap.template, None, ep.state, [ids[i] for i in avail], mem.visited, counters, leaf, steps)
    if aid is None:
        if counters is not None:
            counters.fallback_decisions += 1
        info.update({"intervened": c3 != raw, "trigger": "LOOKAHEAD_EMPTY_TREE_C3" if c3 != raw else None})
        return c3, info
    sel = ids.index(aid)
    info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD" if sel != raw else None, "lookahead_changed_c3": sel != c3})
    return sel, info


def choose_gated(ops, model, ep, snap, logits, mem, tau, counters=None, steps=None):
    """Selective look-ahead (plan 5.2): expand only if |A_C| >= 2 and (top-2 margin <= tau or the one-step choice leads into a locally visible dead end)."""
    ids = snap.candidate_ids
    legal = _legal_idx(snap)
    raw = AC.argmax_tie(logits, ids, legal)
    info = {"raw": ids[raw], "c3": None, "intervened": False, "trigger": None, "available": 0, "gate": None, "margin": None}
    avail = available(ops, ep.state, snap, mem.visited)
    info["available"] = len(avail)
    if not avail:
        info["trigger"] = "NO_UNVISITED_SUCCESSOR"
        return None, info
    first = AC.argmax_tie(logits, ids, avail)
    info["c3"] = ids[first]
    if counters is not None:
        counters.gate_checks += 1
    s1 = ops.apply_id(ep.state, ids[first])
    if len(avail) == 1 or ops.is_goal(s1):
        info["gate"] = "single_root" if len(avail) == 1 else "goal_next"
        info.update({"intervened": first != raw, "trigger": "VISITED_SUCCESSOR" if first != raw else None})
        return first, info
    ordered = sorted(avail, key=lambda i: (-float(logits[i]), ids[i]))
    margin = float(logits[ordered[0]] - logits[ordered[1]])
    info["margin"] = margin
    ancestors = mem.visited | {ep.state, s1}
    nxt = ops.legal_ids(s1)
    if counters is not None:
        counters.trap_unfolds += len(nxt)
    trap = all(ops.apply_id(s1, b) in ancestors for b in nxt)
    small = tau is not None and margin <= tau
    if not (small or trap):
        info["gate"] = "keep_one_step"
        info.update({"intervened": first != raw, "trigger": "VISITED_SUCCESSOR" if first != raw else None})
        return first, info
    info["gate"] = "low_margin" if small and not trap else ("local_trap" if trap and not small else "both")
    if counters is not None:
        counters.gate_expansions += 1
    aid, linfo = two_step_root(ops, model, snap.template, None, ep.state, [ids[i] for i in avail], mem.visited, counters, "model", steps)
    if aid is None:
        if counters is not None:
            counters.fallback_decisions += 1
        info.update({"intervened": first != raw, "trigger": "LOOKAHEAD_EMPTY_TREE_C3" if first != raw else None})
        return first, info
    sel = ids.index(aid)
    info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD" if sel != raw else None, "lookahead_changed_c3": sel != first})
    return sel, info


@torch.no_grad()
def run_episode(policy, ep, ops, chooser, labeler=None, record_states=False, deadline_seconds=None):
    """Generic episode: chooser(ops, ep, snap, logits, mem) -> (index | None, info). ``labeler(state) -> (d*, optimal ids)`` is called AFTER the choice, for scoring only."""
    mem = Memory(ep.state)
    decisions, reason = [], None
    t0 = time.perf_counter()
    lab_t = 0.0                                                                     # time spent in the scoring-only labeler: excluded from the policy wall time and from the deadline
    hidden = policy.initial_hidden()
    while not ep.done:
        if deadline_seconds is not None and time.perf_counter() - t0 - lab_t > deadline_seconds:
            reason = "TIMEOUT"                                                      # wall-clock budget of the problem, checked between decisions
            break
        snap = ep.snapshot()
        out = policy(snap, hidden)
        sel, info = chooser(ops, ep, snap, out.logits, mem)
        if sel is None:
            reason = "NO_UNVISITED_SUCCESSOR"
            break
        ids = snap.candidate_ids
        aid = ids[sel]
        rec = {"decision_index": len(decisions), "raw": info["raw"], "c3": info.get("c3"), "selected": aid, "intervened": info["intervened"], "trigger": info["trigger"], "available": info.get("available"),
               "gate": info.get("gate"), "margin": info.get("margin"), "n_legal": len(_legal_idx(snap)), "raw_survives_c3": (info.get("c3") == info["raw"]) if info.get("c3") else None}
        if record_states:
            rec["state"] = ep.state
        if labeler is not None:
            tl = time.perf_counter()
            d, opt = labeler(ep.state)
            lab_t += time.perf_counter() - tl
            rec.update({"remaining": d, "optimal_actions": opt, "selected_is_optimal": aid in opt, "raw_is_optimal": info["raw"] in opt, "c3_is_optimal": (info.get("c3") in opt) if info.get("c3") else None})
        decisions.append(rec)
        ep.step(aid)
        mem.visited.add(ep.state)
    success = bool(ep.success) and reason is None
    cap = ep.case.step_cap
    return {"case_id": ep.case.case_id, "success": success, "reason": reason or ep.reason, "steps": len(decisions), "optimal_length": ep.case.optimal_length, "step_cap": cap,
            "interventions": sum(d["intervened"] for d in decisions), "decisions": decisions, "wall_seconds": time.perf_counter() - t0 - lab_t, "label_seconds": lab_t,
            "decision_perfect": bool(success and decisions and all(d.get("selected_is_optimal", False) for d in decisions)) if labeler is not None else None}


# ------------------------------------------------------------------------------------------------ ops adapters
class PddlOps:
    def __init__(self, task, model=None, static_true=None):
        self.task, self.model = task, model
        self._legal = {}

    def legal_ids(self, state):
        hit = self._legal.get(state)
        if hit is None:
            hit = [a.aid for a in self.task.legal(state)]
            if len(self._legal) > 200000:
                self._legal.clear()
            self._legal[state] = hit
        return hit

    def apply_id(self, state, aid):
        return self.task.apply(state, self.task.action_by_id[aid])

    def is_goal(self, state):
        return self.task.goal_satisfied(state)

    def unmet(self, state):
        return self.task.unmet_goals(state)

    def leaf_values(self, template, states, steps=None):
        return self.model.state_values(template, self.task, states, steps)
