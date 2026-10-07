"""Zero-training two-step contract look-ahead (runbook 6): roots from the real-history C3 set, every legal second action expanded, non-terminal leaves scored by the learned state value
(MG / W) or by the number of unmet final goal atoms. The same tree is used by both scorers; hypothetical states never enter the real visited set. Planner labels are not used."""
from __future__ import annotations

import time

import torch

from .. import a04p_controls as AC
from .. import state as S


class Counters:
    def __init__(self):
        self.reset()

    def reset(self):
        self.decisions = self.roots = self.leaves = self.unique_leaves = self.model_calls = self.terminal_decisions = self.fallback_decisions = 0
        self.seconds = 0.0

    def as_dict(self):
        return dict(self.__dict__)


def count_unmet(problem, state):
    return sum(1 for x, b in enumerate(problem.goal) if state[x] != b)


def choose_root(model, leaf, template, problem, state, root_ids, action_of, visited, counters=None, steps=None):
    """Returns (root action id or None, info). ``root_ids``: candidate ids of the C3-available roots (already filtered by the REAL visited set)."""
    t0 = time.perf_counter()
    terminals, leaves = [], {}
    for aid in root_ids:
        sa = S.apply(state, action_of[aid])
        if S.goal_satisfied(problem, sa):
            terminals.append((1, aid))
            continue
        ls = []
        for b in S.legal_actions(sa):
            sab = S.apply(sa, b)
            if sab in visited or sab == state or sab == sa:
                continue
            if S.goal_satisfied(problem, sab):
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
        if leaf == "MG":
            vals = model.state_values(template, problem, uniq, steps)
            score = {s: float(v) for s, v in zip(uniq, vals)}
        else:
            score = {s: float(count_unmet(problem, s)) for s in uniq}
        best = {aid: min(score[s] for s in v) for aid, v in leaves.items() if v}
        sel = min(best, key=lambda a: (best[a], a))
        info["unique_leaves"] = len(uniq)
        info["root_best_scores"] = len(best)
    if counters is not None:
        counters.decisions += 1
        counters.roots += len(root_ids)
        counters.leaves += info["n_leaves"]
        counters.unique_leaves += info.get("unique_leaves", 0)
        counters.model_calls += (1 if (leaf == "MG" and info.get("unique_leaves")) else 0)
        counters.terminal_decisions += bool(terminals)
        counters.seconds += time.perf_counter() - t0
    return sel, info


def make_chooser(model, leaf, counters=None, steps=None):
    """Chooser for ``gp_attribution.run_episode_with``: the C3 root set comes from the REAL visited set; empty tree falls back to the one-step C3 argmax of the same scorer."""
    def chooser(_c, ep, snap, logits, mem):
        ids = snap.candidate_ids
        legal = [i for i, m in enumerate(snap.mask) if m]
        raw = AC.argmax_tie(logits, ids, legal)
        info = {"raw": ids[raw], "intervened": False, "trigger": None}
        roots = [i for i in legal if S.apply(ep.state, ep._action_of[ids[i]]) not in mem.visited]
        if not roots:
            info["trigger"] = "NO_UNVISITED_SUCCESSOR"
            return None, info
        aid, linfo = choose_root(model, leaf, snap.template, ep.problem, ep.state, [ids[i] for i in roots], ep._action_of, mem.visited, counters, steps)
        if aid is None:
            sel = AC.argmax_tie(logits, ids, roots)
            if counters is not None:
                counters.fallback_decisions += 1
            info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD_EMPTY_TREE_C3" if sel != raw else None})
            return sel, info
        sel = ids.index(aid)
        info.update({"intervened": sel != raw, "trigger": "LOOKAHEAD" if sel != raw else None})
        return sel, info
    return chooser
