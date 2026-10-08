"""The single search engine shared by all scorers of card C1-DEPOTS-SEARCH-MATCH-V1: eager greedy best-first search (common GBFS adaptation protocol, plan section 6.2).

One OPEN list keyed (h, insertion serial) -> FIFO among equal h; complete-state duplicate detection; CLOSED never reopened; a state still in OPEN whose g can be lowered gets its parent link and g updated in
place (children only exist after expansion, so every parent record that a child points to is final); successors in canonical action-id order; goal tested on the root and on every generated successor (the first
goal successor in action order ends the search); all new successors of one parent are scored together and only then inserted, before the next pop; no beam, no helpful actions, no C3 / visited-route filter, no
step cap derived from a reference length, no dead-end pruning (infinite h only sorts last). Node milestones (1,000 / 10,000 / 100,000 expansions) are snapshots of ONE run: the search is never restarted.
"""
from __future__ import annotations

import heapq
import math
import time
from collections import Counter

from .budget import MILESTONES, Budget, ResourceStop


NEG_INF = float("-inf")


class ModelNonFinite(Exception):
    pass


class SuccessorIndex:
    """Applicable ground actions of a bit-mask state, in canonical (= action index = sorted action id) order, without scanning all actions: every action is filed under its rarest positive precondition."""

    def __init__(self, task):
        self.task = task
        cnt = Counter()
        for a in task.actions:
            m = a.pre_mask
            while m:
                low = m & -m
                cnt[low.bit_length() - 1] += 1
                m ^= low
        self.by_key, self.free = {}, []
        for a in task.actions:
            m, best = a.pre_mask, None
            while m:
                low = m & -m
                i = low.bit_length() - 1
                if best is None or cnt[i] < cnt[best]:
                    best = i
                m ^= low
            if best is None:
                self.free.append(a)
            else:
                self.by_key.setdefault(best, []).append(a)
        self.key_mask = sum(1 << i for i in self.by_key)

    def applicable(self, state):
        out = [a for a in self.free if not (state & a.neg_mask)]
        m = state & self.key_mask
        by_key = self.by_key
        while m:
            low = m & -m
            for a in by_key[low.bit_length() - 1]:
                pm = a.pre_mask
                if (state & pm) == pm and not (state & a.neg_mask):
                    out.append(a)
            m ^= low
        out.sort(key=_index)
        return out


def _index(a):
    return a.index


class Result:
    __slots__ = ("status", "plan", "expanded", "generated", "duplicates", "path_updates", "evaluated_calls", "solved_at_expansion", "snapshots", "open_size", "closed_size", "best_h", "search_seconds",
                 "eval_seconds", "h_root")

    def as_dict(self):
        return {k: getattr(self, k, None) for k in self.__slots__}


def gbfs(task, index, evaluator, budget: Budget, milestones=MILESTONES):
    """Returns a ``Result``. status: SOLVED | OPEN_EXHAUSTED | NODE_LIMIT | TIMEOUT | MEMORY_LIMIT | MODEL_NONFINITE. ``plan`` is a list of action indices (unverified)."""
    r = Result()
    r.plan, r.expanded, r.generated, r.duplicates, r.path_updates, r.evaluated_calls = None, 0, 0, 0, 0, 0
    r.solved_at_expansion, r.snapshots, r.best_h, r.eval_seconds, r.h_root = None, {}, None, 0.0, None
    t_start = time.perf_counter()
    goal = task.goal_mask
    ok_goal = task.solvable_by_relaxation
    init = task.init_mask
    node = {init: (0, None, None)}                    # state -> (g, parent state, action index)
    closed = set()
    heap = []
    serial = 0
    milestones = sorted(milestones)
    max_exp = budget.max_expansions

    def finish(status, plan=None, sol_at=None):
        r.status, r.plan, r.solved_at_expansion = status, plan, sol_at
        r.open_size, r.closed_size = len(heap), len(closed)
        r.search_seconds = time.perf_counter() - t_start
        return r

    def plan_from(state):
        acts = []
        while True:
            g, par, a = node[state]
            if par is None:
                break
            acts.append(a)
            state = par
        acts.reverse()
        return acts

    def score(states):
        t = time.perf_counter()
        vals = evaluator.evaluate(states, budget)
        r.eval_seconds += time.perf_counter() - t
        r.evaluated_calls += 1
        for v in vals:
            if v != v or v == NEG_INF:                                   # +inf is a legal sort-last value (h_add of an unreachable goal); evaluators of learned scores reject it themselves
                raise ModelNonFinite(str(v))
        return vals

    try:
        if ok_goal and (init & goal) == goal:
            return finish("SOLVED", [], 0)
        h0 = score([init])[0]
        r.h_root = r.best_h = h0
        heapq.heappush(heap, (h0, serial, init))
        serial += 1
        while heap:
            budget.check(every=64)
            if r.expanded >= max_exp:
                return finish("NODE_LIMIT")
            h_s, _ser, s = heapq.heappop(heap)
            if s in closed:
                continue
            closed.add(s)
            r.expanded += 1
            g = node[s][0] + 1
            pending = []
            for a in index.applicable(s):
                t = (s & ~a.del_mask) | a.add_mask
                r.generated += 1
                if ok_goal and (t & goal) == goal:
                    node[t] = (g, s, a.index)
                    return finish("SOLVED", plan_from(t), r.expanded)
                if t in closed:
                    r.duplicates += 1
                    continue
                rec = node.get(t)
                if rec is None:
                    node[t] = (g, s, a.index)
                    pending.append(t)
                elif g < rec[0]:
                    node[t] = (g, s, a.index)
                    r.path_updates += 1
                else:
                    r.duplicates += 1
            if pending:
                vals = score(pending)
                for t, v in zip(pending, vals):
                    heapq.heappush(heap, (v, serial, t))
                    serial += 1
                    if v < r.best_h:
                        r.best_h = v
            if milestones and r.expanded == milestones[0]:
                m = milestones.pop(0)
                r.snapshots[m] = {"expanded": r.expanded, "generated": r.generated, "open": len(heap), "elapsed": budget.elapsed(), "best_h": r.best_h}
        return finish("OPEN_EXHAUSTED")
    except ResourceStop as e:
        return finish(e.status)
    except ModelNonFinite:
        return finish("MODEL_NONFINITE")
