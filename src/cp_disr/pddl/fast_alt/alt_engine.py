"""ALT: eager greedy best-first search with TWO open lists ordered by two different scorers and used alternately (plan section 8).

Shared with the single-queue engine of the previous card: complete-state duplicate detection, one CLOSED set, in-place shorter-path update of OPEN states, canonical action order, goal tested at generation,
one discovery serial for FIFO ties, the same budget guard and node milestones. New: every new state is scored by BOTH scorers; heap 0 is keyed (h_main, serial), heap 1 is keyed (h_add, serial); the two scores are
never added, averaged or compared with each other. After every valid expansion the turn passes to the other heap; stale entries (states already expanded through the other heap) are discarded without using a turn.
"""
from __future__ import annotations

import heapq
import math
import time

from ..search_match.budget import MILESTONES, Budget, ResourceStop
from ..search_match.engine import NEG_INF, ModelNonFinite, Result


class AltResult(Result):
    """``Result`` plus a dict of queue statistics (no ``__slots__`` so the extra attribute is allowed)."""


def gbfs_alt(task, index, main, add, budget: Budget, milestones=MILESTONES, observer=None):
    r = AltResult()
    r.plan, r.expanded, r.generated, r.duplicates, r.path_updates, r.evaluated_calls = None, 0, 0, 0, 0, 0
    r.solved_at_expansion, r.snapshots, r.best_h, r.eval_seconds, r.h_root = None, {}, None, 0.0, None
    r.extra = {"valid_pops": [0, 0], "stale_pops": [0, 0], "fallback_pops": 0, "peak_open": [0, 0], "eval_seconds_main": 0.0, "eval_seconds_add": 0.0, "pushed": 0}
    ex = r.extra
    t_start = time.perf_counter()
    goal = task.goal_mask
    ok_goal = task.solvable_by_relaxation
    init = task.init_mask
    node = {init: (0, None, None)}
    serial_of = {init: 0}
    closed = set()
    heaps = ([], [])
    milestones = sorted(milestones)
    max_exp = budget.max_expansions

    def finish(status, plan=None, sol_at=None):
        r.status, r.plan, r.solved_at_expansion = status, plan, sol_at
        r.open_size, r.closed_size = len(heaps[0]) + len(heaps[1]), len(closed)
        r.search_seconds = time.perf_counter() - t_start
        r.eval_seconds = ex["eval_seconds_main"] + ex["eval_seconds_add"]
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
        hm = main.evaluate(states, budget)
        t1 = time.perf_counter()
        ha = add.evaluate(states, budget)
        t2 = time.perf_counter()
        ex["eval_seconds_main"] += t1 - t
        ex["eval_seconds_add"] += t2 - t1
        r.evaluated_calls += 1
        for v in list(hm) + list(ha):
            if v != v or v == NEG_INF:
                raise ModelNonFinite(str(v))
        for v in hm:
            if v == math.inf:
                raise ModelNonFinite("main score is +inf")        # +inf is only a legal sort-last value of h_add
        return hm, ha

    def clean(i):
        h = heaps[i]
        while h and h[0][2] in closed:
            heapq.heappop(h)
            ex["stale_pops"][i] += 1

    try:
        if ok_goal and (init & goal) == goal:
            return finish("SOLVED", [], 0)
        hm, ha = score([init])
        r.h_root = r.best_h = hm[0]
        heapq.heappush(heaps[0], (hm[0], 0, init))
        heapq.heappush(heaps[1], (ha[0], 0, init))
        next_serial = 1
        turn = 0
        while True:
            budget.check(every=64)
            i = turn
            clean(i)
            if not heaps[i]:
                i = 1 - i
                clean(i)
                if not heaps[i]:
                    return finish("OPEN_EXHAUSTED")
                ex["fallback_pops"] += 1
            if r.expanded >= max_exp:
                return finish("NODE_LIMIT")
            h_s, _ser, s = heapq.heappop(heaps[i])
            ex["valid_pops"][i] += 1
            closed.add(s)
            r.expanded += 1
            turn = 1 - i
            if observer is not None:
                observer(s, r.expanded)
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
                    serial_of[t] = next_serial
                    next_serial += 1
                    pending.append(t)
                elif g < rec[0]:
                    node[t] = (g, s, a.index)
                    r.path_updates += 1
                else:
                    r.duplicates += 1
            if pending:
                hm, ha = score(pending)
                for t, vm, va in zip(pending, hm, ha):
                    heapq.heappush(heaps[0], (vm, serial_of[t], t))
                    heapq.heappush(heaps[1], (va, serial_of[t], t))
                    ex["pushed"] += 1
                    if vm < r.best_h:
                        r.best_h = vm
                ex["peak_open"][0] = max(ex["peak_open"][0], len(heaps[0]))
                ex["peak_open"][1] = max(ex["peak_open"][1], len(heaps[1]))
            if milestones and r.expanded == milestones[0]:
                m = milestones.pop(0)
                r.snapshots[m] = {"expanded": r.expanded, "generated": r.generated, "open": len(heaps[0]), "elapsed": budget.elapsed(), "best_h": r.best_h}
    except ResourceStop as e:
        return finish(e.status)
    except ModelNonFinite:
        return finish("MODEL_NONFINITE")
