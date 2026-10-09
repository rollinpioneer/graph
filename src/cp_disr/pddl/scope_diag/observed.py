"""Read-only OPEN observer for the search-scope diagnostic (card CP-DISR-C1-SEARCH-SCOPE-DIAGNOSTIC-V1, plan sections 10.3 / 13).

``gbfs_observed`` is the shared eager GBFS of ``search_match.engine.gbfs`` (single OPEN keyed (h, insertion serial), complete-state duplicate detection, CLOSED never reopened, in-place shorter-path update,
canonical action order, goal tested at generation, all new successors of one parent scored together before the next pop, no beam / helpful actions / pruning) plus bookkeeping that never feeds back into the search:
the enqueue score and serial of every OPEN state, every recorded generating parent of every state, and snapshots taken immediately BEFORE the k-th valid pop for the expansion counts k in ``snapshot_at``.

A snapshot records (all taken while the popped state and every other OPEN state are still in OPEN): the state about to be popped, the active reference anchor (the reference-path state that is still OPEN, not
CLOSED, with the shortest known plan suffix), at most ``n_comp`` further OPEN competitors chosen by a fixed state hash (never by score), and sizes. Fixture 1 of the card checks that the pop sequence, the plan and
all counters equal those of the unmodified engine.
"""
from __future__ import annotations

import hashlib
import heapq
import time

from ..search_match.budget import Budget, ResourceStop
from ..search_match.engine import NEG_INF, ModelNonFinite, Result


def snapshot_schedule(n_cap, n_snap=32):
    """I = unique{ceil(n_cap ** (j / (n_snap - 1))) : j = 0..n_snap-1} (plan 13.3)."""
    out = set()
    for j in range(n_snap):
        v = n_cap ** (j / (n_snap - 1))
        c = int(v)
        if c < v - 1e-12:
            c += 1
        out.add(max(1, min(n_cap, c)))
    return sorted(out)


def state_key(case_id, event, state, nbytes):
    return hashlib.sha256(("%s|%d|" % (case_id, event)).encode() + state.to_bytes(nbytes, "little")).digest()


def gbfs_observed(task, index, evaluator, budget: Budget, snapshot_at=(), ref=None, case_id="", n_comp=3, observer=None):
    """Returns ``(Result, snapshots, extra)``. ``ref`` maps a reference-path state to ``(k, U)`` (position along the plan, remaining plan length); ``snapshots`` is a list of dicts with int states."""
    r = Result()
    r.plan, r.expanded, r.generated, r.duplicates, r.path_updates, r.evaluated_calls = None, 0, 0, 0, 0, 0
    r.solved_at_expansion, r.snapshots, r.best_h, r.eval_seconds, r.h_root = None, {}, None, 0.0, None
    t_start = time.perf_counter()
    goal = task.goal_mask
    ok_goal = task.solvable_by_relaxation
    init = task.init_mask
    nb = max(1, (len(task.dyn_atoms) + 7) // 8)
    node = {init: (0, None, None)}                    # state -> (g, parent state, action index)
    closed = set()
    heap = []
    serial = 0
    meta = {}                                          # OPEN / CLOSED state -> (enqueue h, serial)
    parents = {init: []}                               # state -> distinct recorded generating parents (first = first generation)
    first_event = {init: 0}                            # state -> expansion count of its first generating parent
    ref = ref or {}
    snap_set = set(snapshot_at)
    snap_done = set()
    snaps = []
    extra = {"snapshot_seconds": 0.0, "order_hash": None}
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
            if v != v or v == NEG_INF:
                raise ModelNonFinite(str(v))
        return vals

    def rec(s, role):
        h, ser = meta[s]
        k_u = ref.get(s)
        return {"role": role, "state": s, "h": h, "serial": ser, "g": node[s][0], "parents": list(parents.get(s, ())), "first_event": first_event.get(s), "ref_k": k_u[0] if k_u else None, "ref_U": k_u[1] if k_u else None}

    def take_snapshot(event, popped):
        t0 = time.perf_counter()
        anchor = None
        deepest_gen = deepest_closed = -1
        for st, (k, u) in ref.items():
            if st in node:
                if k > deepest_gen:
                    deepest_gen = k
                if st in closed:
                    if k > deepest_closed:
                        deepest_closed = k
                elif anchor is None or k > ref[anchor][0]:
                    anchor = st
        comps = []
        if n_comp:
            cand = ((state_key(case_id, event, st, nb), st) for (_h, _ser, st) in heap if st != popped and st != anchor and st not in closed)
            comps = [st for _d, st in heapq.nsmallest(n_comp, cand)]
        snap = {"event": event, "open": len(heap), "closed": len(closed), "popped": rec(popped, "POPPED"), "anchor": rec(anchor, "ANCHOR") if anchor is not None else None,
                "competitors": [rec(c, "COMPETITOR") for c in comps], "ref_deepest_generated": deepest_gen, "ref_deepest_closed": deepest_closed, "expanded": r.expanded, "elapsed": budget.elapsed()}
        snaps.append(snap)
        extra["snapshot_seconds"] += time.perf_counter() - t0

    h = hashlib.sha1()
    try:
        if ok_goal and (init & goal) == goal:
            return finish("SOLVED", [], 0), snaps, extra
        h0 = score([init])[0]
        r.h_root = r.best_h = h0
        heapq.heappush(heap, (h0, serial, init))
        meta[init] = (h0, serial)
        serial += 1
        while heap:
            budget.check(every=64)
            if r.expanded >= max_exp:
                return finish("NODE_LIMIT"), snaps, extra
            ev = r.expanded + 1
            if ev in snap_set and ev not in snap_done and heap[0][2] not in closed:
                snap_done.add(ev)
                take_snapshot(ev, heap[0][2])
            h_s, _ser, s = heapq.heappop(heap)
            if s in closed:
                continue
            closed.add(s)
            r.expanded += 1
            h.update(s.to_bytes(nb, "little"))
            if observer is not None:
                observer(s, r.expanded)
            g = node[s][0] + 1
            pending = []
            for a in index.applicable(s):
                t = (s & ~a.del_mask) | a.add_mask
                r.generated += 1
                if ok_goal and (t & goal) == goal:
                    node[t] = (g, s, a.index)
                    return finish("SOLVED", plan_from(t), r.expanded), snaps, extra
                if t in closed:
                    r.duplicates += 1
                    continue
                recd = node.get(t)
                if recd is None:
                    node[t] = (g, s, a.index)
                    pending.append(t)
                    parents[t] = [s]
                    first_event[t] = r.expanded
                else:
                    ps = parents[t]
                    if s not in ps:
                        ps.append(s)
                    if g < recd[0]:
                        node[t] = (g, s, a.index)
                        r.path_updates += 1
                    else:
                        r.duplicates += 1
            if pending:
                vals = score(pending)
                for t, v in zip(pending, vals):
                    heapq.heappush(heap, (v, serial, t))
                    meta[t] = (v, serial)
                    serial += 1
                    if v < r.best_h:
                        r.best_h = v
        return finish("OPEN_EXHAUSTED"), snaps, extra
    except ResourceStop as e:
        return finish(e.status), snaps, extra
    except ModelNonFinite:
        return finish("MODEL_NONFINITE"), snaps, extra
    finally:
        extra["order_hash"] = h.hexdigest()[:16]
