"""Exact Blocksworld planner (runbook 8): A* with an admissible heuristic and fixed lexicographic tie-breaking, plus an independent BFS for cross-validation.

Inputs are only the current symbolic state, the goal configuration and the frozen action contracts (``state.legal_actions`` / ``state.apply``). Nothing here reads a
model, a logit, a Q value or a hidden state.
"""
from __future__ import annotations

import heapq
import time
from collections import deque
from dataclasses import dataclass, field

from . import state as S

MAX_NODES = 2_000_000
CPU_LIMIT = 30.0
MAX_DEPTH = 32


class PlannerLimit(RuntimeError):
    pass


def heuristic(state, goal):
    """Admissible: every block that is not in its final position (right support, recursively right below) must be moved, which costs a pick-type and a place-type action;
    a block already in the hand only needs its place-type action."""
    n = len(state)
    good = [None] * n

    def is_good(x):
        if good[x] is None:
            good[x] = state[x] == goal[x] and (goal[x] == S.TABLE or is_good(goal[x]))
        return good[x]
    h = 0
    for x in range(n):
        if not is_good(x):
            h += 1 if state[x] == S.HELD else 2
    return h


@dataclass
class Stats:
    expanded: int = 0
    generated: int = 0
    cpu: float = 0.0


@dataclass
class Solver:
    """A* with per-solver caches. ``cost_to_go`` is exact; results for (state, goal) are memoised."""
    max_nodes: int = MAX_NODES
    cpu_limit: float = CPU_LIMIT
    max_depth: int = MAX_DEPTH
    cache: dict = field(default_factory=dict)
    stats: Stats = field(default_factory=Stats)

    def cost_to_go(self, state, goal):
        key = (state, goal)
        if key in self.cache:
            return self.cache[key]
        t0 = time.process_time()
        if all(state[i] == goal[i] for i in range(len(goal))):
            self.cache[key] = 0
            return 0
        start_g = {state: 0}
        counter = 0
        heap = [(heuristic(state, goal), 0, counter, state)]
        expanded = 0
        while heap:
            f, g, _c, s = heapq.heappop(heap)
            if start_g.get(s, 1 << 30) < g:
                continue
            if s == goal:
                self.cache[key] = g
                self.stats.cpu += time.process_time() - t0
                return g
            expanded += 1
            self.stats.expanded += 1
            if expanded > self.max_nodes or time.process_time() - t0 > self.cpu_limit:
                raise PlannerLimit("search limit reached")
            if g >= self.max_depth:
                continue
            for a, t in S.successors(s):
                ng = g + 1
                if ng < start_g.get(t, 1 << 30):
                    start_g[t] = ng
                    counter += 1
                    self.stats.generated += 1
                    heapq.heappush(heap, (ng + heuristic(t, goal), ng, counter, t))
        self.cache[key] = None
        return None

    def optimal_actions(self, state, goal):
        """(L*, sorted set of first actions that start some optimal plan). L* is None when unsolvable within limits."""
        L = self.cost_to_go(state, goal)
        if L is None:
            return None, []
        if L == 0:
            return 0, []
        acts = []
        for a, t in S.successors(state):
            c = self.cost_to_go(t, goal)
            if c is not None and c == L - 1:
                acts.append(a)
        return L, sorted(acts)

    def one_optimal_plan(self, state, goal):
        plan, s = [], state
        L = self.cost_to_go(s, goal)
        while L:
            _L, acts = self.optimal_actions(s, goal)
            a = acts[0]
            plan.append(a)
            s = S.apply(s, a)
            L -= 1
        return plan

    def optimal_dag(self, state, goal, limit=200000):
        """States reachable by optimal plans, grouped by depth: {depth: {state: [(action, next_state)]}} (None when it exceeds ``limit``)."""
        L = self.cost_to_go(state, goal)
        layers = [{state: []}]
        total = 1
        for d in range(L):
            nxt = {}
            for s in layers[d]:
                for a, t in S.successors(s):
                    if self.cost_to_go(t, goal) == L - d - 1:
                        layers[d][s].append((a, t))
                        nxt.setdefault(t, [])
            total += len(nxt)
            if total > limit:
                return None
            layers.append(nxt)
        return layers


def bfs_cost(state, goal, limit=40):
    """Independent breadth-first shortest length (used to cross-validate the A* result on small instances)."""
    if state == goal:
        return 0
    seen = {state}
    frontier = deque([(state, 0)])
    while frontier:
        s, d = frontier.popleft()
        if d >= limit:
            continue
        for a, t in S.successors(s):
            if t == goal:
                return d + 1
            if t not in seen:
                seen.add(t)
                frontier.append((t, d + 1))
    return None


def destruction_labels(problem, solver):
    """(requires_goal_destruction, monotone_solution_exists, optimal_length) over ALL optimal plans from the initial state.

    A step destroys a goal when a goal atom TRUE before it is FALSE after it. ``monotone_solution_exists`` = some optimal plan never destroys one.
    """
    goal = problem.goal
    layers = solver.optimal_dag(problem.init, goal)
    if layers is None:
        return None, None, solver.cost_to_go(problem.init, goal)
    L = len(layers) - 1
    ok_states = {problem.init}              # states reachable by monotone optimal prefixes
    for d in range(L):
        nxt = set()
        for s in layers[d]:
            if s not in ok_states:
                continue
            before = S.goal_atom_truth(problem, s)
            for a, t in layers[d][s]:
                after = S.goal_atom_truth(problem, t)
                if all(not before[k] or after[k] for k in before):
                    nxt.add(t)
        ok_states = nxt
    monotone = goal in ok_states if L > 0 else True
    return (not monotone), monotone, L
