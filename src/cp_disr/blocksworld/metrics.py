"""Candidate-to-goal hop distance, decision labels and episode metrics (runbook 8, 9, 10, 13). Analysis only: nothing here feeds a policy."""
from __future__ import annotations

from collections import deque

from . import state as S

INF = float("inf")
STRATA = ("H1", "H2", "H3", "H4")


def stratum(hop):
    if hop == INF:
        return "H4"
    if hop <= 2:
        return "H1"
    if hop <= 4:
        return "H2"
    return "H3"


class HopIndex:
    """Shortest directed message-path lengths on the production message graph (the template's edge set, which already contains the reverse edges)."""

    def __init__(self, template):
        self.template = template
        self.rev = {}
        for a, b, _rel in template.edges:
            self.rev.setdefault(b, []).append(a)
        self.action_ids = {n.id for n in template.nodes if n.kind == "ACTION"}

    def distances_to(self, goal_nodes):
        dist = {g: 0 for g in goal_nodes}
        queue = deque(goal_nodes)
        while queue:
            v = queue.popleft()
            for u in self.rev.get(v, ()):
                if u not in dist:
                    dist[u] = dist[v] + 1
                    queue.append(u)
        return dist

    def candidate_goal_hops(self, problem, state, action_ids):
        """{action id: hops from its ACTION node to the nearest unsatisfied goal proposition} (INF when unreachable)."""
        unmet = [a for a, ok in S.goal_atom_truth(problem, state).items() if not ok]
        dist = self.distances_to(unmet) if unmet else {}
        return {a: dist.get(a, INF) for a in action_ids}


def decision_hop(hops_by_action, optimal_ids):
    """Hop of a decision: smallest hop among the optimal actions (INF when none is reachable)."""
    return min((hops_by_action[a] for a in optimal_ids), default=INF)


def case_labels(problem, solver, hopindex, destruction):
    """Case-level labels along the lexicographically first optimal plan."""
    state = problem.init
    L = solver.cost_to_go(state, problem.goal)
    plan = solver.one_optimal_plan(state, problem.goal)
    decisions = []
    s = state
    for a in plan:
        L_s, acts = solver.optimal_actions(s, problem.goal)
        ids = [S.action_id(problem.names, x) for x in acts]
        hops = hopindex.candidate_goal_hops(problem, s, ids)
        decisions.append({"state": s, "optimal_ids": ids, "hop": min(hops.values()) if hops else INF, "n_optimal": len(acts)})
        s = S.apply(s, a)
    le4 = sum(1 for d in decisions if d["hop"] <= 4)
    requires_destruction, monotone, _L = destruction
    return {"optimal_length": L, "plan": [list(a) for a in plan], "n_decisions": len(decisions), "decisions_hop_le4": le4, "decisions_hop_gt4": len(decisions) - le4,
            "hop_le4_fraction": le4 / len(decisions) if decisions else 1.0, "initial_decision_hop": decisions[0]["hop"] if decisions else INF,
            "requires_goal_destruction": requires_destruction, "monotone_solution_exists": monotone,
            "coordination_slice": bool(requires_destruction) and (decisions[0]["hop"] <= 4 if decisions else False) and (le4 / len(decisions) >= 0.8 if decisions else False)}


# ----------------------------------------------------------------------------- episode / decision metrics
def first_divergence(selected_flags):
    """Index of the first decision whose selected action is not in the optimal set (None when every decision is optimal)."""
    for i, ok in enumerate(selected_flags):
        if not ok:
            return i
    return None


def goal_destroyed(before, after):
    """Goal atoms TRUE before and FALSE after."""
    return sorted(k for k in before if before[k] and not after[k])


def repeated_state_action_cycle(trace):
    """True when some (state, action) pair is executed twice in one episode."""
    seen = set()
    for state, action in trace:
        if (state, action) in seen:
            return True
        seen.add((state, action))
    return False


def excess_steps(success, executed_steps, optimal_length):
    return executed_steps - optimal_length if success else None
