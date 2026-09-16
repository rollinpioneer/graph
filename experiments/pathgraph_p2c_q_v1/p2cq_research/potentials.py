"""Candidate potentials over the same public RewardState. No oracle fields."""
from __future__ import annotations

from .graph_model import phi_cost
from .task_contract import N, TaskContract, dnf_satisfied

REQUIRED = (
    "GEOM_COUNT_EVENTS_PBRS_V1",
    "FLAT_CONTRACT_PBRS_V1",
    "PATHGRAPH_MODEL_COST_PBRS_V1",
    "PATHGRAPH_MODEL_FULL_PBRS_V1",
)
TASK_ONLY = "TASK_ONLY_ZERO_V1"


def _real(contract):
    return contract.present_nodes()


def _g_geom(contract, dyn):
    nodes = dyn["nodes"]
    n = contract.n_real()
    acc = 0.0
    for i in _real(contract):
        k = contract.transport_steps[i]
        p = (nodes[i]["progress"] / k) if k else 0.0
        if nodes[i]["held"] and not nodes[i]["valid"]:
            acc += p
    return acc / n


def geom_phi(contract: TaskContract, dyn):
    nodes = dyn["nodes"]
    n = contract.n_real()
    c = sum(1.0 for i in _real(contract) if nodes[i]["valid"]) / n
    ell = sum(1.0 for i in _real(contract) if nodes[i]["open_loss"]) / n
    g = _g_geom(contract, dyn)
    return c - 1.0 - 0.5 * ell + 0.25 * g


def _ancestors(contract, nodes_of_interest):
    """Recursive boolean support set; no shortest path."""
    seen = set()
    stack = list(nodes_of_interest)
    while stack:
        j = stack.pop()
        if j in seen:
            continue
        seen.add(j)
        for c, on in enumerate(contract.precondition_clause_mask[j]):
            if not on:
                continue
            for i, bit in enumerate(contract.preconditions_dnf[j][c]):
                if bit:
                    stack.append(i)
    return seen


def _clause_gap(contract, dyn, clause, mask_on):
    if not mask_on:
        return None
    nodes = dyn["nodes"]
    missing = []
    for i, bit in enumerate(clause):
        if bit and not nodes[i]["valid"]:
            missing.append(i)
    return missing


def flat_phi(contract: TaskContract, dyn):
    nodes = dyn["nodes"]
    n = contract.n_real()
    gaps = []
    for c, on in enumerate(contract.goal_clause_mask):
        missing = _clause_gap(contract, dyn, contract.goal_dnf[c], on)
        if missing is None:
            continue
        gaps.append((len(missing), c, missing))
    if not gaps:
        return 0.0
    mmin = min(g[0] for g in gaps)
    tied = [g for g in gaps if g[0] == mmin]
    missing_union = set()
    for _, _, miss in tied:
        missing_union.update(miss)
    # direct unsatisfied preconditions of missing goals
    direct = set()
    for j in missing_union:
        flags = [nodes[k]["valid"] for k in range(N)]
        if contract.preconditions_satisfied(j, flags):
            continue
        for c, on in enumerate(contract.precondition_clause_mask[j]):
            if not on:
                continue
            for i, bit in enumerate(contract.preconditions_dnf[j][c]):
                if bit and not nodes[i]["valid"]:
                    direct.add(i)
    support = _ancestors(contract, missing_union)
    # include tied clause members themselves
    for _, c, _ in tied:
        for i, bit in enumerate(contract.goal_dnf[c]):
            if bit:
                support.add(i)
    loss = {i for i in support if nodes[i]["open_loss"]}
    g_rel = 0.0
    denom = 0
    for i in support:
        if not contract.node_mask[i]:
            continue
        denom += 1
        k = contract.transport_steps[i]
        p = (nodes[i]["progress"] / k) if k else 0.0
        if nodes[i]["held"] and not nodes[i]["valid"] and i in support:
            g_rel += p
    g_rel = g_rel / n
    m_goal = float(mmin)
    m_pre = float(len(direct))
    m_loss = float(len(loss))
    return -(m_goal + 0.5 * m_pre + 0.5 * m_loss) / (3.0 * n) + 0.25 * g_rel


def graph_phis(contract, dyn, clip_hits=None):
    pc, _ = phi_cost(contract, dyn, clip_hits)
    g = _g_geom(contract, dyn)
    return pc, pc + 0.25 * g


def evaluate_all(contract: TaskContract, dyn, clip_hits=None):
    pc, pf = graph_phis(contract, dyn, clip_hits)
    return {
        "GEOM_COUNT_EVENTS_PBRS_V1": geom_phi(contract, dyn),
        "FLAT_CONTRACT_PBRS_V1": flat_phi(contract, dyn),
        "PATHGRAPH_MODEL_COST_PBRS_V1": pc,
        "PATHGRAPH_MODEL_FULL_PBRS_V1": pf,
        TASK_ONLY: 0.0,
    }
