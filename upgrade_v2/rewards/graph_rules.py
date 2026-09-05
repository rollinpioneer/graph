"""Explicit oracle/manual-structure references, separated from learned value."""
from __future__ import annotations

import heapq
from collections import defaultdict


def time_fraction(source_step: int, next_step: int, n_transitions: int) -> float | None:
    if n_transitions <= 0:
        return None
    return float(next_step - source_step) / float(n_transitions)


def oracle_topology_cost(node: str, graph: dict) -> float | None:
    """Return the minimum non-negative path cost from ``node`` to a success node.

    This is deliberately an oracle helper: callers must keep its ground-truth
    node input separate from learned scores.  Earlier code used FIFO traversal,
    which is only valid when every edge has equal cost.
    """
    terminal = set(graph.get("terminal_nodes", []))
    if node in terminal:
        return 0.0
    adjacency: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source = edge.get("source", edge.get("src"))
        target = edge.get("target", edge.get("dst"))
        if source is None or target is None:
            continue
        edge_cost = float(edge.get("base_step_cost", 1.0))
        if edge_cost < 0:
            raise ValueError("oracle_topology_cost requires non-negative edge costs")
        adjacency[str(source)].append((str(target), edge_cost))
    queue: list[tuple[float, str]] = [(0.0, node)]
    best = {node: 0.0}
    while queue:
        cost, current = heapq.heappop(queue)
        if cost != best.get(current):
            continue
        if current in terminal:
            return cost
        for target, edge_cost in adjacency.get(current, []):
            next_cost = cost + edge_cost
            if target not in best or next_cost < best[target]:
                best[target] = next_cost
                heapq.heappush(queue, (next_cost, target))
    return None


def graph_from_belief(belief: dict[str, float], graph: dict) -> float | None:
    weighted = 0.0
    mass = 0.0
    for node, probability in belief.items():
        distance = oracle_topology_cost(node, graph)
        if distance is not None:
            weighted += float(probability) * distance
            mass += float(probability)
    return None if mass <= 0 else -weighted / mass


def fixed_chain_from_belief(belief: dict[str, float], chain: list[str]) -> float | None:
    index = {node: position for position, node in enumerate(chain)}
    mass = sum(float(probability) for node, probability in belief.items() if node in index)
    if mass <= 0:
        return None
    return -sum(float(probability) * (len(chain) - 1 - index[node]) for node, probability in belief.items() if node in index) / mass


def legal_success_chains(graph: dict, max_chains: int = 32) -> list[list[str]]:
    """Enumerate simple start-to-success paths from a GraphSpec.

    A chain is emitted only when it is defined by named GraphSpec edges, never
    by classifier label-map order.  Cycles are excluded because a fixed-chain
    reference cannot represent an unbounded recovery loop.
    """
    start = graph.get("start_node")
    terminals = set(graph.get("success_nodes", graph.get("terminal_nodes", [])))
    if not start or not terminals:
        return []
    adjacency: dict[str, list[str]] = defaultdict(list)
    for edge in graph.get("edges", []):
        source = edge.get("source", edge.get("src"))
        target = edge.get("target", edge.get("dst"))
        if source is not None and target is not None:
            adjacency[str(source)].append(str(target))
    chains: list[list[str]] = []

    def visit(current: str, path: list[str]) -> None:
        if len(chains) >= max_chains:
            return
        if current in terminals:
            chains.append(path)
            return
        for target in sorted(set(adjacency.get(current, []))):
            if target not in path:
                visit(target, [*path, target])

    visit(str(start), [str(start)])
    return chains
