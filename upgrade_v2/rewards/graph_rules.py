"""Explicit oracle/manual-structure references, separated from learned value."""
from __future__ import annotations

from collections import deque


def time_fraction(source_step: int, next_step: int, n_transitions: int) -> float | None:
    if n_transitions <= 0:
        return None
    return float(next_step - source_step) / float(n_transitions)


def oracle_topology_cost(node: str, graph: dict) -> float | None:
    terminal = set(graph.get("terminal_nodes", []))
    if node in terminal:
        return 0.0
    edges = graph.get("edges", [])
    queue = deque([(node, 0.0)])
    best = {node: 0.0}
    while queue:
        current, cost = queue.popleft()
        for edge in edges:
            if edge.get("source") != current:
                continue
            next_cost = cost + float(edge.get("base_step_cost", 1.0))
            target = edge.get("target")
            if target in terminal:
                return next_cost
            if target not in best or next_cost < best[target]:
                best[target] = next_cost
                queue.append((target, next_cost))
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
