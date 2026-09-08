"""Safe structured three-valued predicate DSL; arbitrary code is never evaluated."""

from __future__ import annotations

from typing import Any


TRUE, FALSE, UNKNOWN = "true", "false", "unknown"
VALID = {TRUE, FALSE, UNKNOWN}


def tri_not(value: str) -> str:
    if value not in VALID:
        raise ValueError(value)
    return {TRUE: FALSE, FALSE: TRUE, UNKNOWN: UNKNOWN}[value]


def tri_and(values: list[str]) -> str:
    if any(value == FALSE for value in values):
        return FALSE
    if any(value == UNKNOWN for value in values):
        return UNKNOWN
    return TRUE


def tri_or(values: list[str]) -> str:
    if any(value == TRUE for value in values):
        return TRUE
    if any(value == UNKNOWN for value in values):
        return UNKNOWN
    return FALSE


def evaluate(expression: Any, history: list[dict[str, str]], index: int | None = None) -> str:
    if index is None:
        index = len(history) - 1
    if index < 0:
        return UNKNOWN
    if isinstance(expression, str):
        if expression in {"TRUE", "FALSE", "UNKNOWN"}:
            return expression.lower()
        return history[index].get(expression, UNKNOWN)
    if not isinstance(expression, dict):
        raise ValueError("DSL expression must be a predicate string or object")
    op = expression.get("op")
    if op == "NOT":
        return tri_not(evaluate(expression["arg"], history, index))
    if op == "AND":
        return tri_and([evaluate(arg, history, index) for arg in expression.get("args", [])])
    if op == "OR":
        return tri_or([evaluate(arg, history, index) for arg in expression.get("args", [])])
    if op == "CONSECUTIVE":
        n = int(expression["n"])
        if n < 1 or index + 1 < n:
            return UNKNOWN
        return tri_and([evaluate(expression["arg"], history, offset) for offset in range(index - n + 1, index + 1)])
    if op == "EVER":
        window = int(expression["window"])
        if window < 1:
            raise ValueError("history window must be positive")
        return tri_or([evaluate(expression["arg"], history, offset) for offset in range(max(0, index - window + 1), index + 1)])
    if op == "CONFIDENCE_GTE":
        name = str(expression["name"])
        raw = history[index].get(name)
        if raw is None:
            return UNKNOWN
        return TRUE if float(raw) >= float(expression["threshold"]) else FALSE
    raise ValueError(f"unsupported DSL operator: {op!r}")


def satisfied_edges(edges: list[dict[str, Any]], history: list[dict[str, str]], index: int | None = None) -> dict[str, Any]:
    values = [(edge["id"], evaluate(edge.get("condition", "UNKNOWN"), history, index)) for edge in edges]
    active = [edge_id for edge_id, value in values if value == TRUE]
    return {"status": "ambiguous" if len(active) > 1 else "selected" if active else "unknown", "active_edges": active, "values": dict(values)}
