"""Tie-aware, group-aware descriptive statistics used by U0/U1."""
from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable


def average_ranks(values: Iterable[float]) -> list[float]:
    values = [float(v) for v in values]
    ordered = sorted(enumerate(values), key=lambda pair: (pair[1], pair[0]))
    ranks = [0.0] * len(values)
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][1] == ordered[index][1]:
            end += 1
        mean_rank = ((index + 1) + end) / 2.0
        for original, _ in ordered[index:end]:
            ranks[original] = mean_rank
        index = end
    return ranks


def spearman_average_ties(values: Iterable[float], outcomes: Iterable[float]) -> float | None:
    x, y = list(values), list(outcomes)
    if len(x) != len(y) or len(x) < 2:
        return None
    rx, ry = average_ranks(x), average_ranks(y)
    mx, my = sum(rx) / len(rx), sum(ry) / len(ry)
    vx, vy = sum((v - mx) ** 2 for v in rx), sum((v - my) ** 2 for v in ry)
    if vx == 0 or vy == 0:
        return None
    return sum((a - mx) * (b - my) for a, b in zip(rx, ry)) / math.sqrt(vx * vy)


def paired_group_deltas(rows: Iterable[dict], treatment: str, comparator: str) -> list[dict]:
    """Return one stable mean delta per compound independent group."""
    groups: dict[tuple[str, str, str], dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        key = (str(row["task_id"]), str(row["provenance"]), str(row["root_family_id"]))
        groups[key][str(row["method_id"])].append(float(row["value"]))
    output = []
    for key in sorted(groups):
        methods = groups[key]
        if treatment in methods and comparator in methods:
            output.append({"task_id": key[0], "provenance": key[1], "root_family_id": key[2],
                           "delta": sum(methods[treatment]) / len(methods[treatment]) - sum(methods[comparator]) / len(methods[comparator])})
    return output
