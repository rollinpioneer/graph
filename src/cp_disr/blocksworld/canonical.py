"""Exact isomorphism hashes (runbook 11.1). No approximate graph hash is used.

* ``goal_iso_hash``      : colour-preserving isomorphism class of the goal configuration (exact: a complete goal is a set of chains, so the multiset of colour sequences,
                           bottom to top, is a complete invariant; cross-checked by exhaustive enumeration in the tests).
* ``goal_shape_hash``    : colour-ignoring isomorphism class of the goal (multiset of tower heights).
* ``problem_iso_hash``   : colour-preserving isomorphism class of (initial state, goal); minimum over ALL colour-preserving block permutations (n <= 8).
"""
from __future__ import annotations

import hashlib
import itertools
import json

from . import state as S


def _digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def goal_towers(goal):
    """Chains of the goal configuration, bottom to top, as lists of block indices."""
    n = len(goal)
    above = {}
    for x, b in enumerate(goal):
        if b >= 0:
            above[b] = x
    out = []
    for x in range(n):
        if goal[x] == S.TABLE:
            t, y = [x], x
            while y in above:
                y = above[y]
                t.append(y)
            out.append(t)
    return out


def tower_color_sequences(problem):
    return sorted(tuple(problem.colors[b] for b in t) for t in goal_towers(problem.goal))


def goal_iso_hash(problem):
    return _digest({"kind": "goal_color_iso", "towers": tower_color_sequences(problem)})


def goal_shape(problem):
    return tuple(sorted((len(t) for t in goal_towers(problem.goal)), reverse=True))


def goal_shape_hash(problem):
    return _digest({"kind": "goal_shape", "heights": list(goal_shape(problem))})


def _relabel(state, label):
    n = len(state)
    out = [0] * n
    for x in range(n):
        b = state[x]
        out[label[x]] = label[b] if b >= 0 else b
    return tuple(out)


def _color_permutations(colors):
    n = len(colors)
    reds = [i for i in range(n) if colors[i] == S.RED]
    blues = [i for i in range(n) if colors[i] == S.BLUE]
    for pr in itertools.permutations(range(len(reds))):
        for pb in itertools.permutations(range(len(blues))):
            label = [0] * n
            for k, i in enumerate(reds):
                label[i] = pr[k]
            for k, i in enumerate(blues):
                label[i] = len(reds) + pb[k]
            yield label


def problem_iso_hash(problem):
    best = None
    for label in _color_permutations(problem.colors):
        rep = (_relabel(problem.init, label), _relabel(problem.goal, label))
        if best is None or rep < best:
            best = rep
    n_red = sum(1 for c in problem.colors if c == S.RED)
    return _digest({"kind": "problem_color_iso", "n_red": n_red, "n_blue": problem.n - n_red, "rep": best})


def init_iso_hash(problem):
    """Colour-preserving isomorphism class of the initial state alone (used to keep dev initial states disjoint from train)."""
    best = None
    for label in _color_permutations(problem.colors):
        rep = _relabel(problem.init, label)
        if best is None or rep < best:
            best = rep
    n_red = sum(1 for c in problem.colors if c == S.RED)
    return _digest({"kind": "init_color_iso", "n_red": n_red, "n_blue": problem.n - n_red, "rep": best})


def goal_iso_hash_bruteforce(problem):
    """Reference implementation by exhaustive colour-preserving relabelling (used by the tests on small n)."""
    best = None
    for label in _color_permutations(problem.colors):
        rep = _relabel(problem.goal, label)
        if best is None or rep < best:
            best = rep
    n_red = sum(1 for c in problem.colors if c == S.RED)
    return _digest({"kind": "goal_color_iso_bf", "n_red": n_red, "n_blue": problem.n - n_red, "rep": best})


def goal_shape_hash_bruteforce(problem):
    """Colour-ignoring reference: minimum over all n! relabellings."""
    best = None
    n = problem.n
    for perm in itertools.permutations(range(n)):
        rep = _relabel(problem.goal, list(perm))
        if best is None or rep < best:
            best = rep
    return _digest({"kind": "goal_shape_bf", "rep": best})


def rename(problem, new_names, permutation=None):
    """Same instance under new block names (and optionally a colour-preserving permutation of the underlying indices)."""
    if permutation is None:
        return S.Problem(tuple(new_names), problem.colors, problem.init, problem.goal)
    label = list(permutation)
    n = problem.n
    colors = [0] * n
    names = [None] * n
    for x in range(n):
        colors[label[x]] = problem.colors[x]
        names[label[x]] = new_names[x]
    return S.Problem(tuple(names), tuple(colors), _relabel(problem.init, label), _relabel(problem.goal, label))
