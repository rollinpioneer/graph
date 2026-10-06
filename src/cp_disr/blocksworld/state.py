"""CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: pure symbolic Blocksworld engine (no torch, deterministic, fully observable).

A state is a tuple ``below`` with one entry per block: ``TABLE`` (-1) when the block is on the table, ``y >= 0`` when it is on block ``y`` and ``HELD`` (-2) when the
hand holds it. At most one block is held, at most one block rests on a block (enforced by ``is_valid``). All dynamic predicates are derived from ``below``:

    On(x, y) : below[x] == y     OnTable(x) : below[x] == TABLE     Holding(x) : below[x] == HELD
    Clear(x) : x is not held and no block rests on x     HandEmpty : no block is held

Colours are static and live in the problem, never in the state.
"""
from __future__ import annotations

from dataclasses import dataclass

TABLE = -1
HELD = -2
RED, BLUE = 0, 1
COLOR_NAME = {RED: "Red", BLUE: "Blue"}
SCHEMAS = ("PICK_UP", "PUT_DOWN", "UNSTACK", "STACK")
ARITY = {"PICK_UP": 1, "PUT_DOWN": 1, "UNSTACK": 2, "STACK": 2}


@dataclass(frozen=True)
class Problem:
    """One instance: block names, static colours, initial state and a complete goal configuration (hand empty)."""
    names: tuple
    colors: tuple
    init: tuple
    goal: tuple

    @property
    def n(self):
        return len(self.names)


def default_names(n, prefix="b"):
    return tuple("%s%d" % (prefix, i) for i in range(n))


def held(state):
    for x, b in enumerate(state):
        if b == HELD:
            return x
    return -1


def supports(state):
    """Index of the block resting on each block, or -1."""
    on = [-1] * len(state)
    for x, b in enumerate(state):
        if b >= 0:
            on[b] = x
    return on


def is_valid(state):
    n = len(state)
    seen = set()
    holding = 0
    for x, b in enumerate(state):
        if b == HELD:
            holding += 1
        elif b >= 0:
            if b == x or b >= n or b in seen or state[b] == HELD:
                return False
            seen.add(b)
        elif b != TABLE:
            return False
    if holding > 1:
        return False
    # no cycles: every block must reach the table
    for x in range(n):
        y, steps = x, 0
        while state[y] >= 0:
            y = state[y]
            steps += 1
            if steps > n:
                return False
    return True


def clear(state, x):
    return state[x] != HELD and all(b != x for b in state)


def legal_actions(state):
    """Legal grounded actions in a fixed order: ('PICK_UP', x) ('PUT_DOWN', x) ('UNSTACK', x, y) ('STACK', x, y)."""
    h = held(state)
    out = []
    on = supports(state)
    if h < 0:
        for x, b in enumerate(state):
            if on[x] == -1:                      # clear and not held
                if b == TABLE:
                    out.append(("PICK_UP", x))
                elif b >= 0:
                    out.append(("UNSTACK", x, b))
    else:
        out.append(("PUT_DOWN", h))
        for y in range(len(state)):
            if y != h and state[y] != HELD and on[y] == -1:
                out.append(("STACK", h, y))
    return out


def apply(state, action):
    s = list(state)
    kind = action[0]
    x = action[1]
    if kind == "PICK_UP" or kind == "UNSTACK":
        s[x] = HELD
    elif kind == "PUT_DOWN":
        s[x] = TABLE
    elif kind == "STACK":
        s[x] = action[2]
    else:
        raise ValueError(kind)
    return tuple(s)


def successors(state):
    return [(a, apply(state, a)) for a in legal_actions(state)]


def action_name(names, action):
    return ":".join((action[0],) + tuple(names[i] for i in action[1:]))


def action_id(names, action, version="v1"):
    return "a:%s:%s" % (action_name(names, action), version)


def atom_id(names, kind, *blocks):
    return "p:" + ":".join((kind,) + tuple(names[i] for i in blocks))


def fact_values(problem, state):
    """Full propositional description of a state as {atom_id: bool} (the three-valued store only ever sees TRUE / FALSE here)."""
    n, names = problem.n, problem.names
    facts = {}
    on = supports(state)
    h = held(state)
    for x in range(n):
        for y in range(n):
            if x != y:
                facts[atom_id(names, "On", x, y)] = state[x] == y
        facts[atom_id(names, "OnTable", x)] = state[x] == TABLE
        facts[atom_id(names, "Clear", x)] = state[x] != HELD and on[x] == -1
        facts[atom_id(names, "Holding", x)] = state[x] == HELD
        facts[atom_id(names, "Red", x)] = problem.colors[x] == RED
        facts[atom_id(names, "Blue", x)] = problem.colors[x] == BLUE
    facts["p:HandEmpty"] = h < 0
    return facts


def goal_atoms(problem):
    """Goal atoms of a complete configuration: On(x, y) for every stacked block, OnTable(x) for every block on the table."""
    out = []
    for x, b in enumerate(problem.goal):
        out.append(atom_id(problem.names, "On", x, b) if b >= 0 else atom_id(problem.names, "OnTable", x))
    return tuple(sorted(out))


def goal_atom_truth(problem, state):
    """{goal atom id: bool} in ``state``."""
    out = {}
    for x, b in enumerate(problem.goal):
        atom = atom_id(problem.names, "On", x, b) if b >= 0 else atom_id(problem.names, "OnTable", x)
        out[atom] = state[x] == b
    return out


def goal_satisfied(problem, state):
    return all(state[x] == b for x, b in enumerate(problem.goal))


def towers(state):
    """Maximal stacks as bottom-to-top block lists (a held block is not part of any tower)."""
    n = len(state)
    on = supports(state)
    out = []
    for x in range(n):
        if state[x] == TABLE:
            t, y = [x], x
            while on[y] != -1:
                y = on[y]
                t.append(y)
            out.append(t)
    return out
