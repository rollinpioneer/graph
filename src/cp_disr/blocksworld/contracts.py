"""Blocksworld skill contracts (runbook 5.4) expressed with the production ``cp_disr.contracts`` / ``cp_disr.graph`` machinery, grounded per instance.

Colours enter the relational graph exactly as the runbook requires: every grounded action carries the (always TRUE) colour fact of each of its block arguments as a
static ``pre_pos`` atom, so the action node is linked to ``Red(x)`` or ``Blue(x)`` by a PRE_POS edge. Both ``Red(x)`` and ``Blue(x)`` exist as proposition nodes of every
block (one TRUE, one FALSE); no colour is hidden in names or hashes.
"""
from __future__ import annotations

from dataclasses import replace
from functools import lru_cache

from ..contracts import Atom, Effects, SkillContract, TypedArgument
from ..graph import Goal, build_template
from . import state as S

PREDICATE_TYPES = {"On": ("block", "block"), "OnTable": ("block",), "Clear": ("block",), "Holding": ("block",), "HandEmpty": (), "Red": ("block",), "Blue": ("block",)}
PREDICATES = tuple(sorted(PREDICATE_TYPES))
TYPES = ("block",)
SCHEMA_NAMES = S.SCHEMAS
CONTRACT_VERSION = "bw-v1"
TIMEOUT = 1.0


def _atom(pred, *args):
    return Atom(pred, tuple(args))


def schema_contracts():
    """The four lifted schemas (arguments x, y). Preconditions / effects are those of runbook 5.4 (colour preconditions are added when grounding)."""
    x, y = TypedArgument("x", "block"), TypedArgument("y", "block")
    hand = _atom("HandEmpty")

    def mk(name, args, pre, add, delete):
        return SkillContract(name=name, arguments=args, pre_pos=tuple(pre), pre_neg=(), effects=Effects(tuple(add), tuple(delete), ()), version=CONTRACT_VERSION, provenance="runbook-5.4",
                             timeout_seconds=TIMEOUT)
    return {
        "PICK_UP": mk("PICK_UP", (x,), [_atom("OnTable", "x"), _atom("Clear", "x"), hand], [_atom("Holding", "x")], [_atom("OnTable", "x"), _atom("Clear", "x"), hand]),
        "PUT_DOWN": mk("PUT_DOWN", (x,), [_atom("Holding", "x")], [_atom("OnTable", "x"), _atom("Clear", "x"), hand], [_atom("Holding", "x")]),
        "UNSTACK": mk("UNSTACK", (x, y), [_atom("On", "x", "y"), _atom("Clear", "x"), hand], [_atom("Holding", "x"), _atom("Clear", "y")], [_atom("On", "x", "y"), _atom("Clear", "x"), hand]),
        "STACK": mk("STACK", (x, y), [_atom("Holding", "x"), _atom("Clear", "y")], [_atom("On", "x", "y"), _atom("Clear", "x"), hand], [_atom("Holding", "x"), _atom("Clear", "y")]),
    }


def _color_atom(problem, block):
    return Atom(S.COLOR_NAME[problem.colors[block]], (problem.names[block],))


def ground_action(problem, action, schemas=None):
    schemas = schemas or _SCHEMAS
    contract = schemas[action[0]]
    binding = {"x": problem.names[action[1]]}
    if len(action) > 2:
        binding["y"] = problem.names[action[2]]
    g = contract.ground(binding)
    static = tuple(_color_atom(problem, b) for b in action[1:])
    return replace(g, pre_pos=g.pre_pos + static)


_SCHEMAS = schema_contracts()


def all_actions(n):
    out = []
    for x in range(n):
        out.append(("PICK_UP", x))
        out.append(("PUT_DOWN", x))
    for x in range(n):
        for y in range(n):
            if x != y:
                out.append(("UNSTACK", x, y))
                out.append(("STACK", x, y))
    return out


def grounded_contracts(problem):
    """All grounded contracts of an instance (including the static colour preconditions), sorted by contract id."""
    cs = [ground_action(problem, a) for a in all_actions(problem.n)]
    return tuple(sorted(cs, key=lambda c: c.id))


def object_types(problem):
    return {name: "block" for name in problem.names}


def extra_atoms(problem):
    atoms = [Atom("HandEmpty", ())]
    for x in range(problem.n):
        atoms += [Atom("Red", (problem.names[x],)), Atom("Blue", (problem.names[x],)), Atom("OnTable", (problem.names[x],)), Atom("Clear", (problem.names[x],)), Atom("Holding", (problem.names[x],))]
        for y in range(problem.n):
            if x != y:
                atoms.append(Atom("On", (problem.names[x], problem.names[y])))
    return tuple(atoms)


def build_graph_template(problem):
    goals = tuple(Goal(a, 1) for a in S.goal_atoms(problem))
    return build_template(grounded_contracts(problem), goals, PREDICATE_TYPES, object_types(problem), extra_atoms(problem))


@lru_cache(maxsize=4096)
def _template_cached(key):
    names, colors, goal = key
    return build_graph_template(S.Problem(names, colors, tuple([S.TABLE] * len(names)), goal))


def template_for(problem):
    """Graph template of an instance (depends on names, colours and goal only; cached)."""
    return _template_cached((problem.names, problem.colors, problem.goal))


def action_by_id(problem):
    return {S.action_id(problem.names, a): a for a in all_actions(problem.n)}
