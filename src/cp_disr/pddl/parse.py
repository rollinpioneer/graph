"""Minimal typed / untyped STRIPS PDDL reader (domain + problem) for the public-domain adaptation of the CP-DISR relational scorer.

Supported: ``:types`` with inheritance, ``:predicates`` (typed or untyped), ``:action`` with ``:parameters``, conjunctive positive / negative preconditions, add / delete effects, ``:objects`` (typed or untyped),
``:init`` positive atoms, ``:goal`` conjunction of positive atoms. Anything else (conditional effects, quantifiers, numeric fluents, disjunction) raises. Atoms are tuples ``(predicate, arg, ...)``.
Everything is lower-cased (PDDL is case-insensitive); object names are identifiers only and never enter a model feature.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

TOKEN = re.compile(r"\(|\)|[^\s()]+")


class PddlError(ValueError):
    pass


def read_sexpr(text):
    text = re.sub(r";[^\n]*", "", text).lower()
    toks = TOKEN.findall(text)
    pos = 0

    def parse():
        nonlocal pos
        if toks[pos] != "(":
            raise PddlError("expected '(' at token %d" % pos)
        pos += 1
        out = []
        while toks[pos] != ")":
            if toks[pos] == "(":
                out.append(parse())
            else:
                out.append(toks[pos])
                pos += 1
        pos += 1
        return out
    tree = parse()
    if pos != len(toks):
        raise PddlError("trailing tokens")
    return tree


def typed_list(items):
    """['a','b','-','t','c'] -> [('a','t'),('b','t'),('c','object')]; items may be atoms (str) only."""
    out, pending, i = [], [], 0
    while i < len(items):
        x = items[i]
        if x == "-":
            t = items[i + 1]
            if isinstance(t, list):
                raise PddlError("either-types are not supported")
            out += [(p, t) for p in pending]
            pending = []
            i += 2
            continue
        pending.append(x)
        i += 1
    out += [(p, "object") for p in pending]
    return out


@dataclass(frozen=True)
class Action:
    name: str
    params: tuple                 # ((var, type), ...)
    pre_pos: tuple                # ((pred, (var, ...)), ...)
    pre_neg: tuple
    add: tuple
    delete: tuple


@dataclass
class Domain:
    name: str
    parents: dict                 # type -> parent type
    predicates: dict              # name -> tuple(arg types)
    actions: dict                 # name -> Action
    typed: bool

    def subtype_of(self, t, anc):
        while True:
            if t == anc:
                return True
            if t not in self.parents or t == "object":
                return anc == "object"
            t = self.parents[t]


@dataclass
class Problem:
    name: str
    objects: dict                 # object -> declared type
    init: frozenset               # atoms (pred, args)
    goal: tuple                   # atoms


def _atoms_of(cond, positive_only=False):
    """Flatten a conjunction into (positive atoms, negative atoms)."""
    pos, neg = [], []

    def go(c):
        if not c:
            return
        head = c[0]
        if head == "and":
            for sub in c[1:]:
                go(sub)
        elif head == "not":
            inner = c[1]
            if inner[0] in ("and", "not", "or", "forall", "exists", "when"):
                raise PddlError("nested negation / quantifier not supported")
            neg.append((inner[0], tuple(inner[1:])))
        elif head in ("or", "forall", "exists", "when", "imply"):
            raise PddlError("unsupported construct: %s" % head)
        elif head == "=":
            raise PddlError("equality not supported")
        else:
            pos.append((head, tuple(c[1:])))
    go(cond)
    return tuple(pos), tuple(neg)


def parse_domain(text):
    tree = read_sexpr(text)
    if tree[0] != "define":
        raise PddlError("not a PDDL file")
    name, parents, predicates, actions, typed = "", {}, {}, {}, False
    for sec in tree[1:]:
        key = sec[0]
        if key == "domain":
            name = sec[1]
        elif key == ":types":
            typed = True
            for t, p in typed_list(sec[1:]):
                parents[t] = p
        elif key == ":predicates":
            for pr in sec[1:]:
                predicates[pr[0]] = tuple(t for _v, t in typed_list(pr[1:]))
        elif key == ":action":
            aname = sec[1]
            body = dict(zip(sec[2::2], sec[3::2]))
            params = tuple(typed_list(body[":parameters"]))
            pre = body.get(":precondition", [])
            eff = body.get(":effect", [])
            pre_pos, pre_neg = _atoms_of(pre)
            add, delete = _atoms_of(eff)
            actions[aname] = Action(aname, params, pre_pos, pre_neg, add, tuple(delete))
        elif key in (":requirements", ":constants", ":functions"):
            if key != ":requirements":
                raise PddlError("unsupported section %s" % key)
    return Domain(name, parents, predicates, actions, typed)


def parse_problem(text):
    tree = read_sexpr(text)
    name, objects, init, goal = "", {}, [], ()
    for sec in tree[1:]:
        key = sec[0]
        if key == "problem":
            name = sec[1]
        elif key == ":objects":
            for o, t in typed_list(sec[1:]):
                objects[o] = t
        elif key == ":init":
            for a in sec[1:]:
                if a[0] in ("=", "not"):
                    raise PddlError("numeric / negative init not supported")
                init.append((a[0], tuple(a[1:])))
        elif key == ":goal":
            pos, neg = _atoms_of(sec[1])
            if neg:
                raise PddlError("negative goals not supported")
            goal = pos
    return Problem(name, objects, frozenset(init), goal)
