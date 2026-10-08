"""Depots instance families built with the PUBLIC generator (AI-Planning/pddl-generators, ``depots``): objects and initial layout come from the generator; the goal structure is selected by explicit rules.

The generator draws a partial random goal (every crate gets a goal with probability ~0.87). A family keeps an instance only if EVERY crate has a goal support (complete layout) and the goal stacks have the required
height profile. Nothing in the filter looks at a model. Provenance (command line, seed) is stored with every instance.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from collections import Counter
from pathlib import Path

from .parse import parse_problem

EXT = Path(os.environ.get("CPDISR_EXT", str(Path.home() / "ext")))
GEN = EXT / "pddl-generators" / "depots" / "depots"
DOMAIN_TYPED = EXT / "pddl-generators" / "depots" / "domain.pddl"
DOMAIN_IPC = EXT / "downward-benchmarks" / "depot" / "domain.pddl"
LOCS = 2                                    # 1 depot + 1 distributor, 1 truck, 2 hoists in every Track-B family (places / trucks stay in the training range)


def generator_command(n_crates, seed, pallets=None, depots=1, distributors=1, trucks=1, hoists=2):
    p = pallets or n_crates
    return [str(GEN), "-e", str(depots), "-i", str(distributors), "-t", str(trucks), "-p", str(max(p, depots + distributors)), "-h", str(hoists), "-c", str(n_crates), "-s", str(seed)]


def generate(n_crates, seed, **kw):
    cmd = generator_command(n_crates, seed, **kw)
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    return out, cmd


def analyse(text):
    """Structure of a generated problem: goal / initial stack heights, completeness, transport need."""
    p = parse_problem(text)
    otype = p.objects
    crates = sorted(o for o, t in otype.items() if t == "crate")
    pallets = sorted(o for o, t in otype.items() if t == "pallet")
    at = {}
    on_init, on_goal = {}, {}
    for pred, args in p.init:
        if pred == "at":
            at[args[0]] = args[1]
        elif pred == "on":
            on_init[args[0]] = args[1]
    for pred, args in p.goal:
        if pred == "on":
            on_goal.setdefault(args[0], []).append(args[1])
    complete = all(len(on_goal.get(c, [])) == 1 for c in crates)
    res = {"n_crates": len(crates), "n_pallets": len(pallets), "goal_complete": complete, "goal_atoms": len(p.goal)}
    if not complete:
        return res
    sup_g = {c: on_goal[c][0] for c in crates}
    if len(set(sup_g.values())) != len(crates):
        res["goal_complete"] = False                                  # two crates on one surface: not a layout
        return res

    def stacks(sup):
        base = {}
        for c in crates:
            x, d = c, 0
            while x in sup:
                x = sup[x]
                d += 1
                if d > len(crates) + 1:
                    return None
            base[c] = x
        # height of the stack standing on a pallet = number of crates whose base is that pallet
        return sorted(Counter(base.values()).values(), reverse=True), base
    g = stacks(sup_g)
    i = stacks(on_init)
    if g is None or i is None:
        res["goal_complete"] = False
        return res
    res["goal_heights"], goal_base = g
    res["init_heights"], init_base = i
    res["max_goal_height"] = max(g[0])
    res["k_goal_towers"] = sum(1 for x in g[0] if x >= 2)
    res["max_init_height"] = max(i[0])
    pallet_place = {pl: at[pl] for pl in pallets}
    res["needs_transport"] = any(pallet_place[goal_base[c]] != at[c] for c in crates)
    res["init_satisfies_goal"] = all(on_init.get(c) == sup_g[c] for c in crates)
    res["goal_crates_already_in_place"] = sum(1 for c in crates if on_init.get(c) == sup_g[c])
    return res


def canonical_hash(text):
    """Isomorphism-class hash of (typed objects, init atoms, goal atoms) under object renaming (colour refinement, 5 rounds)."""
    p = parse_problem(text)
    objs = sorted(p.objects)
    atoms = [("i",) + a for a in sorted(p.init)] + [("g",) + a for a in sorted(p.goal)]
    h = lambda x: hashlib.sha1(repr(x).encode()).hexdigest()[:16]
    col = {o: h(p.objects[o]) for o in objs}
    for _ in range(5):
        sig = {o: [] for o in objs}
        for a in atoms:
            tag, pred, args = a[0], a[1], a[2]
            for pos, o in enumerate(args):
                sig[o].append((tag, pred, pos, tuple(col[x] for x in args)))
        col = {o: h((col[o], sorted(sig[o]))) for o in objs}
    return h(sorted(col.values()))


def shape_matches(heights, spec):
    """spec: exact sorted-descending goal stack-height multiset (singletons included) or a callable."""
    return bool(spec(heights)) if callable(spec) else list(heights) == list(spec)


def one_tower(allowed_heights):
    return lambda hs: sum(1 for x in hs if x >= 2) == 1 and max(hs) in allowed_heights and all(x == 1 for x in hs[1:])


JOINT_CELLS = {"n6k1h2": (6, [2, 1, 1, 1, 1]), "n6k1h4": (6, [4, 1, 1]), "n6k2h2": (6, [2, 2, 1, 1]), "n6k2h4": (6, [4, 2]),
               "n8k1h2": (8, [2, 1, 1, 1, 1, 1, 1]), "n8k1h4": (8, [4, 1, 1, 1, 1]), "n8k2h2": (8, [2, 2, 1, 1, 1, 1]), "n8k2h4": (8, [4, 2, 1, 1])}
