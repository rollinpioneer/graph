"""Paired task changes of plan 4.C (old DENSE and old WL only): P goal relaxation, R extra empty pallet, W extra place with hoist and pallet receiving the first initial stack.

Base problems: the Dev n4 / n5 groups of the earlier card, 4 per n, one per (needs_transport x max_goal_height) cell in round-robin order (cells ordered by transport desc, height asc), smallest problem sha256 first inside a cell.
Every transformation is applied once, exactly as defined, and is never replaced because of a score. Text is re-emitted from the parsed problem (lower case, canonical order)."""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from .. import data as DT
from .. import depots as DP
from .. import task as T
from ..parse import parse_problem
from .diagb import emit_problem
from .train import load_old


def pick_bases(man, per_n=4):
    out = []
    for n in (4, 5):
        cells = defaultdict(list)
        for c in sorted(man["dev"], key=lambda c: c["sha256"]):
            if c["analysis"]["n_crates"] == n:
                cells[(not c["analysis"]["needs_transport"], c["analysis"]["max_goal_height"])].append(c)
        keys = sorted(cells)
        got, r = [], 0
        while len(got) < per_n and any(cells[k] for k in keys):
            for k in keys:
                if cells[k] and len(got) < per_n:
                    got.append(cells[k].pop(0))
            r += 1
        out += got
    return out


def goal_structure(p):
    sup = {}
    for pred, args in p.goal:
        if pred == "on":
            sup[args[0]] = args[1]
    crates = sorted(o for o, t in p.objects.items() if t == "crate")

    def base(c):
        x = c
        while x in sup and sup[x] in sup:
            x = sup[x]
        return sup.get(x, x)
    stacks = defaultdict(list)
    for c in crates:
        stacks[base(c)].append(c)
    return sup, crates, stacks


def variant_P(p):
    sup, crates, stacks = goal_structure(p)
    tower = max(stacks.values(), key=len)
    in_tower = set(tower) if len(tower) >= 2 else set()
    others = [c for c in crates if c not in in_tower and c in sup]
    goal = list(p.goal)
    if others:
        victim = others[0]
        removed = ("on", (victim, sup[victim]))
        why = "support goal of a crate outside the non-trivial tower"
    else:
        tops = [c for c in tower if not any(sup.get(d) == c for d in tower)]
        crate_on_crate = [c for c in tower if sup[c] in crates]
        if len(crate_on_crate) < 2:
            return None, {"available": False, "reason": "no removable goal atom keeps a crate-on-crate goal"}
        victim = sorted(tops)[0]
        removed = ("on", (victim, sup[victim]))
        why = "top 'on' atom of the tower (at least one crate-on-crate goal remains)"
    goal.remove(removed)
    return goal, {"available": True, "removed_goal_atom": list(map(str, (removed[0],) + removed[1])), "rule": why}


def variant_R(p):
    at = {a[0]: a[1] for pred, a in p.init if pred == "at"}
    crates = [o for o, t in p.objects.items() if t == "crate"]
    cnt = defaultdict(int)
    for c in crates:
        cnt[at[c]] += 1
    place = sorted(cnt, key=lambda x: (-cnt[x], x))[0]
    pallets = [o for o, t in p.objects.items() if t == "pallet"]
    name = "pallet%d" % len(pallets)
    return name, place


def variant_W(p):
    at = {a[0]: a[1] for pred, a in p.init if pred == "at"}
    on = {a[0]: a[1] for pred, a in p.init if pred == "on"}
    pallets = sorted(o for o, t in p.objects.items() if t == "pallet")
    crates = sorted(o for o, t in p.objects.items() if t == "crate")

    def base(c):
        x = c
        while x in on:
            x = on[x]
        return x
    stack_of = defaultdict(list)
    for c in crates:
        stack_of[base(c)].append(c)
    movers = None
    for pl in pallets:
        if stack_of.get(pl):
            movers = [pl] + stack_of[pl]
            break
    if movers is None:
        return None
    dist = [o for o, t in p.objects.items() if t == "distributor"]
    hoists = [o for o, t in p.objects.items() if t == "hoist"]
    newplace, newhoist, newpallet = "distributor%d" % len(dist), "hoist%d" % len(hoists), "pallet%d" % len(pallets)
    return movers, newplace, newhoist, newpallet


def build(base_case):
    text = Path(base_case["file"]).read_text()
    p = parse_problem(text)
    name = base_case["case_id"]
    out = {}
    goal, info = variant_P(p)
    if goal is not None:
        out["P"] = (emit_problem(name + "-P", dict(p.objects), sorted(p.init), goal), info)
    else:
        out["P"] = (None, info)
    pn, place = variant_R(p)
    objs = dict(p.objects)
    objs[pn] = "pallet"
    init = sorted(p.init) + [("at", (pn, place)), ("clear", (pn,))]
    out["R"] = (emit_problem(name + "-R", objs, init, list(p.goal)), {"available": True, "new_pallet": pn, "place": place})
    w = variant_W(p)
    if w is None:
        out["W"] = (None, {"available": False, "reason": "no initial stack with a crate"})
    else:
        movers, npl, nh, npa = w
        objs = dict(p.objects)
        objs[npl], objs[nh], objs[npa] = "distributor", "hoist", "pallet"
        moved = set(movers)
        old = {a[0]: a[1] for pred, a in p.init if pred == "at"}[movers[0]]
        init = []
        for pred, args in sorted(p.init):
            if pred == "at" and args[0] in moved:
                init.append(("at", (args[0], npl)))
            else:
                init.append((pred, args))
        init += [("at", (nh, npl)), ("available", (nh,)), ("at", (npa, npl)), ("clear", (npa,))]
        out["W"] = (emit_problem(name + "-W", objs, init, list(p.goal)), {"available": True, "moved_objects": movers, "from_place": old, "new_place": npl, "new_hoist": nh, "new_empty_pallet": npa})
    return out


def make_all(root, cfg, outdir, workdir, fd_seconds=60):
    """Writes problem files and the transform manifest; reference plans: P / R replay the base optimal plan, W: one bounded LAMA call. Returns the manifest rows."""
    man, exact, root = load_old(root)
    outdir, workdir = Path(outdir), Path(workdir)
    outdir.mkdir(parents=True, exist_ok=True)
    domain_text = DP.DOMAIN_TYPED.read_text()
    rows, solves = [], 0
    for b in pick_bases(man):
        cid = b["case_id"]
        base_task = T.depots_task(DP.DOMAIN_TYPED, b["file"])
        base_plan = exact[cid]["plan"]
        rows.append({"problem_id": cid, "base": cid, "kind": "BASE", "file": b["file"], "sha256": b["sha256"], "feasible": True, "reference_length": len(base_plan), "reference_source": "exact optimal plan (earlier card)",
                     "reference_optimal": True, "n_crates": b["analysis"]["n_crates"], "needs_transport": b["analysis"]["needs_transport"], "max_goal_height": b["analysis"]["max_goal_height"]})
        for kind, (text, info) in build(b).items():
            pid = "%s-%s" % (cid, kind)
            row = {"problem_id": pid, "base": cid, "kind": kind, **info}
            if text is None:
                row["feasible"] = False
                rows.append(row)
                continue
            path = outdir / (pid + ".pddl")
            path.write_text(text)
            row["file"], row["sha256"] = str(path), hashlib.sha256(text.encode()).hexdigest()
            task = T.depots_task(DP.DOMAIN_TYPED, path)
            row["n_ground_actions"], row["n_dyn_atoms"] = len(task.actions), len(task.dyn_atoms)
            if kind in ("P", "R"):
                fin = task.replay(base_plan)
                ok = bool(fin is not None and task.goal_satisfied(fin))
                row.update({"feasible": ok, "reference_length": len(base_plan) if ok else None, "reference_source": "replay of the base optimal plan (upper bound, not an optimum)", "reference_optimal": False})
            else:
                solves += 1
                assert solves <= cfg["variants_C"]["reference_solves"]["max"]
                r = DT.run_fd(DP.DOMAIN_TYPED, path, workdir / pid, "lama-first", fd_seconds, 4096, tag="lamafirst")
                ok = bool(r["solved"])
                row.update({"feasible": ok, "reference_length": r.get("best_length"), "reference_source": "lama-first <= %d s (upper bound, not an optimum)" % fd_seconds, "reference_optimal": False, "reference_plan_file": r.get("best_plan_file"),
                            "reference_wall_seconds": r.get("wall_seconds"), "feasibility_unproven_means_not_unsolvable": not ok})
            rows.append(row)
    return rows, solves
