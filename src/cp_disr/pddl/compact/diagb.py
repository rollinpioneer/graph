"""Stage code of the shared state library: build + label, score with any scorer, metrics, equivariance check (plan 4.B)."""
from __future__ import annotations

import hashlib
import json
import math
import random
import tempfile
import time
from collections import Counter, defaultdict
from pathlib import Path

from .. import depots as DP
from .. import stages as ST
from .. import task as T
from ..parse import parse_problem
from ..search_match import evaluators as EV
from ..search_match.budget import Budget
from . import statelib as SL
from .train import load_old, rj

TOL_ABS, TOL_REL = 1e-5, 1e-6


def tol(a, b):
    return TOL_ABS + TOL_REL * max(abs(a), abs(b))


def jdump(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=1, sort_keys=True, default=str) + "\n", encoding="utf-8")


def build(rr, root, cfg):
    t0 = time.time()
    per, included, uniq = SL.build_library(root, cfg)
    labels, queries, notes = SL.label_library(per, included, root, cfg)
    pairs = SL.select_pairs(per, included, labels, cfg)
    d = Path(rr) / "diagnostics"
    d.mkdir(parents=True, exist_ok=True)
    mem = {}
    pk_out = {}
    for cid, p in per.items():
        pk_out[cid] = {"family": p["case"]["family"], "case_file": p["case"]["file"], "problem_sha256": p["case"]["sha256"], "reference_kind": p["ref_kind"], "reference_length": p["L"], "oracle": notes.get(cid),
                       "packages": [{"k": k, "kind": p["packages"][k]["kind"], "ref_pos": p["packages"][k]["ref_pos"], "parent": SL.hx(p["packages"][k]["parent"]), "successors": [[SL.hx(t), a] for t, a in p["packages"][k]["successors"]]} for k in included[cid]],
                       "skipped_packages": [k for k in range(len(p["packages"])) if k not in included[cid]]}
        for k in included[cid]:
            pkg = p["packages"][k]
            for s in [pkg["parent"]] + [t for t, _a in pkg["successors"]]:
                mem.setdefault((cid, s), []).append(k)
    jdump(d / "packages.json", pk_out)
    with open(d / "state_manifest.jsonl", "w") as f:
        for (cid, s), ks in sorted(mem.items(), key=lambda kv: (kv[0][0], kv[0][1])):
            lab = labels[(cid, s)]
            f.write(json.dumps({"problem": cid, "state": SL.hx(s), "packages": sorted(set(ks)), "state_sha": SL.sha("%s|%s" % (cid, SL.hx(s))), **lab}, sort_keys=True) + "\n")
    with open(d / "pairs.jsonl", "w") as f:
        for r in pairs:
            f.write(json.dumps(r, sort_keys=True) + "\n")
    summ = Counter()
    for r in pairs:
        summ[(r["kind"], r["relation"])] += 1
    stats = {"problems": len(per), "unique_states": len(uniq), "new_exact_queries": queries, "pairs": {"%s|%s" % k: v for k, v in sorted(summ.items())}, "label_sources": dict(Counter(l["source"] for l in labels.values())),
             "packages_included": {cid: len(v) for cid, v in included.items()}, "packages_total": {cid: len(p["packages"]) for cid, p in per.items()}, "seconds": round(time.time() - t0, 1)}
    jdump(d / "library_stats.json", stats)
    return stats


def load_library(rr):
    d = Path(rr) / "diagnostics"
    pk = json.loads((d / "packages.json").read_text())
    man = [json.loads(l) for l in (d / "state_manifest.jsonl").read_text().splitlines() if l.strip()]
    pairs = [json.loads(l) for l in (d / "pairs.jsonl").read_text().splitlines() if l.strip()]
    return pk, man, pairs


def score(rr, name, make_eval, tag=None):
    """Scores every library state of every problem with one scorer; writes diagnostics/scores_<name>.json {problem: {hex: value}}."""
    pk, man, _pairs = load_library(rr)
    by = defaultdict(list)
    for m in man:
        by[m["problem"]].append(m["state"])
    out, t0 = {}, time.time()
    ev = make_eval()
    for cid, states in sorted(by.items()):
        task = ST.get_task(DP.DOMAIN_TYPED, pk[cid]["case_file"])
        b = Budget(900, 10 ** 9, 16 * 2 ** 30, 60.0)
        ev.prepare(task, {"domain": str(DP.DOMAIN_TYPED), "problem": pk[cid]["case_file"]}, b)
        vals = ev.evaluate([int(h, 16) for h in states], b)
        out[cid] = dict(zip(states, vals))
        try:
            ev.close()
        except Exception:
            pass
    jdump(Path(rr) / "diagnostics" / ("scores_%s.json" % name), out)
    return {"model": name, "states": sum(len(v) for v in out.values()), "seconds": round(time.time() - t0, 1)}


def _labels(man):
    return {(m["problem"], m["state"]): m for m in man}


def metrics(rr, names):
    """Same-parent decisions, same-parent and cross-parent pair orders, macro-averaged over problems; plus pairwise repair / damage between models."""
    pk, man, pairs = load_library(rr)
    lab = _labels(man)
    sc = {n: json.loads((Path(rr) / "diagnostics" / ("scores_%s.json" % n)).read_text()) for n in names}
    rows_dec, rows_pair = [], []
    decisions = defaultdict(dict)                                     # (model) -> {(problem, package): (correct, regret)}
    for cid, p in pk.items():
        for pkg in p["packages"]:
            succ = [h for h, _a in pkg["successors"]]
            if not succ:
                continue
            labs = [lab[(cid, h)] for h in succ]
            known = all(l["source"] == "EXACT" for l in labs)
            if not known:
                for n in names:
                    rows_dec.append({"model": n, "problem": cid, "family": p["family"], "package": pkg["k"], "kind": pkg["kind"], "n_successors": len(succ), "status": "UNKNOWN"})
                continue
            dist = [l["lower"] for l in labs]
            dmin = min(dist)
            for n in names:
                vals = [sc[n][cid][h] for h in succ]
                best = min(range(len(succ)), key=lambda i: (vals[i], i))
                ties = sum(1 for v in vals if abs(v - vals[best]) <= tol(v, vals[best]))
                correct = dist[best] == dmin
                regret = dist[best] - dmin
                rows_dec.append({"model": n, "problem": cid, "family": p["family"], "package": pkg["k"], "kind": pkg["kind"], "n_successors": len(succ), "status": "OK", "correct": correct, "regret": regret,
                                 "optimal_set_size": sum(1 for d in dist if d == dmin), "tie_size": ties})
                decisions[n][(cid, pkg["k"])] = (correct, regret)
    for r in pairs:
        cid = r["problem"]
        for n in names:
            vx, vy = sc[n][cid][r["x"]], sc[n][cid][r["y"]]
            if r["relation"] == "X_BETTER":
                better, worse = vx, vy
            elif r["relation"] == "Y_BETTER":
                better, worse = vy, vx
            else:
                rows_pair.append({"model": n, "problem": cid, "kind": r["kind"], "relation": r["relation"], "outcome": "UNDECIDED", "x": r["x"], "y": r["y"]})
                continue
            o = "TIE" if abs(better - worse) <= tol(better, worse) else ("CORRECT" if better < worse else "INVERTED")
            lx, ly = lab[(cid, r["x"])], lab[(cid, r["y"])]
            gap = abs(lx["lower"] - ly["lower"]) if lx["source"] == "EXACT" and ly["source"] == "EXACT" else None
            rows_pair.append({"model": n, "problem": cid, "kind": r["kind"], "relation": r["relation"], "outcome": o, "margin": worse - better, "x": r["x"], "y": r["y"], "distance_gap": gap})
    return rows_dec, rows_pair, decisions


def summarise(rows_dec, rows_pair, pk, names):
    """Macro (over problems) and micro summaries per model and kind."""
    out = {}
    for n in names:
        d = [r for r in rows_dec if r["model"] == n]
        s = {"decisions_total": len(d), "decisions_exact": sum(1 for r in d if r["status"] == "OK")}
        byp = defaultdict(list)
        for r in d:
            if r["status"] == "OK":
                byp[r["problem"]].append(r)
        if byp:
            s["action_accuracy_macro"] = sum(sum(x["correct"] for x in v) / len(v) for v in byp.values()) / len(byp)
            s["regret_macro"] = sum(sum(x["regret"] for x in v) / len(v) for v in byp.values()) / len(byp)
            s["problems_with_decisions"] = len(byp)
        for kind in ("same_parent", "cross_parent"):
            pr = [r for r in rows_pair if r["model"] == n and r["kind"] == kind]
            dec = [r for r in pr if r["outcome"] != "UNDECIDED"]
            s[kind + "_pairs"] = len(pr)
            s[kind + "_decidable"] = len(dec)
            s[kind + "_decidable_rate"] = len(dec) / len(pr) if pr else None
            s[kind + "_inverted"] = sum(1 for r in dec if r["outcome"] == "INVERTED")
            s[kind + "_ties"] = sum(1 for r in dec if r["outcome"] == "TIE")
            s[kind + "_correct"] = sum(1 for r in dec if r["outcome"] == "CORRECT")
            byp = defaultdict(list)
            for r in dec:
                byp[r["problem"]].append(r)
            if byp:
                s[kind + "_inversion_rate_macro"] = sum(sum(1 for x in v if x["outcome"] == "INVERTED") / len(v) for v in byp.values()) / len(byp)
                s[kind + "_problems_with_decidable"] = len(byp)
        out[n] = s
    return out


# ------------------------------------------------------------------------------------------------ equivariance check
def emit_problem(name, objects, init, goal):
    by = defaultdict(list)
    for o, t in sorted(objects.items()):
        by[t].append(o)
    lines = ["(define (problem %s) (:domain depots)" % name, "(:objects"]
    for t in sorted(by):
        lines.append("\t%s - %s" % (" ".join(by[t]), t))
    lines.append(")")
    lines.append("(:init")
    for pred, args in init:
        lines.append("\t(%s)" % " ".join((pred,) + tuple(args)))
    lines.append(")")
    lines.append("(:goal (and")
    for pred, args in goal:
        lines.append("\t(%s)" % " ".join((pred,) + tuple(args)))
    lines.append("))")
    lines.append(")")
    return "\n".join(lines) + "\n"


def transformed_problem(text, kind, seed):
    """kind 'rename': permute object names inside each type (deterministic by seed); 'goalperm': reorder the goal atoms. Returns (new text, name map)."""
    p = parse_problem(text)
    rng = random.Random(seed)
    phi = {o: o for o in p.objects}
    goal = list(p.goal)
    if kind == "rename":
        bytype = defaultdict(list)
        for o, t in p.objects.items():
            bytype[t].append(o)
        for t, objs in bytype.items():
            objs = sorted(objs)
            perm = objs[:]
            rng.shuffle(perm)
            for a, b in zip(objs, perm):
                phi[a] = b
    else:
        rng.shuffle(goal)
    obj2 = {phi[o]: t for o, t in p.objects.items()}
    init2 = [(pr, tuple(phi[x] for x in args)) for pr, args in sorted(p.init)]
    goal2 = [(pr, tuple(phi[x] for x in args)) for pr, args in goal]
    return emit_problem(p.name + "-" + kind, obj2, init2, goal2), phi


def map_state(task_a, state, task_b, phi):
    m = 0
    for i, (pred, args) in enumerate(task_a.dyn_atoms):
        if (state >> i) & 1:
            j = task_b.dyn_index[(pred, tuple(phi[x] for x in args))]
            m |= 1 << j
    return m


def equivariance_states(rr, n=12):
    pk, man, _ = load_library(rr)
    return sorted(man, key=lambda m: m["state_sha"])[:n], pk


def equivariance(rr, name, make_eval, n=12, renamings=2):
    """12 hash-selected library states: original scored twice (numerical reference), two type-preserving renamings and one goal permutation. Compares VALUES, never action ids."""
    sel, pk = equivariance_states(rr, n)
    ev = make_eval()
    rows = []
    domain_text = DP.DOMAIN_TYPED.read_text()
    for m in sel:
        cid = m["problem"]
        base_file = pk[cid]["case_file"]
        text = Path(base_file).read_text()
        task_a = ST.get_task(DP.DOMAIN_TYPED, base_file)
        s = int(m["state"], 16)
        variants = [("rename%d" % i, "rename", int(m["state_sha"][:8], 16) + i) for i in range(renamings)] + [("goalperm", "goalperm", int(m["state_sha"][8:16], 16))]
        b = Budget(300, 10 ** 9, 16 * 2 ** 30, 60.0)
        ev.prepare(task_a, {"domain": str(DP.DOMAIN_TYPED), "problem": base_file}, b)
        v0 = ev.evaluate([s], b)[0]
        v0b = ev.evaluate([s], b)[0]
        try:
            ev.close()
        except Exception:
            pass
        for tag, kind, seed in variants:
            ntext, phi = transformed_problem(text, kind, seed)
            task_b = T.StripsTask(domain_text, ntext, type_map=T.DEPOTS_TYPE_MAP, type_preds=T.DEPOTS_TYPE_PREDS, type_order=T.DEPOTS_TYPE_ORDER)
            sb = map_state(task_a, s, task_b, phi)
            with tempfile.NamedTemporaryFile("w", suffix=".pddl", delete=False) as f:
                f.write(ntext)
                path = f.name
            b2 = Budget(300, 10 ** 9, 16 * 2 ** 30, 60.0)
            ev.prepare(task_b, {"domain": str(DP.DOMAIN_TYPED), "problem": path}, b2)
            v1 = ev.evaluate([sb], b2)[0]
            try:
                ev.close()
            except Exception:
                pass
            rows.append({"model": name, "problem": cid, "state": m["state"], "variant": tag, "v_original": v0, "v_repeat": v0b, "v_variant": v1, "abs_repeat_diff": abs(v0 - v0b), "abs_variant_diff": abs(v0 - v1),
                         "goals_equal": sorted(task_a.goal_atoms) == sorted(tuple((pr, tuple(phi[x] for x in a)) for pr, a in task_a.goal_atoms)) if kind == "goalperm" else None})
    return rows
