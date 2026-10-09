"""Shared state library B of card CP-DISR-C1-COMPACT-DIAGNOSIS-V2 (plan 4.B): 64 existing problems (Train32, Struct16, Joint16), packages of 'parent + all legal successors', distance labels with explicit EXACT / UPPER_ONLY / UNKNOWN status,
selected comparison pairs. One library serves the four old models and the final selected new models.

Selection rules (all deterministic, no model score is looked at):
  problems  : Train 11 / 11 / 10 for n3 / n4 / n5, Struct 4 per (n_crates x needs_transport) cell, Joint 2 per cell; always the smallest problem-file sha256 first.
  parents   : reference-trajectory positions 0, L//3, (2L)//3 and the last up to 4 non-terminal positions (unique), then up to 4 off-path non-terminal legal successors of those parents with the smallest state sha256, the total
              capped at 8 per problem (reference positions first).
  packages  : added round-robin over problems in the order [reference positions ascending, off-path by sha]; a package that does not fit into the 4,096 unique-state cap is skipped whole.
  labels    : Train / Struct: exact oracle rebuild (one 'new exact query' per problem); Joint n6: states on the proven-optimal plan are EXACT (a suffix of an optimal plan is optimal), other states UNKNOWN (an oracle build above 1M states is
              refused); Joint n8: states on the best saved plan are UPPER_ONLY, others UNKNOWN.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path

from .. import data as DT
from .. import depots as DP
from .. import stages as ST
from .. import task as T
from ..search_match.engine import SuccessorIndex
from .train import load_old, rj


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def hx(s):
    return format(s, "x")


def select_problems(man, cfg):
    B = cfg["state_library_B"]
    out = []
    byn = defaultdict(list)
    for c in sorted(man["train"], key=lambda c: c["sha256"]):
        byn[c["analysis"]["n_crates"]].append(c)
    for n, k in ((3, B["train32"]["n3"]), (4, B["train32"]["n4"]), (5, B["train32"]["n5"])):
        out += [dict(c, family="train") for c in byn[n][:k]]
    cells = defaultdict(list)
    for c in sorted(man["struct"], key=lambda c: c["sha256"]):
        cells[(c["analysis"]["n_crates"], bool(c["analysis"]["needs_transport"]))].append(c)
    for key in sorted(cells):
        out += [dict(c, family="struct") for c in cells[key][:4]]
    jc = defaultdict(list)
    for c in sorted(man["joint"], key=lambda c: c["sha256"]):
        jc[c["cell"]].append(c)
    for cell in sorted(jc):
        out += [dict(c, family="joint") for c in jc[cell][:B["joint16"]["per_cell"]]]
    return out


def reference_ids(root, c, exact, refs_joint):
    cid = c["case_id"]
    ex = exact.get(cid)
    if ex and ex.get("status") == "OK":
        return ex["plan"], "OPTIMAL"
    r = refs_joint[cid]["lama"]
    return DT.plan_ids_from_file(r["best_plan_file"]), "BEST_SAVED"


def path_seq(task, ids):
    s = task.init_mask
    seq = [s]
    for aid in ids:
        s = task.apply(s, task.action_by_id[aid])
        seq.append(s)
    assert task.goal_satisfied(seq[-1])
    return seq


def parent_positions(L):
    pos = {0, L // 3, (2 * L) // 3} | {L - k for k in range(1, 5)}
    return sorted(p for p in pos if 0 <= p < L)


def build_library(root, cfg):
    man, exact, root = load_old(root)
    refs_joint = rj(root / "data" / "refs_joint.json")
    probs = select_problems(man, cfg)
    B = cfg["state_library_B"]
    per = {}
    for c in probs:
        task = ST.get_task(DP.DOMAIN_TYPED, c["file"])
        index = SuccessorIndex(task)
        ids, kind = reference_ids(root, c, exact, refs_joint)
        seq = path_seq(task, ids)
        L = len(ids)
        ref_pos = parent_positions(L)
        parents = [("ref", p, seq[p]) for p in ref_pos]
        succ_all = {}
        for _k, _p, s in parents:
            for a in index.applicable(s):
                t = task.apply(s, a)
                if t not in succ_all and not task.goal_satisfied(t):
                    succ_all[t] = a.index
        extra = sorted((t for t in succ_all if t not in {s for _k, _p, s in parents}), key=lambda t: sha("%s|%s" % (c["case_id"], hx(t))))
        room = max(0, B["parents_per_problem"] - len(parents))
        parents += [("off", None, t) for t in extra[:min(B["successor_states_per_problem"], room)]]
        pk = []
        for k, p, s in parents:
            succ, seen = [], set()
            for a in index.applicable(s):
                t = task.apply(s, a)
                if t not in seen:
                    seen.add(t)
                    succ.append((t, a.index))
            pk.append({"kind": k, "ref_pos": p, "parent": s, "successors": succ})
        per[c["case_id"]] = {"case": c, "task": task, "ids": ids, "ref_kind": kind, "seq": seq, "packages": pk, "L": L}
    # round-robin package admission under the unique-state cap
    cap = B["max_unique_states"]
    included = {cid: [] for cid in per}
    uniq = set()
    rnd = 0
    while True:
        progressed = False
        for cid, d in per.items():
            if rnd < len(d["packages"]):
                progressed = True
                p = d["packages"][rnd]
                members = {(cid, p["parent"])} | {(cid, t) for t, _a in p["successors"]}
                if len(uniq | members) <= cap:
                    uniq |= members
                    included[cid].append(rnd)
        if not progressed:
            break
        rnd += 1
    return per, included, uniq


def label_library(per, included, root, cfg):
    """Distance bounds per (problem, state). Returns {(cid, state): dict(source, lower, upper)} and the number of new exact queries."""
    labels, queries, notes = {}, 0, {}
    B = cfg["state_library_B"]["new_exact_queries"]
    for cid, d in per.items():
        fam = d["case"]["family"]
        states = set()
        for k in included[cid]:
            p = d["packages"][k]
            states.add(p["parent"])
            states |= {t for t, _a in p["successors"]}
        task, L, seq = d["task"], d["L"], d["seq"]
        pathd = {}
        for k, s in enumerate(seq):
            pathd[s] = L - k                                   # remaining plan length (exact if the plan is optimal)
        oracle = None
        if fam in ("train", "struct") or (fam == "joint" and d["case"]["analysis"]["n_crates"] == 6):
            queries += 1
            assert queries <= B["max"], "new exact query budget exceeded"
            oracle = T.ExactOracle(task, B["search_states"], timeout=B["seconds"])
            notes[cid] = {"oracle_ok": bool(oracle.ok), "oracle_states": oracle.n_states}
        for s in states:
            if oracle is not None and oracle.ok:
                dd, _ = oracle.query(s)
                labels[(cid, s)] = {"source": "EXACT", "lower": dd, "upper": dd, "via": "oracle"}
            elif s in pathd and d["ref_kind"] == "OPTIMAL":
                labels[(cid, s)] = {"source": "EXACT", "lower": pathd[s], "upper": pathd[s], "via": "optimal-plan suffix"}
            elif s in pathd:
                labels[(cid, s)] = {"source": "UPPER_ONLY", "lower": None, "upper": pathd[s], "via": "best saved plan suffix"}
            else:
                labels[(cid, s)] = {"source": "UNKNOWN", "lower": None, "upper": None, "via": None}
        if oracle is not None:
            oracle.close()
    return labels, queries, notes


def relation(lab_x, lab_y):
    """Strict order from bounds: 'X_BETTER' if U(x) < L(y), 'Y_BETTER' if U(y) < L(x), 'EQUAL' for equal exact distances, else 'UNKNOWN'."""
    ux, ly, uy, lx = lab_x["upper"], lab_y["lower"], lab_y["upper"], lab_x["lower"]
    if ux is not None and ly is not None and ux < ly:
        return "X_BETTER"
    if uy is not None and lx is not None and uy < lx:
        return "Y_BETTER"
    if lab_x["source"] == "EXACT" and lab_y["source"] == "EXACT" and lab_x["lower"] == lab_y["lower"]:
        return "EQUAL"
    return "UNKNOWN"


def select_pairs(per, included, labels, cfg):
    B = cfg["state_library_B"]
    same, cross = [], []
    for cid, d in per.items():
        members = []
        mem_sets = []
        for k in included[cid]:
            p = d["packages"][k]
            succ = [t for t, _a in p["successors"]]
            mem_sets.append(set(succ) | {p["parent"]})
            members.append((k, p["parent"], succ))
        for k, par, succ in members:
            for i in range(len(succ)):
                for j in range(i + 1, len(succ)):
                    a, b = sorted((succ[i], succ[j]))
                    same.append((sha("%s|%s|%s|SP" % (cid, hx(a), hx(b))), cid, a, b, k))
        allst = sorted(set().union(*mem_sets)) if mem_sets else []
        cand = []
        for i in range(len(allst)):
            for j in range(i + 1, len(allst)):
                x, y = allst[i], allst[j]
                if any(x in m and y in m for m in mem_sets):
                    continue                                    # shares a package: not a cross-parent pair
                cand.append((sha("%s|%s|%s|CP" % (cid, hx(x), hx(y))), cid, x, y, -1))
        cand.sort()
        cross += cand[:B["cross_parent_pairs_per_problem"]]
    same.sort()
    same = same[:B["max_same_parent_pairs"]]
    cross.sort()
    cross = cross[:B["cross_parent_pairs_total"]]
    rows = []
    for kind, lst in (("same_parent", same), ("cross_parent", cross)):
        for h, cid, x, y, k in lst:
            r = relation(labels[(cid, x)], labels[(cid, y)])
            rows.append({"kind": kind, "problem": cid, "x": hx(x), "y": hx(y), "package": k, "relation": r, "sha": h})
    return rows
