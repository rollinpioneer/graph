#!/usr/bin/env python
"""A1 probe (symbolic only): what does a-m pruning remove? For the first selected candidates (both states in the training set), which unpruned features separate the pair and does an equal-column kept feature exist?"""
import csv
import json
import sys
from collections import Counter

import numpy as np
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator
from wlplan.planning import Atom, Domain, Predicate, Problem, State

OUT = sys.argv[1]
R = "/home/xushijie3/work/graph_cp_disr/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z"
d = json.load(open(R + "/training/W1/data.json"))
preds = {n: Predicate(n, a) for n, a in d["predicates"]}
DOM = Domain("domain", list(preds.values()), [], [], [])
sts, pos = [], {}
dsl = []
for p in d["problems"]:
    st = [State([Atom(preds[n], list(a)) for n, a in s]) for s in p["states"]]
    dsl.append(ProblemDataset(problem=Problem(DOM, p["objects"], [Atom(preds[n], list(a)) for n, a in p["goals"]], []), states=st))
    for i, h in enumerate(p["state_masks_hex"]):
        pos[(p["case_id"], h)] = len(sts) + i
    sts += st
ds = DomainDataset(DOM, dsl)
res = {}
for name, pr in (("am", "a-m"), ("none", "none")):
    g = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=DOM, iterations=2, pruning=pr, multiset_hash=False)
    g.collect(ds)
    res[name] = (g, np.array(g.embed(ds), dtype=np.float64))
Xa, Xn = res["am"][1], res["none"][1]
ga, gn = res["am"][0], res["none"][0]
la = dict(ga.get_colour_to_layer())
ln = dict(gn.get_colour_to_layer())
cols_a = {tuple(Xa[:, i].tolist()): i for i in range(Xa.shape[1])}
rows = list(csv.DictReader(open(OUT + "/a1/candidates.csv")))
out = []
for r in rows[:24]:
    key_b, key_w = (r["problem"], r["better_state"]), (r["problem"], r["worse_state"])
    if key_b not in pos or key_w not in pos:
        out.append({"cand": r["candidate"], "note": "state not in training set"})
        continue
    ib, iw = pos[key_b], pos[key_w]
    dn = np.nonzero(Xn[ib] != Xn[iw])[0]
    da = np.nonzero(Xa[ib] != Xa[iw])[0]
    eq = []
    for j in dn:
        col = tuple(Xn[:, j].tolist())
        eq.append({"feat": int(j), "layer": ln.get(int(j)), "equal_column_kept": col in cols_a, "n_states_nonzero": int((Xn[:, j] != 0).sum())})
    out.append({"cand": r["candidate"], "problem": r["problem"], "diff_unpruned": len(dn), "diff_pruned": len(da), "details": eq})
json.dump(out, open(OUT + "/a1/pruning_probe.json", "w"), indent=1)
print(json.dumps(out[:6])[:3000])
print("kept", Xa.shape[1], "none", Xn.shape[1])
# which unpruned features have an equal column among kept ones?
eqcount = sum(1 for j in range(Xn.shape[1]) if tuple(Xn[:, j].tolist()) in cols_a)
print("unpruned features with equal-column kept feature:", eqcount, "of", Xn.shape[1])
