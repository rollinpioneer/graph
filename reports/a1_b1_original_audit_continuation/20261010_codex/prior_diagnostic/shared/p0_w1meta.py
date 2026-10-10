#!/usr/bin/env python
"""P0 (shared, cpdisr env): provenance of the W1 training pairs. Re-derives, from the saved Train96 exact label cache (no query, no search), parent / exact distances / optimal-set membership of every unique state pair of
training/W1/data.json, and checks that the re-derived pair set equals the saved one. Output: <OUT>/shared/w1_pair_meta.json.gz"""
import gzip
import json
import os
import sys
import time
from collections import defaultdict

REPO = os.path.expanduser("~/work/graph_cp_disr")
sys.path.insert(0, REPO + "/src")
sys.dont_write_bytecode = True
OUT = sys.argv[1]
t0, cpu0 = time.time(), time.process_time()
from cp_disr.pddl import depots as DP, stages as ST  # noqa: E402

V2 = REPO + "/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z"
OLD = REPO + "/runs/final_master/c1_route_b/public_depots_rel_v1/20261008T063544Z_9898106e"
data = json.load(open(V2 + "/training/W1/data.json"))
man = json.load(open(OLD + "/data/manifest.json"))
exact = json.load(open(OLD + "/data/exact.json"))
files = {c["case_id"]: c for c in man["train"]}
res, tot = {}, defaultdict(int)
for p in data["problems"]:
    cid = p["case_id"]
    c = files[cid]
    task = ST.get_task(DP.DOMAIN_TYPED, c["file"])
    ex = exact[cid]
    hexes = p["state_masks_hex"]
    index = {int(h, 16): i for i, h in enumerate(hexes)}
    wdec = defaultdict(float)
    for t in ex["trajectories"]:
        n = len(t["actions"])
        for s in t["states"][:-1]:
            wdec[",".join(map(str, s))] += 1.0 / n
    pairs = {}
    for k, w in wdec.items():
        dist = ex["rank_labels"]["%s|%s" % (cid, k)]
        parent = task.state_from_list([int(x) for x in k.split(",")]) if k else 0
        succ = {aid: task.apply(parent, task.action_by_id[aid]) for aid in dist}
        dmin = min(dist.values())
        strict = [(a, b) for a in dist for b in dist if dist[a] < dist[b]]
        for a, b in strict:
            if succ[a] == succ[b]:
                continue
            key = (index[succ[a]], index[succ[b]])
            r = pairs.setdefault(key, {"parents": [], "d_better": dist[a], "d_worse": dist[b], "better_in_opt": False, "n_action_pairs": 0})
            r["parents"].append(k)
            r["better_in_opt"] = r["better_in_opt"] or dist[a] == dmin
            r["n_action_pairs"] += 1
            if r["d_better"] != dist[a] or r["d_worse"] != dist[b]:
                r.setdefault("conflict", True)
    saved = {(i, j) for i, j, _w in p["pairs"]}
    res[cid] = {"pairs": [[i, j, v["d_better"], v["d_worse"], int(v["better_in_opt"]), len(v["parents"]), v["n_action_pairs"], int(v.get("conflict", False))] for (i, j), v in sorted(pairs.items())],
                "same_set_as_saved": saved == set(pairs), "n_saved": len(saved), "n_rederived": len(pairs)}
    tot["problems"] += 1
    tot["saved_pairs"] += len(saved)
    tot["rederived_pairs"] += len(pairs)
    tot["problems_set_equal"] += int(saved == set(pairs))
    tot["action_pairs"] += sum(v["n_action_pairs"] for v in pairs.values())
print(dict(tot))
with gzip.open(OUT + "/shared/w1_pair_meta.json.gz", "wt") as f:
    json.dump({"fields": ["i_better", "j_worse", "d_better", "d_worse", "better_in_optimal_set", "n_parents", "n_action_pairs", "label_conflict"], "problems": res, "totals": dict(tot)}, f)
json.dump({"cpu_seconds": time.process_time() - cpu0, "wall_seconds": time.time() - t0}, open(OUT + "/shared/p0_w1meta_ledger.json", "w"))
