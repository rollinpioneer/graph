#!/usr/bin/env python3
"""B-E (read-only): is each audited D0->T1 pair constructible by the published Opt Rank constraint set (Hao et al., IJCAI 2024: parent-child + sibling constraints along an OPTIMAL plan, cross-path by transitivity)?
Only saved library-B packages are used (reference-plan parents with ref_pos and all their successors). No state is generated, no search, no replay. Result is a coarse classification with explicit caveats."""
import csv
import gzip
import json
import os
import sys
from collections import Counter, defaultdict

AUD = sys.argv[1]
PREV = open(os.path.expanduser("~/work/rr_a1b1.txt")).read().strip()
idx = json.load(gzip.open(PREV + "/shared/b_index.json.gz", "rt"))
LAB = {(m["problem"], m["state"]): m for m in idx["states"]}
PK = idx["packages"]
info = {}
kinds = Counter()
refkinds = Counter()
for cid, p in PK.items():
    refkinds[p["reference_kind"] if "reference_kind" in p else "NA"] += 1
    d = {}
    for pg in p["packages"]:
        kinds[pg["kind"]] += 1
        labs = [LAB[(cid, h)] for h, _a in pg["successors"]]
        exact = all(l["source"] == "EXACT" for l in labs)
        dmin = min((l["lower"] for l in labs if l["lower"] is not None), default=None) if exact else None
        for h, _a in pg["successors"]:
            d.setdefault(h, []).append({"kind": pg["kind"], "ref_pos": pg["ref_pos"], "pk": pg["k"], "exact_pkg": exact, "is_opt_succ": exact and LAB[(cid, h)]["lower"] == dmin})
    info[cid] = d
rows = list(csv.DictReader(open(AUD + "/B1/pair_to_search_evidence.csv")))
out = []
for r in rows:
    cid, b, w = r["problem"], r["better_state"], r["worse_state"]
    ref_kind = PK[cid].get("reference_kind", "NA")
    cb = [e for e in info[cid].get(b, []) if e["kind"] == "ref"]
    cw = [e for e in info[cid].get(w, []) if e["kind"] == "ref"]
    status, note = "NOT_COVERED_BY_PLAN_ONLY_RULES", ""
    if not cb:
        status, note = "NOT_COVERED_BY_PLAN_ONLY_RULES", "better state is not a successor of a reference-plan parent in the library"
    elif not any(e["exact_pkg"] for e in cb):
        status, note = "NOT_DETERMINABLE", "package labels incomplete"
    elif not any(e["is_opt_succ"] for e in cb):
        status, note = "NOT_COVERED_BY_PLAN_ONLY_RULES", "better state is not an optimal successor (plan successor candidate) of its reference parent"
    elif r["kind"] == "same_parent":
        status, note = "COVERED_SIBLING_CONSTRAINT", "better = optimal successor of a reference-plan parent, worse = its sibling (Hao s3/s4.3 sibling constraint; the specific plan successor is not identified)"
    else:
        pa = min(e["ref_pos"] for e in cb if e["is_opt_succ"])
        if cw and min(e["ref_pos"] for e in cw) <= max(e["ref_pos"] for e in cb if e["is_opt_succ"]):
            status, note = "COVERED_BY_TRANSITIVITY_IF_PLAN_OPTIMAL", "worse is a successor of an earlier-or-equal reference parent (cross-path by transitivity, Hao s3)"
        else:
            status, note = "NOT_COVERED_BY_PLAN_ONLY_RULES", "worse state is not a successor of an earlier-or-equal reference-plan parent"
    out.append(dict(r, optrank_status=status, optrank_note=note, reference_kind=ref_kind))
with open(AUD + "/B1/pair_to_search_evidence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(out[0]))
    w.writeheader()
    w.writerows(out)
sm = {"package_kinds": dict(kinds), "reference_kinds": dict(refkinds),
      "status_by_kind_and_transition": {"%s|%s|%s" % k: v for k, v in sorted(Counter((x["kind"], x["transition_D0_to_T1"], x["optrank_status"]) for x in out).items())},
      "caveat": "coarse: reference plans in library B are the ones recorded in packages.json; a pair marked COVERED is constructible by the Opt Rank rule only if that reference plan is optimal and the plan successor is the stated one; T1-S pairs were drawn from Train96 exact labels, not from this rule"}
json.dump(sm, open(AUD + "/B1/optrank_coverage_summary.json", "w"), indent=1)
print(json.dumps(sm, indent=1))
