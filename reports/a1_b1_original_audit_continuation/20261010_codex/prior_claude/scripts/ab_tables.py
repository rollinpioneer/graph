#!/usr/bin/env python3
"""Audit card CP-DISR-A1-B1-ORIGINAL-AUDIT-V1: READ-ONLY table work on already saved files (no feature recomputation, no model, no planner).
Inputs: previous card output (derived tables, hashes recorded in 00_receipt), V2 saved W1 data (states per training problem), SCOPE saved candidate pairs.
Writes A1/loss_pipeline_evidence.csv, A1/zero_diff_836_profile.json, B1/pair_to_search_evidence.csv, B1/pair_audit_summary.json."""
import csv
import gzip
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict

AUD = sys.argv[1]
PREV = open(os.path.expanduser("~/work/rr_a1b1.txt")).read().strip()
REPO = os.path.expanduser("~/work/graph_cp_disr")
V2 = REPO + "/runs/final_master/c1_route_b/compact_diagnosis_v2/20261009T165825Z"
SC = REPO + "/runs/final_master/c1_route_b/search_scope_diagnostic_v1/20261009T063905Z_511b117"

# ---------------------------------------------------------------- A1: what the 836 are
meta = json.load(gzip.open(PREV + "/shared/w1_pair_meta.json.gz", "rt"))
data = json.load(open(V2 + "/training/W1/data.json"))
hexes = {p["case_id"]: p["state_masks_hex"] for p in data["problems"]}
zero = []
with gzip.open(PREV + "/a1/pair_readout_classes.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        if r["model"] == "W1" and r["pool"] == "P1" and r["category"] == "FEATURE_EQUAL":
            zero.append((r["problem"], r["better_state"], r["worse_state"]))
rows = {}
for cid, t in meta["problems"].items():
    idx = {h: i for i, h in enumerate(hexes[cid])}
    for i, j, db, dw, opt, npar, nact, conf in t["pairs"]:
        rows[(cid, hexes[cid][i], hexes[cid][j])] = {"d_better": db, "d_worse": dw, "better_in_opt": opt, "n_parents": npar, "n_action_pairs": nact, "label_conflict": conf}
prof = {"zero_feature_difference_unique_state_pairs": len(zero), "distinct_problems": len({z[0] for z in zero}), "all_in_Train96_typed_track": True, "all_same_problem_same_parent_construction": True}
sel = [rows[z] for z in zero]
prof["action_level_strict_pairs_covered"] = sum(r["n_action_pairs"] for r in sel)
prof["better_in_exact_optimal_set_pairs"] = sum(r["better_in_opt"] for r in sel)
prof["label_conflicts"] = sum(r["label_conflict"] for r in sel)
prof["pairs_arising_from_more_than_one_parent_decision"] = sum(1 for r in sel if r["n_parents"] > 1)
prof["distance_gap_histogram"] = dict(sorted(Counter(r["d_worse"] - r["d_better"] for r in sel).items()))
prof["per_problem_counts_top10"] = Counter(z[0] for z in zero).most_common(10)
prof["total_unique_pairs"] = sum(len(t["pairs"]) for t in meta["problems"].values())
prof["share_of_unique_pairs"] = len(zero) / prof["total_unique_pairs"]
prof["definition"] = "unique ordered (problem, better successor state, worse successor state) pairs of the D0-aligned strict supervision; duplicates of the same pair from several decisions are merged (n_parents>1); action-level count = pairs of actions"
json.dump(prof, open(AUD + "/A1/zero_diff_836_profile.json", "w"), indent=1)

# chain table for the 24 explored candidates (saved previous-card outputs only)
cands = list(csv.DictReader(open(PREV + "/a1/candidates.csv")))
out = []
for c in cands:
    first = c["first_separating_rung"]
    out.append({
        "candidate": c["candidate"], "problem": c["problem"], "problem_sha256": c["problem_sha256"], "better_state": c["better_state"], "worse_state": c["worse_state"], "sampling_stratum": c["sampling_stratum"],
        "source_pools": c["source_pools"], "label_basis": c["label_basis"],
        "L1_pddl_state_and_task": "DIFFERENT (distinct bit-mask states, exact labels differ)",
        "L2_adapter_atoms": "DIFFERENT" if c["atom_sets_identical"] == "False" else "SAME",
        "L3_initial_ilg_colour_histogram": "SAME" if c["pair_vocab_set_separated_it0"] == "False" else "DIFFERENT",
        "L4_wl_colours_unpruned_training_vocab": ("DIFFERENT" if first == "R1x_no_pruning_training_vocabulary" else ("UNKNOWN_NOT_SEPARATED_BY_LADDER" if not first else "DIFFERENT_AFTER_" + first)),
        "L5_after_a-m_projection": "SAME" if c["phi_equal_W1"] == "True" else "DIFFERENT",
        "L6_readout_scores_raw": "SAME" if c["raw_better_W1"] == c["raw_worse_W1"] else "DIFFERENT",
        "L7_deployed_rounded_score": "SAME" if c["round_better_W1"] == c["round_worse_W1"] else "DIFFERENT",
        "L8_search_comparison": "UNKNOWN (no recorded search event for this pair)",
        "graph_isomorphism_checked": c.get("iso_status", "") if first == "" else "NOT_RUN (separated earlier)",
        "tag": "ORIGINAL_GRAPH_ISOMORPHISM_UNKNOWN_EXCEPT_CANDIDATE_16",
        "evidence_origin": "previous-card A1 run (frozen output, sha in 00_receipt); not recomputed here"})
cols = list(out[0])
with open(AUD + "/A1/loss_pipeline_evidence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(out)

# ---------------------------------------------------------------- B1: the 148 audited D0->T1 pairs
idxB = json.load(gzip.open(PREV + "/shared/b_index.json.gz", "rt"))
LAB = {(m["problem"], m["state"]): m for m in idxB["states"]}
scope = list(csv.DictReader(open(SC + "/results/candidate_pairs_all.csv")))
sc_states = defaultdict(set)
for r in scope:
    sc_states[r["case_id"]] |= {r["better_hash"], r["worse_hash"], r["actual_popped"]}
sc_pairs = {(r["case_id"], frozenset((r["better_hash"], r["worse_hash"]))) for r in scope}
train_states = {cid: set(h) for cid, h in hexes.items()}
audited = {}
with gzip.open(PREV + "/b1/pair_changes.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        if r["model_a"] == "D0" and r["model_b"] == "T1" and r["transition_diag"] in ("INV>COR", "COR>INV"):
            k = (r["kind"], r["problem"], r["better_state"], r["worse_state"])
            audited.setdefault(k, r)
sv = {}
with gzip.open(V2 + "/diagnostics/b_pair_outcomes.csv.gz", "rt") as f:
    for r in csv.DictReader(f):
        if r["model"] in ("D0", "T1"):
            sv[(r["model"], r["problem"], r["x"], r["y"], r["kind"])] = r["outcome"]
rows_b = []
for (kind, cid, b, w), r in sorted(audited.items()):
    fam = idxB["packages"][cid]["family"]
    lb, lw = LAB[(cid, b)], LAB[(cid, w)]
    in_tr = cid in train_states and b in train_states[cid] and w in train_states[cid]
    seen_scope = (b in sc_states.get(cid, ())) or (w in sc_states.get(cid, ()))
    rows_b.append({"pair_id": hashlib.sha256(("%s|%s|%s|%s" % (kind, cid, b, w)).encode()).hexdigest()[:16], "kind": kind, "problem": cid, "family": fam, "better_state": b, "worse_state": w, "transition_D0_to_T1": r["transition_diag"],
                   "label_basis": r["label_basis"], "L_better": lb["lower"], "U_better": lb["upper"], "L_worse": lw["lower"], "U_worse": lw["upper"], "distance_gap": r["distance_gap"],
                   "D0_margin": r["a_margin"], "T1_margin": r["b_margin"], "tie_involved": r["a_outcome_diag"] == "TIE" or r["b_outcome_diag"] == "TIE",
                   "both_states_are_Train96_training_states_of_this_problem": in_tr, "problem_role": "Train96 (in-sample data role)" if cid.startswith("train_") else ("Struct (diagnostic role)" if cid.startswith("struct_") else "Joint (diagnostic role)"),
                   "evidence_level": "E0", "E1_same_run_state_record": "NONE for D0/T1 (no trajectories); old DENSE/WL Scope sample only" + (" (a state occurs in a Scope snapshot of an OLD model)" if seen_scope else ""),
                   "pair_recorded_in_scope_candidate_pairs": (cid, frozenset((b, w))) in sc_pairs, "mapped_search_event": "NONE (task/state/model/checkpoint/protocol key not matched)"})
cols = list(rows_b[0])
with open(AUD + "/B1/pair_to_search_evidence.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(rows_b)
sm = {"audited_unique_pairs": len(rows_b), "repairs_INV_to_COR": sum(1 for r in rows_b if r["transition_D0_to_T1"] == "INV>COR"), "damages_COR_to_INV": sum(1 for r in rows_b if r["transition_D0_to_T1"] == "COR>INV"),
      "pair_ids_unique": len({r["pair_id"] for r in rows_b}), "ties_involved": sum(1 for r in rows_b if r["tie_involved"]),
      "by_kind_transition": {"%s|%s" % k: v for k, v in sorted(Counter((r["kind"], r["transition_D0_to_T1"]) for r in rows_b).items())},
      "by_family_transition": {"%s|%s" % k: v for k, v in sorted(Counter((r["family"], r["transition_D0_to_T1"]) for r in rows_b).items())},
      "both_states_in_Train96_training_states": {"%s" % t: sum(1 for r in rows_b if r["transition_D0_to_T1"] == t and r["both_states_are_Train96_training_states_of_this_problem"]) for t in ("INV>COR", "COR>INV")},
      "label_basis": dict(Counter(r["label_basis"] for r in rows_b)),
      "distance_gap_hist_repairs": dict(sorted(Counter(int(r["distance_gap"]) for r in rows_b if r["transition_D0_to_T1"] == "INV>COR" and r["distance_gap"] != "").items())),
      "distance_gap_hist_damages": dict(sorted(Counter(int(r["distance_gap"]) for r in rows_b if r["transition_D0_to_T1"] == "COR>INV" and r["distance_gap"] != "").items())),
      "pairs_with_any_scope_record": sum(1 for r in rows_b if r["pair_recorded_in_scope_candidate_pairs"]),
      "pairs_with_E1_or_higher_for_D0_T1": 0}
# saved-outcome cross check
bad = 0
for r in rows_b:
    cid, b, w = r["problem"], r["better_state"], r["worse_state"]
    for key in ((cid, b, w), (cid, w, b)):
        a, bb = sv.get(("D0", key[0], key[1], key[2], r["kind"])), sv.get(("T1", key[0], key[1], key[2], r["kind"]))
        if a is not None and bb is not None:
            exp = r["transition_D0_to_T1"].split(">")
            if {"COR": "CORRECT", "INV": "INVERTED"}[exp[0]] != a or {"COR": "CORRECT", "INV": "INVERTED"}[exp[1]] != bb:
                bad += 1
            break
sm["saved_b_pair_outcomes_recheck_mismatches"] = bad
json.dump(sm, open(AUD + "/B1/pair_audit_summary.json", "w"), indent=1)
print(json.dumps(prof, indent=1)[:1500])
print(json.dumps(sm, indent=1)[:2500])
