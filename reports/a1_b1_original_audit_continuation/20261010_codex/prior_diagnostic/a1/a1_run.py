#!/usr/bin/env python
"""A1: where does the fixed WL pipeline lose information? (goose env, wlplan 2.2.0; symbolic only: no fit, no search, no neural forward.)
Reads the shared read-only index of <OUT>/shared and writes only into <OUT>/a1."""
import csv
import gzip
import hashlib
import json
import math
import os
import signal
import sys
import time
from collections import Counter, defaultdict

import numpy as np
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator, load_feature_generator, get_available_pruning_methods
from wlplan.planning import Atom, Domain, Predicate, Problem, State

OUT = sys.argv[1]
A1 = OUT + "/a1"
REPO = os.path.expanduser("~/work/graph_cp_disr")
R = REPO + "/runs/final_master/c1_route_b/"
V2 = R + "compact_diagnosis_v2/20261009T165825Z"
OLD = R + "public_depots_rel_v1/20261008T063544Z_9898106e"
MODELS = {"W1": V2 + "/training/W1/wl_goose_w1.model.params", "WL": OLD + "/goose/train_typed/wl_goose.model.params"}
TOL_ABS, TOL_REL = 1e-5, 1e-6
T0, CPU0 = time.time(), time.process_time()


def tol(a, b):
    return TOL_ABS + TOL_REL * max(abs(a), abs(b))


def cpp_round(x):
    return int(math.floor(x + 0.5)) if x >= 0 else -int(math.floor(-x + 0.5))


def sha(s):
    return hashlib.sha256(s.encode()).hexdigest()


def sha_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


# ------------------------------------------------------------------ inputs
data = json.load(open(V2 + "/training/W1/data.json"))
idx = json.load(gzip.open(OUT + "/shared/b_index.json.gz", "rt"))
atoms_b = json.load(gzip.open(OUT + "/shared/b_atoms.json.gz", "rt"))
meta = json.load(gzip.open(OUT + "/shared/w1_pair_meta.json.gz", "rt"))
old_man = json.load(open(OLD + "/data/manifest.json"))
train_sha = {c["case_id"]: c.get("sha256") for c in old_man["train"]}
panels = json.load(open(V2 + "/registration/panels.json"))
preds = {n: Predicate(n, a) for n, a in data["predicates"]}
assert [[n, a] for n, a in data["predicates"]] == atoms_b["predicates"]
DOM = Domain("domain", list(preds.values()), [], [], [])
GEN = {k: load_feature_generator(p) for k, p in MODELS.items()}
WGT = {k: np.array(g.get_weights(), dtype=np.float64) for k, g in GEN.items()}
LAYER = {k: {int(c): int(l) for c, l in json.load(open(p))["colour_to_layer"]} for k, p in MODELS.items()}
print("pruning methods", get_available_pruning_methods(), {k: (len(w), int((w != 0).sum())) for k, w in WGT.items()}, flush=True)


def mk_state(atoms):
    return State([Atom(preds[n], list(a)) for n, a in atoms])


def mk_problem(objects, goals):
    return Problem(DOM, objects, [Atom(preds[n], list(a)) for n, a in goals], [])


def embed(gen, objects, goals, state_atoms):
    ds = DomainDataset(DOM, [ProblemDataset(problem=mk_problem(objects, goals), states=[mk_state(a) for a in state_atoms])])
    return np.array(gen.embed(ds), dtype=np.float64)


# ------------------------------------------------------------------ 1. fixed features of every saved state
TRAIN = {}                      # cid -> dict(objects, goals, hex[], atoms[], X{W1,WL})
for p in data["problems"]:
    X = {k: embed(GEN[k], p["objects"], p["goals"], p["states"]) for k in GEN}
    TRAIN[p["case_id"]] = {"objects": p["objects"], "goals": p["goals"], "hex": p["state_masks_hex"], "atoms": p["states"], "X": X, "pairs": p["pairs"]}
BST = {}
for cid, pr in atoms_b["problems"].items():
    hs = sorted(pr["states"])
    BST[cid] = {"objects": pr["objects"], "goals": pr["goals"], "hex": hs, "atoms": [pr["states"][h] for h in hs]}
    BST[cid]["X"] = {k: embed(GEN[k], pr["objects"], pr["goals"], BST[cid]["atoms"]) for k in GEN}
print("features computed", flush=True)
STAT = json.load(open(OUT + "/shared/statics.json"))
TRAIN_DS = DomainDataset(DOM, [ProblemDataset(problem=mk_problem(t["objects"], t["goals"]), states=[mk_state(a) for a in t["atoms"]]) for t in TRAIN.values()])
# rebuild of the actual pipeline's vocabulary (consistency, not a relaxation) and the same vocabulary without pruning
g_am = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=DOM, iterations=2, pruning="a-m", multiset_hash=False)
g_am.collect(TRAIN_DS)
X_am = np.array(g_am.embed(TRAIN_DS), dtype=np.float64)
X_loaded = np.vstack([TRAIN[c]["X"]["W1"] for c in TRAIN])
G_NONE = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=DOM, iterations=2, pruning="none", multiset_hash=False)
G_NONE.collect(TRAIN_DS)
rebuild = {"rebuilt_a-m_features": int(g_am.get_n_features()), "loaded_W1_features": int(GEN["W1"].get_n_features()), "rebuilt_embedding_equals_loaded": bool(X_am.shape == X_loaded.shape and np.array_equal(X_am, X_loaded)),
           "train_vocabulary_without_pruning_features": int(G_NONE.get_n_features()), "train_states": int(X_am.shape[0]),
           "distinct_vectors_a-m": len({r.astype(np.float32).tobytes() for r in X_am}),
           "features_by_layer_a-m": dict(Counter(LAYER["W1"].values())), "features_by_layer_no_pruning_training_vocabulary": dict(Counter(dict(G_NONE.get_colour_to_layer()).values())), "nonzero_weight_features_by_layer_W1": dict(Counter(LAYER["W1"][i] for i in np.nonzero(WGT["W1"])[0])),
           "nonzero_weight_features_by_layer_WL": dict(Counter(LAYER["WL"][i] for i in np.nonzero(WGT["WL"])[0]))}
json.dump(rebuild, open(A1 + "/pipeline_rebuild_check.json", "w"), indent=1)
print(rebuild, flush=True)

# consistency: the saved W1 fit statistics are reproduced from our own embeddings
rep = json.load(open(V2 + "/training/W1/fit_report.json"))
cls = {}
for k in GEN:
    d2c = {}
    for cid, t in TRAIN.items():
        X = t["X"][k]
        for i, j, w in t["pairs"]:
            D = X[j] - X[i]
            key = D.astype(np.float32).tobytes()
            e = d2c.setdefault(key, {"n": 0, "mass": 0.0, "D": D, "cases": Counter()})
            e["n"] += 1
            e["mass"] += w
            e["cases"][cid] += 1
    cls[k] = d2c
w1c = cls["W1"]
nz = [e for e in w1c.values() if np.any(e["D"] != 0)]
mass_nz = sum(e["mass"] for e in nz)
acc = sum(e["mass"] * (float(e["D"] @ WGT["W1"]) > 0) for e in nz) / mass_nz
consistency = {"W1_classes_total": len(w1c), "W1_classes_nonzero": len(nz), "W1_zero_classes": len(w1c) - len(nz), "saved_unique_nonzero": rep["pairs_unique_nonzero"], "saved_identical_dropped": rep["pairs_identical_features_dropped"],
               "weighted_accuracy_recomputed": acc, "weighted_accuracy_saved": rep["train_pair_accuracy_weighted"], "nonzero_weights_here": int((WGT["W1"] != 0).sum()), "nonzero_weights_saved": rep["weights_nonzero"],
               "mass_before_rescale_recomputed": mass_nz, "mass_before_rescale_saved": rep["weight_mass_before_rescale"]}
json.dump(consistency, open(A1 + "/consistency_w1_fit.json", "w"), indent=1)
print(consistency, flush=True)

# ------------------------------------------------------------------ 2. zero-difference tracing on the training pairs
rows_cls = []
for k, d2c in cls.items():
    for n, (key, e) in enumerate(sorted(d2c.items(), key=lambda kv: -kv[1]["mass"])):
        D = e["D"]
        sup = np.nonzero(D)[0]
        rows_cls.append({"model": k, "class": n, "state_pairs": e["n"], "weight_mass": round(e["mass"], 6), "zero_difference": int(len(sup) == 0), "support_size": len(sup), "support_layers": sorted({LAYER[k][int(i)] for i in sup}),
                         "support_nonzero_weight_features": int(sum(WGT[k][int(i)] != 0 for i in sup)), "readout_delta": float(D @ WGT[k]), "problems": len(e["cases"])})
with open(A1 + "/diff_classes_train.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows_cls[0]))
    w.writeheader()
    w.writerows(rows_cls)
tot_pairs = sum(len(t["pairs"]) for t in TRAIN.values())
tot_mass = sum(w for t in TRAIN.values() for _i, _j, w in t["pairs"])
act_pairs = sum(r[6] for t in meta["problems"].values() for r in t["pairs"])
zero_summary = {}
for k, d2c in cls.items():
    z = [e for e in d2c.values() if not np.any(e["D"] != 0)]
    zero_summary[k] = {"unique_state_pairs": tot_pairs, "action_level_strict_pairs": act_pairs, "weight_mass_total": tot_mass, "difference_vector_classes": len(d2c), "zero_classes": len(z),
                       "zero_state_pairs": sum(e["n"] for e in z), "zero_weight_mass": sum(e["mass"] for e in z), "zero_mass_share": sum(e["mass"] for e in z) / tot_mass,
                       "nonzero_weight_features": int((WGT[k] != 0).sum()), "features": int(len(WGT[k]))}
json.dump(zero_summary, open(A1 + "/zero_difference_summary_train.json", "w"), indent=1)
print(json.dumps(zero_summary), flush=True)


# ------------------------------------------------------------------ 3. read-out classification of every pair (train strict pairs and library B decided pairs)
def category(Xb, Xw, w):
    """Xb/Xw: feature vectors of the label-better and label-worse state."""
    D = Xw - Xb
    sb, sw = float(Xb @ w), float(Xw @ w)
    if not np.any(D != 0):
        return "FEATURE_EQUAL", sb, sw
    if sb == sw or abs(sw - sb) <= 1e-9:
        return "FEATURES_DIFFER_READOUT_EXACT_EQUAL", sb, sw
    if abs(sw - sb) <= tol(sb, sw):
        return "FEATURES_DIFFER_READOUT_NEAR_EQUAL", sb, sw
    rb, rw = cpp_round(sb), cpp_round(sw)
    if rb == rw:
        return "RAW_DISTINCT_ROUNDED_EQUAL", sb, sw
    return ("ROUNDED_CORRECT" if rb < rw else "ROUNDED_INVERTED"), sb, sw


pair_rows = []
cat_count = defaultdict(Counter)
for k in GEN:
    for cid, t in TRAIN.items():
        for i, j, wgt in t["pairs"]:
            c, sb, sw = category(t["X"][k][i], t["X"][k][j], WGT[k])
            cat_count[(k, "train_strict")][c] += 1
            pair_rows.append((k, "P1", cid, "train_strict", t["hex"][i], t["hex"][j], c, sb, sw))
state_pos = {cid: {h: n for n, h in enumerate(t["hex"])} for cid, t in BST.items()}
sample_mismatch = Counter()
for k in GEN:
    for cid, t in BST.items():
        sc = idx["scores"][k][cid]
        for n, h in enumerate(t["hex"]):
            raw = float(t["X"][k][n] @ WGT[k])
            sample_mismatch[(k, "match" if cpp_round(raw) == int(round(sc[h])) else "mismatch")] += 1
    for r in idx["pairs"]:
        if r["relation"] not in ("X_BETTER", "Y_BETTER"):
            continue
        cid = r["problem"]
        t = BST[cid]
        b, w_ = (r["x"], r["y"]) if r["relation"] == "X_BETTER" else (r["y"], r["x"])
        ib, iw = state_pos[cid][b], state_pos[cid][w_]
        c, sb, sw = category(t["X"][k][ib], t["X"][k][iw], WGT[k])
        cat_count[(k, "B_%s" % r["kind"])][c] += 1
        pair_rows.append((k, "P2", cid, "B_" + r["kind"], b, w_, c, sb, sw))
with gzip.open(A1 + "/pair_readout_classes.csv.gz", "wt", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["model", "pool", "problem", "pair_kind", "better_state", "worse_state", "category", "raw_better", "raw_worse"])
    wr.writerows(pair_rows)
tc = {"%s|%s" % k: dict(v) for k, v in cat_count.items()}
tc["score_reproduction_round_vs_saved"] = {"%s|%s" % k: v for k, v in sample_mismatch.items()}
json.dump(tc, open(A1 + "/readout_categories.json", "w"), indent=1, sort_keys=True)
print(json.dumps(tc, sort_keys=True), flush=True)

with gzip.open(A1 + "/b_state_features.csv.gz", "wt", newline="") as f:
    wr = csv.writer(f)
    wr.writerow(["problem", "state", "phi_hash_W1", "phi_hash_WL", "raw_W1", "raw_WL", "saved_W1", "saved_WL"])
    for cid, t_ in BST.items():
        for n_, h_ in enumerate(t_["hex"]):
            wr.writerow([cid, h_, hashlib.sha1(t_["X"]["W1"][n_].astype(np.float32).tobytes()).hexdigest()[:16], hashlib.sha1(t_["X"]["WL"][n_].astype(np.float32).tobytes()).hexdigest()[:16],
                         repr(float(t_["X"]["W1"][n_] @ WGT["W1"])), repr(float(t_["X"]["WL"][n_] @ WGT["WL"])), idx["scores"]["W1"][cid][h_], idx["scores"]["WL"][cid][h_]])
# population: distinct fixed feature vectors per problem
pop = []
for pool, S in (("train", TRAIN), ("B", BST)):
    for k in GEN:
        n_states = n_distinct = in_shared = 0
        for cid, t in S.items():
            X = t["X"][k]
            keys = Counter(r.astype(np.float32).tobytes() for r in X)
            n_states += len(X)
            n_distinct += len(keys)
            in_shared += sum(v for v in keys.values() if v > 1)
        pop.append({"pool": pool, "model": k, "states": n_states, "distinct_feature_vectors": n_distinct, "states_sharing_vector": in_shared})
atom_ident = 0
for cid, t in BST.items():
    keys = Counter(json.dumps(sorted(a)) for a in t["atoms"])
    atom_ident += sum(v for v in keys.values() if v > 1)
json.dump({"population": pop, "B_states_with_identical_atom_set_to_another_state": atom_ident}, open(A1 + "/population_distinct_vectors.json", "w"), indent=1)
print(pop, atom_ident, flush=True)

# ------------------------------------------------------------------ 4. candidate pairs
panel_joint = set(panels.get("joint32", []))
state_lab = {(m["problem"], m["state"]): m for m in idx["states"]}
pk = idx["packages"]
univ = {}


def add(cid, b, w_, src, info):
    key = (cid, frozenset((b, w_)))
    e = univ.setdefault(key, {"problem": cid, "better": b, "worse": w_, "sources": [], "info": {}})
    e["sources"].append(src)
    e["info"].update(info)


for cid, t in TRAIN.items():
    X = t["X"]["W1"]
    mrows = {(r[0], r[1]): r for r in meta["problems"][cid]["pairs"]}
    for i, j, _w in t["pairs"]:
        if np.any(X[j] != X[i]):
            continue
        r = mrows[(i, j)]
        add(cid, t["hex"][i], t["hex"][j], "P1", {"label_basis": "EXACT_TRAIN_CACHE", "d_better": r[2], "d_worse": r[3], "better_in_opt": bool(r[4]), "same_parent": True, "n_parents": r[5], "kind": "train_strict"})
pkg_index = {}
for cid, p in pk.items():
    for pg in p["packages"]:
        pkg_index[(cid, pg["k"])] = pg
for r in idx["pairs"]:
    if r["relation"] not in ("X_BETTER", "Y_BETTER"):
        continue
    cid = r["problem"]
    t = BST[cid]
    b, w_ = (r["x"], r["y"]) if r["relation"] == "X_BETTER" else (r["y"], r["x"])
    if np.any(t["X"]["W1"][state_pos[cid][b]] != t["X"]["W1"][state_pos[cid][w_]]):
        continue
    lb, lw = state_lab[(cid, b)], state_lab[(cid, w_)]
    in_opt = False
    if r["kind"] == "same_parent":
        pg = pkg_index[(cid, r["package"])]
        labs = [state_lab[(cid, h)] for h, _a in pg["successors"]]
        if all(l["source"] == "EXACT" for l in labs):
            dmin = min(l["lower"] for l in labs)
            in_opt = lb["lower"] == dmin and lw["lower"] > dmin
    add(cid, b, w_, "P2", {"label_basis": "B_%s_%s" % (lb["source"], lw["source"]), "L_better": lb["lower"], "U_better": lb["upper"], "L_worse": lw["lower"], "U_worse": lw["upper"], "same_parent": r["kind"] == "same_parent",
                           "better_in_opt_B": in_opt, "kind": "B_" + r["kind"], "package": r["package"]})
for e in univ.values():
    e["S1"] = bool(e["info"].get("better_in_opt")) or bool(e["info"].get("better_in_opt_B"))
    e["sha"] = sha("A1|%s|%s|%s" % (e["problem"], e["better"], e["worse"]))
S1 = sorted((e for e in univ.values() if e["S1"]), key=lambda e: e["sha"])
S2 = sorted((e for e in univ.values() if not e["S1"]), key=lambda e: e["sha"])
sel = [dict(e, stratum="S1") for e in S1[:12]]
sel += [dict(e, stratum="S2") for e in (S2[:24 - len(sel)])]
# if S1 has fewer than 12, the remaining slots are filled from S2 as registered; if S1 has more, the surplus stays unsampled
sel_summary = {"universe": len(univ), "S1_total": len(S1), "S2_total": len(S2), "selected": len(sel), "selected_S1": sum(1 for e in sel if e["stratum"] == "S1"), "selected_S2": sum(1 for e in sel if e["stratum"] == "S2"),
               "universe_by_source": dict(Counter("+".join(sorted(set(e["sources"]))) for e in univ.values())), "universe_by_problem_count": len({e["problem"] for e in univ.values()})}
json.dump(sel_summary, open(A1 + "/candidate_selection_summary.json", "w"), indent=1)
print(sel_summary, flush=True)


# ------------------------------------------------------------------ 5/6/7. layered localisation, nested relaxation ladder, isomorphism
class Timeout(Exception):
    pass


def _alarm(_s, _f):
    raise Timeout()


def atoms_of(cid, h):
    if cid in TRAIN and h in TRAIN[cid]["hex"]:
        return TRAIN[cid]["atoms"][TRAIN[cid]["hex"].index(h)], TRAIN[cid]["objects"], TRAIN[cid]["goals"]
    t = BST[cid]
    return t["atoms"][t["hex"].index(h)], t["objects"], t["goals"]


def feat(cid, h, k):
    if cid in TRAIN and h in TRAIN[cid]["hex"]:
        return TRAIN[cid]["X"][k][TRAIN[cid]["hex"].index(h)]
    return BST[cid]["X"][k][BST[cid]["hex"].index(h)]


def mk_problem_s(objects, goals, statics):
    return Problem(DOM, objects, [Atom(preds[n], list(a)) for n, a in goals], [Atom(preds[n], list(a)) for n, a in statics])


def fresh(iterations, pruning, multiset, objects, goals, atoms_list, statics=()):
    g = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=DOM, iterations=iterations, pruning=pruning, multiset_hash=multiset)
    ds = DomainDataset(DOM, [ProblemDataset(problem=mk_problem_s(objects, goals, statics), states=[mk_state(a) for a in atoms_list])])
    g.collect(ds)
    return g, ds, np.array(g.embed(ds), dtype=np.float64)


OTYPES = json.load(open(OUT + "/shared/object_types.json"))
TYPE_PREDS = {"type_" + t: Predicate("type_" + t, 1) for t in sorted({t for d in OTYPES.values() for t in d.values()})}
DOM_ALL = Domain("domain", list(preds.values()) + list(TYPE_PREDS.values()), [], [], [])
PREDS_ALL = dict(preds)
PREDS_ALL.update(TYPE_PREDS)


def fresh_types(iterations, pruning, multiset, objects, goals, atoms_list, statics, cid):
    """statics and unary object-type atoms are ordinary atoms of every state (wlplan connects Problem.statics only as isolated nodes)."""
    extra = list(statics) + [("type_" + OTYPES[cid][o], [o]) for o in objects]
    prob = Problem(DOM_ALL, objects, [Atom(PREDS_ALL[n], list(a)) for n, a in goals], [])
    g = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=DOM_ALL, iterations=iterations, pruning=pruning, multiset_hash=multiset)
    ds = DomainDataset(DOM_ALL, [ProblemDataset(problem=prob, states=[State([Atom(PREDS_ALL[n], list(a)) for n, a in list(s) + extra]) for s in atoms_list])])
    g.collect(ds)
    return g, ds, np.array(g.embed(ds), dtype=np.float64)


def iso_check(g, ds, cid=None):
    """Typed/goal-marked ILG isomorphism (networkx, <=10 s). If isomorphic, also enumerates (<= 20000) isomorphisms and counts those that preserve PDDL object types (a legal renaming must)."""
    from wlplan.graph_generator import to_networkx
    import networkx as nx
    from networkx.algorithms.isomorphism import DiGraphMatcher
    graphs = g.to_graphs(ds)
    G0, G1 = to_networkx(graphs[0]), to_networkx(graphs[1])
    info = {"nodes": G0.number_of_nodes(), "edges": G0.number_of_edges()}
    signal.signal(signal.SIGALRM, _alarm)
    signal.alarm(10)
    try:
        m = DiGraphMatcher(G0, G1, node_match=lambda a, b: a.get("colour") == b.get("colour"), edge_match=lambda a, b: a.get("relation") == b.get("relation"))
        if not m.is_isomorphic():
            return "NOT_ISOMORPHIC", info
        n_maps = n_typed = 0
        if cid is not None:
            for mp in m.isomorphisms_iter():
                n_maps += 1
                if all(OTYPES[cid].get(u) == OTYPES[cid].get(v) for u, v in mp.items() if u in OTYPES[cid]):
                    n_typed += 1
                if n_maps >= 20000:
                    break
        info.update({"isomorphisms_enumerated": n_maps, "type_preserving": n_typed, "enumeration_capped": n_maps >= 20000})
        return "ISOMORPHIC", info
    except Timeout:
        return "UNKNOWN_TIMEOUT", info
    finally:
        signal.alarm(0)

# nested relaxation ladder (registered): R1x pruning removed (training vocabulary kept) -> R1y vocabulary rebuilt on the pair (colours unseen in training allowed) -> R2 multiset -> R3 4 -> R4 8 -> R5 16 iterations
LADDER = [("R2_multiset", 2, "none", True), ("R3_iterations4", 4, "none", True), ("R4_iterations8", 8, "none", True), ("R5_iterations16", 16, "none", True)]
STAGE_OF = {"R1x_no_pruning_training_vocabulary": "PRUNING_a-m", "R1y_vocabulary_rebuilt_on_pair": "VOCABULARY_TRUNCATION_OOV", "R2_multiset": "SET_AGGREGATION", "R3_iterations4": "ITERATION_DEPTH",
            "R4_iterations8": "ITERATION_DEPTH", "R5_iterations16": "ITERATION_DEPTH"}
cand_rows = []
for n, e in enumerate(sel):
    cid = e["problem"]
    ax, obj, goals = atoms_of(cid, e["better"])
    ay, _o, _g = atoms_of(cid, e["worse"])
    row = {"candidate": n, "problem": cid, "problem_sha256": idx["problems"][cid]["declared_sha256"] if cid in idx["problems"] else train_sha.get(cid), "better_state": e["better"], "worse_state": e["worse"],
           "better_state_sha": sha("%s|%s" % (cid, e["better"])), "worse_state_sha": sha("%s|%s" % (cid, e["worse"])), "source_pools": "+".join(sorted(set(e["sources"]))), "sampling_stratum": e["stratum"], "sampling_hash": e["sha"],
           "label_basis": e["info"].get("label_basis"), "d_better": e["info"].get("d_better"), "d_worse": e["info"].get("d_worse"), "L_better": e["info"].get("L_better"), "U_better": e["info"].get("U_better"),
           "L_worse": e["info"].get("L_worse"), "U_worse": e["info"].get("U_worse"), "same_parent": e["info"].get("same_parent"), "better_vs_suboptimal_choice": e["S1"],
           "atom_sets_identical": sorted(map(json.dumps, ax)) == sorted(map(json.dumps, ay)), "atom_symmetric_difference": len(set(map(json.dumps, ax)) ^ set(map(json.dumps, ay)))}
    for k in GEN:
        xb, xw = feat(cid, e["better"], k), feat(cid, e["worse"], k)
        row["phi_equal_" + k] = bool(not np.any(xb != xw))
        row["raw_better_" + k], row["raw_worse_" + k] = float(xb @ WGT[k]), float(xw @ WGT[k])
        row["round_better_" + k], row["round_worse_" + k] = cpp_round(row["raw_better_" + k]), cpp_round(row["raw_worse_" + k])
    # layer profile with a vocabulary rebuilt on the pair, no pruning, set hash: separated after 0 / 1 / 2 iterations?
    prof = {}
    for it in (0, 1, 2):
        _g, _ds, X = fresh(it, "none", False, obj, goals, [ax, ay])
        prof[it] = bool(np.any(X[0] != X[1]))
    row["pair_vocab_set_separated_it0"], row["pair_vocab_set_separated_it1"], row["pair_vocab_set_separated_it2"] = prof[0], prof[1], prof[2]
    # R1x: same training vocabulary, pruning removed
    ds_pair = DomainDataset(DOM, [ProblemDataset(problem=mk_problem(obj, goals), states=[mk_state(ax), mk_state(ay)])])
    Xx = np.array(G_NONE.embed(ds_pair), dtype=np.float64)
    first, last_ok = None, "R0_fixed_generator"
    sep = bool(np.any(Xx[0] != Xx[1]))
    row["ladder_R1x_no_pruning_training_vocabulary"] = sep
    if sep:
        first = "R1x_no_pruning_training_vocabulary"
    else:
        last_ok = "R1x_no_pruning_training_vocabulary"
        _g, _ds, X = fresh(2, "none", False, obj, goals, [ax, ay])
        sep = bool(np.any(X[0] != X[1]))
        row["ladder_R1y_vocabulary_rebuilt_on_pair"] = sep
        if sep:
            first = "R1y_vocabulary_rebuilt_on_pair"
        else:
            last_ok = "R1y_vocabulary_rebuilt_on_pair"
            for name, it, pr, ms in LADDER:
                _g, _ds, X = fresh(it, pr, ms, obj, goals, [ax, ay])
                sep = bool(np.any(X[0] != X[1]))
                row["ladder_" + name] = sep
                if sep:
                    first = name
                    break
                last_ok = name
    row["first_separating_rung"] = first
    row["control_rung_before"] = last_ok
    row["first_loss_stage"] = "ADAPTER_INPUT_IDENTICAL" if row["atom_sets_identical"] else (STAGE_OF[first] if first else "NOT_SEPARATED_BY_LADDER")
    if first is None:
        # step 7: isomorphism of the goal-marked ILG built from the adapter atoms (objects carry no type colour), <=10 s; type-preservation of the isomorphisms
        try:
            g, ds, _X = fresh(16, "none", True, obj, goals, [ax, ay])
            row["iso_status"], info = iso_check(g, ds, cid)
            row["iso_info"] = json.dumps(info)
        except Exception as ex:  # noqa: BLE001
            row["iso_status"] = "UNKNOWN_ERR:" + repr(ex)[:80]
        # step 8, published encoding (rationale registered in the report): the lifted-planning ILG contains the static facts and object types; the deployed adapter shows only the translation atoms, which omit
        # always-true facts (hoist / pallet positions) and carry no type. Two NESTED single-change steps on the same pair: (a) + static positions, (b) + object types as unary static atoms.
        # Same set / 2-iteration colouring (and the multiset / 16-iteration graph for the isomorphism check); no weights, no scores.
        st = STAT[cid]["statics"]
        try:
            ax_s, ay_s = [list(x) for x in ax] + [list(x) for x in st], [list(x) for x in ay] + [list(x) for x in st]
            g, ds, X = fresh(2, "none", False, obj, goals, [ax_s, ay_s])
            row["step8a_statics_as_state_atoms_separated_it2_set"] = bool(np.any(X[0] != X[1]))
            g, ds, X = fresh(2, "none", True, obj, goals, [ax_s, ay_s])
            row["step8a_statics_as_state_atoms_multiset_it2_separated"] = bool(np.any(X[0] != X[1]))
            g, ds, _X = fresh(16, "none", True, obj, goals, [ax_s, ay_s])
            row["step8a_statics_iso_status"], info = iso_check(g, ds, cid)
            row["step8a_info"] = json.dumps(info)
            if True:
                g, ds, X = fresh_types(2, "none", False, obj, goals, [ax, ay], st, cid)
                row["step8b_statics_types_separated_it2_set"] = bool(np.any(X[0] != X[1]))
                g, ds, _X = fresh_types(16, "none", True, obj, goals, [ax, ay], st, cid)
                row["step8b_statics_types_iso_status"], info = iso_check(g, ds, cid)
                row["step8b_info"] = json.dumps(info)
        except Exception as ex:  # noqa: BLE001
            row["step8_error"] = repr(ex)[:120]
    else:
        row["iso_status"] = "NOT_NEEDED_SEPARATED_BY_" + first
    row["actual_search_evidence_status"] = ("PROBLEM_IN_JOINT32_PANEL_NO_TRAJECTORY" if cid in panel_joint else "NO_SEARCH_RECORD_FOR_THIS_PAIR")
    row["models_involved"] = "W1(fixed)+WL(fixed)"
    row["conclusion_boundary"] = "symbolic feature level; no GBFS causal claim; iso 'UNKNOWN' = not decided"
    cand_rows.append(row)
    print("cand", n, cid, row["first_loss_stage"], first, flush=True)
cols = []
for r in cand_rows:
    for c in r:
        if c not in cols:
            cols.append(c)
with open(A1 + "/candidates.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=cols)
    w.writeheader()
    w.writerows(cand_rows)
json.dump({"cpu_seconds": time.process_time() - CPU0, "wall_seconds": time.time() - T0}, open(A1 + "/a1_ledger.json", "w"))
print("A1 done", time.time() - T0)

