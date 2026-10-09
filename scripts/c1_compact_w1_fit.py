#!/usr/bin/env python
"""W1 fit (goose environment, no project imports): fixed L2 WL features, rank-SVM exactly as the author's trainer (sklearn LinearSVC, hinge, C=1, no intercept, max_iter 1e6), but with the sample weights of the D0-aligned pair list
(the author's ``rank_classifier`` computes pair counts and then does not pass them to ``fit``). The weights are rescaled so that their sum equals the number of unique pairs, i.e. the mass the author's unweighted fit assigns, which keeps C = 1
comparable. Pairs with identical feature differences are merged (their weights add), which is exactly equivalent for the hinge loss. Signs are flipped for a seeded half of the pairs (both classes are needed; the loss is symmetric).
ONE fit, no hyper-parameter, no validation-based choice."""
import argparse
import json
import random
import sys
import time

import numpy as np
import scipy.sparse as sp
from sklearn import svm
from wlplan.data import DomainDataset, ProblemDataset
from wlplan.feature_generator import init_feature_generator
from wlplan.planning import Atom, Domain, Predicate, Problem, State


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data")
    ap.add_argument("out_params")
    ap.add_argument("--report")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    t0 = time.time()
    d = json.load(open(a.data))
    preds = {n: Predicate(n, ar) for n, ar in d["predicates"]}
    domain = Domain("domain", list(preds.values()), [], [], [])
    data, offsets, off = [], [], 0
    for p in d["problems"]:
        goals = [Atom(preds[n], list(args)) for n, args in p["goals"]]
        prob = Problem(domain, p["objects"], goals, [])
        states = [State([Atom(preds[n], list(args)) for n, args in st]) for st in p["states"]]
        data.append(ProblemDataset(problem=prob, states=states))
        offsets.append(off)
        off += len(states)
    dataset = DomainDataset(domain, data)
    gen = init_feature_generator(feature_algorithm="wl", graph_representation="ilg", domain=domain, iterations=2, pruning="a-m", multiset_hash=False)
    gen.collect(dataset)
    X = np.array(gen.embed(dataset), dtype=np.float32)
    print("X", X.shape, "collect+embed %.1fs" % (time.time() - t0), flush=True)
    rows_b, rows_w, wts = [], [], []
    for p, o in zip(d["problems"], offsets):
        for i, j, w in p["pairs"]:
            rows_b.append(o + i)
            rows_w.append(o + j)
            wts.append(w)
    rows_b, rows_w, wts = np.array(rows_b), np.array(rows_w), np.array(wts)
    D = sp.csr_matrix(X[rows_w] - X[rows_b])
    # merge identical differences (weights add)
    key = {}
    uniq_rows, uniq_w = [], []
    for k in range(D.shape[0]):
        row = D.getrow(k)
        h = (tuple(row.indices.tolist()), tuple(row.data.tolist()))
        if h in key:
            uniq_w[key[h]] += wts[k]
        else:
            key[h] = len(uniq_rows)
            uniq_rows.append(k)
            uniq_w.append(wts[k])
    Du = D[uniq_rows]
    wu = np.array(uniq_w)
    nz = np.array([Du.getrow(i).nnz for i in range(Du.shape[0])])
    keep = nz > 0                                                                  # identical feature vectors (WL-indistinguishable states) carry no preference
    Du, wu = Du[keep], wu[keep]
    mass_raw = float(wu.sum())
    wu = wu * (len(wu) / wu.sum())
    rng = random.Random(a.seed)
    sign = np.array([1 if rng.random() > 0.5 else -1 for _ in range(Du.shape[0])])
    Xi = sp.diags(sign.astype(np.float32)) @ Du
    if a.dry_run:                                                                  # fixture: NO fit; random weights only to test the save / deploy-adapter path
        w = np.random.RandomState(0).randn(X.shape[1])
        gen.save(a.out_params, w)
        json.dump({"scores": (X @ w).tolist(), "offsets": offsets, "pairs_unique_nonzero": int(Du.shape[0]), "features": int(X.shape[1]), "dry_run": True, "fits": 0}, open(a.report, "w"))
        print("dry run: no fit")
        return 0
    model = svm.LinearSVC(loss="hinge", fit_intercept=False, max_iter=1000000, C=1.0)
    t1 = time.time()
    model.fit(Xi, sign, sample_weight=wu)
    w = model.coef_.reshape(-1)
    acc_w = float(np.sum(wu * ((Du @ w) > 0)) / wu.sum())
    acc_u = float(np.mean((Du @ w) > 0))
    gen.save(a.out_params, w)
    rep = {"states": int(X.shape[0]), "features": int(X.shape[1]), "pairs_raw": int(len(wts)), "pairs_unique_nonzero": int(Du.shape[0]), "pairs_identical_features_dropped": int((~keep).sum()), "weight_mass_before_rescale": mass_raw,
           "weight_mass_after_rescale": float(wu.sum()), "train_pair_accuracy_weighted": acc_w, "train_pair_accuracy_unweighted": acc_u, "fit_seconds": round(time.time() - t1, 1), "total_seconds": round(time.time() - t0, 1),
           "weights_nonzero": int((w != 0).sum()), "seed": a.seed, "svm_iterations": int(model.n_iter_), "fits": 1}
    if a.report:
        json.dump(rep, open(a.report, "w"), indent=1)
    print(json.dumps(rep))


if __name__ == "__main__":
    sys.exit(main())
