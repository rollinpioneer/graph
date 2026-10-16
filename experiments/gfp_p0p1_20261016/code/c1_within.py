"""Post-hoc check (not in the frozen card): c1 AUC with west/east pairs taken only within the same start state.

The card's pooled AUC mixes candidates from 10 start states whose V differs by 6-14, so a start state that
both has a higher V and produces more west candidates can drive the pooled AUC on its own.
usage: python c1_within.py C1_JSON
"""
import sys
import json

import numpy as np
from scipy.stats import spearmanr

r = json.load(open(sys.argv[1]))
K, n, thr = r["K"], r["n_start"], r["thr"]
print("model | src | AUC within start | pairs | Q std within start | start-mean Q range | Spearman(Q,dx) within start")
for m, v in r["models"].items():
    for src in ("actor", "flow", "data"):
        q = np.array(v[src]["q"]).reshape(n, K)
        dx = np.array(v[src]["dx"] if src != "data" else r["data_labels"]["dx"]).reshape(n, K)
        gt = eq = tot = 0
        for i in range(n):
            w, e = q[i][dx[i] <= -thr], q[i][dx[i] >= thr]
            if len(w) and len(e):
                d = w[:, None] - e[None, :]
                gt, eq, tot = gt + (d > 0).sum(), eq + (d == 0).sum(), tot + d.size
        auc = (gt + 0.5 * eq) / tot if tot else float("nan")
        rho = np.mean([spearmanr(q[i], dx[i])[0] for i in range(n)])
        print(f"{m} | {src} | {auc:.3f} | {tot} | {q.std(1).mean():.2f} | {np.ptp(q.mean(1)):.2f} | {rho:+.2f}")
