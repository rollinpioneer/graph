"""Stage-2 confirmation: judgement per CARD_CONFIRM_SCRATCH_20261016.md section 4 (no GPU).
usage: python judge_scratch.py SCAN_JSON OUT_JSON I_RUN_DIR... -- G_RUN_DIR...   (RUN_DIR = .../out/*/date_*)
"""
import sys
import csv
import json
import numpy as np


def final_actor(rd):
    rows = list(csv.DictReader(open(f"{rd}/success.csv")))
    return float(rows[-1]["final_Actor"])


def v_minus_q(rd):
    rows = list(csv.DictReader(open(f"{rd}/train.csv")))
    v = [float(r["training/value/v_minus_q"]) for r in rows if 500000 <= float(r["step"]) <= 1000000]
    return float(np.mean(v)), len(v)


def island(scan, name):
    L = sorted((v["step"], v) for v in scan.values() if v["run"] == name)
    assert len(L) == 10, (name, len(L))
    hits = sorted({b for i, (a, va) in enumerate(L) for (b, vb) in L[i + 1:]
                   if a >= 200000 and vb["contrast"] - va["contrast"] >= 15
                   and vb["dqda_norm_start"] >= 2 * va["dqda_norm_start"]})
    return hits


def main():
    scan = json.load(open(sys.argv[1]))
    out = sys.argv[2]
    k = sys.argv.index("--")
    I, G = sys.argv[3:k], sys.argv[k + 1:]
    seed = lambda rd: int(rd.rstrip("/").split("seed_")[-1])
    I = {seed(r): r for r in I}
    G = {seed(r): r for r in G}
    assert sorted(I) == sorted(G) == list(range(8))
    rows = []
    for s in range(8):
        vq, n = v_minus_q(I[s])
        r = dict(seed=s, I=final_actor(I[s]), G=final_actor(G[s]), I_v_minus_q=vq, I_v_rows=n,
                 I_valid=vq <= 2.0, I_island=island(scan, f"I_s{s}"), G_island=island(scan, f"G_s{s}"))
        r["d"] = r["I"] - r["G"]
        r["healthy"] = r["G"] >= 60
        r["harm"] = r["healthy"] and r["d"] < -15
        rows.append(r)
        print(json.dumps(r))
    valid = [r for r in rows if r["I_valid"]]
    d = np.array([r["d"] for r in valid])
    rng = np.random.RandomState(20261016)
    boot = np.array([rng.choice(d, len(d)).mean() for _ in range(10000)])
    summ = dict(n_valid=len(valid), mean_I=float(np.mean([r["I"] for r in valid])),
                mean_G=float(np.mean([r["G"] for r in valid])), mean_d=float(d.mean()),
                boot95=[float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))],
                n_I_better=int((d > 0).sum()), n_I_worse=int((d < 0).sum()),
                mean_d_seeds0_3=float(np.mean([r["d"] for r in valid if r["seed"] < 4])),
                mean_d_seeds4_7=float(np.mean([r["d"] for r in valid if r["seed"] >= 4])),
                harm_seeds=[r["seed"] for r in valid if r["harm"]],
                n_island_I=sum(bool(r["I_island"]) for r in rows), n_island_G=sum(bool(r["G_island"]) for r in rows),
                invalid_I=[r["seed"] for r in rows if not r["I_valid"]])
    if summ["harm_seeds"] or summ["mean_d"] <= -5:
        summ["verdict"] = "harm"
    elif summ["mean_d"] >= 5:
        summ["verdict"] = "overall improvement"
    else:
        summ["verdict"] = "no clear difference"
    print(json.dumps(summ, indent=1))
    json.dump(dict(rows=rows, summary=summ), open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
