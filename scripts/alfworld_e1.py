"""E1 task-admission test (zero training) on the dev split. Spec section 6/E1.

Prior-dependent criteria are evaluated only when --prior points to a frozen prior cache.
The catalog's oracle partition is read ONLY by the analysis helpers at the bottom (ceiling,
calibration, prior-conditions); policies/scripts never see it.
"""
import argparse
import json
import math
import os
import random
import statistics as st
import sys
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
import numpy as np  # noqa: E402

from cp_disr.analysis import alfworld_qref as Q  # noqa: E402
from cp_disr.baselines import alfworld_scripts as S  # noqa: E402
from cp_disr.platforms.alfworld import data  # noqa: E402
from cp_disr.platforms.alfworld.adapter import AlfEpisode  # noqa: E402
from cp_disr.platforms.alfworld.observation import type_of  # noqa: E402
from cp_disr.platforms.alfworld.prior_provider import PriorCache  # noqa: E402

CFG = {}
RANDOM_SEEDS = 10
BOOT = 10000


def init(cfg):
    CFG.update(cfg)
    CFG["tables"] = data.Tables(json.load(open(cfg["run_dir"] + "/tables.json")))
    CFG["prior"] = PriorCache.load(cfg["prior"]) if cfg.get("prior") else None


def _g(rel):
    return os.path.join(CFG["data_root"], rel)


def job_script(args):
    rel, name, seed = args
    r = S.run_episode(_g(rel), CFG["tables"], S.SCRIPTS[name], prior=CFG["prior"], seed=seed)
    r.pop("trace")
    r["rel"], r["script"], r["seed"] = rel, name, seed
    return r


def job_trace(args):
    rel, name = args
    r = S.run_episode(_g(rel), CFG["tables"], S.SCRIPTS[name], prior=CFG["prior"], seed=0, record_states=True)
    states = []
    for t in r["trace"]:
        pub = t["pub"]
        checks = [a for a in pub.legal if a[0] == "CHECK"]
        if not checks:
            continue
        row = {"n_check_legal": len(checks), "belief_pick": list(S.belief_optimal(pub, CFG | {"tables": CFG["tables"]})),
               "contract_pick": list(S.contract_search(pub, CFG))}
        if CFG["prior"] is not None:
            row["prior_pick"] = list(S.fixed_prior_search(pub, {"tables": CFG["tables"], "prior": CFG["prior"]}))
        states.append(row)
    return {"rel": rel, "script": name, "states": states}


def job_qref_state(args):
    """Pre-registered state: Random Legal prefix of k in {0,1,2} checks without finding the target."""
    rel, seed = args
    rng = random.Random(seed)
    k = rng.choice([0, 1, 2])
    ep = AlfEpisode(_g(rel), CFG["tables"])
    prefix = []
    try:
        for _ in range(k):
            pub = ep.public()
            checks = [a for a in pub.legal if a[0] == "CHECK"]
            if len(checks) < 3 or ep.done:
                break
            a = rng.choice(sorted(checks))
            ep.execute(*a)
            prefix.append(a)
            if ep.target_found:
                return None
        pub = ep.public()
        checks = [a for a in pub.legal if a[0] == "CHECK"]
        if len(checks) < 2 or ep.done or ep.target_found:
            return None
        q = Q.q_ref(CFG["tables"], pub)
        ranked = sorted(checks, key=lambda a: (-q[a], a))
        top1, top2 = ranked[0], ranked[1]
        bel = Q.belief(CFG["tables"], pub)
    finally:
        ep.close()

    def real(first):
        e = AlfEpisode.replay(_g(rel), CFG["tables"], prefix)
        try:
            e.execute(*first)
            ctx = {"tables": CFG["tables"], "prior": None, "rng": random.Random(0)}
            while not e.done:
                e.execute(*S.belief_optimal(e.public(), ctx))
            return 2.0 ** (-e.success_step / 50.0) if e.won else 0.0
        finally:
            e.close()

    return {"rel": rel, "k": len(prefix), "top1": list(top1), "top2": list(top2), "q1": q[top1], "q2": q[top2],
            "J1": real(top1), "J2": real(top2), "belief": {r: v for r, v in bel.items()}, "prefix": [list(a) for a in prefix]}


def boot_ci(x, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), size=(BOOT, len(x)))
    means = x[idx].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------- oracle-side analysis (never feeds a policy)
def clairvoyant_J(row):
    by = {x["name"]: x for x in row["receptacles"]}
    holders = {t["receptacle"] for t in row["oracle"]["targets"] if t["receptacle"]}
    goal = sorted(x["name"] for x in row["receptacles"] if x["rtype"] == row["goal_rtype"])[0]
    T = min(1 + int(by[h]["openable"]) for h in holders) + 1 + 1 + int(by[goal]["openable"]) + 1
    return 2.0 ** (-T / 50.0)


def prior_conditions(row, tables, prior):
    ep_room = sorted({x["rtype"] for x in row["receptacles"]})
    feas = [x["name"] for x in row["receptacles"] if tables.can_contain(x["rtype"], row["goal_otype"])]
    cs = prior.class_scores(row["goal_otype"], ep_room)
    classes = sorted({type_of(r) for r in feas})
    rank = sorted(classes, key=lambda c: (-(cs or {}).get(c, 0.0), c)) if cs else []
    by = {x["name"]: x for x in row["receptacles"]}
    hclasses = {by[t["receptacle"]]["rtype"] for t in row["oracle"]["targets"] if t["receptacle"]}
    k = min(3, len(classes))
    best = min((rank.index(c) + 1 for c in hclasses if c in rank), default=None)
    cat = "miss" if (not rank or best is None or best > k) else "top1" if best == 1 else "topk_only"
    vals = [cs.get(c, 0.0) for c in classes] if cs else []
    nonuniform = bool(cs) and len({round(v, 9) for v in vals}) > 1
    # contract redundancy: does the ranking add anything beyond canContain? (feasible set is already filtered)
    return {"cat": cat, "nonuniform": nonuniform, "n_classes": len(classes)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", default="runs/alfworld_prior_reliance")
    ap.add_argument("--data-root", default="/home/xushijie2/xsj2_alf/data")
    ap.add_argument("--prior", default=None)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    cfg = {"run_dir": a.run_dir, "data_root": a.data_root, "prior": a.prior}
    splits = json.load(open(a.run_dir + "/splits.json"))
    rows = {r["gamefile"]: r for r in data.load_catalog(a.run_dir + "/catalog.json")}
    dev = splits["dev"]
    init(cfg)
    names = ["contract_search", "belief_optimal"] + (["fixed_prior"] if a.prior else [])
    with Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        det = pool.map(job_script, [(g, n, 0) for g in dev for n in names])
        rnd = pool.map(job_script, [(g, "random_legal", s) for g in dev for s in range(RANDOM_SEEDS)])
        traces = pool.map(job_trace, [(g, n) for g in dev for n in names])
        qstates = [x for x in pool.map(job_qref_state, [(g, 1000 + i) for i, g in enumerate(dev)]) if x]
    res = {"n_dev": len(dev), "prior": a.prior}
    J = {n: {r["rel"]: r["J"] for r in det if r["script"] == n} for n in names}
    J["random_legal"] = {g: st.mean(r["J"] for r in rnd if r["rel"] == g) for g in dev}
    J["clairvoyant_ceiling"] = {g: clairvoyant_J(rows[g]) for g in dev}
    res["mean_J"] = {n: boot_ci([J[n][g] for g in dev]) for n in J}
    res["success_rate"] = {n: st.mean(float(r["won"]) for r in det if r["script"] == n) for n in names}
    res["mean_raw_steps"] = {n: st.mean(r["raw_steps"] for r in det if r["script"] == n) for n in names}
    res["mean_checks"] = {n: st.mean(r["n_checks"] for r in det if r["script"] == n) for n in names}
    # criterion 1: search decisions with >=2 legal CHECK
    c1 = {}
    for n in names:
        s = [x for t in traces if t["script"] == n for x in t["states"]]
        c1[n] = {"search_decisions": len(s), "frac_two_or_more_checks": sum(x["n_check_legal"] >= 2 for x in s) / max(1, len(s))}
    res["criterion1"] = c1
    # criterion 5: belief-optimal vs best static (paired, bootstrap)
    statics = [n for n in ("contract_search", "fixed_prior", "random_legal") if n in J]
    best_static = max(statics, key=lambda n: st.mean(J[n][g] for g in dev))
    d_best = [J["belief_optimal"][g] - J[best_static][g] for g in dev]
    res["criterion5"] = {"best_static": best_static, "paired_mean_gain_vs_best_static": boot_ci(d_best),
                         "gain_vs_contract": boot_ci([J["belief_optimal"][g] - J["contract_search"][g] for g in dev]),
                         "gain_vs_random": boot_ci([J["belief_optimal"][g] - J["random_legal"][g] for g in dev]),
                         "ceiling_gain_over_best_static": boot_ci([J["clairvoyant_ceiling"][g] - J[best_static][g] for g in dev]),
                         "threshold": 0.03}
    # criterion 6 (needs prior): belief vs fixed-prior disagreement on shared search states
    if a.prior:
        ds = [x for t in traces for x in t["states"] if "prior_pick" in x]
        res["criterion6"] = {"states": len(ds), "disagree_rate": sum(x["belief_pick"] != x["prior_pick"] for x in ds) / max(1, len(ds))}
        cond = [prior_conditions(rows[g], CFG["tables"], CFG["prior"]) for g in dev]
        cats = {c: sum(x["cat"] == c for x in cond) for c in ("top1", "topk_only", "miss")}
        res["criterion3"] = {"frac_nonuniform_within_feasible": sum(x["nonuniform"] for x in cond) / len(cond)}
        res["criterion4"] = {"counts": cats, "n": len(cond)}
    res["disagree_belief_vs_contract"] = {"states": sum(len(t["states"]) for t in traces if t["script"] == "belief_optimal"),
                                          "rate": (lambda ss: sum(x["belief_pick"] != x["contract_pick"] for x in ss) / max(1, len(ss)))(
                                              [x for t in traces if t["script"] == "belief_optimal" for x in t["states"]])}
    # Q_ref replay consistency
    diffs = [x["J1"] - x["J2"] for x in qstates]
    res["qref_consistency"] = {"n_states": len(qstates), "paired_mean_J_top1_minus_top2": boot_ci(diffs) if diffs else None,
                               "frac_top1_better": sum(d > 0 for d in diffs) / max(1, len(diffs)),
                               "frac_tie": sum(d == 0 for d in diffs) / max(1, len(diffs)), "pass_lower_bound_gt_0": bool(diffs) and boot_ci(diffs)[1] > 0}
    # belief calibration on dev initial states
    bins = [[0, 0.0, 0] for _ in range(10)]
    for g in dev:
        row = rows[g]
        feas = [x["name"] for x in row["receptacles"] if CFG["tables"].can_contain(x["rtype"], row["goal_otype"])]
        b = data.instance_belief(CFG["tables"], row["goal_otype"], feas)
        holders = {t["receptacle"] for t in row["oracle"]["targets"] if t["receptacle"]}
        for r, p in b.items():
            i = min(9, int(p * 10))
            bins[i][0] += 1
            bins[i][1] += p
            bins[i][2] += int(r in holders)
    res["belief_calibration"] = [{"bin": "%.1f-%.1f" % (i / 10, (i + 1) / 10), "n": n, "mean_pred": (sp / n if n else None), "freq": (h / n if n else None)} for i, (n, sp, h) in enumerate(bins)]
    out = a.out or os.path.join(a.run_dir, "e1_result.json" if a.prior else "e1_result_noprior.json")
    json.dump({"result": res, "qref_states": qstates}, open(out, "w"), indent=1, default=str)
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
