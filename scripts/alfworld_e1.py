"""E1 task/prior admission (zero training) on the dev split, v2 (multi-holder Q_ref, frozen prior cache).

Sections written to the result file:
  task_prior_admission : criteria 1-6 (+ non-redundancy statistics)
  qref_validation      : same-observation-signature conditional-expectation check (reported SEPARATELY, 'pending' status)
The catalog's oracle partition is read ONLY by analysis helpers (ceiling, calibration, prior conditions, replay
consistency); scripts/policies never see it.
"""
import argparse
import hashlib
import json
import os
import random
import statistics as st
import sys
from collections import defaultdict
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
EPS_Q = 0.001  # pre-defined: expected-J gaps below this are 'neutral/unknown' in the Q_ref validation


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
    ctx = {"tables": CFG["tables"], "prior": CFG["prior"], "rng": random.Random(0)}
    for t in r["trace"]:
        pub = t["pub"]
        checks = [a for a in pub.legal if a[0] == "CHECK"]
        if not checks:
            continue
        row = {"n_check_legal": len(checks), "belief_pick": list(S.belief_optimal(pub, ctx)), "contract_pick": list(S.contract_search(pub, ctx))}
        if CFG["prior"] is not None:
            row["prior_pick"] = list(S.fixed_prior_search(pub, ctx))
            s = CFG["prior"].scores(pub, CFG["tables"])
            row["prior_scores_distinct"] = len({round(s[r], 9) for r in pub.feasible})
        states.append(row)
    return {"rel": rel, "script": name, "states": states}


# ----------------------------------------------------------------- Q_ref validation (same observation signature)
def signature(row):
    return (tuple(sorted(x["name"] for x in row["receptacles"])), row["goal_otype"], row["goal_rtype"])


def job_qref_group(args):
    """One (group, depth) cell: trials sharing the same public start, a fixed prefix of empty checks, then
    real J of the Q_ref top-2 actions followed by mu_ref, averaged over the hidden worlds consistent with the signature."""
    group_id, rels, depth, seed = args
    tables = CFG["tables"]
    rng = random.Random(seed)
    ep0 = AlfEpisode(_g(rels[0]), tables)
    feas = sorted(ep0.feasible)
    ep0.close()
    rng.shuffle(feas)
    prefix = [("CHECK", r) for r in feas[:depth]]
    consistent, q = [], None
    for rel in rels:
        e = AlfEpisode(_g(rel), tables)
        try:
            for a in prefix:
                if e.done or e.target_found:
                    break
                e.execute(*a)
            if e.done or e.target_found:
                continue
            pub = e.public()
            if sum(1 for a in pub.legal if a[0] == "CHECK") < 2:
                continue
            qq = Q.q_ref(tables, pub)
            if q is None:
                q = qq
            elif any(abs(q[k] - qq.get(k, -1)) > 1e-12 for k in q):
                continue  # different reduced public state (e.g. different openability): not the same signature
            consistent.append(rel)
        finally:
            e.close()
    if len(consistent) < 2:
        return None
    ranked = sorted((k for k in q if k[0] == "CHECK"), key=lambda a: (-q[a], a))
    a1, a2 = ranked[0], ranked[1]

    def real(rel, first):
        e = AlfEpisode.replay(_g(rel), tables, prefix)
        try:
            e.execute(*first)
            ctx = {"tables": tables, "prior": None, "rng": random.Random(0)}
            while not e.done:
                e.execute(*S.belief_optimal(e.public(), ctx))
            return 2.0 ** (-e.success_step / 50.0) if e.won else 0.0
        finally:
            e.close()

    j1 = [real(r, a1) for r in consistent]
    j2 = [real(r, a2) for r in consistent]
    return {"group": group_id, "depth": depth, "n_trials": len(consistent), "a1": list(a1), "a2": list(a2), "q1": q[a1], "q2": q[a2],
            "J1": float(np.mean(j1)), "J2": float(np.mean(j2)), "gap": q[a1] - q[a2], "scene": rels[0]}


def cluster_boot(items, key, value, seed=0):
    groups = defaultdict(list)
    for it in items:
        groups[key(it)].append(value(it))
    ks = sorted(groups)
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(BOOT):
        pick = rng.integers(0, len(ks), len(ks))
        means.append(float(np.mean([v for i in pick for v in groups[ks[i]]])))
    allv = [v for k in ks for v in groups[k]]
    return float(np.mean(allv)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(ks)


def boot_ci(x, seed=0):
    x = np.asarray(x, dtype=float)
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(BOOT, len(x)))].mean(1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


# ----------------------------------------------------------------- oracle-side analysis (never feeds a policy)
def clairvoyant_J(row):
    by = {x["name"]: x for x in row["receptacles"]}
    can = {tuple(p) for p in row["cancontain"]}
    holders = {t["receptacle"] for t in row["oracle"]["targets"] if t["receptacle"] and (by[t["receptacle"]]["rtype"], row["goal_otype"]) in can}
    goal = sorted(x["name"] for x in row["receptacles"] if x["rtype"] == row["goal_rtype"])[0]
    T = min(1 + int(by[h]["openable"]) for h in holders) + 1 + 1 + int(by[goal]["openable"]) + 1
    return 2.0 ** (-T / 50.0)


def prior_conditions(row, tables, prior):
    room = sorted({x["rtype"] for x in row["receptacles"]})
    by = {x["name"]: x for x in row["receptacles"]}
    feas = [x["name"] for x in row["receptacles"] if tables.can_contain(x["rtype"], row["goal_otype"])]
    cs = prior.class_scores(row["goal_otype"], room)
    classes = sorted({type_of(r) for r in feas})
    rank = sorted(classes, key=lambda c: (-(cs or {}).get(c, 0.0), c)) if cs else []
    hclasses = {by[t["receptacle"]]["rtype"] for t in row["oracle"]["targets"] if t["receptacle"] and by[t["receptacle"]]["name"] in feas}
    k = min(3, len(classes))
    best = min((rank.index(c) + 1 for c in hclasses if c in rank), default=None)
    cat = "miss" if (not rank or best is None or best > k) else "top1" if best == 1 else "topk_only"
    feas_mass = sum((cs or {}).get(c, 0.0) for c in classes)
    all_mass = sum((cs or {}).values())
    unf_top = bool(cs) and max(cs, key=lambda c: (cs[c], c)) not in classes
    fvals = [(cs or {}).get(c, 0.0) for c in classes]
    return {"cat": cat, "nonuniform_within_feasible": bool(cs) and len({round(v, 9) for v in fvals}) > 1, "n_classes": len(classes),
            "infeasible_mass_share": (1 - feas_mass / all_mass) if all_mass > 0 else None, "top1_unfiltered_is_infeasible": unf_top,
            "ranking_changed_by_canContain": bool(cs) and (sorted(cs, key=lambda c: (-cs[c], c))[:len(classes)] != rank) }


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
    groups = defaultdict(list)
    for g in dev:
        groups[signature(rows[g])].append(g)
    cells = []
    for gi, (sig, rels) in enumerate(sorted(groups.items(), key=lambda kv: kv[1][0])):
        if len(rels) >= 2:
            for d in (0, 1, 2):
                cells.append((gi, rels, d, int(hashlib.sha256(("%d|%d" % (gi, d)).encode()).hexdigest()[:8], 16)))
    with Pool(a.workers, initializer=init, initargs=(cfg,)) as pool:
        det = pool.map(job_script, [(g, n, 0) for g in dev for n in names])
        rnd = pool.map(job_script, [(g, "random_legal", s) for g in dev for s in range(RANDOM_SEEDS)])
        traces = pool.map(job_trace, [(g, n) for g in dev for n in names])
        qcells = [x for x in pool.map(job_qref_group, cells) if x]
    J = {n: {r["rel"]: r["J"] for r in det if r["script"] == n} for n in names}
    J["random_legal"] = {g: st.mean(r["J"] for r in rnd if r["rel"] == g) for g in dev}
    J["clairvoyant_ceiling"] = {g: clairvoyant_J(rows[g]) for g in dev}
    adm = {"n_dev": len(dev), "prior": a.prior}
    adm["mean_J"] = {n: boot_ci([J[n][g] for g in dev]) for n in J}
    adm["success_rate"] = {n: st.mean(float(r["won"]) for r in det if r["script"] == n) for n in names}
    adm["mean_raw_steps"] = {n: st.mean(r["raw_steps"] for r in det if r["script"] == n) for n in names}
    adm["mean_checks"] = {n: st.mean(r["n_checks"] for r in det if r["script"] == n) for n in names}
    c1 = {}
    for n in names:
        s = [x for t in traces if t["script"] == n for x in t["states"]]
        c1[n] = {"search_decisions": len(s), "frac_two_or_more_checks": sum(x["n_check_legal"] >= 2 for x in s) / max(1, len(s))}
    adm["criterion1_multiple_legal_checks"] = c1
    statics = [n for n in ("contract_search", "fixed_prior", "random_legal") if n in J]
    best_static = max(statics, key=lambda n: st.mean(J[n][g] for g in dev))
    adm["criterion5_belief_vs_best_static"] = {
        "best_static": best_static, "static_mean_J": {n: st.mean(J[n][g] for g in dev) for n in statics},
        "paired_gain_vs_best_static": boot_ci([J["belief_optimal"][g] - J[best_static][g] for g in dev]),
        "gain_vs_contract": boot_ci([J["belief_optimal"][g] - J["contract_search"][g] for g in dev]),
        "gain_vs_random": boot_ci([J["belief_optimal"][g] - J["random_legal"][g] for g in dev]),
        "ceiling_gain_over_best_static": boot_ci([J["clairvoyant_ceiling"][g] - J[best_static][g] for g in dev]), "threshold": 0.03}
    if a.prior:
        adm["criterion5_belief_vs_fixed_prior"] = boot_ci([J["belief_optimal"][g] - J["fixed_prior"][g] for g in dev])
        adm["criterion5_fixed_prior_vs_contract"] = boot_ci([J["fixed_prior"][g] - J["contract_search"][g] for g in dev])
        adm["criterion5_fixed_prior_vs_random"] = boot_ci([J["fixed_prior"][g] - J["random_legal"][g] for g in dev])
        ds = [x for t in traces for x in t["states"] if "prior_pick" in x]
        dis = [x for x in ds if x["belief_pick"] != x["prior_pick"]]
        adm["criterion6_disagreement_belief_vs_fixed_prior"] = {"search_states": len(ds), "disagree_rate": len(dis) / max(1, len(ds))}
        adm["criterion2_soft_choice_share"] = {"disagreements": len(dis),
                                               "frac_where_both_actions_legal_and_multiple_checks_available": sum(x["n_check_legal"] >= 2 for x in dis) / max(1, len(dis)),
                                               "note": "both picks are drawn from the public legal set by construction; the share is the fraction of disagreements with >=2 legal CHECK"}
        cond = [prior_conditions(rows[g], CFG["tables"], CFG["prior"]) for g in dev]
        adm["criterion3_prior_nonuniform_within_feasible"] = sum(x["nonuniform_within_feasible"] for x in cond) / len(cond)
        cats = {c: sum(x["cat"] == c for x in cond) for c in ("top1", "topk_only", "miss")}
        adm["criterion4_prior_conditions"] = {"counts": cats, "n": len(cond), "fractions": {c: v / len(cond) for c, v in cats.items()}}
        adm["non_redundancy"] = {
            "mean_prior_mass_on_canContain_infeasible_classes": float(np.mean([x["infeasible_mass_share"] for x in cond if x["infeasible_mass_share"] is not None])),
            "frac_episodes_unfiltered_top1_is_infeasible": float(np.mean([x["top1_unfiltered_is_infeasible"] for x in cond])),
            "frac_episodes_canContain_filter_changes_ranking": float(np.mean([x["ranking_changed_by_canContain"] for x in cond])),
            "frac_states_with_distinct_scores_within_feasible_set": float(np.mean([x["prior_scores_distinct"] > 1 for t in traces if t["script"] == "fixed_prior" for x in t["states"]]))}
    adm["disagree_belief_vs_contract"] = (lambda ss: {"states": len(ss), "rate": sum(x["belief_pick"] != x["contract_pick"] for x in ss) / max(1, len(ss))})(
        [x for t in traces if t["script"] == "belief_optimal" for x in t["states"]])
    # belief calibration on dev initial states (marginal P(holder | >=1 holder))
    bins = [[0, 0.0, 0] for _ in range(10)]
    for g in dev:
        row = rows[g]
        feas = [x["name"] for x in row["receptacles"] if CFG["tables"].can_contain(x["rtype"], row["goal_otype"])]
        class P:  # minimal public state for the belief
            pass
        p = P()
        p.goal_otype, p.unchecked = row["goal_otype"], tuple(feas)
        b = Q.belief(CFG["tables"], p)
        by = {x["name"]: x for x in row["receptacles"]}
        holders = {t["receptacle"] for t in row["oracle"]["targets"] if t["receptacle"]}
        for r, pr in b.items():
            i = min(9, int(pr * 10))
            bins[i][0] += 1; bins[i][1] += pr; bins[i][2] += int(r in holders)
    adm["belief_calibration_multi_holder"] = [{"bin": "%.1f-%.1f" % (i / 10, (i + 1) / 10), "n": n, "mean_pred": (sp / n if n else None), "freq": (h / n if n else None)} for i, (n, sp, h) in enumerate(bins)]
    # ---- Q_ref validation (separate, pending)
    dec = [c for c in qcells if c["gap"] >= EPS_Q]
    neu = [c for c in qcells if c["gap"] < EPS_Q]
    scene_of = lambda c: rows[c["scene"]]["scene"]
    qv = {"status": "PENDING_RESEARCH_DECISION", "epsilon_Q": EPS_Q, "n_signature_cells": len(qcells), "n_groups_ge2_trials": len({c["group"] for c in qcells}),
          "trials_per_cell": {"mean": float(np.mean([c["n_trials"] for c in qcells])) if qcells else None}, "n_decided": len(dec), "n_neutral_unknown": len(neu)}
    if dec:
        m, lo, hi, nc = cluster_boot(dec, scene_of, lambda c: c["J1"] - c["J2"])
        qv["decided_cells"] = {"mean_dJ_top1_minus_top2": m, "scene_cluster_ci95": [lo, hi], "n_scene_clusters": nc,
                               "frac_top1_better": float(np.mean([c["J1"] > c["J2"] for c in dec])), "frac_top1_worse": float(np.mean([c["J1"] < c["J2"] for c in dec])),
                               "mean_predicted_gap": float(np.mean([c["gap"] for c in dec]))}
        qv["absolute_calibration_top1"] = {"mean_Q_pred": float(np.mean([c["q1"] for c in qcells])), "mean_J_realised": float(np.mean([c["J1"] for c in qcells])),
                                           "corr_pred_vs_realised_over_all_actions": float(np.corrcoef([c["q1"] for c in qcells] + [c["q2"] for c in qcells], [c["J1"] for c in qcells] + [c["J2"] for c in qcells])[0, 1])}
    if neu:
        qv["neutral_cells"] = {"mean_dJ": float(np.mean([c["J1"] - c["J2"] for c in neu])), "n": len(neu)}
    out = {"task_prior_admission": adm, "qref_validation": qv, "qref_cells": qcells}
    path = a.out or os.path.join(a.run_dir, "e1_result.json" if a.prior else "e1_result_noprior_v2.json")
    json.dump(out, open(path, "w"), indent=1, default=str)
    print(json.dumps({"task_prior_admission": adm, "qref_validation": qv}, indent=1, default=str))


if __name__ == "__main__":
    main()
