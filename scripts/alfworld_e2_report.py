"""E2-rescue comparison report for B2 / Full / PRIOR_BIAS (seed 0). Dev only."""
import json
import os
import sys

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "/home/xushijie2/xsj2_alf/e2_rescue"
METHODS = ["B2", "Full", "PRIOR_BIAS"]
BOOT = 10000


def jl(path):
    return [json.loads(l) for l in open(path)] if os.path.exists(path) else []


def auc(points, key="J"):
    xs = [p["transitions"] for p in points]
    ys = [p[key] for p in points]
    if len(xs) < 2:
        return None
    span = xs[-1] - xs[0]
    mean_level = float(np.trapz(ys, xs) / span)
    above_init = float(np.trapz([y - ys[0] for y in ys], xs) / span)
    return {"mean_J_over_curve": mean_level, "mean_gain_over_init": above_init, "span": span}


def boot_game_ci(diffs_by_game, seed=0):
    games = sorted(diffs_by_game)
    rng = np.random.default_rng(seed)
    vals = [np.mean(diffs_by_game[g]) for g in games]
    means = [float(np.mean([vals[i] for i in rng.integers(0, len(vals), len(vals))])) for _ in range(BOOT)]
    return float(np.mean(vals)), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def per_game(path):
    recs = json.load(open(path))
    d = {}
    for r in recs:
        d.setdefault(r["game"], []).append(r["J"])
    return {g: float(np.mean(v)) for g, v in d.items()}


def main():
    out = {"methods": {}}
    for m in METHODS:
        d = os.path.join(BASE, m + "_s0")
        ev = jl(os.path.join(d, "eval_log.jsonl"))
        tr = jl(os.path.join(d, "train_log.jsonl"))
        orig = sorted([e for e in ev if e["mode"] == "original"], key=lambda e: e["transitions"])
        absn = sorted([e for e in ev if e["mode"] == "absent"], key=lambda e: e["transitions"])
        gate = json.load(open(os.path.join(d, "gate_2048.json"))) if os.path.exists(os.path.join(d, "gate_2048.json")) else None
        done = json.load(open(os.path.join(d, "done.json"))) if os.path.exists(os.path.join(d, "done.json")) else None
        last = tr[-1] if tr else {}
        out["methods"][m] = {
            "gate": gate, "done": done,
            "dev_curve_original": [{k: e.get(k) for k in ("transitions", "J", "success", "raw_steps", "n_checks", "tv_mean", "beta", "tanh_beta")} for e in orig],
            "dev_absent": [{k: e.get(k) for k in ("transitions", "J", "success", "raw_steps", "n_checks")} for e in absn],
            "residual_at_last_eval": {k: orig[-1].get(k) for k in ("residual_candidate_spread", "residual_abs_mean")} if orig else None,
            "auc": auc(orig), "train_strata_final": last.get("strata"), "train_checks_per_episode_final": last.get("train_checks_per_episode"),
            "beta_trace": [(r["transitions"], r["beta"]) for r in tr if "beta" in r][::4],
            "throughput": {"transitions": last.get("transitions"), "episodes": last.get("episodes"), "wall_s": last.get("wall"), "t_collect": last.get("t_collect"),
                           "t_update": last.get("t_update"), "t_eval": last.get("t_eval"), "trans_per_hour_train_only": last.get("trans_per_hour_train_only")}}
    # paired comparisons on identical (game, repeat) evaluation episodes
    pairs = {}
    for mark_name in ("2048", "4096", "8192"):
        for m in METHODS:
            d = os.path.join(BASE, m + "_s0")
            cands = [f for f in os.listdir(d) if f.startswith("eval_") and f.endswith("_original.json")] if os.path.isdir(d) else []
            tr = [json.loads(l) for l in open(os.path.join(d, "train_log.jsonl"))] if os.path.exists(os.path.join(d, "train_log.jsonl")) else []
            marks = sorted(int(f.split("_")[1]) for f in cands)
            sel = [x for x in marks if x >= int(mark_name)]
            pairs.setdefault(mark_name, {})[m] = os.path.join(d, "eval_%09d_original.json" % sel[0]) if sel else None
    out["paired_dev_J"] = {}
    for mark_name, files in pairs.items():
        if not all(files.values()):
            continue
        pg = {m: per_game(f) for m, f in files.items()}
        games = sorted(pg["B2"])
        cmp = {}
        for a, b in (("Full", "B2"), ("PRIOR_BIAS", "B2"), ("Full", "PRIOR_BIAS")):
            cmp["%s - %s" % (a, b)] = boot_game_ci({g: [pg[a][g] - pg[b][g]] for g in games})
        out["paired_dev_J"][mark_name] = {"files": {m: os.path.basename(f) for m, f in files.items()}, "paired_gain_mean_ci95_over_games": cmp}
    json.dump(out, open(os.path.join(BASE, "e2_rescue_report.json"), "w"), indent=1)
    f = lambda x: "-" if x is None else ("%.4f" % x)
    for m in METHODS:
        r = out["methods"][m]
        print("==", m, "gate:", (r["gate"] or {}).get("technically_valid"), "| done:", bool(r["done"]))
        for e in r["dev_curve_original"]:
            print("  t=%5s J=%s succ=%s steps=%.2f checks=%.2f TV=%s%s" % (e["transitions"], f(e["J"]), f(e["success"]), e["raw_steps"], e["n_checks"], ("%.2e" % e["tv_mean"]) if e["tv_mean"] is not None else "-", ("  beta=%.3f tanh=%.3f" % (e["beta"], e["tanh_beta"])) if e.get("beta") is not None else ""))
        for e in r["dev_absent"]:
            print("  [absent-prior eval] t=%5s J=%s steps=%.2f checks=%.2f" % (e["transitions"], f(e["J"]), e["raw_steps"], e["n_checks"]))
        print("  AUC:", r["auc"], "| train strata:", r["train_strata_final"])
        print("  residual @last eval:", r["residual_at_last_eval"])
        print("  throughput:", r["throughput"])
    print(json.dumps(out["paired_dev_J"], indent=1))


if __name__ == "__main__":
    main()
