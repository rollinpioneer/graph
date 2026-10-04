"""Corrected E2-rescue comparison (archive version). Dev only; single training seed per method.

Corrections w.r.t. alfworld_e2_report.py: checkpoints/eval points are labelled by each method's REAL transition count;
AUC is given on each method's own grid and on a common span; paired intervals are reported both clustered by game
(original) and by scene; the single-training-seed caveat is part of every table.
"""
import json
import os
import sys

import numpy as np

BASE = sys.argv[1] if len(sys.argv) > 1 else "runs/alfworld_prior_reliance/e2_rescue"
CATALOG = "runs/alfworld_prior_reliance/catalog.json"
METHODS = ["B2", "Full", "PRIOR_BIAS"]
BOOT = 10000
NOMINAL = (2048, 4096, 8192)


def jl(p):
    return [json.loads(l) for l in open(p)]


def trap(xs, ys):
    return float(sum((xs[i + 1] - xs[i]) * (ys[i] + ys[i + 1]) / 2 for i in range(len(xs) - 1)))


def auc_native(pts):
    xs, ys = [p["transitions"] for p in pts], [p["J"] for p in pts]
    span = xs[-1] - xs[0]
    return {"mean_J": trap(xs, ys) / span, "mean_gain_over_init": trap(xs, [y - ys[0] for y in ys]) / span, "span_transitions": span}


def auc_common(pts, span):
    xs, ys = [p["transitions"] for p in pts], [p["J"] for p in pts]
    grid = np.linspace(0, span, 400)
    yy = np.interp(grid, xs, ys)
    return {"mean_J": float(np.trapz(yy, grid) / span), "mean_gain_over_init": float(np.trapz(yy - ys[0], grid) / span), "span_transitions": span}


def cluster_ci(vals_by_cluster, seed=0):
    ks = sorted(vals_by_cluster)
    per = np.array([np.mean(vals_by_cluster[k]) for k in ks])
    rng = np.random.default_rng(seed)
    means = [float(per[rng.integers(0, len(per), len(per))].mean()) for _ in range(BOOT)]
    return [float(per.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)), len(ks)]


def main():
    scene = {r["gamefile"]: r["scene"] for r in json.load(open(CATALOG))}
    out = {"caveat": "ALL intervals come from ONE training seed per method (seed 0); they quantify dev-episode/scene sampling only and say nothing about stability across training seeds.", "methods": {}}
    curves = {}
    for m in METHODS:
        d = os.path.join(BASE, m + "_s0")
        ev = jl(os.path.join(d, "eval_log.jsonl"))
        orig = sorted([e for e in ev if e["mode"] == "original"], key=lambda e: e["transitions"])
        absn = sorted([e for e in ev if e["mode"] == "absent"], key=lambda e: e["transitions"])
        curves[m] = orig
        tr = jl(os.path.join(d, "train_log.jsonl"))[-1]
        out["methods"][m] = {"eval_points_real_transitions": [e["transitions"] for e in orig], "absent_eval_real_transitions": [e["transitions"] for e in absn],
                             "checkpoint_real_transitions": {str(n): next(e["transitions"] for e in orig if e["transitions"] >= n) for n in NOMINAL},
                             "dev_original": [{k: e.get(k) for k in ("transitions", "J", "success", "raw_steps", "n_checks", "tv_mean", "beta", "tanh_beta")} for e in orig],
                             "dev_absent": [{k: e.get(k) for k in ("transitions", "J", "success", "raw_steps", "n_checks")} for e in absn],
                             "residual_last_eval": {k: orig[-1].get(k) for k in ("residual_candidate_spread", "residual_abs_mean")},
                             "train_strata_final": tr["strata"], "final_transitions": tr["transitions"], "final_episodes": tr["episodes"],
                             "throughput": {k: tr.get(k) for k in ("wall", "t_collect", "t_update", "t_eval", "trans_per_hour_train_only")}}
    span = min(c[-1]["transitions"] for c in curves.values())
    for m in METHODS:
        out["methods"][m]["auc_native_grid"] = auc_native(curves[m])
        out["methods"][m]["auc_common_span"] = auc_common(curves[m], span)
    out["paired_dev_J"] = {}
    for n in NOMINAL:
        files, recs = {}, {}
        for m in METHODS:
            t = out["methods"][m]["checkpoint_real_transitions"][str(n)]
            files[m] = {"nominal": n, "real_transitions": t}
            recs[m] = {(r["game"], r["repeat"]): r["J"] for r in json.load(open(os.path.join(BASE, m + "_s0", "eval_%09d_original.json" % t)))}
        keys = sorted(recs["B2"])
        cmp = {}
        for a, b in (("Full", "B2"), ("PRIOR_BIAS", "B2"), ("Full", "PRIOR_BIAS")):
            bg, bs = {}, {}
            for g, r in keys:
                dv = recs[a][(g, r)] - recs[b][(g, r)]
                bg.setdefault(g, []).append(dv)
                bs.setdefault(scene[g], []).append(dv)
            cmp["%s - %s" % (a, b)] = {"mean": float(np.mean([x for v in bg.values() for x in v])), "game_cluster_ci95": cluster_ci(bg)[1:3] + [len(bg)], "scene_cluster_ci95": cluster_ci(bs)[1:3] + [len(bs)]}
        out["paired_dev_J"][str(n)] = {"real_transitions_by_method": files, "differences": cmp}
    json.dump(out, open(os.path.join(BASE, "e2_rescue_report_v2.json"), "w"), indent=1)
    L = ["# E2-rescue 对比报告（更正版；dev；每方法单训练 seed）", "", "> " + out["caveat"], "",
         "超参数来源：PPO lr=3e-4、4 个 epoch、minibatch 64、序列长度 16、残差上界 B=0.5 沿用仓库既有默认值（`torch_rl.PPO`、`neural.Policy`）；rollout=256、评价间隔 1024、评价重复 2 次、2048 关口由本轮 pilot 运行器设定，并非仓库默认。", "",
         "Belief-Optimal 的研究描述：基于训练集多-holder 模型与固定成本的搜索参照；它相对 Fixed Prior 的差距不是动态依赖收益。", ""]
    for m in METHODS:
        r = out["methods"][m]
        L += ["## %s" % m, "", "| 真实 transition 数 | J | 成功率 | raw steps | CHECK 数 | prior 引起的 TV |", "|---|---|---|---|---|---|"]
        for e in r["dev_original"]:
            L.append("| %d | %.4f | %.3f | %.2f | %.2f | %s |" % (e["transitions"], e["J"], e["success"], e["raw_steps"], e["n_checks"], ("%.1e" % e["tv_mean"]) if e["tv_mean"] is not None else "-"))
        L += ["", "AUC（本方法自身网格，平均 J / 相对初始增益）：%.4f / %.4f；共同区间 0–%d：%.4f / %.4f" % (r["auc_native_grid"]["mean_J"], r["auc_native_grid"]["mean_gain_over_init"], span, r["auc_common_span"]["mean_J"], r["auc_common_span"]["mean_gain_over_init"]),
              "checkpoint 实际 transition 数：%s" % json.dumps(r["checkpoint_real_transitions"]), ""]
    L += ["## 配对差（同一 dev episode；J）", ""]
    for n, v in out["paired_dev_J"].items():
        L += ["### 名义 %s（各方法实际 transition：%s）" % (n, ", ".join("%s=%d" % (m, x["real_transitions"]) for m, x in v["real_transitions_by_method"].items())), "", "| 比较 | 均值差 | game-cluster 95% CI | scene-cluster 95% CI |", "|---|---|---|---|"]
        for k, c in v["differences"].items():
            L.append("| %s | %+.4f | [%+.4f, %+.4f] (%d games) | [%+.4f, %+.4f] (%d scenes) |" % (k, c["mean"], *c["game_cluster_ci95"], *c["scene_cluster_ci95"]))
        L.append("")
    open(os.path.join(BASE, "e2_rescue_report.md"), "w").write("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
