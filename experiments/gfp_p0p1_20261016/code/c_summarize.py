"""Summarize c1 / c2 / P0-fixedref outputs and apply the frozen criteria of ANALYSIS_CARD_C_P0REF.md.

usage: python c_summarize.py CDIAG_DIR
"""
import sys
import json
import glob

import numpy as np

D = sys.argv[1]


def c1():
    r = json.load(open(f"{D}/c1_rank.json"))
    print(f"## c1  (n_start={r['n_start']} K={r['K']} H={r['H']} thr={r['thr']})")
    print("model | src | n_west | n_east | n_still | AUC(W>E) | Qwest-Qeast | dx median")
    verdict = {}
    for src in ("actor", "flow", "data"):
        aucs = []
        for m, v in r["models"].items():
            s = v[src]
            dec = s["n_west"] >= 10 and s["n_east"] >= 10
            dq = None if s["q_west"] is None or s["q_east"] is None else s["q_west"] - s["q_east"]
            print(f"{m} | {src} | {s['n_west']} | {s['n_east']} | {s['n_still']} | "
                  f"{'n/a' if s['auc_west_gt_east'] is None else round(s['auc_west_gt_east'], 3)}{'' if dec else ' (不可判)'} | "
                  f"{'n/a' if dq is None else round(dq, 2)} | {round(s['dx_quartiles'][1], 2)}")
            if dec:
                aucs.append((m, s["auc_west_gt_east"]))
        n = len(aucs)
        hi = sum(a >= 0.6 for _, a in aucs)
        lo = sum(a <= 0.4 for _, a in aucs)
        v = "critic 偏好贴墙" if n and hi >= 0.75 * n else ("critic 偏好绕行" if n and lo >= 0.75 * n else "排序不明确")
        verdict[src] = dict(n_decidable=n, n_auc_ge_0_6=hi, n_auc_le_0_4=lo, verdict=v)
    print("verdict", json.dumps(verdict, ensure_ascii=False))
    return verdict


def c2():
    res = {}
    for f in sorted(glob.glob(f"{D}/c2_*.json")):
        res.update(json.load(open(f)))
    rows = {}
    for k, v in res.items():
        rows.setdefault((v["name"].split("_")[1], v["arm"]), []).append(v)
    print("\n## c2 per model")
    print("model | arm | n | success | left_start | min_bfs")
    for k in sorted(res):
        v = res[k]
        print(f"{v['name']} | {v['arm']} | {v['n']} | {v['success']:.1f} | {v['left_start']:.1f} | {v['min_bfs_mean']:.2f}")
    print("\n## c2 pooled")
    pooled = {}
    for (task, arm), vs in sorted(rows.items()):
        eps = [e for v in vs for e in v["episodes"]]
        p = dict(n_models=len(vs), n=len(eps), success=100 * np.mean([e["success"] for e in eps]),
                 left_start=100 * np.mean([e["min_bfs"] < e["start_bfs"] for e in eps]),
                 min_bfs=float(np.mean([e["min_bfs"] for e in eps])))
        pooled[(task, arm)] = p
        print(f"{task} | {arm} | models={p['n_models']} n={p['n']} | success={p['success']:.1f} | "
              f"left_start={p['left_start']:.1f} | min_bfs={p['min_bfs']:.2f}")
    ls = {a: pooled[("task4", a)]["left_start"] for (t, a) in pooled if t == "task4"}
    out = {}
    if all(a in ls for a in ("A0", "A1", "A2", "A3", "A4")):
        out["Q排序有用"] = ls["A2"] - ls["A4"] >= 10 and ls["A2"] - ls["A1"] >= 10 and ls["A3"] < ls["A0"]
        out["Q排序推向墙"] = ls["A4"] - ls["A2"] >= 10 and ls["A3"] > ls["A0"]
        out["多采样或去随机有用"] = (ls["A4"] - ls["A0"] >= 10 or ls["A1"] - ls["A0"] >= 10) and abs(ls["A2"] - ls["A4"]) < 10
    if all(a in ls for a in ("F0", "F2", "F3")):
        out["流:Q排序有用(F2-F0>=10且F3<F0)"] = ls["F2"] - ls["F0"] >= 10 and ls["F3"] < ls["F0"]
        out["流:Q排序推向墙(F0-F2>=10且F3>F0)"] = ls["F0"] - ls["F2"] >= 10 and ls["F3"] > ls["F0"]
    print("verdict(task4 left_start)", json.dumps({k: bool(v) for k, v in out.items()}, ensure_ascii=False))
    return out


def p0():
    r = json.load(open(f"{D}/p0_fixedref.json"))
    print("\n## P0 fixed-reference (start region (1,1),(2,1),(1,2),(2,2))")
    print("pair | total | critic | actor | inter | data | action_shift_l2 | verdict")
    for k, v in r.items():
        s = v["start_region"]
        if abs(s["critic"]) >= 2 * abs(s["actor"]) and np.sign(s["data"]) == np.sign(s["critic"]):
            vd = "critic"
        elif abs(s["actor"]) >= 2 * abs(s["critic"]):
            vd = "actor"
        else:
            vd = "both"
        print(f"{k} | {s['total']:.2f} | {s['critic']:.2f} | {s['actor']:.2f} | {s['inter']:.2f} | {s['data']:.2f} | "
              f"{v['action_shift_l2']:.3f} | {vd}")


if __name__ == "__main__":
    c1()
    c2()
    p0()
