"""EGRA stage 1: window-rule anomaly points and swap/ratchet summaries from train_*.json (no GPU).

usage: python judge_s1.py S1_DIR OUT_JSON
Anomaly point (exp D rule, relative to the fork probe): start contrast (V_online_start - V_online_all) rise >= 15
AND start ||dQ/da|| >= 2x fork value. Ratchet: contrast change across consecutive swaps (pre-swap probe to the probe
10k later), cumulative.
"""
import sys, os, json, glob

def summarize(path):
    d = json.load(open(path))
    L = d["log"]
    c0 = L[0]["V_online_start"] - L[0]["V_online_all"]
    n0 = L[0]["dqda_norm_start"]
    pts, anomalies = [], []
    for r in L[1:]:
        rise = r["V_online_start"] - r["V_online_all"] - c0
        ratio = r["dqda_norm_start"] / n0
        pts.append((r["step"], rise, ratio))
        if rise >= 15 and ratio >= 2:
            anomalies.append(r["step"])
    run = 0; best = 0
    steps = [p[0] for p in pts]
    for s in steps:
        run = run + 1 if s in anomalies else 0
        best = max(best, run)
    byk = {r["step"]: r["V_online_start"] - r["V_online_all"] - c0 for r in L}
    ratchet = []
    for sw in d["swaps"]:
        s = sw["step"]
        pre = max([k for k in byk if k <= s], default=None)
        post = min([k for k in byk if k >= s + 10000], default=None)
        if pre is not None and post is not None:
            ratchet.append(byk[post] - byk[pre])
    last = L[-1]
    return dict(arm=d["arm"], run=d["run"], n_probe=len(pts), anomalies=anomalies, longest_run=best,
                max_rise=max(p[1] for p in pts), max_norm_ratio=max(p[2] for p in pts),
                final_rise=pts[-1][1], final_norm_ratio=pts[-1][2],
                n_swaps=len(d["swaps"]), swap_kinds={k: sum(1 for s in d["swaps"] if s["why"] == k) for k in set(s["why"] for s in d["swaps"])},
                n_brake=d["n_brake"], ratchet_cum=sum(ratchet), ratchet_max_step=max(ratchet) if ratchet else None,
                action_shift_start_per_probe=[round(r["action_shift_start"], 3) for r in L[::4]],
                value_lag_gap_mean=sum(r["value_lag_gap"] for r in L[1:]) / max(1, len(L) - 1),
                final_V_start=last["V_online_start"], final_lam=last["lam"])

def main():
    out = {}
    for p in sorted(glob.glob(f"{sys.argv[1]}/*/train_*.json")):
        s = summarize(p)
        out[f"{s['run']}:{s['arm']}"] = s
        print(f"{s['run']}:{s['arm']}", json.dumps({k: (round(v, 2) if isinstance(v, float) else v) for k, v in s.items() if k not in ("action_shift_start_per_probe",)}))
    json.dump(out, open(sys.argv[2], "w"), indent=1)

if __name__ == "__main__":
    main()
