"""Step 2 screening: window-rule anomaly points and validity checks V1-V3 from each arm train.json (no GPU),
same rules as the X3 card (CARD_SCREEN_20261016.md section 5).

usage: python judge_screen.py S2_DIR OUT_JSON      (S2_DIR/<RUN>_<ARM><PARAM>/train.json)
Anomaly point (exp D rule, as judge_s1.py): start contrast (V_online_start - V_online_all) rise >= 15 vs the fork
probe AND start ||dQ/da|| >= 2x fork value. Validity at fork + 100k/150k/200k (all probe states):
V1 shift_50k >= 0.27, V2 gap_vabc <= 0.88, V3 vabc_shift >= 0.29.
"""
import sys, json, glob

V1, V2, V3 = 0.27, 0.88, 0.29


def summarize(path):
    d = json.load(open(path))
    if "invalid" in d:
        return dict(run=d["run"], invalid=d["invalid"], valid=False)
    L = d["log"]
    c0 = L[0]["V_online_start"] - L[0]["V_online_all"]
    n0 = L[0]["dqda_norm_start"]
    fork = L[0]["step"]
    pts, anomalies = [], []
    for r in L[1:]:
        rise = r["V_online_start"] - r["V_online_all"] - c0
        ratio = r["dqda_norm_start"] / n0
        pts.append((r["step"], rise, ratio))
        if rise >= 15 and ratio >= 2:
            anomalies.append(r["step"])
    by = {r["step"]: r for r in L}
    checks = {}
    for k in (100000, 150000, 200000):
        r = by[fork + k]
        checks[f"+{k // 1000}k"] = dict(shift_50k=r["shift_50k"][0], gap_vabc=r["gap_vabc"][0], vabc_shift=r["vabc_shift"][0],
                                       qadv=r["qadv"][0], V1=r["shift_50k"][0] >= V1, V2=r["gap_vabc"][0] <= V2,
                                       V3=r["vabc_shift"][0] >= V3)
    valid = all(c["V1"] and c["V2"] and c["V3"] for c in checks.values())
    # pre-registered "may only be delayed" flag (card section 6): at +200k the start contrast rise is >= 10 and still
    # rising over the last 50k, or the dQ/da ratio is >= 1.5
    last = [p for p in pts if p[0] >= fork + 150000]
    delayed = (pts[-1][1] >= 10 and pts[-1][1] > last[0][1]) or pts[-1][2] >= 1.5
    return dict(run=d["run"], fork=fork, n_probe=len(pts), anomalies=anomalies, max_rise=max(p[1] for p in pts),
                max_norm_ratio=max(p[2] for p in pts), final_rise=pts[-1][1], final_norm_ratio=pts[-1][2],
                V_start_fork=L[0]["V_online_start"], V_start_final=L[-1]["V_online_start"],
                checks=checks, valid=valid, delayed_flag=bool(delayed), sec=d.get("total_sec"))


def main():
    out = {}
    for p in sorted(glob.glob(f"{sys.argv[1]}/*/train.json")):
        s = summarize(p)
        out[p.split("/")[-2]] = s
        print(p.split("/")[-2], json.dumps(s))
    json.dump(out, open(sys.argv[2], "w"), indent=1)


if __name__ == "__main__":
    main()
