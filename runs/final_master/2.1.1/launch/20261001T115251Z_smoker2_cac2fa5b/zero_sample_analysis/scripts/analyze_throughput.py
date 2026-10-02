#!/usr/bin/env python3
import json, os, sys, glob, calendar, time, statistics as st
OUT = open("/home/xushijie2/tmp/r2/R2_OUT.txt").read().strip()
launch_hms = open(os.path.join(OUT, "throughput", "k1_launch_time.txt")).read().strip()
day = "2026-10-01"
def ep(hms): return calendar.timegm(time.strptime(day + "T" + hms, "%Y-%m-%dT%H:%M:%S"))
L = ep(launch_hms)
rows = [json.loads(l) for l in open(os.path.join(OUT, "throughput", "samples.jsonl")) if l.strip()]
plans = ["R-TB-E-0", "R-TB-DK-1", "R-TB-K-1"]
def rates(rs, plan):
    pts = [(r["t"], r["transition_lines"].get(plan)) for r in rs if r["transition_lines"].get(plan) is not None]
    if len(pts) < 2: return None
    dt = (pts[-1][0] - pts[0][0]) / 60.0; dn = pts[-1][1] - pts[0][1]
    act_n = act_t = 0.0
    for (t0, n0), (t1, n1) in zip(pts, pts[1:]):
        if n1 > n0: act_n += n1 - n0; act_t += (t1 - t0) / 60.0
    return {"minutes": round(dt, 2), "new_transitions": dn, "all_in_per_min": round(dn / dt, 2) if dt > 0 else None,
            "collection_phase_per_min": round(act_n / act_t, 2) if act_t > 0 else None, "collection_minutes": round(act_t, 2)}
def window(a, b):
    rs = [r for r in rows if a <= r["t"] <= b]
    if len(rs) < 2: return {"samples": len(rs)}
    out = {"samples": len(rs), "utc": [rs[0]["utc"], rs[-1]["utc"]], "per_worker": {p: rates(rs, p) for p in plans}}
    agg_all = sum((out["per_worker"][p] or {}).get("all_in_per_min") or 0 for p in plans)
    agg_col = sum((out["per_worker"][p] or {}).get("collection_phase_per_min") or 0 for p in plans)
    out["aggregate_all_in_per_min"] = round(agg_all, 2); out["aggregate_collection_phase_per_min"] = round(agg_col, 2)
    gp = {}
    for g in ("0", "1", "2"):
        u = [r["gpus"][g][0] for r in rs if isinstance(r.get("gpus"), dict) and g in r["gpus"]]
        m = [r["gpus"][g][1] for r in rs if isinstance(r.get("gpus"), dict) and g in r["gpus"]]
        gp[g] = {"util_mean": round(st.mean(u), 1), "util_max": max(u), "mem_mib_max": max(m), "mem_mib_min": min(m)}
    out["gpu"] = gp
    io = [r["iowait_frac"] for r in rs]; out["iowait"] = {"mean": round(st.mean(io), 4), "max": max(io)}
    out["loadavg1_max"] = max(r["loadavg"][0] for r in rs); out["free_bytes_min"] = min(r["free_bytes"] for r in rs)
    return out
def ppo_events():
    ev = {}
    for p, pat in (("R-TB-E-0", "T_B/B1-K+E/seed_0/R-TB-E-0-*"), ("R-TB-DK-1", "T_B/B2/seed_1/R-TB-DK-1-*"), ("R-TB-K-1", "T_B/B1-K/seed_1/R-TB-K-1-*")):
        d = glob.glob("/home/xushijie2/graph_cp_disr_final_tb_launch/runs/final_master/2.1.1/" + pat)
        if not d: continue
        pub = {}
        try:
            for l in open(os.path.join(d[0], "persistence", "raw_events.jsonl")):
                j = json.loads(l)
                if j.get("event") == "checkpoint_generation_published": pub[j["N"]] = j["time"]
        except Exception: pass
        rowsx = []
        for l in open(os.path.join(OUT, "logs", "train_%s.log" % p)):
            if "PPO complete update" in l:
                ts = l.split()[1].replace("Z", ""); n = int(l.split("N=")[1].split()[0])
                tstart = calendar.timegm(time.strptime(ts, "%Y-%m-%dT%H:%M:%S"))
                pt = pub.get(n); tend = calendar.timegm(time.strptime(pt, "%Y-%m-%dT%H:%M:%SZ")) if pt else None
                rowsx.append({"N": n, "ppo_start_utc": ts, "published_utc": pt, "wall_min": round((tend - tstart) / 60.0, 2) if tend else "IN_PROGRESS"})
        ev[p] = rowsx
    return ev
rep = {"k1_launch_utc": launch_hms,
       "baseline_2_workers_prelaunch": window(rows[0]["t"], L),
       "window_0_15min": window(L, L + 15 * 60), "window_0_30min": window(L, L + 30 * 60), "window_so_far": window(L, rows[-1]["t"]),
       "ppo_updates": ppo_events(), "now_utc": rows[-1]["utc"]}
json.dump(rep, open(os.path.join(OUT, "throughput", "window_report.json"), "w"), indent=1, sort_keys=True)
print(json.dumps(rep, indent=1, sort_keys=True))
