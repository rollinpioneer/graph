"""Summarize the GFP eta=0 ablation: arm G (official GFP) vs arm F (GFP with eta_temperature=0), paired by seed (stdlib only).
Usage: python summarize.py RUNS_DIR OUT_DIR  (RUNS_DIR holds G_task*_s* and F_task*_s*)"""
import csv, glob, json, math, os, re, sys

T95 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365}
runs_dir, out_dir = sys.argv[1], sys.argv[2]
os.makedirs(out_dir, exist_ok=True)

def read_csv(path):
    if not os.path.exists(path):
        return []
    with open(path) as f:
        return list(csv.DictReader(f))

def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None

# success.csv already stores percentages (utils/evaluation.py: 100 * eval_info["success"]).
rows = []
for d in sorted(glob.glob(os.path.join(runs_dir, "*_task*_s*"))):
    m = re.match(r"([GF])_task(\d)_s(\d+)$", os.path.basename(d))
    if not m:
        continue
    arm, task, seed = m.group(1), int(m.group(2)), int(m.group(3))
    outs = glob.glob(os.path.join(d, "out", "*", "*"))
    out = outs[0] if outs else None
    succ = read_csv(os.path.join(out, "success.csv")) if out else []
    train = read_csv(os.path.join(out, "train.csv")) if out else []
    curve = [num(r.get("success_Actor")) for r in succ]
    final = next((num(r.get("final_Actor")) for r in succ if num(r.get("final_Actor")) is not None), None)
    last = train[-1] if train else {}
    exitj = json.load(open(os.path.join(d, "exit.json"))) if os.path.exists(os.path.join(d, "exit.json")) else None
    rss = wall = None
    tv = os.path.join(d, "time_v.txt")
    if os.path.exists(tv):
        for line in open(tv):
            if "Maximum resident set size" in line:
                rss = int(line.split(":")[1]) / 1024 / 1024
            if "Elapsed (wall clock)" in line:
                wall = line.split("):")[-1].strip()
    rows.append(dict(
        arm=arm, task=task, seed=seed,
        step=int(float(last.get("step", 0) or 0)),
        n_evals=len(curve),
        success_by_100k=[None if c is None else round(c, 2) for c in curve],
        final_actor=None if final is None else round(final, 2),
        last3_mean=round(sum(curve[-3:]) / 3, 2) if len(curve) >= 10 else None,
        train_time_s=num(last.get("time/total_time")),
        val_q_mean=num(last.get("validation/critic/q_mean")),
        val_tgt_q_mean=num(last.get("validation/critic/tgt_q_mean")),
        val_distill=num(last.get("validation/actor/distill_loss")),
        val_vabc=num(last.get("validation/actor/VaBC_loss")),
        # Guidance keys exist only in arm G (eta=0 skips compute_guidance, agents/gfp.py:176-178).
        val_guidance=num(last.get("validation/actor/guidance")),
        val_g_above_10=num(last.get("validation/actor/guidance_above_10")),
        val_g_above_75=num(last.get("validation/actor/guidance_above_75")),
        exit=exitj, max_rss_gib=None if rss is None else round(rss, 2), wall=wall,
    ))

json.dump(rows, open(os.path.join(out_dir, "per_run.json"), "w"), indent=1)

paired = {}
for task in (1, 4):
    diffs = []
    for seed in sorted({r["seed"] for r in rows if r["task"] == task}):
        g = next((r for r in rows if r["task"] == task and r["seed"] == seed and r["arm"] == "G"), None)
        f = next((r for r in rows if r["task"] == task and r["seed"] == seed and r["arm"] == "F"), None)
        if g and f and g["final_actor"] is not None and f["final_actor"] is not None:
            diffs.append(dict(seed=seed, G=g["final_actor"], F=f["final_actor"], d=round(g["final_actor"] - f["final_actor"], 2)))
    res = dict(pairs=diffs, n=len(diffs))
    if diffs:
        ds = [x["d"] for x in diffs]
        mean = sum(ds) / len(ds)
        res.update(mean_G=round(sum(x["G"] for x in diffs) / len(diffs), 2),
                   mean_F=round(sum(x["F"] for x in diffs) / len(diffs), 2), mean_d=round(mean, 2))
        if len(ds) > 1:
            sd = math.sqrt(sum((x - mean) ** 2 for x in ds) / (len(ds) - 1))
            half = T95[len(ds) - 1] * sd / math.sqrt(len(ds))
            res.update(sd_d=round(sd, 2), ci95=[round(mean - half, 2), round(mean + half, 2)])
    paired[f"task{task}"] = res
json.dump(paired, open(os.path.join(out_dir, "paired.json"), "w"), indent=1)
print(json.dumps(paired, indent=1))
for r in rows:
    print(r["arm"], r["task"], r["seed"], "step", r["step"], "evals", r["n_evals"], "final", r["final_actor"],
          "curve", r["success_by_100k"], "g", r["val_guidance"], "q", r["val_q_mean"], "exit", r["exit"])
