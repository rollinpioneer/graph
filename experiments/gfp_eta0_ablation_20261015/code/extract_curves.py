"""Extract light training/eval curves from GFP eta=0 ablation runs (stdlib only).
Usage: python extract_curves.py RUNS_DIR OUT_DIR [STEP_STRIDE]
Writes curves.csv (validation metrics every STEP_STRIDE updates) and success_long.csv."""
import csv, glob, os, re, sys

runs_dir, out_dir = sys.argv[1], sys.argv[2]
stride = int(sys.argv[3]) if len(sys.argv) > 3 else 25000
os.makedirs(out_dir, exist_ok=True)
# Keys logged by agents/gfp.py (critic_loss, actor_loss, compute_guidance); guidance keys are absent in arm F.
COLS = {
    "q_mean": "validation/critic/q_mean", "q_max": "validation/critic/q_max",
    "tgt_q_mean": "validation/critic/tgt_q_mean",
    "distill_loss": "validation/actor/distill_loss", "VaBC_loss": "validation/actor/VaBC_loss",
    "q_loss": "validation/actor/q_loss",
    "guidance": "validation/actor/guidance", "g_above_10": "validation/actor/guidance_above_10",
    "g_above_75": "validation/actor/guidance_above_75",
    "q_data": "validation/actor/q_data", "q_ref": "validation/actor/q_ref",
}

curve_rows, succ_rows = [], []
for d in sorted(glob.glob(os.path.join(runs_dir, "*_task*_s*"))):
    m = re.match(r"([GF])_task(\d)_s(\d+)$", os.path.basename(d))
    if not m:
        continue
    key = dict(arm=m.group(1), task=int(m.group(2)), seed=int(m.group(3)))
    outs = glob.glob(os.path.join(d, "out", "*", "*"))
    if not outs:
        continue
    with open(os.path.join(outs[0], "train.csv")) as f:
        for r in csv.DictReader(f):
            step = int(float(r["step"]))
            if step % stride == 0:
                curve_rows.append(dict(key, step=step, **{k: r.get(c, "") for k, c in COLS.items()}))
    sp = os.path.join(outs[0], "success.csv")
    if os.path.exists(sp):
        with open(sp) as f:
            for i, r in enumerate(csv.DictReader(f)):
                # One row per evaluation, every eval_interval=100k updates; values are percentages.
                succ_rows.append(dict(key, step=100000 * (i + 1), success=r["success_Actor"], final=r.get("final_Actor", "")))

for name, rows in (("curves.csv", curve_rows), ("success_long.csv", succ_rows)):
    with open(os.path.join(out_dir, name), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
print(len(curve_rows), len(succ_rows))
