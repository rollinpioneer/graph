"""Aggregate probe / frozen-eval / coverage outputs into small tables (CPU only).

usage: python aggregate.py ANALYSIS_DIR
writes ANALYSIS_DIR/out/summary/{probe_summary.csv, cell_maps.json, rollout_summary.json,
                                 frozen_eval_summary.csv}
"""
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

from common import MAZE_MAP, MAZE_UNIT, TASKS, bfs_dist, xy_to_ij, run_dir

GAMMA = 0.995


def cell_center(i, j):
    return np.array([j * MAZE_UNIT - 4.0, i * MAZE_UNIT - 4.0])


def bfs_direction(task):
    """Unit vector, per free cell, pointing along a shortest path to the goal."""
    info = TASKS[task]
    d = bfs_dist(info["goal_ij"])
    dirs = np.zeros(MAZE_MAP.shape + (2,))
    for i in range(MAZE_MAP.shape[0]):
        for j in range(MAZE_MAP.shape[1]):
            if MAZE_MAP[i, j] or d[i, j] < 0:
                continue
            if d[i, j] == 0:
                v = np.array(info["goal_xy"]) - cell_center(i, j)
            else:
                v = np.zeros(2)
                for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    a, b = i + di, j + dj
                    if MAZE_MAP[a, b] == 0 and d[a, b] == d[i, j] - 1:
                        u = cell_center(a, b) - cell_center(i, j)
                        v += u / np.linalg.norm(u)
            n = np.linalg.norm(v)
            dirs[i, j] = v / n if n > 0 else 0
    return d, dirs


def spearman(a, b):
    ra = np.argsort(np.argsort(a))
    rb = np.argsort(np.argsort(b))
    return float(np.corrcoef(ra, rb)[0, 1])


def implied_steps(v):
    # V = -(1 - gamma^T) / (1 - gamma)  =>  T = log(1 + V (1 - gamma)) / log(gamma)
    x = np.clip(1 + v * (1 - GAMMA), 1e-6, 1)
    return np.log(x) / np.log(GAMMA)


def main():
    A = sys.argv[1]
    P = f"{A}/out/probe"
    S = f"{A}/out/summary"
    os.makedirs(S, exist_ok=True)
    m = np.load(f"{P}/states.npz")
    xy = m["obs"][:, :2]
    ci, cj = m["cell_i"], m["cell_j"]
    disp = m["next_xy"] - xy
    dn = np.linalg.norm(disp, axis=1)
    moving = dn > 1e-4

    task_info = {}
    for t in (1, 4):
        d, dirs = bfs_direction(t)
        prog = np.where(moving, np.einsum("nk,nk->n", disp, dirs[ci, cj]) / np.maximum(dn, 1e-8), np.nan)
        eucl = np.linalg.norm(xy - np.array(TASKS[t]["goal_xy"]), axis=1)
        task_info[t] = dict(bfs=d, bfs_s=d[ci, cj], prog=prog, eucl=eucl)

    rows, cell_maps = [], {}
    for f in sorted(glob.glob(f"{P}/*_[0-9]*.npz")):
        base = os.path.basename(f)[:-4]
        name, step = base.rsplit("_", 1)
        arm, task, seed = name.split("_")
        t = int(task[4:])
        r = np.load(f)
        ti = task_info[t]
        g = r["g"]
        V = r["Qt_pi"]
        adv = r["Qt_data"] - np.minimum(r["Qt_pi"], r["Qt_flow"])
        ok = ~np.isnan(ti["prog"])
        fwd, bwd = ok & (ti["prog"] > 0.5), ok & (ti["prog"] < -0.5)
        row = dict(
            run=name, arm=arm, task=t, seed=int(seed[1:]), step=int(step),
            lam=float(r["lam"]), V_mean=float(V.mean()), Qdata_mean=float(r["Qt_data"].mean()),
            adv_mean=float(adv.mean()), g_mean=float(g.mean()), g_gt_075=float((g > 0.75).mean()),
            g_lt_025=float((g < 0.25).mean()),
            g_ess_frac=float(g.sum() ** 2 / (len(g) * (g ** 2).sum())),
            g_fwd=float(g[fwd].mean()), g_bwd=float(g[bwd].mean()),
            corr_g_prog=float(np.corrcoef(g[ok], ti["prog"][ok])[0, 1]),
            corr_V_bfs=float(np.corrcoef(V, ti["bfs_s"])[0, 1]),
            corr_V_eucl=float(np.corrcoef(V, ti["eucl"])[0, 1]),
            pi_flow_gap=float(r["pi_flow_gap"].mean()), pi_data_gap=float(r["pi_data_gap"].mean()),
            Qo_data_sd=float(r["Qo_data_sd"].mean()),
            d_a_pi=float(r["d_a_pi"].mean()) if "d_a_pi" in r else "",
            d_a_flow=float(r["d_a_flow"].mean()) if "d_a_flow" in r else "",
            d_V=float(r["d_Qt_pi"].mean()) if "d_Qt_pi" in r else "",
            d_V_absmean=float(np.abs(r["d_Qt_pi"]).mean()) if "d_Qt_pi" in r else "",
        )
        rows.append(row)
        # cell maps for the final checkpoint and for the s0 collapse window
        if int(step) == 1000000 or (name == "G_task1_s0" and int(step) in (600000, 700000)) or \
                (name == "F_task1_s0" and int(step) in (600000, 700000)):
            cm = {}
            for (i, j) in zip(*np.nonzero(MAZE_MAP == 0)):
                sel = (ci == i) & (cj == j)
                if sel.sum() < 20:
                    continue
                k = f"{i},{j}"
                cm[k] = dict(n=int(sel.sum()), V=float(V[sel].mean()),
                             T=float(implied_steps(V[sel]).mean()), g=float(g[sel].mean()),
                             adv=float(adv[sel].mean()), bfs=int(ti["bfs"][i, j]),
                             d_a_pi=float(r["d_a_pi"][sel].mean()) if "d_a_pi" in r else None)
            ks = list(cm)
            bfs_c = np.array([cm[k]["bfs"] for k in ks])
            eu_c = np.array([np.linalg.norm(cell_center(*map(int, k.split(","))) - np.array(TASKS[t]["goal_xy"])) for k in ks])
            V_c = np.array([cm[k]["V"] for k in ks])
            cell_maps[f"{name}:{step}"] = dict(
                cells=cm, spearman_V_bfs=spearman(-V_c, bfs_c), spearman_V_eucl=spearman(-V_c, eu_c))
            row["cell_spearman_negV_bfs"] = cell_maps[f"{name}:{step}"]["spearman_V_bfs"]
            row["cell_spearman_negV_eucl"] = cell_maps[f"{name}:{step}"]["spearman_V_eucl"]

    keys = list(rows[0].keys()) + ["cell_spearman_negV_bfs", "cell_spearman_negV_eucl"]
    with open(f"{S}/probe_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow(row)
    json.dump(cell_maps, open(f"{S}/cell_maps.json", "w"), indent=1)

    # frozen evals
    fe_rows, roll = [], {}
    for f in sorted(glob.glob(f"{A}/out/*.json")):
        if os.path.basename(f) in ("coverage.json", "precision.json", "smoke_eval.json"):
            continue
        res = json.load(open(f))
        for k, v in res.items():
            name, step = v["name"], v["step"]
            t = int(name.split("_")[1][4:])
            # training-time eval at the same step
            sc = list(csv.DictReader(open(f"{run_dir(name)}/success.csv")))
            train_sr = float(sc[step // 100000 - 1]["success_Actor"])
            eps = v["episodes"]
            d = task_info[t]["bfs"]
            fin_bfs = [int(d[e["final_ij"][0], e["final_ij"][1]]) for e in eps]
            min_bfs = []
            for e in eps:
                pi, pj = xy_to_ij(np.array(e["path"]))
                min_bfs.append(int(d[pi, pj].min()))
            fails = [e for e in eps if e["success"] < 0.5]
            end_cells = {}
            for e in fails:
                kk = f"{e['final_ij'][0]},{e['final_ij'][1]}"
                end_cells[kk] = end_cells.get(kk, 0) + 1
            fe_rows.append(dict(file=os.path.basename(f), run=name, step=step, n=v["n_episodes"],
                                frozen_success=v["success"], train_eval_success=train_sr,
                                mean_len_success=float(np.mean([e["length"] for e in eps if e["success"] > 0.5]))
                                if len(fails) < len(eps) else "",
                                fail_final_bfs_mean=float(np.mean([b for b, e in zip(fin_bfs, eps) if e["success"] < 0.5]))
                                if fails else "",
                                fail_min_bfs_mean=float(np.mean([b for b, e in zip(min_bfs, eps) if e["success"] < 0.5]))
                                if fails else ""))
            roll[k] = dict(success=v["success"], n=v["n_episodes"], fail_end_cells=end_cells,
                           fail_min_bfs_hist={int(b): int(sum(1 for x, e in zip(min_bfs, eps)
                                                              if x == b and e["success"] < 0.5))
                                              for b in sorted(set(min_bfs))})
    with open(f"{S}/frozen_eval_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(fe_rows[0].keys()))
        w.writeheader()
        for row in fe_rows:
            w.writerow(row)
    json.dump(roll, open(f"{S}/rollout_summary.json", "w"), indent=1)
    print("rows", len(rows), "cell_maps", len(cell_maps), "frozen", len(fe_rows))


if __name__ == "__main__":
    main()
