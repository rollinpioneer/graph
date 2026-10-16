"""Stage-1 confirmation: task-aware summary of exp_c_eval.py outputs (no GPU, no env).
exp_c_eval.py's own final_in_start_region field uses the task1 cells, so it is ignored here; every metric is
recomputed from the saved episodes (final_xy, final_ij, path sampled every XY_EVERY steps).

usage: python eval_t4.py TASK OUT_JSON EVAL_JSON [EVAL_JSON ...]
Per checkpoint (card section 4):
  success           % episodes with success
  final_start_cell  % episodes ending in the init cell
  west_wall_trap    % episodes ending in the init cell with final x <= init x - 1.0 (task4: x <= 15.0, the wall at
                    x = 14 is the trap seen in P0/P1)
  final_start_reg   % episodes ending within 2 maze steps of the init cell
  left_start_reg    % episodes whose sampled path ever reaches a cell more than 2 maze steps from the init cell
  min_bfs, final_bfs  mean over episodes of the smallest / final maze distance (cells) to the goal cell
"""
import sys
import json

sys.path.insert(0, "/home/xushijie3/work/gfp_eta0_ablation_20261015/analysis_p0p1/code")
import numpy as np
from common import bfs_dist, xy_to_ij, TASKS


def main():
    task, out = int(sys.argv[1]), sys.argv[2]
    T = TASKS[task]
    d_init, d_goal = bfs_dist(T["init_ij"]), bfs_dist(T["goal_ij"])
    res = {}
    for f in sys.argv[3:]:
        for k, r in json.load(open(f)).items():
            eps = r["episodes"]
            fin = [tuple(e["final_ij"]) for e in eps]
            path_ij = [list(zip(*xy_to_ij(np.asarray(e["path"])))) for e in eps]
            pc = lambda m: float(100 * np.mean(m))
            res[k] = dict(
                pkl=r["pkl"], branch=r["branch"], n=len(eps),
                success=pc([e["success"] for e in eps]),
                final_start_cell=pc([c == T["init_ij"] for c in fin]),
                west_wall_trap=pc([c == T["init_ij"] and e["final_xy"][0] <= T["init_xy"][0] - 1.0
                                   for c, e in zip(fin, eps)]),
                final_start_reg=pc([0 <= d_init[c] <= 2 for c in fin]),
                left_start_reg=pc([any(d_init[c] > 2 for c in p) for p in path_ij]),
                min_bfs=float(np.mean([min(d_goal[c] for c in p if d_goal[c] >= 0) for p in path_ij])),
                final_bfs=float(np.mean([d_goal[c] for c in fin])))
            print(k, json.dumps({a: round(b, 2) if isinstance(b, float) else b for a, b in res[k].items() if a != "pkl"}))
    json.dump(res, open(out, "w"), indent=1)


if __name__ == "__main__":
    main()
