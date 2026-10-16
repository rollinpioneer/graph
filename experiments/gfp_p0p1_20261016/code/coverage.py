"""Dataset coverage for task1 vs task4 (full training dataset, CPU only).

usage: python coverage.py OUT_JSON
"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import ogbench

from common import MAZE_MAP, TASKS, GOAL_TOL, xy_to_ij, bfs_dist


def main():
    out = sys.argv[1]
    _, tr, _ = ogbench.make_env_and_datasets("humanoidmaze-medium-navigate-singletask-task1-v0")
    _, tr4, _ = ogbench.make_env_and_datasets("humanoidmaze-medium-navigate-singletask-task4-v0")
    xy = tr["observations"][:, :2]
    term = tr["terminals"].astype(bool)
    N = len(xy)
    ci, cj = xy_to_ij(xy)
    # episode boundaries: terminals mark the last transition of each episode
    ends = np.flatnonzero(term)
    starts = np.concatenate([[0], ends[:-1] + 1])
    res = dict(n_transitions=int(N), n_episodes=int(len(ends)),
               episode_len=[int(np.min(ends - starts + 1)), int(np.max(ends - starts + 1))])
    counts = np.zeros(MAZE_MAP.shape, int)
    np.add.at(counts, (ci, cj), 1)
    res["cell_counts"] = counts.tolist()

    for t, ds in ((1, tr), (4, tr4)):
        info = TASKS[t]
        gi, gj = info["goal_ij"]
        si, sj = info["init_ij"]
        # OGBench relabels with the CURRENT state's qpos xy (= observations[:, :2])
        dg = np.linalg.norm(xy - np.array(info["goal_xy"]), axis=1)
        succ = dg <= GOAL_TOL
        assert np.array_equal(succ, ds["rewards"] == 0), "reward relabel mismatch"
        bfs = bfs_dist(info["goal_ij"])
        r = dict(bfs_dist=bfs.tolist(),
                 n_success_transitions=int(succ.sum()),
                 n_within_1=int((dg <= 1.0).sum()), n_within_2=int((dg <= 2.0).sum()),
                 n_goal_cell=int(((ci == gi) & (cj == gj)).sum()),
                 n_start_cell=int(((ci == si) & (cj == sj)).sum()))
        # per-episode: reaches goal tol? passes start cell then goal tol later? reaches from where?
        reach, start_to_goal, steps_s2g, from_cells = 0, 0, [], {}
        for a, b in zip(starts, ends + 1):
            s = succ[a:b]
            if not s.any():
                continue
            reach += 1
            first = a + int(np.argmax(s))
            # cell at which this episode was 300 env steps (= 6 cells at ~?) before reaching
            seg_i, seg_j = ci[a:first + 1], cj[a:first + 1]
            at_start = np.flatnonzero((seg_i == si) & (seg_j == sj))
            if len(at_start):
                start_to_goal += 1
                steps_s2g.append(int(first - (a + at_start[-1])))
            # where the episode entered the goal's BFS-distance-1 neighbourhood from
            d = bfs[seg_i, seg_j]
            far = np.flatnonzero(d >= 2)
            key = f"{int(seg_i[far[-1]])},{int(seg_j[far[-1]])}" if len(far) else "start<2"
            from_cells[key] = from_cells.get(key, 0) + 1
        r.update(n_episodes_reaching_goal=reach, n_episodes_start_cell_then_goal=start_to_goal,
                 steps_start_to_goal=[int(np.min(steps_s2g)), int(np.median(steps_s2g)),
                                      int(np.max(steps_s2g))] if steps_s2g else None,
                 last_cell_with_bfs_ge2_before_goal=from_cells)
        # transitions per BFS distance band
        d_all = bfs[ci, cj]
        r["transitions_by_bfs"] = {int(k): int((d_all == k).sum()) for k in np.unique(d_all)}
        res[f"task{t}"] = r
    json.dump(res, open(out, "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "cell_counts"}, indent=1)[:4000])


if __name__ == "__main__":
    main()
