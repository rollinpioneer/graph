"""Shared helpers for the P0/P1 analysis (no training).

Run with cwd = official GFP source (source/gfp) so that `agents` imports the
unmodified official code.
"""
import glob
import json
import os
import pickle
from collections import deque

import sys

import numpy as np

# official source tree (cwd) must be importable for `agents`
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())

ROOT = "/home/xushijie3/work/gfp_eta0_ablation_20261015"
ACT_DIM = 21
OB_DIM = 69

# OGBench humanoidmaze-medium (probed on the remote, ogbench 1.1.0)
MAZE_MAP = np.array([
    [1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 1, 1, 0, 0, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 0, 0, 0, 1, 1, 1],
    [1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 1, 0, 0, 1, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1],
])
MAZE_UNIT = 4.0
OFFSET = 4.0
GOAL_TOL = 0.5
TASKS = {
    1: dict(init_ij=(1, 1), goal_ij=(6, 6), init_xy=(0.0, 0.0), goal_xy=(20.0, 20.0)),
    4: dict(init_ij=(6, 5), goal_ij=(6, 1), init_xy=(16.0, 20.0), goal_xy=(0.0, 20.0)),
}


def xy_to_ij(xy):
    xy = np.asarray(xy)
    i = np.floor((xy[..., 1] + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT).astype(int)
    j = np.floor((xy[..., 0] + OFFSET + 0.5 * MAZE_UNIT) / MAZE_UNIT).astype(int)
    return i, j


def bfs_dist(goal_ij):
    """Cell distance to goal over free cells (4-neighbour)."""
    d = np.full(MAZE_MAP.shape, -1, dtype=int)
    d[goal_ij] = 0
    q = deque([goal_ij])
    while q:
        i, j = q.popleft()
        for di, dj in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            a, b = i + di, j + dj
            if MAZE_MAP[a, b] == 0 and d[a, b] < 0:
                d[a, b] = d[i, j] + 1
                q.append((a, b))
    return d


def run_dir(name):
    c = glob.glob(f"{ROOT}/runs/formal/{name}/out/*/date_*")
    assert len(c) == 1, c
    return c[0]


def load_agent(name, step):
    from agents import agents
    import flax

    rd = run_dir(name)
    cfg = json.load(open(f"{rd}/config.json"))
    ca = dict(cfg["agent"])
    agent = agents[ca["agent_name"]].create(
        cfg["seed"],
        np.zeros((1, OB_DIM), np.float32),
        np.zeros((1, ACT_DIM), np.float32),
        ca,
    )
    path = f"{rd}/params_{step}.pkl"
    with open(path, "rb") as f:
        d = pickle.load(f)
    agent = flax.serialization.from_state_dict(agent, d["agent"])
    return agent, cfg, path
