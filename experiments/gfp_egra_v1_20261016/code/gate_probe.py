"""Label-free gate probe set for EGRA: N_P dataset states (uniform, fixed seed) and fixed noise z.

The gate only ever sees these observations, the critic and the reference actor; no map, cell label or success.
Cell labels are attached separately (gate_probe_cells) for reporting only.
"""
import numpy as np

GATE_SEED = 20261017
N_P = 4096
ACT_DIM = 21


def make_gate_probe(train_obs, n=N_P, seed=GATE_SEED):
    rng = np.random.default_rng(seed)
    idx = np.sort(rng.choice(train_obs.shape[0], n, replace=False))
    z = rng.standard_normal((n, ACT_DIM)).astype(np.float32)
    return idx, np.asarray(train_obs[idx], np.float32), z


def gate_probe_cells(obs):
    from common import xy_to_ij
    return xy_to_ij(obs[:, :2])
