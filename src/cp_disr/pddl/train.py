"""Training of MG-G / DENSE-G / REL-G on grounded-task trajectories: the action-set likelihood + pairwise rank loss of the Blocksworld cards (same SerialTrainer, same batching / weighting / clipping),
all parameters trained from a common random initialisation with one learning rate."""
from __future__ import annotations

import random

import torch

from ..blocksworld.method_serial import trainer as ST
from .task import snapshot_at


class PddlTrainer(ST.SerialTrainer):
    def __init__(self, model, cases, rank_labels, device, lr=3e-4, chunk=48):
        super().__init__(model, "ACTION", cases, rank_labels, device)
        groups = [{"params": model.encoder_params() + model.existing_head_params(), "lr": lr}]
        if model.new_params():
            groups.append({"params": model.new_params(), "lr": lr})
        self.optimizer = torch.optim.Adam(groups, betas=(0.9, 0.999), eps=1e-8, weight_decay=0)
        self.params = [p for g in groups for p in g["params"]]
        self.chunk = chunk

    def snapshots(self, traj):
        got = self._snaps.get(traj["tid"])
        if got is None:
            c = self.cases[traj["case_id"]]
            got = [snapshot_at(c, s, t) for t, s in enumerate(traj["states"][:-1])]
            self._snaps[traj["tid"]] = got
        return got

    def epoch(self, trajs, seed):
        return super().epoch(trajs, seed)

    def update_batches(self, trajs, seed):
        """One shuffled pass as a list of batches of 32 trajectories (the last batch may be smaller)."""
        order = list(range(len(trajs)))
        random.Random(seed).shuffle(order)
        return [[trajs[i] for i in order[k:k + ST.BATCH]] for k in range(0, len(order), ST.BATCH)]
