"""Method-independent contract sampler. Rank-local shuffle, no cyclic reset."""
from __future__ import annotations
import hashlib
import random
from . import MOTIFS

def sampler_seed(run_seed, rank):
    text = f"sampler|{int(run_seed)}|{int(rank)}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") & 0x7fffffff


def run_seed(draw, policy_seed):
    text = f"P2CRL_V1|{draw}|{policy_seed}"
    return int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:4], "big") & 0x7fffffff


class BlockSampler:
    """Shuffle 32 motif-balanced blocks of 8 contracts (4 motifs x 2 sides)."""
    def __init__(self, contracts, seed):
        self.contracts = list(contracts)
        self.rng = random.Random(int(seed))
        self.blocks = self._make_blocks()
        self.flat = []
        self.i = 0
        self.exposures = [0] * len(self.contracts)
        self._refill()

    def _make_blocks(self):
        groups = {}
        for idx, c in enumerate(self.contracts):
            key = (c.motif, c.family_id)
            groups.setdefault(key, []).append(idx)
        per_motif = {m: [] for m in MOTIFS}
        for (motif, fid), idxs in groups.items():
            if len(idxs) != 2:
                raise ValueError("expected left/right pair")
            per_motif[motif].append(tuple(idxs))
        n = min(len(v) for v in per_motif.values())
        for m in MOTIFS:
            self.rng.shuffle(per_motif[m])
            if len(per_motif[m]) != n:
                raise ValueError("unbalanced motifs")
        blocks = []
        for i in range(n):
            block = []
            for m in MOTIFS:
                block.extend(per_motif[m][i])
            blocks.append(block)
        return blocks

    def _refill(self):
        order = list(range(len(self.blocks)))
        self.rng.shuffle(order)
        self.flat = [j for i in order for j in self.blocks[i]]
        self.i = 0

    def next_index(self):
        if self.i >= len(self.flat):
            self._refill()
        idx = self.flat[self.i]
        self.i += 1
        self.exposures[idx] += 1
        return idx

    def fingerprint(self):
        return {"i": self.i, "exposures": list(self.exposures), "rng": self.rng.getstate()[1][0]}
