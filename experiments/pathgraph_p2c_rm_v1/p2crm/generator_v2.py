"""Independent Reward-V2 confirmation / smoke families.

Namespace 1310000-1310399 confirmation; 1311000-1311099 smoke.
Does not import oracle analysis. Slot prefixes are topology-agnostic.
"""
from __future__ import annotations

from p2cq_research.generator import (
    MOTIFS, ROOTS_PER_MOTIF, SLOTS_PER_ROOT, MAKERS, slot_prefix_plan,
)
from p2cq_research.task_contract import N, TaskContract, make_empty_arrays

CONF_START = 1310000
SMOKE_START = 1311000


def family_id(split, motif, root):
    mi = MOTIFS.index(motif)
    if split == "confirmation":
        fid = CONF_START + mi * 100 + int(root)
        if not (CONF_START <= fid <= 1310399):
            raise ValueError("confirmation family overflow")
        return fid
    if split == "smoke":
        # Reserved window is only 1311000-1311099; keep motif stride 10.
        fid = SMOKE_START + mi * 10 + int(root)
        if not (SMOKE_START <= fid <= 1311099):
            raise ValueError("smoke family overflow")
        return fid
    raise ValueError("split")


def _ks_v2(root, n):
    cycle = (2, 1, 3, 2, 1, 3, 2, 1)
    rot = root % len(cycle)
    cyc = cycle[rot:] + cycle[:rot]
    return [cyc[i % len(cyc)] for i in range(n)]


def _horizon_v2(root):
    return (28, 36, 44)[root % 3]


def _geom_v2(root, i):
    off = 1.7 + 0.11 * (root % 11) + 0.02 * i
    ang = i * 0.91 + 0.07 * root
    sx, sy = off + 0.13 * i, 1.1 + 0.17 * i
    tx, ty = sx + 1.25 + 0.04 * i, sy + 0.95 + 0.025 * ang
    return (sx, sy), (tx, ty)


def _apply_schedule(contract: TaskContract, split, motif, root):
    data = contract.public_dict()
    data["split"] = split
    data["family_id"] = family_id(split, motif, root)
    data["horizon"] = _horizon_v2(root)
    data["version"] = "P2CRM_TASK_CONTRACT_V1"
    present = [i for i in range(N) if data["node_mask"][i]]
    ks = _ks_v2(root, len(present))
    rot = root % max(1, len(present))
    order = present[rot:] + present[:rot]
    # Rotate K and geometry across present indices; keep topology on original indices.
    old_k = list(data["transport_steps"])
    old_s = [list(x) for x in data["start_xy"]]
    old_t = [list(x) for x in data["target_xy"]]
    old_ids = list(data["node_ids"])
    for j, i in enumerate(present):
        src = order[j]
        data["transport_steps"][i] = ks[j]
        s, t = _geom_v2(root, j)
        data["start_xy"][i] = [float(s[0]), float(s[1])]
        data["target_xy"][i] = [float(t[0]), float(t[1])]
        data["node_ids"][i] = old_ids[src] + f"_r{root}"
    if motif == "ALTERNATIVE_COST":
        pairs = ((1, 3), (1, 2), (2, 3))
        kc, kd = pairs[root % 3]
        data["transport_steps"][2] = kc
        data["transport_steps"][3] = kd
    if motif == "INVALIDATION_RECOVERY":
        data["transport_steps"][3] = max(2, data["transport_steps"][3])
        thr = 1 + (root % data["transport_steps"][3])
        rule = dict(data["disturbance_rule"])
        if rule.get("kind") == "ONE_SHOT":
            rule["trigger_progress"] = int(thr)
            data["disturbance_rule"] = rule
    return TaskContract(data)


def pair_contracts(split, motif, root):
    left, right = MAKERS[motif]("development", root, "left"), MAKERS[motif]("development", root, "right")
    return _apply_schedule(left, split, motif, root), _apply_schedule(right, split, motif, root)


def iter_split(split):
    if split == "confirmation":
        roots = ROOTS_PER_MOTIF
    elif split == "smoke":
        roots = 2
    else:
        raise ValueError("split")
    for motif in MOTIFS:
        for root in range(roots):
            left, right = pair_contracts(split, motif, root)
            yield {
                "split": split,
                "motif": motif,
                "root": root,
                "family": str(left.family_id),
                "family_id": left.family_id,
                "left": left,
                "right": right,
                "slots": slot_prefix_plan(left),
            }


def planned_pairs(split):
    n_root = ROOTS_PER_MOTIF if split == "confirmation" else 2
    return len(MOTIFS) * n_root * SLOTS_PER_ROOT