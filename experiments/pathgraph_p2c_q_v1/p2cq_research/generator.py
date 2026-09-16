"""Four motif families with frozen root/parameter control.

Development namespace 1301000, confirmation 1302000. Slot prefixes are
topology-agnostic so left/right local facts stay matched. Oracle sets are
NOT consulted when registering slots.
"""
from __future__ import annotations

from copy import deepcopy

from .task_contract import (
    N, TaskContract, add_clause, add_goal_clause, make_empty_arrays,
    set_empty_precondition, set_invalidation_from_preconditions,
)

MOTIFS = ("PRECEDENCE", "SHARED_PREREQUISITE", "ALTERNATIVE_COST", "INVALIDATION_RECOVERY")
ROOTS_PER_MOTIF = 32
SLOTS_PER_ROOT = 16
DEV_START = 1301000
CONF_START = 1302000


def family_id(split, motif, root):
    base = DEV_START if split == "development" else CONF_START
    return base + MOTIFS.index(motif) * 100 + root


def _place_node(data, idx, name, start, target, k):
    data["node_mask"][idx] = True
    data["node_ids"][idx] = name
    data["start_xy"][idx] = [float(start[0]), float(start[1])]
    data["target_xy"][idx] = [float(target[0]), float(target[1])]
    data["transport_steps"][idx] = int(k)
    set_empty_precondition(data, idx)


def _geom(root, split, i):
    off = 0.15 * (root % 8) + (0.4 if split == "confirmation" else 0.0)
    ang = i * 0.7 + 0.05 * root
    sx, sy = off + 0.1 * i, 0.2 * i
    tx, ty = sx + 1.0 + 0.05 * i, sy + 0.8 + 0.03 * ang
    return (sx, sy), (tx, ty)


def _ks(root, n):
    cycle = (1, 1, 2, 1, 2, 3, 1, 2)
    return [cycle[(root + i) % len(cycle)] for i in range(n)]


def _horizon(root):
    return 32 if root % 2 == 0 else 40


def _base(split, motif, root, n_real, names):
    data = make_empty_arrays()
    ks = _ks(root, n_real)
    for i, name in enumerate(names):
        s, t = _geom(root, split, i)
        _place_node(data, i, name, s, t, ks[i])
    data.update({
        "version": "P2CQ_TASK_CONTRACT_V1",
        "motif": motif,
        "family_id": family_id(split, motif, root),
        "split": split,
        "gamma": 0.99,
        "horizon": _horizon(root),
        "disturbance_rule": {"kind": "NONE"},
    })
    return data


def make_precedence(split, root, side):
    data = _base(split, "PRECEDENCE", root, 2, ["A", "B"])
    # clear default empty pre; rebuild
    data["preconditions_dnf"] = [[[False] * N for _ in range(2)] for _ in range(N)]
    data["precondition_clause_mask"] = [[False] * 2 for _ in range(N)]
    set_empty_precondition(data, 0)
    set_empty_precondition(data, 1)
    if side == "left":
        # A free, B requires A
        data["precondition_clause_mask"][1] = [False, False]
        data["preconditions_dnf"][1] = [[False] * N for _ in range(2)]
        add_clause(data, 1, [0])
    else:
        data["precondition_clause_mask"][0] = [False, False]
        data["preconditions_dnf"][0] = [[False] * N for _ in range(2)]
        add_clause(data, 0, [1])
        set_empty_precondition(data, 1)
    add_goal_clause(data, [0, 1])
    set_invalidation_from_preconditions(data)
    return TaskContract(data)


def make_shared(split, root, side):
    data = _base(split, "SHARED_PREREQUISITE", root, 4, ["P", "Q", "A", "B"])
    data["preconditions_dnf"] = [[[False] * N for _ in range(2)] for _ in range(N)]
    data["precondition_clause_mask"] = [[False] * 2 for _ in range(N)]
    for i in range(4):
        set_empty_precondition(data, i)
    # A,B require P (left) or Q (right)
    src = 0 if side == "left" else 1
    for node in (2, 3):
        data["precondition_clause_mask"][node] = [False, False]
        data["preconditions_dnf"][node] = [[False] * N for _ in range(2)]
        add_clause(data, node, [src])
    add_goal_clause(data, [2, 3])
    set_invalidation_from_preconditions(data)
    return TaskContract(data)


def make_alternative(split, root, side):
    names = ["A", "B", "C", "D", "G"]
    data = _base(split, "ALTERNATIVE_COST", root, 5, names)
    # C cheap, D expensive: force K_C=1, K_D=3 regardless of cycle, geometry unchanged
    data["transport_steps"][2] = 1
    data["transport_steps"][3] = 3
    data["preconditions_dnf"] = [[[False] * N for _ in range(2)] for _ in range(N)]
    data["precondition_clause_mask"] = [[False] * 2 for _ in range(N)]
    for i in range(5):
        set_empty_precondition(data, i)
    # left: C<-A, D<-B; right: C<-B, D<-A. G = C OR D
    if side == "left":
        c_src, d_src = 0, 1
    else:
        c_src, d_src = 1, 0
    data["precondition_clause_mask"][2] = [False, False]
    data["preconditions_dnf"][2] = [[False] * N for _ in range(2)]
    add_clause(data, 2, [c_src])
    data["precondition_clause_mask"][3] = [False, False]
    data["preconditions_dnf"][3] = [[False] * N for _ in range(2)]
    add_clause(data, 3, [d_src])
    data["precondition_clause_mask"][4] = [False, False]
    data["preconditions_dnf"][4] = [[False] * N for _ in range(2)]
    add_clause(data, 4, [2])
    add_clause(data, 4, [3])
    add_goal_clause(data, [4])
    set_invalidation_from_preconditions(data)
    return TaskContract(data)


def make_invalidation(split, root, side):
    data = _base(split, "INVALIDATION_RECOVERY", root, 4, ["P", "Q", "A", "B"])
    # B needs K>=2 so mid-transport exists
    data["transport_steps"][3] = max(2, data["transport_steps"][3])
    data["preconditions_dnf"] = [[[False] * N for _ in range(2)] for _ in range(N)]
    data["precondition_clause_mask"] = [[False] * 2 for _ in range(N)]
    for i in range(4):
        set_empty_precondition(data, i)
    src = 0 if side == "left" else 1
    for node in (2, 3):
        data["precondition_clause_mask"][node] = [False, False]
        data["preconditions_dnf"][node] = [[False] * N for _ in range(2)]
        add_clause(data, node, [src])
    add_goal_clause(data, [2, 3])
    set_invalidation_from_preconditions(data)
    data["disturbance_rule"] = {
        "kind": "ONE_SHOT",
        "trigger_node": 3,
        "target_node": 0,
        "trigger_progress": 1,
        "one_shot": True,
    }
    return TaskContract(data)


MAKERS = {
    "PRECEDENCE": make_precedence,
    "SHARED_PREREQUISITE": make_shared,
    "ALTERNATIVE_COST": make_alternative,
    "INVALIDATION_RECOVERY": make_invalidation,
}


def pair_contracts(split, motif, root):
    fn = MAKERS[motif]
    return fn(split, root, "left"), fn(split, root, "right")


def slot_prefix_plan(contract):
    """Topology-agnostic action sequences. Invalid padding actions are no-ops."""
    pad = 5
    # ACQUIRE padding is invalid if node 5 unused
    from .task_contract import action_index
    w = 0
    a0 = action_index(0, "ACQUIRE")
    r0 = action_index(0, "RELEASE")
    d0 = action_index(0, "ADVANCE")
    a1 = action_index(1, "ACQUIRE")
    r1 = action_index(1, "RELEASE")
    ap = action_index(pad, "ACQUIRE")
    dp = action_index(pad, "ADVANCE")
    return [
        [],
        [w],
        [w, w],
        [w, w, w],
        [ap],
        [w, w, w, w],
        [a0],
        [a0, w],
        [a0, r0],
        [a1],
        [a1, w],
        [a1, r1],
        [w, w, w, w, w],
        [dp],
        [a0, d0],
        [a0, d0, r0],
    ]


def iter_split(split):
    if split not in ("development", "confirmation"):
        raise ValueError("split")
    for motif in MOTIFS:
        for root in range(ROOTS_PER_MOTIF):
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
