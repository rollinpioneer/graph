"""Strict TaskContract / public-state types for P2C-Q.

No graph, oracle, or potential imports. Canonical hashes treat node renaming
as the same topology; geometry and durations are parameter hashes.
"""
from __future__ import annotations

import hashlib
import json
from copy import deepcopy
from itertools import permutations

N = 6
MAX_CLAUSES = 2
K_ALLOWED = (1, 2, 3)
OPS = ("ACQUIRE", "ADVANCE", "PLACE", "START_RECOVERY", "REGRASP", "RELEASE")
ACTION_NAMES = ("WAIT",) + tuple(f"{op}_{j}" for j in range(N) for op in OPS)
assert len(ACTION_NAMES) == 37


def _finite_num(x):
    if type(x) is bool or not isinstance(x, (int, float)):
        raise ValueError("numeric field must be a real number")
    if x != x or x in (float("inf"), float("-inf")):
        raise ValueError("non-finite")
    return float(x)


def _bool(x, name="bool"):
    if type(x) is not bool:
        raise ValueError(f"{name} must be bool")
    return x


def _int(x, name="int"):
    if type(x) is bool or type(x) is not int:
        raise ValueError(f"{name} must be int")
    return x


def canonical(obj):
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def digest(obj):
    return hashlib.sha256(canonical(obj).encode("ascii")).hexdigest()


def action_index(node, op):
    return 1 + 6 * node + OPS.index(op)


def node_op(action):
    if action == 0:
        return None, "WAIT"
    if not 1 <= action <= 36:
        raise ValueError("action")
    j, op = divmod(action - 1, 6)
    return j, OPS[op]


def _shape(seq, dims, leaf=None):
    if len(dims) == 0:
        return leaf(seq) if leaf else seq
    if not isinstance(seq, (list, tuple)) or len(seq) != dims[0]:
        raise ValueError("shape")
    return [_shape(x, dims[1:], leaf) for x in seq]


def dnf_clause_satisfied(clause, mask_on, valid):
    if not mask_on:
        return False
    for i, bit in enumerate(clause):
        if bit and not valid[i]:
            return False
    return True


def dnf_satisfied(dnf, clause_mask, valid):
    return any(dnf_clause_satisfied(dnf[c], clause_mask[c], valid) for c in range(len(clause_mask)))


def empty_clause(clause):
    return all(not bit for bit in clause)


class TaskContract:
    __slots__ = (
        "node_mask", "node_ids", "start_xy", "target_xy", "transport_steps",
        "preconditions_dnf", "precondition_clause_mask", "goal_dnf",
        "goal_clause_mask", "invalidation_dependency", "disturbance_rule",
        "horizon", "gamma", "motif", "family_id", "split", "version",
    )

    def __init__(self, data):
        d = data
        self.version = d.get("version", "P2CQ_TASK_CONTRACT_V1")
        self.motif = d["motif"]
        self.family_id = _int(d["family_id"], "family_id")
        self.split = d["split"]
        self.gamma = _finite_num(d.get("gamma", 0.99))
        if not 0 < self.gamma < 1:
            raise ValueError("gamma")
        self.horizon = _int(d["horizon"], "horizon")
        if not 1 <= self.horizon <= 128:
            raise ValueError("horizon")
        self.node_mask = _shape(d["node_mask"], [N], lambda x: _bool(x, "node_mask"))
        self.node_ids = _shape(d["node_ids"], [N], lambda x: x if isinstance(x, str) else (_ for _ in ()).throw(ValueError("id")))
        self.start_xy = _shape(d["start_xy"], [N, 2], _finite_num)
        self.target_xy = _shape(d["target_xy"], [N, 2], _finite_num)
        self.transport_steps = _shape(d["transport_steps"], [N], _int)
        self.preconditions_dnf = _shape(d["preconditions_dnf"], [N, MAX_CLAUSES, N], lambda x: _bool(x, "pre_dnf"))
        self.precondition_clause_mask = _shape(d["precondition_clause_mask"], [N, MAX_CLAUSES], lambda x: _bool(x, "pre_mask"))
        self.goal_dnf = _shape(d["goal_dnf"], [MAX_CLAUSES, N], lambda x: _bool(x, "goal_dnf"))
        self.goal_clause_mask = _shape(d["goal_clause_mask"], [MAX_CLAUSES], lambda x: _bool(x, "goal_mask"))
        self.invalidation_dependency = _shape(d["invalidation_dependency"], [N, N], lambda x: _bool(x, "inv"))
        self.disturbance_rule = dict(d["disturbance_rule"])
        self.validate()

    def present_nodes(self):
        return [i for i in range(N) if self.node_mask[i]]

    def n_real(self):
        return len(self.present_nodes())

    def validate(self):
        ids = []
        for i in range(N):
            if self.node_mask[i]:
                if self.transport_steps[i] not in K_ALLOWED:
                    raise ValueError("K")
                if not self.node_ids[i] or self.node_ids[i] in ids:
                    raise ValueError("node id")
                ids.append(self.node_ids[i])
            else:
                if self.transport_steps[i] != 0:
                    raise ValueError("padding K")
                if any(self.precondition_clause_mask[i]) or any(any(row) for row in self.preconditions_dnf[i]):
                    raise ValueError("padding preconditions")
                if any(self.invalidation_dependency[i]) or any(self.invalidation_dependency[j][i] for j in range(N)):
                    raise ValueError("padding invalidation")
        if not 2 <= self.n_real() <= 6:
            raise ValueError("node count")
        if not any(self.goal_clause_mask):
            raise ValueError("empty goal")
        for c, on in enumerate(self.goal_clause_mask):
            if on:
                if empty_clause(self.goal_dnf[c]):
                    raise ValueError("empty goal clause")
                for i, bit in enumerate(self.goal_dnf[c]):
                    if bit and not self.node_mask[i]:
                        raise ValueError("goal padding")
            elif any(self.goal_dnf[c]):
                raise ValueError("masked goal bits")
        for i in self.present_nodes():
            if not any(self.precondition_clause_mask[i]):
                raise ValueError("need explicit empty-precondition clause")
            for c, on in enumerate(self.precondition_clause_mask[i]):
                if on:
                    for j, bit in enumerate(self.preconditions_dnf[i][c]):
                        if bit:
                            if not self.node_mask[j]:
                                raise ValueError("pre padding")
                            if j == i:
                                raise ValueError("self pre")
                elif any(self.preconditions_dnf[i][c]):
                    raise ValueError("masked pre bits")
            # at least one active clause; empty clause is allowed
        self._check_acyclic()
        rule = self.disturbance_rule
        kind = rule.get("kind")
        if kind not in ("NONE", "ONE_SHOT"):
            raise ValueError("disturbance kind")
        if kind == "NONE":
            extra = {k: v for k, v in rule.items() if k != "kind"}
            if extra:
                raise ValueError("NONE extras")
        else:
            trig = _int(rule["trigger_node"], "trigger")
            tgt = _int(rule["target_node"], "target")
            thr = _int(rule["trigger_progress"], "thr")
            if not self.node_mask[trig] or not self.node_mask[tgt]:
                raise ValueError("disturbance padding")
            if not 1 <= thr <= self.transport_steps[trig]:
                raise ValueError("trigger threshold")
            if not _bool(rule.get("one_shot", True), "one_shot"):
                raise ValueError("one_shot")

    def _check_acyclic(self):
        # Edge i->j if j's precondition mentions i (j depends on i).
        adj = {i: [] for i in range(N)}
        for j in self.present_nodes():
            for c, on in enumerate(self.precondition_clause_mask[j]):
                if not on:
                    continue
                for i, bit in enumerate(self.preconditions_dnf[j][c]):
                    if bit:
                        adj[i].append(j)
        temp, done = set(), set()

        def dfs(u):
            if u in done:
                return
            if u in temp:
                raise ValueError("cyclic preconditions")
            temp.add(u)
            for v in adj[u]:
                dfs(v)
            temp.remove(u)
            done.add(u)

        for i in self.present_nodes():
            dfs(i)

    def preconditions_satisfied(self, node, valid):
        return dnf_satisfied(self.preconditions_dnf[node], self.precondition_clause_mask[node], valid)

    def goal_satisfied(self, valid):
        return dnf_satisfied(self.goal_dnf, self.goal_clause_mask, valid)

    def public_dict(self):
        return {
            "version": self.version,
            "motif": self.motif,
            "family_id": self.family_id,
            "split": self.split,
            "gamma": self.gamma,
            "horizon": self.horizon,
            "node_mask": list(self.node_mask),
            "node_ids": list(self.node_ids),
            "start_xy": deepcopy(self.start_xy),
            "target_xy": deepcopy(self.target_xy),
            "transport_steps": list(self.transport_steps),
            "preconditions_dnf": deepcopy(self.preconditions_dnf),
            "precondition_clause_mask": deepcopy(self.precondition_clause_mask),
            "goal_dnf": deepcopy(self.goal_dnf),
            "goal_clause_mask": list(self.goal_clause_mask),
            "invalidation_dependency": deepcopy(self.invalidation_dependency),
            "disturbance_rule": dict(self.disturbance_rule),
        }

    def parameter_payload(self):
        return {
            "start_xy": self.start_xy,
            "target_xy": self.target_xy,
            "transport_steps": self.transport_steps,
            "horizon": self.horizon,
            "disturbance_progress": self.disturbance_rule.get("trigger_progress"),
        }

    def structure_payload(self, perm=None):
        """Topology without names/geometry. perm maps old index -> new index."""
        if perm is None:
            perm = list(range(N))
        inv = [0] * N
        for a, b in enumerate(perm):
            inv[b] = a

        def map_bits(bits):
            out = [False] * N
            for i, bit in enumerate(bits):
                if bit:
                    out[perm[i]] = True
            return out

        nodes = [i for i in range(N) if self.node_mask[inv[i]]] if False else None
        # Build in new index order: new_i = perm[old_i], we iterate new indices
        node_mask = [False] * N
        pre_dnf = [[[False] * N for _ in range(MAX_CLAUSES)] for _ in range(N)]
        pre_mask = [[False] * MAX_CLAUSES for _ in range(N)]
        goal = [[False] * N for _ in range(MAX_CLAUSES)]
        goal_mask = [False] * MAX_CLAUSES
        invdep = [[False] * N for _ in range(N)]
        for old in range(N):
            new = perm[old]
            node_mask[new] = self.node_mask[old]
            if not self.node_mask[old]:
                continue
            pre_mask[new] = list(self.precondition_clause_mask[old])
            for c in range(MAX_CLAUSES):
                pre_dnf[new][c] = map_bits(self.preconditions_dnf[old][c])
            for old_j in range(N):
                invdep[new][perm[old_j]] = self.invalidation_dependency[old][old_j]
        for c in range(MAX_CLAUSES):
            goal_mask[c] = self.goal_clause_mask[c]
            goal[c] = map_bits(self.goal_dnf[c])
        dist = {"kind": self.disturbance_rule["kind"]}
        if dist["kind"] == "ONE_SHOT":
            dist["trigger_node"] = perm[self.disturbance_rule["trigger_node"]]
            dist["target_node"] = perm[self.disturbance_rule["target_node"]]
        return {
            "node_mask": node_mask,
            "preconditions_dnf": pre_dnf,
            "precondition_clause_mask": pre_mask,
            "goal_dnf": goal,
            "goal_clause_mask": goal_mask,
            "invalidation_dependency": invdep,
            "disturbance": dist,
            "n": self.n_real(),
        }

    def structure_hash(self):
        present = self.present_nodes()
        # Pad remaining indices stay identity; permute only real nodes.
        others = [i for i in range(N) if i not in present]
        best = None
        for p in permutations(present):
            perm = [None] * N
            for new_pos, old in enumerate(p):
                perm[old] = new_pos
            # map padding to remaining slots
            used = set(perm[i] for i in present)
            slots = [i for i in range(N) if i not in used]
            for old, slot in zip(others, slots):
                perm[old] = slot
            payload = self.structure_payload(perm)
            ser = canonical(payload)
            if best is None or ser < best:
                best = ser
        return hashlib.sha256(best.encode("ascii")).hexdigest()

    def parameter_hash(self):
        return digest(self.parameter_payload())

    def task_hash(self):
        return digest(self.public_dict())


def make_empty_arrays():
    return {
        "node_mask": [False] * N,
        "node_ids": [""] * N,
        "start_xy": [[0.0, 0.0] for _ in range(N)],
        "target_xy": [[0.0, 0.0] for _ in range(N)],
        "transport_steps": [0] * N,
        "preconditions_dnf": [[[False] * N for _ in range(MAX_CLAUSES)] for _ in range(N)],
        "precondition_clause_mask": [[False] * MAX_CLAUSES for _ in range(N)],
        "goal_dnf": [[False] * N for _ in range(MAX_CLAUSES)],
        "goal_clause_mask": [False] * MAX_CLAUSES,
        "invalidation_dependency": [[False] * N for _ in range(N)],
    }


def set_empty_precondition(data, node):
    data["precondition_clause_mask"][node][0] = True
    data["preconditions_dnf"][node][0] = [False] * N


def add_clause(data, node, members):
    masks = data["precondition_clause_mask"][node]
    for c in range(MAX_CLAUSES):
        if not masks[c]:
            masks[c] = True
            for m in members:
                data["preconditions_dnf"][node][c][m] = True
            return
    raise ValueError("clause overflow")


def add_goal_clause(data, members):
    for c in range(MAX_CLAUSES):
        if not data["goal_clause_mask"][c]:
            data["goal_clause_mask"][c] = True
            for m in members:
                data["goal_dnf"][c][m] = True
            return
    raise ValueError("goal overflow")


def set_invalidation_from_preconditions(data):
    for j in range(N):
        for c in range(MAX_CLAUSES):
            if not data["precondition_clause_mask"][j][c]:
                continue
            for i, bit in enumerate(data["preconditions_dnf"][j][c]):
                if bit:
                    data["invalidation_dependency"][i][j] = True
