"""Prospective P2CRL instance registry. No reward-based resampling."""
from __future__ import annotations
import hashlib
import random
from copy import deepcopy
from pathlib import Path

from p2cq_research.generator import MAKERS, slot_prefix_plan
from p2cq_research.task_contract import N, TaskContract, action_index, node_op

from . import INSTANCE_VERSION, MOTIFS, SPLITS
from .contracts import family_id, load_protocol, roots_for_split, strip_identity
from .io_utils import canonical, hash_json, write_new

DATA_SALT = "P2CRL_DATA_V1|2026091801"
HORIZONS = [32, 40, 48, 56, 64, 72, 80, 88]


def split_rng(split, motif, root):
    text = f"{DATA_SALT}|{split}|{motif}|{root}"
    seed = int.from_bytes(hashlib.sha256(text.encode("utf-8")).digest()[:8], "big")
    return random.Random(seed)


def sample_bundle(rng, motif, present):
    K = [0] * N
    for i in present:
        if motif == "ALTERNATIVE_COST" and i in (2, 3):
            continue
        if motif == "INVALIDATION_RECOVERY" and i == 3:
            continue
        K[i] = rng.choice([1, 2, 3])
    trigger = None
    if motif == "ALTERNATIVE_COST":
        kc, kd = rng.choice([(1, 2), (1, 3), (2, 3)])
        K[2], K[3] = int(kc), int(kd)
    if motif == "INVALIDATION_RECOVERY":
        K[3] = rng.choice([2, 3])
        trigger = rng.randint(1, K[3])
    horizon = rng.choice(HORIZONS)
    geom = {}
    for i in present:
        sx = round(rng.uniform(-1.0, 1.0), 8)
        sy = round(rng.uniform(-1.0, 1.0), 8)
        dx = round(rng.uniform(0.7, 1.3), 8)
        dy = round(rng.uniform(0.5, 1.1), 8)
        geom[i] = ([sx, sy], [round(sx + dx, 8), round(sy + dy, 8)])
    perm = list(range(N))
    rng.shuffle(perm)
    return {"K": K, "horizon": horizon, "geom": geom, "trigger": trigger, "perm": perm}


def apply_params(contract, split, motif, fid, bundle):
    d = contract.public_dict()
    d["split"] = split
    d["motif"] = motif
    d["family_id"] = int(fid)
    d["horizon"] = int(bundle["horizon"])
    d["version"] = INSTANCE_VERSION
    d["gamma"] = 0.99
    for i, k in enumerate(bundle["K"]):
        d["transport_steps"][i] = int(k)
    for i, (s, t) in bundle["geom"].items():
        d["start_xy"][int(i)] = [float(s[0]), float(s[1])]
        d["target_xy"][int(i)] = [float(t[0]), float(t[1])]
    if bundle["trigger"] is not None:
        rule = dict(d["disturbance_rule"])
        if rule.get("kind") != "ONE_SHOT":
            raise ValueError("expected ONE_SHOT disturbance")
        rule["trigger_progress"] = int(bundle["trigger"])
        d["disturbance_rule"] = rule
    return TaskContract(d)


def permute_dict(data, perm):
    def map_bits(bits):
        out = [False] * N
        for i, bit in enumerate(bits):
            if bit:
                out[perm[i]] = True
        return out
    out = deepcopy(data)
    out["node_mask"] = [False] * N
    out["node_ids"] = [""] * N
    out["start_xy"] = [[0.0, 0.0] for _ in range(N)]
    out["target_xy"] = [[0.0, 0.0] for _ in range(N)]
    out["transport_steps"] = [0] * N
    out["preconditions_dnf"] = [[[False] * N for _ in range(2)] for _ in range(N)]
    out["precondition_clause_mask"] = [[False] * 2 for _ in range(N)]
    out["goal_dnf"] = [[False] * N for _ in range(2)]
    out["goal_clause_mask"] = list(data["goal_clause_mask"])
    out["invalidation_dependency"] = [[False] * N for _ in range(N)]
    for old in range(N):
        new = perm[old]
        out["node_mask"][new] = data["node_mask"][old]
        out["node_ids"][new] = data["node_ids"][old]
        out["start_xy"][new] = list(data["start_xy"][old])
        out["target_xy"][new] = list(data["target_xy"][old])
        out["transport_steps"][new] = data["transport_steps"][old]
        out["precondition_clause_mask"][new] = list(data["precondition_clause_mask"][old])
        for c in range(2):
            out["preconditions_dnf"][new][c] = map_bits(data["preconditions_dnf"][old][c])
        for old_j in range(N):
            out["invalidation_dependency"][new][perm[old_j]] = data["invalidation_dependency"][old][old_j]
    for c in range(2):
        out["goal_dnf"][c] = map_bits(data["goal_dnf"][c])
    dist = dict(data["disturbance_rule"])
    if dist.get("kind") == "ONE_SHOT":
        dist["trigger_node"] = perm[dist["trigger_node"]]
        dist["target_node"] = perm[dist["target_node"]]
    out["disturbance_rule"] = dist
    return out


def map_action(action, perm):
    if action == 0:
        return 0
    j, op = node_op(action)
    return action_index(perm[j], op)


def map_prefix(prefix, perm):
    return [map_action(int(a), perm) for a in prefix]


def signatures(contract: TaskContract):
    full = hash_json(strip_identity(contract.public_dict()))
    dyn_src = strip_identity(contract.public_dict())
    dyn_src.pop("start_xy", None)
    dyn_src.pop("target_xy", None)
    return {
        "full": full,
        "dynamics": hash_json(dyn_src),
        "topology": contract.structure_hash(),
        "task": contract.task_hash(),
        "parameter": contract.parameter_hash(),
    }


def build_family(protocol, split, motif, root):
    rng = split_rng(split, motif, root)
    left_t = MAKERS[motif]("development", int(root), "left")
    right_t = MAKERS[motif]("development", int(root), "right")
    present = left_t.present_nodes()
    if present != right_t.present_nodes():
        raise ValueError("left/right present mismatch")
    bundle = sample_bundle(rng, motif, present)
    fid = family_id(protocol, split, motif, root)
    left_u = apply_params(left_t, split, motif, fid, bundle)
    right_u = apply_params(right_t, split, motif, fid, bundle)
    prefixes = [map_prefix(p, bundle["perm"]) for p in slot_prefix_plan(left_u)]
    if len(prefixes) != 16:
        raise ValueError("prefix count")
    left = TaskContract(permute_dict(left_u.public_dict(), bundle["perm"]))
    right = TaskContract(permute_dict(right_u.public_dict(), bundle["perm"]))
    if left.structure_hash() != left_u.structure_hash():
        raise RuntimeError("permutation changed topology")
    if right.structure_hash() != right_u.structure_hash():
        raise RuntimeError("permutation changed topology")
    return {
        "split": split,
        "motif": motif,
        "root": int(root),
        "family_id": int(fid),
        "perm": list(bundle["perm"]),
        "horizon": int(bundle["horizon"]),
        "left": left,
        "right": right,
        "prefixes": prefixes,
        "left_sig": signatures(left),
        "right_sig": signatures(right),
    }


def iter_planned(protocol):
    for split in SPLITS:
        n = roots_for_split(protocol, split)
        for motif in MOTIFS:
            for root in range(n):
                yield build_family(protocol, split, motif, root)


def dump_registry(protocol, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    families = []
    contracts_dir = out / "contracts"
    contracts_dir.mkdir(exist_ok=True)
    rejected = []
    for fam in iter_planned(protocol):
        rec = {
            "split": fam["split"],
            "motif": fam["motif"],
            "root": fam["root"],
            "family_id": fam["family_id"],
            "perm": fam["perm"],
            "horizon": fam["horizon"],
            "prefixes": fam["prefixes"],
            "left_sig": fam["left_sig"],
            "right_sig": fam["right_sig"],
        }
        try:
            for side in ("left", "right"):
                c = fam[side]
                payload = c.public_dict()
                write_new(contracts_dir / f"{c.family_id}_{side}.json", payload)
                rec[f"{side}_task_hash"] = c.task_hash()
            families.append(rec)
            if len(families) % 32 == 0:
                print(f"REGISTRY {len(families)} {rec['split']} {rec['motif']} {rec['root']}", flush=True)
        except Exception as exc:
            rejected.append({"family_id": fam["family_id"], "error": str(exc)})
            break
    if rejected:
        write_new(out / "rejected_registry.json", {"rejected": rejected, "families_written": len(families)})
        raise RuntimeError("RL_DATASET_REGISTRATION_FAILED")
    write_new(out / "families.json", {"schema": "P2CRL_FAMILY_REGISTRY_V1", "n": len(families), "families": families})
    return families
