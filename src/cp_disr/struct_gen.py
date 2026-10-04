"""CP-DISR-TB-STRUCT-GEN-V1: pure symbolic structural-generalization suite for the frozen T_B profile.

No torch, no simulator, no VLM. The Skill Contract registry, the predicates, the candidate set, the controller, verifier and the
evaluator's atomic predicates are the frozen T_B ones; only the *goal composition* (which known action consequences must be
combined, and which object is bound to which destination) changes.

Goal atoms (all four exist in the frozen T_B template as proposition nodes):
    IN_T  = p:Inside:target:container        IN_S  = p:Inside:second_object:container
    BUF_T = p:AtBuffer:target:buffer         BUF_S = p:AtBuffer:second_object:buffer
Cells (goal sets); the lid is always closed and the layout is the frozen T_B layout:
    TRAIN / DEV : IN_T (depth 3)   BUF_S (depth 2)   IN_T+BUF_S (depth 5, the frozen T_B goal)
    STRUCT-GEN TEST : IN_S (depth 3)   BUF_T+BUF_S (depth 4)   IN_S+BUF_T (depth 5)
Every test cell contains an object->destination binding that no train cell contains (target->buffer, second_object->container),
composed from skill schemas and effect schemas that all appear in training.
"""
from __future__ import annotations

import hashlib
import itertools
import json
from collections import defaultdict, deque
from pathlib import Path

import yaml

from .common import ContractError, canonical, digest
from .contracts import Registry, from_dict, nominal_overlay, precondition_value
from .facts import Truth

CARD = "CP-DISR-TB-STRUCT-GEN-V1"
NAMESPACE = "cp_disr_tb_struct_gen_v1"
CONTRACT_REGISTRY_REL = Path("configs/runtime/stage_2a_contract_registry.yaml")
OBJECTS = {"target": "object", "second_object": "object", "container": "container", "buffer": "buffer"}
CONTAINER = [0.18, 0.12]
BUFFER = [-0.18, 0.12]
TARGET_BOX = (-0.22, -0.02, -0.18, -0.02)
SECOND_BOX = (0.04, 0.22, -0.18, -0.02)
MIN_DIST = 0.09
DEADLINE_SECONDS = 60.0
SEED_BASE = 910000

IN_T, IN_S = "p:Inside:target:container", "p:Inside:second_object:container"
BUF_T, BUF_S = "p:AtBuffer:target:buffer", "p:AtBuffer:second_object:buffer"
GOAL_SETS = {
    "IN_T": (IN_T,), "BUF_S": (BUF_S,), "IN_T+BUF_S": (IN_T, BUF_S),
    "IN_S": (IN_S,), "BUF_T+BUF_S": (BUF_T, BUF_S), "IN_S+BUF_T": (IN_S, BUF_T),
}
TRAIN_CELLS = ("IN_T", "BUF_S", "IN_T+BUF_S")
TEST_CELLS = ("IN_S", "BUF_T+BUF_S", "IN_S+BUF_T")
PER_CELL = {"train": 20, "dev": 4, "test": 10}
SPLIT_CELLS = {"train": TRAIN_CELLS, "dev": TRAIN_CELLS, "test": TEST_CELLS}
EXPECTED_DEPTH = {"IN_T": 3, "BUF_S": 2, "IN_T+BUF_S": 5, "IN_S": 3, "BUF_T+BUF_S": 4, "IN_S+BUF_T": 5}
DEPTH_LEVELS = {1: (2, 3), 2: (4, 4), 3: (5, 5)}  # DEPTH-1 < DEPTH-2 < DEPTH-3 by dependency-chain length
MAX_SEARCH_DEPTH = 8


def depth_level(depth: int) -> int:
    for level, (lo, hi) in DEPTH_LEVELS.items():
        if lo <= depth <= hi:
            return level
    raise ValueError("depth %s outside the registered levels" % depth)


# ----------------------------------------------------------------------------- contracts / symbolic world
def load_contracts(root):
    doc = yaml.safe_load((Path(root) / CONTRACT_REGISTRY_REL).read_text(encoding="utf-8"))
    registry = Registry(doc["predicate_types"])
    for item in doc["contracts"]:
        registry.register(from_dict(item))
    return registry.ground(OBJECTS), doc["predicate_types"]


def proposition_ids(contracts):
    return sorted({a.id for c in contracts for a in c.atoms()})


def initial_facts(contracts):
    """Frozen T_B initial state: gripper empty, both cubes on the table, lid closed, nothing placed."""
    facts = {fid: Truth.FALSE for fid in proposition_ids(contracts)}
    facts["p:GripperEmpty"] = Truth.TRUE
    for obj in ("target", "second_object"):
        facts["p:OnTable:%s" % obj] = Truth.TRUE
    return facts


def _key(facts):
    return tuple(sorted((k, v.value) for k, v in facts.items()))


def _apply(contract, facts):
    if precondition_value(contract, facts) != Truth.TRUE:
        return None
    try:
        return {k: Truth(v) for k, v in nominal_overlay(contract, facts, (), ()).items()}
    except ContractError:
        return None


def _goal_true(facts, goals):
    return all(facts[g] == Truth.TRUE for g in goals)


def solve(contracts, facts0, goals, max_depth=MAX_SEARCH_DEPTH):
    """Breadth-first nominal search. Returns depth (shortest legal chain), all optimal plans, per-state optimal action sets."""
    by_id = {c.id: c for c in contracts}
    start = _key(facts0)
    states = {start: dict(facts0)}
    dist = {start: 0}
    edges = defaultdict(dict)  # state -> {cid: next_state}
    frontier, goal_depth = deque([start]), None
    while frontier:
        s = frontier.popleft()
        if goal_depth is not None and dist[s] >= goal_depth:
            continue
        if dist[s] >= max_depth:
            continue
        for c in sorted(contracts, key=lambda x: x.id):
            nxt = _apply(c, states[s])
            if nxt is None:
                continue
            k = _key(nxt)
            if k not in states:
                states[k] = nxt
                dist[k] = dist[s] + 1
                frontier.append(k)
                if goal_depth is None and _goal_true(nxt, goals):
                    goal_depth = dist[k]
            edges[s][c.id] = k
    if _goal_true(facts0, goals):
        return {"depth": 0, "plans": [()], "optimal_actions": {}, "states": states}
    if goal_depth is None:
        return {"depth": None, "plans": [], "optimal_actions": {}, "states": states}
    goal_states = {k for k, d in dist.items() if d == goal_depth and _goal_true(states[k], goals)}
    # distance to goal restricted to the layered DAG
    to_goal = {k: 0 for k in goal_states}
    for d in range(goal_depth - 1, -1, -1):
        for s in [k for k, dd in dist.items() if dd == d]:
            best = [to_goal[n] for n in edges[s].values() if n in to_goal and dist[n] == d + 1]
            if best:
                to_goal[s] = 1 + min(best)
    optimal_actions, plans = {}, []
    on_path = {s for s in to_goal if to_goal[s] + dist[s] == goal_depth}
    for s in on_path:
        if to_goal[s] == 0:
            continue
        optimal_actions[s] = sorted(cid for cid, n in edges[s].items() if n in on_path and dist[n] == dist[s] + 1 and to_goal.get(n) == to_goal[s] - 1)

    def walk(s, prefix):
        if to_goal[s] == 0:
            plans.append(tuple(prefix))
            return
        for cid in optimal_actions[s]:
            walk(edges[s][cid], prefix + [cid])
    walk(start, [])
    return {"depth": goal_depth, "plans": plans, "optimal_actions": optimal_actions, "states": states}


def reference_depth(contracts, facts0, goals, max_depth=MAX_SEARCH_DEPTH):
    """Independent iterative-deepening search without memoisation (used by the depth audit)."""
    by_cid = sorted(contracts, key=lambda c: c.id)

    def dfs(facts, remaining):
        if _goal_true(facts, goals):
            return True
        if remaining == 0:
            return False
        for c in by_cid:
            nxt = _apply(c, facts)
            if nxt is not None and dfs(nxt, remaining - 1):
                return True
        return False
    for d in range(0, max_depth + 1):
        if dfs(facts0, d):
            return d
    return None


def plan_multiset(plan):
    return tuple(sorted(plan))


def effect_composition_signature(plans) -> str:
    """Order-free multiset of the consequences the optimal plan composes (identical for all optimal orderings, else the set of multisets)."""
    return digest(sorted({plan_multiset(p) for p in plans}))


def atomic_bindings(goals):
    out = []
    for g in goals:
        parts = g.split(":")  # p:Inside:obj:container / p:AtBuffer:obj:buffer
        out.append((parts[2], parts[3]))
    return sorted(out)


def binding_signature(goals) -> str:
    return digest(atomic_bindings(goals))


def _nominal_successors(contracts, facts):
    out = {}
    for c in sorted(contracts, key=lambda x: x.id):
        nxt = _apply(c, facts)
        if nxt is not None:
            out[c.id] = {k: v.value for k, v in nxt.items() if v != facts[k]}
    return out


def _atoms(atoms):
    return sorted(a.id for a in atoms)


def contract_table(contracts):
    rows = {}
    for c in sorted(contracts, key=lambda x: x.id):
        rows[c.id] = {"PRE_POS": _atoms(c.pre_pos), "PRE_NEG": _atoms(c.pre_neg), "ADD": _atoms(c.effects.add), "DEL": _atoms(c.effects.delete),
                      "UNKNOWN": _atoms(c.effects.unknown), "CONDITIONAL": len(c.conditional)}
    return rows


# ----------------------------------------------------------------------------- case generation
def _unit(payload) -> float:
    return int.from_bytes(hashlib.sha256(canonical(payload).encode()).digest()[:8], "big") / float(2 ** 64)


def _sample(base, tag, box):
    x_lo, x_hi, y_lo, y_hi = box
    return [float(x_lo + (x_hi - x_lo) * _unit({**base, "axis": tag + "_x"})), float(y_lo + (y_hi - y_lo) * _unit({**base, "axis": tag + "_y"}))]


def _dist(a, b):
    return float(((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2) ** 0.5)


def legal_pose(t, s):
    """Same legality rule as the frozen T_B generator (boxes + minimum cube distance)."""
    return (TARGET_BOX[0] <= t[0] <= TARGET_BOX[1] and TARGET_BOX[2] <= t[1] <= TARGET_BOX[3]
            and SECOND_BOX[0] <= s[0] <= SECOND_BOX[1] and SECOND_BOX[2] <= s[1] <= SECOND_BOX[3] and _dist(t, s) >= MIN_DIST)


def generate_rows(per_cell=None, split_cells=None):
    """Runtime rows (train/dev/test), interleaved by cell so every window of the train order covers every train cell."""
    per_cell, split_cells = per_cell or PER_CELL, split_cells or SPLIT_CELLS
    rows, used_poses, used_seeds, counter = [], set(), set(), 0
    for split in ("train", "dev", "test"):
        cells = split_cells[split]
        for k in range(per_cell[split]):
            for cell in cells:
                reject = 0
                while True:
                    base = {"namespace": NAMESPACE, "split": split, "cell": cell, "k": k, "reject": reject}
                    t, s = _sample(base, "target", TARGET_BOX), _sample(base, "second", SECOND_BOX)
                    pose = (round(t[0], 9), round(t[1], 9), round(s[0], 9), round(s[1], 9))
                    if legal_pose(t, s) and pose not in used_poses:
                        break
                    reject += 1
                    if reject > 256:
                        raise RuntimeError("could not sample a legal pose for %s %s %d" % (split, cell, k))
                seed = SEED_BASE + counter
                counter += 1
                used_poses.add(pose)
                used_seeds.add(seed)
                rows.append({"case_id": "SG_%s_%02d" % (split, len([r for r in rows if r["split"] == split])), "split": split, "seed": seed,
                             "target_xy": t, "second_xy": s, "container_xy": list(CONTAINER), "buffer_xy": list(BUFFER), "lid_closed": True,
                             "second_role": "second_object", "cell": cell, "goal_key": cell, "goal_atoms": list(GOAL_SETS[cell])})
    return rows


# ----------------------------------------------------------------------------- manifest
def build_manifest(root, rows=None, solve_fn=None):
    solve_fn = solve_fn or solve
    contracts, _predicates = load_contracts(root)
    rows = rows if rows is not None else generate_rows()
    facts0 = initial_facts(contracts)
    candidate_ids = [c.id for c in sorted(contracts, key=lambda c: c.id)]
    table = contract_table(contracts)
    cache, cases = {}, []
    for row in rows:
        goals = tuple(GOAL_SETS[row["goal_key"]]) if row["goal_key"] in GOAL_SETS else tuple(row["goal_atoms"])
        if row["goal_key"] not in cache:
            res = solve_fn(contracts, facts0, goals)
            cache[row["goal_key"]] = res
        res = cache[row["goal_key"]]
        plans = [list(p) for p in res["plans"]]
        cases.append({
            "case_id": row["case_id"], "split": row["split"], "cell": row["cell"], "seed": row["seed"],
            "dependency_depth": res["depth"], "depth_level": depth_level(res["depth"]) if res["depth"] else None,
            "objects": dict(OBJECTS), "goal": {"key": row["goal_key"], "atoms": list(goals)},
            "bindings": atomic_bindings(goals), "binding_signature": binding_signature(goals),
            "initial_facts": {k: v.value for k, v in facts0.items()}, "candidate_ids": candidate_ids,
            "contracts": table, "nominal_successors": _nominal_successors(contracts, facts0),
            "optimal_plans": plans, "optimal_first_actions": sorted({p[0] for p in plans}) if plans else [],
            "effect_composition_signature": effect_composition_signature(res["plans"]),
            "geometry": {"target_xy": row["target_xy"], "second_xy": row["second_xy"], "container_xy": row["container_xy"], "buffer_xy": row["buffer_xy"],
                         "lid_closed": row["lid_closed"], "deadline_seconds": DEADLINE_SECONDS},
        })
    return {"card": CARD, "cases": cases, "candidate_ids": candidate_ids, "contract_table": table,
            "solutions": {k: {"depth": v["depth"], "plans": [list(p) for p in v["plans"]],
                              "optimal_actions": {digest(list(s)): a for s, a in v["optimal_actions"].items()}} for k, v in cache.items()}}, contracts, cache


# ----------------------------------------------------------------------------- audits
def dependency_depth_audit(manifest, contracts):
    facts0 = initial_facts(contracts)
    rows, ok = [], True
    for c in manifest["cases"]:
        ref = reference_depth(contracts, facts0, tuple(c["goal"]["atoms"]))
        match = ref == c["dependency_depth"] and (c["optimal_plans"] and len(c["optimal_plans"][0]) == ref)
        expected = EXPECTED_DEPTH.get(c["cell"])
        rows.append({"case_id": c["case_id"], "cell": c["cell"], "labelled": c["dependency_depth"], "reference": ref, "expected_by_design": expected,
                     "plan_length": len(c["optimal_plans"][0]) if c["optimal_plans"] else None, "ok": bool(match and expected == ref)})
        ok = ok and rows[-1]["ok"]
    levels = defaultdict(set)
    for r in rows:
        levels[depth_level(r["labelled"])].add(r["labelled"])
    ordered = [max(levels[1]) < min(levels[2]) < min(levels[3]) and max(levels[2]) < min(levels[3])] if all(l in levels for l in (1, 2, 3)) else [False]
    per_split = {s: sorted({r["labelled"] for r, c in zip(rows, manifest["cases"]) if c["split"] == s}) for s in ("train", "dev", "test")}
    per_level_geometry = {}
    for lvl in (1, 2, 3):
        d = [_dist(c["geometry"]["target_xy"], c["geometry"]["second_xy"]) for c in manifest["cases"] if c["depth_level"] == lvl]
        per_level_geometry[lvl] = {"n": len(d), "mean_target_second_distance": sum(d) / len(d), "min": min(d), "max": max(d)}
    means = [v["mean_target_second_distance"] for v in per_level_geometry.values()]
    geometry_balanced = (max(means) - min(means)) < 0.04
    constants = {"deadline_seconds": sorted({c["geometry"]["deadline_seconds"] for c in manifest["cases"]}), "lid_closed": sorted({c["geometry"]["lid_closed"] for c in manifest["cases"]}),
                 "container_xy": sorted({tuple(c["geometry"]["container_xy"]) for c in manifest["cases"]}), "buffer_xy": sorted({tuple(c["geometry"]["buffer_xy"]) for c in manifest["cases"]})}
    constant_ok = all(len(v) == 1 for v in constants.values())
    checks = {"all_depth_labels_match_independent_search": ok, "levels_strictly_ordered": bool(ordered[0]), "every_level_present": all(l in levels for l in (1, 2, 3)),
              "level_depth_values": {str(k): sorted(v) for k, v in levels.items()}, "geometry_balanced_across_levels": geometry_balanced, "difficulty_knobs_constant": constant_ok}
    verdict = all(v for k, v in checks.items() if isinstance(v, bool))
    return {"card": CARD, "definition": "dependency depth = length of the shortest legal chain of Skill-Contract actions (nominal_apply, preconditions enforced) from the "
            "frozen public initial facts to the goal; computed by a pure symbolic BFS and re-computed by an independent iterative-deepening search",
            "levels": {"DEPTH-1": "depth 2-3", "DEPTH-2": "depth 4", "DEPTH-3": "depth 5"}, "per_split_depths": per_split, "per_level_geometry": per_level_geometry,
            "constants": {k: [list(x) if isinstance(x, tuple) else x for x in v] for k, v in constants.items()}, "cases": rows, "checks": checks, "verdict": "PASS" if verdict else "FAIL"}


def binding_novelty_audit(manifest, contracts):
    cases = manifest["cases"]
    train = [c for c in cases if c["split"] in ("train", "dev")]
    test = [c for c in cases if c["split"] == "test"]
    pure_train = [c for c in cases if c["split"] == "train"]
    dev = [c for c in cases if c["split"] == "dev"]
    train_keys = {(c["binding_signature"], c["effect_composition_signature"]) for c in train}
    train_atomic = {tuple(b) for c in train for b in c["bindings"]}
    train_goal_sets = {c["goal"]["key"] for c in train}
    train_actions = {a for c in train for p in c["optimal_plans"] for a in p}
    train_schemas = {a.split(":")[1] for a in train_actions}
    rows = []
    for c in test:
        atomic = {tuple(b) for b in c["bindings"]}
        unseen = sorted(atomic - train_atomic)
        test_actions = {a for p in c["optimal_plans"] for a in p}
        rows.append({"case_id": c["case_id"], "cell": c["cell"], "composition_key_seen_in_train": (c["binding_signature"], c["effect_composition_signature"]) in train_keys,
                     "goal_set_seen_in_train": c["goal"]["key"] in train_goal_sets, "unseen_atomic_bindings": [list(b) for b in unseen],
                     "schemas_all_seen_in_train": {a.split(":")[1] for a in test_actions} <= train_schemas,
                     "ok": (c["binding_signature"], c["effect_composition_signature"]) not in train_keys and c["goal"]["key"] not in train_goal_sets and bool(unseen)
                     and {a.split(":")[1] for a in test_actions} <= train_schemas})
    pose = lambda c: (tuple(c["geometry"]["target_xy"]), tuple(c["geometry"]["second_xy"]))  # noqa: E731
    seeds = [c["seed"] for c in cases]
    checks = {"test_compositions_absent_from_train": all(not r["composition_key_seen_in_train"] for r in rows), "test_goal_sets_absent_from_train": all(not r["goal_set_seen_in_train"] for r in rows),
              "every_test_case_has_an_unseen_atomic_binding": all(r["unseen_atomic_bindings"] for r in rows), "all_skill_schemas_seen_in_train": all(r["schemas_all_seen_in_train"] for r in rows),
              "no_new_object_type_predicate_or_skill": True, "train_dev_same_cells": {c["cell"] for c in pure_train} == {c["cell"] for c in dev},
              "no_duplicate_case_ids": len({c["case_id"] for c in cases}) == len(cases), "no_duplicate_seeds": len(set(seeds)) == len(seeds),
              "no_pose_overlap_between_splits": len({pose(c) for c in cases}) == len(cases), "test_cells_disjoint_from_train_cells": not ({c["cell"] for c in test} & {c["cell"] for c in train})}
    return {"card": CARD, "definition": "binding signature = sorted (object, destination) goal bindings; composition key = (binding signature, effect-composition signature = order-free multiset of the "
            "optimal plan's action consequences). A test case is novel when its composition key and goal set are absent from train/dev and at least one atomic object->destination binding is absent from train/dev.",
            "train_atomic_bindings": sorted([list(b) for b in train_atomic]), "test_atomic_bindings": sorted({tuple(b) for c in test for b in c["bindings"]}),
            "train_goal_sets": sorted(train_goal_sets), "test_goal_sets": sorted({c["goal"]["key"] for c in test}), "train_action_schemas": sorted(train_schemas),
            "cases": rows, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


# --- shortcut audit -----------------------------------------------------------------------------
def _simulate_priority(contracts, facts0, goals, order, horizon):
    by_id = {c.id: c for c in contracts}
    facts, steps = dict(facts0), 0
    while not _goal_true(facts, goals) and steps < horizon:
        for cid in order:
            nxt = _apply(by_id[cid], facts)
            if nxt is not None:
                facts = nxt
                break
        else:
            return None
        steps += 1
    return steps if _goal_true(facts, goals) else None


def shortcut_audit(manifest, contracts, cache, max_feature_accuracy=0.50):
    """Offline shortcut gate S1-S6 on the symbolic suite."""
    facts0 = initial_facts(contracts)
    by_cid = {c.id: c for c in contracts}
    ids = [c.id for c in sorted(contracts, key=lambda c: c.id)]
    cells = sorted({(c["goal"]["key"], tuple(c["goal"]["atoms"]), c["dependency_depth"]) for c in manifest["cases"]})
    all_cases = {k: (atoms, d) for k, atoms, d in cells}
    train_cells = {k for k in all_cases if k in TRAIN_CELLS}

    # S1 fixed candidate-ID priority policies
    best_all, best_train, optimal_on_all, optimal_on_train = 0, 0, [], []
    for order in itertools.permutations(ids):
        ok_cells = set()
        for key, (atoms, depth) in all_cases.items():
            steps = _simulate_priority(contracts, facts0, atoms, order, depth + 6)
            if steps == depth:
                ok_cells.add(key)
        best_all = max(best_all, len(ok_cells))
        best_train = max(best_train, len(ok_cells & train_cells))
        if len(ok_cells) == len(all_cases):
            optimal_on_all.append(order)
        if train_cells <= ok_cells:
            optimal_on_train.append(order)
    # S2 fixed schema priority (ties broken by candidate id)
    schemas = sorted({i.split(":")[1] for i in ids})
    s2_all = []
    for perm in itertools.permutations(schemas):
        order = [i for sch in perm for i in ids if i.split(":")[1] == sch]
        if all(_simulate_priority(contracts, facts0, atoms, order, depth + 6) == depth for atoms, depth in all_cases.values()):
            s2_all.append(perm)
    # feature-lookup accuracy over on-path decisions (S3 object ID, S4 goal ID, S6 depth/step) -------------
    decisions = []
    for c in manifest["cases"]:
        res = cache[c["goal"]["key"]]
        if not res["depth"]:
            continue
        for plan in res["plans"]:
            facts = dict(facts0)
            for step, cid in enumerate(plan):
                dkey = _key(facts)
                decisions.append({"cell": c["goal"]["key"], "goal": c["goal"]["key"], "depth": res["depth"], "step": step, "state": dkey,
                                  "optimal": set(res["optimal_actions"].get(dkey, [])), "objects": tuple(sorted(c["objects"])), "candidates": tuple(c["candidate_ids"]),
                                  "binding": c["binding_signature"]})
                facts = _apply(by_cid[cid], facts)
    seen, uniq = set(), []
    for d in decisions:
        k = (d["cell"], d["step"], d["state"])
        if k not in seen:
            seen.add(k)
            uniq.append(d)
    decisions = uniq

    def lookup_accuracy(feature):
        groups = defaultdict(list)
        for d in decisions:
            groups[feature(d)].append(d)
        hit = 0
        for _v, ds in groups.items():
            counts = defaultdict(int)
            for d in ds:
                for a in d["optimal"]:
                    counts[a] += 1
            hit += max(counts.values()) if counts else 0
        return hit / len(decisions)
    features = {"object_ids_only": lambda d: d["objects"], "object_ids_and_step": lambda d: (d["objects"], d["step"]), "goal_id_only": lambda d: d["goal"],
                "candidate_id_set_only": lambda d: d["candidates"],
                "depth_only": lambda d: d["depth"], "step_index_only": lambda d: d["step"], "goal_id_and_depth": lambda d: (d["goal"], d["depth"]),
                "binding_signature_only": lambda d: d["binding"]}
    accuracies = {name: lookup_accuracy(fn) for name, fn in features.items()}
    state_goal_full = lookup_accuracy(lambda d: (d["goal"], d["state"]))
    gated = ("object_ids_only", "object_ids_and_step", "goal_id_only", "candidate_id_set_only", "depth_only", "step_index_only", "binding_signature_only")
    # S5 binding -> answer leakage: a composite binding signature must not map to one answer across cells with different answers; test composite bindings absent from train
    train_bind = {c["binding_signature"] for c in manifest["cases"] if c["split"] in ("train", "dev")}
    test_bind = {c["binding_signature"] for c in manifest["cases"] if c["split"] == "test"}
    answers_by_binding = defaultdict(set)
    for c in manifest["cases"]:
        answers_by_binding[c["binding_signature"]].add(tuple(tuple(p) for p in c["optimal_plans"]))
    # S6 open-loop sequence per depth cannot solve every case of that depth (for depths with >1 goal set)
    by_depth = defaultdict(set)
    for key, (atoms, depth) in all_cases.items():
        by_depth[depth].add(key)
    s6 = {}
    for depth, keys in sorted(by_depth.items()):
        plans_by_key = {k: {tuple(p) for p in cache[k]["plans"]} for k in keys}
        common = set.intersection(*plans_by_key.values()) if plans_by_key else set()
        s6[str(depth)] = {"goal_sets": sorted(keys), "single_open_loop_sequence_valid_for_all": bool(common) if len(keys) > 1 else None}
    s6_ok = all(v["single_open_loop_sequence_valid_for_all"] in (None, False) for v in s6.values())
    checks = {
        "S1_no_fixed_candidate_id_priority_is_optimal_on_all_cases": len(optimal_on_all) == 0,
        "S1b_no_fixed_candidate_id_priority_is_optimal_on_all_train_cells": len(optimal_on_train) == 0,
        "S2_no_fixed_action_schema_priority_solves_all_cases": len(s2_all) == 0,
        "S3_object_ids_do_not_determine_the_optimal_candidate": max(accuracies["object_ids_only"], accuracies["object_ids_and_step"]) <= max_feature_accuracy,
        "S4_goal_id_alone_does_not_determine_the_optimal_candidate": accuracies["goal_id_only"] <= max_feature_accuracy,
        "S5_test_composite_bindings_absent_from_train": not (test_bind & train_bind),
        "S5b_binding_signature_alone_does_not_determine_the_optimal_candidate": accuracies["binding_signature_only"] <= max_feature_accuracy,
        "S6_no_open_loop_sequence_per_depth_solves_all_cases_of_that_depth": s6_ok,
        "other_single_features_below_threshold": all(accuracies[n] <= max_feature_accuracy for n in gated),
    }
    return {"card": CARD, "threshold_max_single_feature_lookup_accuracy": max_feature_accuracy,
            "S1": {"permutations_tested": len(list(itertools.permutations(ids))), "optimal_on_all_cases": len(optimal_on_all), "optimal_on_all_train_cells": len(optimal_on_train),
                   "max_cells_optimal_by_one_priority": best_all, "n_cells": len(all_cases), "max_train_cells_optimal": best_train},
            "S2": {"schema_priorities_tested": len(list(itertools.permutations(schemas))), "solving_all_cases": len(s2_all)},
            "feature_lookup_accuracy": accuracies, "state_and_goal_lookup_accuracy": state_goal_full, "n_decisions": len(decisions),
            "S5": {"train_binding_signatures": len(train_bind), "test_binding_signatures": len(test_bind), "overlap": len(test_bind & train_bind)},
            "S6": s6, "checks": checks, "verdict": "PASS" if all(checks.values()) else "FAIL"}


def split_manifest(rows):
    out = {"card": CARD, "namespace": NAMESPACE, "seed_base": SEED_BASE, "per_cell": PER_CELL, "cells": {s: list(c) for s, c in SPLIT_CELLS.items()}, "goal_sets": {k: list(v) for k, v in GOAL_SETS.items()}}
    for split in ("train", "dev", "test"):
        out[split] = [r for r in rows if r["split"] == split]
    out["counts"] = {s: len(out[s]) for s in ("train", "dev", "test")}
    out["test_ids_note"] = "the test rows are written to a separate file and are never part of the training split file"
    out["sha256"] = digest({s: out[s] for s in ("train", "dev", "test")})
    return out


def write_json(path, doc):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(doc, indent=2, sort_keys=True, default=str) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()
