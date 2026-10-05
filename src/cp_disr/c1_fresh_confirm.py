"""CP-DISR-C1-FRESH-CONFIRM-V1: the new, frozen, untouched confirmation suite for the C1 mechanism card (pure symbolic; no torch, no simulator).

Reuses the CP-DISR-TB-STRUCT-GEN-V1 machinery (contracts, goal atoms, symbolic solver, case generator, audits). Only the goal set, the object->destination binding
and the legal pose seed may differ from the frozen T_B profile. Cells: IN_S, BUF_T, BUF_T+BUF_S, IN_S+BUF_T (BUF_T alone is the one new goal set).
Frozen untouched confirmation set, not a strict blind benchmark: it is generated and frozen (hashes) before any model sees it.
"""
from __future__ import annotations

import hashlib
import json
from collections import defaultdict
from pathlib import Path

from . import struct_gen as G
from .common import canonical, digest

CARD = "CP-DISR-C1-MECH-CONFIRM-V1"
SUITE = "CP-DISR-C1-FRESH-CONFIRM-V1"
NAMESPACE = "cp_disr_c1_fresh_confirm_v1"
SEED_BASE = 920000
CELLS = ("IN_S", "BUF_T", "BUF_T+BUF_S", "IN_S+BUF_T")
PER_CELL = {"qualification": 2, "test": 8}
QUAL_REL = Path("configs/splits/c1_fresh_confirm_v1_qualification.json")
TEST_REL = Path("configs/splits/c1_fresh_confirm_v1_test.json")
NEW_GOAL_SETS = {"BUF_T": (G.BUF_T,)}
NEW_EXPECTED_DEPTH = {"BUF_T": 2}
MAX_CONSTRUCTION_ATTEMPTS = 12
POSE_DECIMALS = 9


def install_goal_sets():
    """Idempotent in-process extension of the frozen goal-set table with the single new goal set BUF_T (no tracked file changes)."""
    for key, atoms in NEW_GOAL_SETS.items():
        G.GOAL_SETS.setdefault(key, atoms)
    for key, depth in NEW_EXPECTED_DEPTH.items():
        G.EXPECTED_DEPTH.setdefault(key, depth)


def uninstall_goal_sets():
    """Restore the frozen struct-gen tables (used by tests so the old suite's tests see the original module state)."""
    for key in NEW_GOAL_SETS:
        G.GOAL_SETS.pop(key, None)
    for key in NEW_EXPECTED_DEPTH:
        G.EXPECTED_DEPTH.pop(key, None)


# NOTE: nothing is installed at import time. Entry points (build, the evaluation runtime, qualification) call install_goal_sets() explicitly, so importing this
# module never changes the old structural-generalization suite for a process that only wants the old suite.


def _pose(row):
    return (round(row["target_xy"][0], POSE_DECIMALS), round(row["target_xy"][1], POSE_DECIMALS), round(row["second_xy"][0], POSE_DECIMALS), round(row["second_xy"][1], POSE_DECIMALS))


def read_all_rows(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    rows = []
    for key, value in doc.items():
        if isinstance(value, list):
            rows.extend(r for r in value if isinstance(r, dict) and "case_id" in r)
    return rows


def old_rows(root):
    root = Path(root)
    rows = []
    # every old structural-generalization split file (train/dev and the old test rows), read for the dedupe audit only; never by a training path
    for path in sorted((root / "configs/splits").glob("struct_gen_v1_*.json")):
        rows.extend(read_all_rows(path))
    return rows


def old_seeds_anywhere(root):
    """Every integer 'seed' that appears in any tracked split file (any task), for the dedupe audit."""
    seeds = set()
    for path in sorted((Path(root) / "configs/splits").glob("*.json")):
        try:
            doc = json.loads(path.read_text(encoding="utf-8"))
        except ValueError:
            continue
        stack = [doc]
        while stack:
            item = stack.pop()
            if isinstance(item, dict):
                if isinstance(item.get("seed"), int) and not isinstance(item.get("seed"), bool):
                    seeds.add(item["seed"])
                stack.extend(item.values())
            elif isinstance(item, list):
                stack.extend(item)
    return seeds


def generate_rows(forbidden_poses, forbidden_seeds, attempt=0, per_cell=None, cells=CELLS):
    """Rows for the qualification and test splits, interleaved by cell. Deterministic in (namespace, attempt)."""
    install_goal_sets()
    per_cell = per_cell or PER_CELL
    ns = NAMESPACE if attempt == 0 else "%s-r%d" % (NAMESPACE, attempt)
    rows, used_poses, used_seeds, counter = [], set(forbidden_poses), set(forbidden_seeds), 0
    for split in ("qualification", "test"):
        for k in range(per_cell[split]):
            for cell in cells:
                reject = 0
                while True:
                    base = {"namespace": ns, "split": split, "cell": cell, "k": k, "reject": reject}
                    t, s = G._sample(base, "target", G.TARGET_BOX), G._sample(base, "second", G.SECOND_BOX)
                    pose = (round(t[0], POSE_DECIMALS), round(t[1], POSE_DECIMALS), round(s[0], POSE_DECIMALS), round(s[1], POSE_DECIMALS))
                    if G.legal_pose(t, s) and pose not in used_poses:
                        break
                    reject += 1
                    if reject > 256:
                        raise RuntimeError("could not sample a legal pose for %s %s %d" % (split, cell, k))
                seed = SEED_BASE + counter
                while seed in used_seeds:
                    seed += 1000
                counter += 1
                used_poses.add(pose)
                used_seeds.add(seed)
                n_split = len([r for r in rows if r["split"] == split])
                rows.append({"case_id": "C1_%s_%02d" % ("qual" if split == "qualification" else "test", n_split), "split": split, "seed": seed, "target_xy": t, "second_xy": s,
                             "container_xy": list(G.CONTAINER), "buffer_xy": list(G.BUFFER), "lid_closed": True, "second_role": "second_object", "cell": cell, "goal_key": cell,
                             "goal_atoms": list(G.GOAL_SETS[cell])})
    return rows


def dedupe_audit(root, rows):
    root = Path(root)
    old = old_rows(root)
    old_poses = {_pose(r) for r in old}
    new_poses = [_pose(r) for r in rows]
    new_seeds = [r["seed"] for r in rows]
    other_seeds = old_seeds_anywhere(root)
    qual = {_pose(r) for r in rows if r["split"] == "qualification"}
    final = {_pose(r) for r in rows if r["split"] == "test"}
    checks = {"no_new_pose_equals_an_old_train60_dev12_test30_or_old_qualification_pose": not (set(new_poses) & old_poses),
              "no_new_seed_equals_any_seed_in_any_tracked_split_file": not (set(new_seeds) & other_seeds), "new_poses_unique": len(set(new_poses)) == len(new_poses),
              "new_seeds_unique": len(set(new_seeds)) == len(new_seeds), "qualification_and_final_poses_disjoint": not (qual & final),
              "new_case_ids_do_not_collide_with_old": not ({r["case_id"] for r in rows} & {r["case_id"] for r in old}), "namespace_differs_from_old": NAMESPACE != G.NAMESPACE}
    nearest = []
    for r in rows:
        p = (r["target_xy"], r["second_xy"])
        nearest.append(min(max(G._dist(p[0], o["target_xy"]), G._dist(p[1], o["second_xy"])) for o in old))
    return {"checks": checks, "old_rows_compared": len(old), "tracked_split_seed_count": len(other_seeds), "min_chebyshev_object_distance_to_any_old_pose": min(nearest),
            "verdict": "PASS" if all(checks.values()) else "FAIL"}


def build(root, attempt=0):
    """Rows + manifest + audits for one construction attempt (no files written)."""
    install_goal_sets()
    root = Path(root)
    old = old_rows(root)
    forbidden_poses = {_pose(r) for r in old}
    forbidden_seeds = old_seeds_anywhere(root)
    rows = generate_rows(forbidden_poses, forbidden_seeds, attempt)
    dedupe = dedupe_audit(root, rows)
    contracts_rows_old = [dict(r) for r in old if r["split"] in ("train", "dev")]
    final_rows = [r for r in rows if r["split"] == "test"]
    qual_rows = [r for r in rows if r["split"] == "qualification"]
    audits = {}
    for name, new in (("final", final_rows), ("qualification", qual_rows)):
        combined = contracts_rows_old + [dict(r, split="test") for r in new]
        manifest, contracts, cache = G.build_manifest(root, combined)
        depth = G.dependency_depth_audit(manifest, contracts)
        novelty = G.binding_novelty_audit(manifest, contracts)
        audits[name] = {"depth": depth, "novelty": novelty}
    return rows, dedupe, audits


def geometry_by_level(rows):
    install_goal_sets()
    from .struct_gen import EXPECTED_DEPTH, depth_level
    levels = defaultdict(list)
    for r in rows:
        levels[depth_level(EXPECTED_DEPTH[r["cell"]])].append(G._dist(r["target_xy"], r["second_xy"]))
    return {str(k): {"n": len(v), "mean_target_second_distance": sum(v) / len(v), "min": min(v), "max": max(v)} for k, v in sorted(levels.items())}


def split_documents(rows):
    docs = {}
    for split, rel in (("qualification", QUAL_REL), ("test", TEST_REL)):
        part = [r for r in rows if r["split"] == split]
        doc = {"task_id": "T_B", "suite": SUITE, "namespace": NAMESPACE, "train": [], "dev": [], "test": part if split == "test" else [], "qualification": part if split == "qualification" else [],
               "note": "frozen untouched confirmation rows; never opened by a training code path", "cells": list(CELLS), "per_cell": PER_CELL[split]}
        docs[rel] = doc
    return docs


def write_documents(root, docs):
    out = {}
    for rel, doc in docs.items():
        path = Path(root) / rel
        out[str(rel)] = G.write_json(path, doc)
    return out


def case_hash_table(rows):
    return {r["case_id"]: digest({k: r[k] for k in ("seed", "target_xy", "second_xy", "cell", "goal_atoms")}) for r in rows}
