"""Reference plans and reference-path states (plan section 12).

The reference plan of an IPC problem is the shortest plan among all plan files already saved in the three run roots of the base commit (any producer: learned scorers, WL, LAMA anytime plans, native
planner), replayed under the project semantics; ties by file SHA-256, then by normalised relative path. No planner is run to find a more favourable reference. Its length is an UPPER bound on d*(init); the
optimality is UNKNOWN unless a proof exists in the earlier cards. For a state on the reference path the remaining plan length is a valid upper bound U(state) (the shortest suffix if the state repeats).
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

PLAN_NAME = re.compile(r"(\.plan(\.\d+)?$|^sas_plan$)")


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def case_pattern(case_id):
    return re.compile(re.escape(case_id) + r"(?![0-9])")


def plan_candidates(repo_root, run_roots, case_id):
    """All plan-looking files of the run roots whose path names the case (deterministic order)."""
    pat = case_pattern(case_id)
    out = []
    for rr in run_roots:
        base = Path(repo_root) / rr
        for p in sorted(base.rglob("*")):
            if p.is_file() and PLAN_NAME.search(p.name) and pat.search(str(p.relative_to(base)).replace("\\", "/")):
                out.append(p)
    return out


def parse_plan_steps(path):
    return [ln.strip()[1:-1].strip().lower() for ln in Path(path).read_text().splitlines() if ln.strip().startswith("(")]


def steps_to_ids(task, steps):
    """Case-insensitive mapping of ``name arg ...`` plan lines to project action ids; None if a step names an unknown action."""
    lower = {a.aid.lower(): a.aid for a in task.actions}
    ids = []
    for s in steps:
        parts = s.split()
        key = "a:" + parts[0] + (":" + ":".join(parts[1:]) if len(parts) > 1 else "") + ":v1"
        aid = lower.get(key)
        if aid is None:
            return None
        ids.append(aid)
    return ids


def evaluate_candidate(task, path):
    ids = steps_to_ids(task, parse_plan_steps(path))
    fin = task.replay(ids) if ids is not None else None
    ok = bool(fin is not None and task.goal_satisfied(fin))
    return {"path": str(path), "length": len(ids) if ids is not None else None, "valid": ok, "sha256": sha_bytes(Path(path).read_bytes()), "ids": ids if ok else None}


def select_reference(task, repo_root, run_roots, case_id):
    cands = [evaluate_candidate(task, p) for p in plan_candidates(repo_root, run_roots, case_id)]
    for c in cands:
        c["rel"] = str(Path(c["path"]).relative_to(repo_root)).replace("\\", "/")
    valid = [c for c in cands if c["valid"]]
    valid.sort(key=lambda c: (c["length"], c["sha256"], c["rel"]))
    return (valid[0] if valid else None), cands


def path_states(task, ids):
    """States w_0 .. w_L along the plan and ``ref`` = {state: (k, U)} with the shortest remaining suffix U = L - k for repeated states."""
    s = task.init_mask
    seq = [s]
    for aid in ids:
        s = task.apply(s, task.action_by_id[aid])
        seq.append(s)
    L = len(ids)
    ref = {}
    for k, st in enumerate(seq):
        ref[st] = (k, L - k)                      # later k overwrites earlier k: the shortest remaining suffix
    return seq, ref
