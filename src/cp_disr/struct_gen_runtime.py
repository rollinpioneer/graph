"""CP-DISR-TB-STRUCT-GEN-V1 runtime adaptor: per-case goal composition on the frozen T_B runtime.

The controller, verifier, perception, safety, clock, reward and the evaluator's atomic checks (`TaskEvaluator._inside`,
`TaskEvaluator._at_buffer`, thresholds, deadline and first-success reward logic) are reused unchanged; `task_evaluator.py`
is not modified. Only two things vary per case: the goal atoms of the policy snapshot template and the same atoms inside the
success check. Contracts, candidate IDs, hard mask and the 17 template nodes are identical for every case.
"""
from __future__ import annotations

import json
from dataclasses import fields
from pathlib import Path

from . import struct_gen as G
from .graph import Goal, build_template
from .platforms.libero.runtime_factory import PREDICATES, RuntimeBundle
from .platforms.libero.task_evaluator import TaskEvaluator


class StructGenEvaluator(TaskEvaluator):
    """TaskEvaluator whose success check is the conjunction of the case's goal atoms, each evaluated by the frozen atomic check."""

    def __init__(self, env, deadline_seconds, goal_atoms, task_id="T_B"):
        super().__init__(env, deadline_seconds, task_id=task_id)
        self.goal_atoms = tuple(goal_atoms)

    def goal_true(self) -> bool:
        h = self.env.hidden_truth()
        role = self._second_role()
        for atom in self.goal_atoms:
            parts = atom.split(":")  # p:Inside:<obj>:container | p:AtBuffer:<obj>:buffer
            obj = "target" if parts[2] == "target" else role
            if parts[1] == "Inside":
                ok = self._inside(h, obj)
            elif parts[1] == "AtBuffer":
                ok = self._at_buffer(h, obj)
            else:
                raise ValueError("unsupported goal atom %s" % atom)
            if not ok:
                return False
        return True


class StructGenBundle(RuntimeBundle):
    """RuntimeBundle that binds the goal template and the success check to the case about to be started."""

    def start_case(self, case_id: str, restore_seed=None):
        key = self.goal_by_case[case_id]
        self.template = self.templates[key]
        self.current_goal_atoms = G.GOAL_SETS[key]
        return super().start_case(case_id, restore_seed)


def wrap_bundle(bundle, rows):
    new = StructGenBundle(**{f.name: getattr(bundle, f.name) for f in fields(bundle)})
    new.goal_by_case = {r["case_id"]: r["goal_key"] for r in rows}
    contracts = bundle.template.contracts
    new.templates = {key: build_template(contracts, tuple(Goal(a, 1) for a in atoms), PREDICATES, G.OBJECTS) for key, atoms in G.GOAL_SETS.items()}
    new.current_goal_atoms = G.GOAL_SETS["IN_T+BUF_S"]
    new.evaluator_factory = lambda env, deadline, task_id, _b=new: StructGenEvaluator(env, deadline, _b.current_goal_atoms)
    return new


def read_rows(path):
    doc = json.loads(Path(path).read_text(encoding="utf-8"))
    return [r for part in ("train", "dev", "test") for r in (doc.get(part) or [])]


def install(v11):
    """Idempotently route `make_bundle` through the structural-generalization adaptor (in-process; no tracked file changes)."""
    if getattr(v11.make_bundle, "__structgen__", False):
        return v11
    original = v11.make_bundle

    def make_bundle(root, task_id, split_rel=None):
        bundle = original(root, task_id, split_rel)
        path = Path(split_rel or v11.ENABLED_SPLITS[task_id])
        path = path if path.is_absolute() else Path(root) / path
        return wrap_bundle(bundle, read_rows(path))
    make_bundle.__structgen__ = True
    make_bundle.__wrapped_production__ = original
    v11.make_bundle = make_bundle
    return v11
