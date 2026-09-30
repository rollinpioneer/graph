"""Production RuntimeBundle binding for T_P_FB; no simulator at import/factory time."""
from __future__ import annotations

import json
from pathlib import Path
import yaml

from cp_disr.common import BindingError
from cp_disr.contracts import Atom, Registry, from_dict
from cp_disr.graph import Goal, build_template

from .family_b_adapters import FamilyBEvaluator, FamilyBPerception, FamilyBVerifier
from .family_b_env import FamilyBCaseSpec, FamilyBEnv, make_family_b_env
from .runtime_factory import RuntimeBundle
from .tp_so_mvp_runtime import _NullEnv
from .tp_sr_instrumentation import (InstrumentedSkillExecutor,
                                    RecordingEvaluatorProxy,
                                    RecordingSnapshotBuilderProxy, RunRecorder)

TASK_ID = "T_P_FB"
OBJECTS = {
    "carrier": "object", "obj_b": "object", "obj_c": "object",
    "receiver": "container", "pad_u": "buffer", "pad_v": "buffer",
}
ACTION_IDS = (
    "a:PICK:carrier:v1", "a:PICK:obj_b:v1", "a:PICK:obj_c:v1",
    "a:PLACE_BUFFER:carrier:pad_u:v1",
    "a:PLACE_BUFFER:carrier:pad_v:v1",
    "a:PLACE:obj_b:receiver:v1", "a:PLACE:obj_c:receiver:v1",
)
GOALS = (
    Goal("p:Inside:obj_b:receiver", 1),
    Goal("p:Inside:obj_c:receiver", 1),
    Goal("p:GripperEmpty", 1),
)
VERIFIER_FACT_IDS = (
    "p:GripperEmpty", "p:Open:receiver",
    *(f"p:{predicate}:{obj}" for obj in ("carrier", "obj_b", "obj_c")
      for predicate in ("Held", "OnTable")),
    *(f"p:Inside:{obj}:receiver" for obj in ("carrier", "obj_b", "obj_c")),
    *(f"p:AtBuffer:{obj}:{pad}" for obj in ("carrier", "obj_b", "obj_c")
      for pad in ("pad_u", "pad_v")),
)


def _atom(fid):
    parts = fid.split(":")
    return Atom(parts[1], tuple(parts[2:]))


def load_task_contracts(path):
    doc = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    reg = Registry(doc["predicate_types"])
    for row in doc["contracts"]:
        reg.register(from_dict(row))
    grounded = reg.ground(OBJECTS)
    selected = tuple(c for c in grounded if c.id in ACTION_IDS)
    if {c.id for c in selected} != set(ACTION_IDS):
        raise BindingError("Family B contract action set mismatch")
    return selected, reg.predicate_types


def build_task_template(contract_path):
    contracts, predicates = load_task_contracts(contract_path)
    in_contracts = {a.id for c in contracts for a in c.atoms()}
    extra = tuple(_atom(fid) for fid in VERIFIER_FACT_IDS if fid not in in_contracts)
    template = build_template(contracts, GOALS, predicates, OBJECTS, extra_atoms=extra)
    propositions = {n.id for n in template.nodes if n.kind == "PROPOSITION"}
    if propositions != set(VERIFIER_FACT_IDS):
        raise BindingError("Family B verifier/template proposition mismatch")
    return template


class FamilyBRecorder(RunRecorder):
    def write_initial(self, env, bundle):
        super().write_initial(env, bundle)
        self._wj(
            "initial_object_bindings.json",
            {"objects": {name: name for name in OBJECTS},
             "public_blobs": {
                 name: (None if env.last_perception.get(name) is None
                        else env.last_perception[name].tolist())
                 for name in ("carrier", "obj_b", "obj_c")},
             "qa_hidden_used": False},
            self.root,
        )


class FamilyBBundle(RuntimeBundle):
    recorder = None
    branch = None
    out_root = None

    def configure(self, out_root, branch):
        self.out_root, self.branch = Path(out_root), dict(branch)
        self.recorder = FamilyBRecorder(out_root, branch["branch_id"], branch)
        return self.recorder

    def start_case(self, case_id, restore_seed=None):
        snap = RuntimeBundle.start_case(self, case_id, restore_seed)
        if not isinstance(self.environment, FamilyBEnv):
            raise BindingError("Family B environment binding lost")
        if not isinstance(self.perception, FamilyBPerception):
            raise BindingError("Family B perception binding lost")
        if not isinstance(self.verifier, FamilyBVerifier):
            raise BindingError("Family B verifier binding lost")
        if not isinstance(self.evaluator, FamilyBEvaluator):
            raise BindingError("Family B evaluator binding lost")
        if not callable(getattr(self.environment, "resolve_skill_destination", None)):
            raise BindingError("Family B destination resolver binding lost")
        if self.recorder is not None:
            self._install_recorders()
        return snap

    def _install_recorders(self):
        rec, env = self.recorder, self.environment
        env.recorder = rec
        self.perception.recorder = rec
        self.verifier.recorder = rec
        inner = self.executor.inner
        inner.__class__ = InstrumentedSkillExecutor
        inner.recorder = rec
        self.evaluator = RecordingEvaluatorProxy(self.evaluator, rec)
        self.snapshot_builder = RecordingSnapshotBuilderProxy(
            self.snapshot_builder, rec)
        rec.write_initial(env, self)

    def env_counts(self):
        env = self.environment
        return {
            "constructions": 1,
            "reset_calls": int(getattr(env, "reset_calls", -1)),
            "internal_resets": int(getattr(env, "internal_resets", -1)),
            "bootstrap_constructions": 0,
        }


def create_family_b_runtime(manifest):
    runtime = manifest["runtime"]
    if runtime.get("active_task_id") != TASK_ID:
        raise BindingError("Family B runtime active_task_id mismatch")
    root = Path(runtime["repository_path"])
    template = build_task_template(root / runtime["contract_path"])
    deadline = float(runtime["task_deadlines"][TASK_ID])
    layouts = json.loads((root / runtime["layouts_path"]).read_text())
    cases = {}
    for layout in layouts["layouts"]:
        case_id = layout["layout_id"]
        cases[case_id] = FamilyBCaseSpec(
            case_id=case_id, split="dev", seed=0,
            target_xy=tuple(layout["carrier_xy"]),
            second_xy=tuple(layout["obj_b_xy"]),
            obj_c_xy=tuple(layout["obj_c_xy"]),
            container_xy=tuple(layout["receiver_xy"]),
            buffer_xy=tuple(layout["pad_u_xy"]),
            pad_v_xy=tuple(layout["pad_v_xy"]),
            lid_closed=False, task_id=TASK_ID, deadline=deadline,
        )
    null = _NullEnv()
    return FamilyBBundle(
        environment=null, executor=None, observations=None, perception=None,
        verifier=FamilyBVerifier(null),
        evaluator=FamilyBEvaluator(null, deadline, TASK_ID),
        safety=None, clock=None, snapshot_builder=None, task_id=TASK_ID,
        template=template, cases=cases, caches={},
        env_factory=make_family_b_env,
        perception_cls=FamilyBPerception,
        verifier_factory=FamilyBVerifier,
        evaluator_factory=FamilyBEvaluator,
    )


def create(manifest):
    return create_family_b_runtime(manifest)
