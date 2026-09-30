"""RuntimeFactory for T_P_SOFT_ORDERING_MVP_V1 (S4 Family A soft-ordering minimal mechanism).

Reuses the production RuntimeBundle.start_case restore, SkillExecutor, FactVerifier, TaskEvaluator, SnapshotBuilder and the
T_P_SOF_relocation PICK/PLACE contract grounding. Differences from the T_P_SR V2 pilot (tp_sr_v2_runtime.py, untouched):
  * the second object is a SECOND OBJECT, not an interferer: both cubes are identical in geometry, mass, friction and
    controller, and both are goal objects (Inside(target,container) and Inside(second_object,container));
  * OPEN is not registered (the container starts open), PLACE_BUFFER is not registered and MOVE is absent;
  * the task evaluator subclass here requires BOTH objects inside the container;
  * no process-wide perception monkeypatch: the V1-compatible nearest-palette assignment lives only in
    TPSRNearestPalettePerceptionAdapter;
  * no bootstrap environment: the first environment of a branch is the one created by start_case (one restore chain);
  * passive recorders installed after start_case (they never change actions, facts, masks, planner input or seeds).
"""
from __future__ import annotations

from pathlib import Path

from cp_disr.common import BindingError
from cp_disr.contracts import Atom
from cp_disr.facts import Truth
from cp_disr.graph import Goal, build_template

from .d0_env import CaseSpec
from .runtime_factory import PREDICATES, RuntimeBundle, _read, _ground_contracts_for
from .task_evaluator import TaskEvaluator
from .verifier import FactVerifier
from .tp_sr_instrumentation import (InstrumentedSkillExecutor, RecordingEvaluatorProxy, RecordingSnapshotBuilderProxy, RecordingVerifier, RunRecorder,
                                    TPSRNearestPalettePerceptionAdapter)
from .tp_sr_v2_env import InstrumentedD0Env

RUNTIME_IDENTITY = "tp-so-mvp-runtime-v1"

TASK_ID = "T_P_SO_MVP"
TASK_FAMILY_ID = "T_P_SOFT_ORDERING_MVP_V1"
SECOND_ROLE = "second_object"
OBJECTS = {"target": "object", "second_object": "object", "container": "container", "buffer": "buffer"}
GOAL_FACTS = ("p:Inside:target:container", "p:Inside:second_object:container")

ALLOWED_ACTION_IDS = ("a:PICK:target:v1", "a:PICK:second_object:v1", "a:PLACE:target:container:v1", "a:PLACE:second_object:container:v1")
EXCLUDED_GROUNDING_PREFIXES = ("a:OPEN:", "a:PLACE_BUFFER:", "a:MOVE:")
FORBIDDEN_HARD_FACTS = ("ClearPath", "Occluded", "Blocked", "EasyGrasp", "SafeToPick")

# Facts produced by the production verifier for both cubes; kept as template propositions so FactStore and template align.
VERIFIER_FACT_IDS = ("p:GripperEmpty", "p:Held:target", "p:Held:second_object", "p:OnTable:target", "p:OnTable:second_object",
                     "p:Open:container", "p:Inside:target:container", "p:Inside:second_object:container",
                     "p:AtBuffer:target:buffer", "p:AtBuffer:second_object:buffer")

INITIAL_TRUE_FACTS = ("p:GripperEmpty", "p:OnTable:target", "p:OnTable:second_object", "p:Open:container")
INITIAL_FALSE_FACTS = ("p:Inside:target:container", "p:Inside:second_object:container", "p:Held:target", "p:Held:second_object")

TASK_EVALUATOR_VERSION = "cp-disr-tp-so-mvp-task-evaluator-v1"


class StaticInvariantContradiction(RuntimeError):
    """Public task invariant contradicted by measured evidence."""


class TPSOMVPFactVerifier(FactVerifier):
    """Production verifier with the task's public-open invariant overlay.

    Only the Open fact may be overlaid, and only from public manifest/reset
    metadata plus the registered contract effects. No hidden simulator state is
    consulted.
    """

    def _open_writers(self):
        writers = []
        for contract in getattr(self.env, "task_contracts", ()):
            for attr, label in (("add", "ADD"), ("delete", "DEL"), ("unknown", "UNKNOWN")):
                for atom in getattr(contract.effects, attr, ()):
                    if atom.id == "p:Open:container":
                        writers.append((contract.id, label))
            for conditional in getattr(contract, "conditional", ()):
                for attr, label in (("add", "ADD"), ("delete", "DEL"), ("unknown", "UNKNOWN")):
                    for atom in getattr(conditional.effects, attr, ()):
                        if atom.id == "p:Open:container":
                            writers.append((contract.id, label))
        return tuple(writers)

    def verify(self, measurement, execution=None):
        recs = list(super().verify(measurement, execution))
        open_rec = next((r for r in recs if r.fact_id == "p:Open:container"), None)
        if open_rec is None or open_rec.value != Truth.UNKNOWN:
            if open_rec is not None and open_rec.value == Truth.FALSE:
                raise StaticInvariantContradiction("STOPPED_STATIC_INVARIANT_CONTRADICTION")
            return tuple(recs)

        manifest = getattr(self.env, "task_manifest", None)
        reset_identity = getattr(self.env, "reset_identity", None)
        if manifest is None or reset_identity is None or not getattr(self.env, "task_contracts", ()):
            raise StaticInvariantContradiction("task manifest/reset identity/contracts missing")
        case = getattr(self.env, "case", None)
        expected = {"case_id": manifest.case_id, "seed": int(manifest.seed),
                    "lid_closed": bool(manifest.lid_closed)}
        if (case is None or reset_identity != expected or case.case_id != manifest.case_id
                or int(case.seed) != int(manifest.seed) or bool(case.lid_closed) != bool(manifest.lid_closed)
                or manifest.task_id != TASK_ID or case.task_id != TASK_ID):
            raise StaticInvariantContradiction("reset identity mismatch")
        if bool(getattr(manifest, "lid_closed", True)):
            raise StaticInvariantContradiction("lid_closed=true; static open invariant unavailable")
        if self._open_writers():
            raise StaticInvariantContradiction("registered action writes p:Open:container")

        evidence = ["task_manifest", "reset_config"]
        if self.prev.get("p:Open:container", Truth.UNKNOWN) == Truth.TRUE:
            evidence.append("previous_public_confirmation")
        replacement = type(open_rec)(
            fact_id=open_rec.fact_id, value=Truth.TRUE,
            capture_time=open_rec.capture_time, available_time=open_rec.available_time,
            evidence_ids=tuple(evidence), last_confirmed_value=Truth.TRUE,
            last_confirmed_time=open_rec.available_time,
            reason="tp-so-mvp-static-public-invariant:container_declared_open_and_no_registered_writer",
            quality=open_rec.quality, hold_epoch=open_rec.hold_epoch,
        )
        self.prev["p:Open:container"] = Truth.TRUE
        return tuple(replacement if r.fact_id == replacement.fact_id else r for r in recs)


def make_tp_so_evaluator(env, deadline, task_id):
    assert task_id == TASK_ID
    return TPSOMVPTaskEvaluator(env, deadline, task_id=task_id)


class TPSOMVPTaskEvaluator(TaskEvaluator):
    """T_P_SO_MVP goal: BOTH objects inside the container. D0/T_P_SR semantics are not modified.

    TaskEvaluator.goal_true() dispatches on task_id and has no branch for T_P_SO_MVP, so the subclass overrides
    goal_true only; deadline/reward/termination semantics are inherited unchanged.
    """

    def goal_true(self) -> bool:
        h = self.env.hidden_truth()
        return bool(self._inside(h, "target") and self._inside(h, SECOND_ROLE))


class RecordingTPSOMVPVerifier(TPSOMVPFactVerifier):
    recorder = None

    def verify(self, measurement, execution=None):
        recs = super().verify(measurement, execution)
        if self.recorder is not None:
            self.recorder.facts(recs)
        return recs


class _NullEnv:
    """Placeholder so RuntimeBundle.start_case can close 'the previous environment'; no simulator is constructed."""

    def close(self):
        return None


class TPSOMVPBundle(RuntimeBundle):
    recorder = None
    branch = None
    out_root = None
    out_root_phys = None

    def configure(self, out_root, phys_dir, branch):
        """out_root: experiment output root (captures/ live here); phys_dir: witness dir for this wave."""
        self.out_root, self.out_root_phys, self.branch = Path(out_root), Path(phys_dir), dict(branch)
        self.recorder = RunRecorder(self.out_root, branch["branch_id"], {"case_id": branch.get("case_id"), "route": branch.get("route")})
        return self.recorder

    def start_case(self, case_id, restore_seed=None):
        snap = super().start_case(case_id, restore_seed=restore_seed)
        try:
            self._install_recorders()
        except Exception as exc:  # noqa: BLE001
            self.recorder.errors.append({"where": "install", "error": f"{type(exc).__name__}: {exc}"})
        if self.branch is not None:
            from cp_disr.analysis.s1_e4_recovery import ProbedBundle
            ProbedBundle(self, self.out_root_phys, self.branch)._write_sidecars()
        if not isinstance(self.evaluator._inner if hasattr(self.evaluator, "_inner") else self.evaluator, TPSOMVPTaskEvaluator):
            raise BindingError("T_P_SO_MVP evaluator binding lost after start_case")
        if not isinstance(self.verifier, TPSOMVPFactVerifier):
            raise BindingError("T_P_SO_MVP verifier binding lost after start_case")
        return snap

    def _install_recorders(self):
        rec, env = self.recorder, self.environment
        env.recorder = rec
        self.perception.recorder = rec
        if isinstance(self.verifier, TPSOMVPFactVerifier):
            self.verifier.__class__ = RecordingTPSOMVPVerifier
            self.verifier.recorder = rec
        else:
            self.verifier.__class__ = RecordingVerifier          # generic verifier path
            self.verifier.recorder = rec
        inner = self.executor.inner
        inner.__class__ = InstrumentedSkillExecutor
        inner.recorder = rec
        self.evaluator = RecordingEvaluatorProxy(self.evaluator, rec)
        self.snapshot_builder = RecordingSnapshotBuilderProxy(self.snapshot_builder, rec)
        rec.write_initial(env, self)

    def env_counts(self):
        env = self.environment
        return {"env_class": type(env).__name__, "constructions": 1, "bootstrap_constructions": 0,
                "reset_calls": int(getattr(env, "reset_calls", -1)), "internal_resets": int(getattr(env, "internal_resets", -1)),
                "restore_chain": "start_case: make_env -> env.reset() -> _apply_case_poses -> _open_gripper_reset (20 gripper-open steps)"}


def ground_task_contracts(contract_path, timeouts):
    """Production grounding helper, then restriction to ALLOWED_ACTION_IDS (the other groundings are absent, not merely masked)."""
    grounded = _ground_contracts_for(contract_path, timeouts, OBJECTS)
    ids = {c.id for c in grounded}
    if not set(ALLOWED_ACTION_IDS) <= ids:
        raise BindingError("T_P_SO_MVP grounding is missing an allowed action")
    if any(str(i).startswith(p) for i in ids for p in EXCLUDED_GROUNDING_PREFIXES):
        raise BindingError("T_P_SO_MVP grounding contains an excluded action")
    kept = tuple(c for c in grounded if c.id in ALLOWED_ACTION_IDS)
    if tuple(sorted(c.id for c in kept)) != tuple(sorted(ALLOWED_ACTION_IDS)):
        raise BindingError("T_P_SO_MVP allowed action set mismatch")
    return kept


def _atom_from_id(fact_id):
    parts = fact_id.split(":")
    return Atom(parts[1], tuple(parts[2:]))


def build_task_template(contracts):
    have = {a.id for c in contracts for a in c.atoms()}
    extra = tuple(_atom_from_id(f) for f in VERIFIER_FACT_IDS if f not in have)
    goals = tuple(Goal(g, 1) for g in GOAL_FACTS)
    template = build_template(contracts, goals, PREDICATES, OBJECTS, extra_atoms=extra)
    props = {n.id for n in template.nodes if n.kind == "PROPOSITION"}
    if props != set(VERIFIER_FACT_IDS):
        raise BindingError("T_P_SO_MVP template propositions differ from verifier facts")
    if any(any(h in p for h in FORBIDDEN_HARD_FACTS) for p in props):
        raise BindingError("forbidden hard fact registered")
    return template


def create_tp_so_mvp_runtime(manifest: dict):
    runtime = manifest["runtime"]
    root = Path(runtime["repository_path"])
    if runtime.get("active_task_id") != TASK_ID:
        raise BindingError("runtime.active_task_id must be T_P_SO_MVP")
    timeouts = runtime["skill_timeouts"][TASK_ID]
    deadline = float(runtime["task_deadlines"][TASK_ID])
    contract_path = root / runtime["stage_2a_contract_path"]
    contracts = ground_task_contracts(contract_path, timeouts)
    for c in contracts:
        object.__setattr__(c, "timeout_seconds", float(timeouts[c.name]))
    template = build_task_template(contracts)
    split = _read(root / runtime["task_splits"][TASK_ID])
    cases = {}
    for row in split["train"] + split["dev"]:
        if row.get("lid_closed") is not False or row.get("second_role") != SECOND_ROLE:
            raise BindingError("T_P_SO_MVP rows must have an open container and the second_object role")
        cases[row["case_id"]] = CaseSpec(case_id=row["case_id"], split=row["split"], seed=int(row["seed"]),
                                         target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
                                         container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]),
                                         lid_closed=False, task_id=TASK_ID, second_role=SECOND_ROLE, deadline=deadline)
    null_env = _NullEnv()
    bundle = TPSOMVPBundle(environment=null_env, executor=None, observations=None, perception=None,
                         verifier=TPSOMVPFactVerifier(null_env),
                         evaluator=make_tp_so_evaluator(null_env, deadline, TASK_ID), safety=None,
                         clock=None, snapshot_builder=None, task_id=TASK_ID, template=template, cases=cases, caches={},
                         env_factory=InstrumentedD0Env, perception_cls=TPSRNearestPalettePerceptionAdapter,
                         verifier_factory=TPSOMVPFactVerifier, evaluator_factory=make_tp_so_evaluator)
    assert isinstance(bundle.evaluator, TPSOMVPTaskEvaluator)
    return bundle


def create(manifest: dict):
    return create_tp_so_mvp_runtime(manifest)