"""RuntimeFactory for the T_P_SR V2 pilot (Wave D diagnostics and Wave P prototypes).

Reuses the production RuntimeBundle.start_case restore, SkillExecutor, FactVerifier, TaskEvaluator, SnapshotBuilder and
the V1 T_P_SR contract grounding. Differences from V1 (tp_sr_runtime.py, untouched):
  * NO process-wide perception monkeypatch: the V1-compatible nearest-palette assignment lives only in
    TPSRNearestPalettePerceptionAdapter;
  * no bootstrap environment: the first environment of a branch is the one created by start_case (one restore chain);
  * passive recorders installed after start_case (they never change actions, facts, masks, planner input or seeds).
"""
from __future__ import annotations

from pathlib import Path

from cp_disr.common import BindingError

from .d0_env import CaseSpec
from .runtime_factory import RuntimeBundle, _read
from .tp_sr_instrumentation import (InstrumentedSkillExecutor, RecordingEvaluatorProxy, RecordingSnapshotBuilderProxy, RecordingVerifier, RunRecorder,
                                    TPSRNearestPalettePerceptionAdapter, LOCAL_PERCEPTION_VARIANT, INSTRUMENTATION_VERSION)
from .tp_sr_runtime import (SECOND_ROLE, TASK_FAMILY_ID, TASK_ID, ground_task_contracts, build_task_template)
from .tp_sr_v2_env import CaseSpecV2, make_v2_env

RUNTIME_IDENTITY = "tp-sr-v2-runtime-v1"


class _NullEnv:
    """Placeholder so RuntimeBundle.start_case can close 'the previous environment'; no simulator is constructed."""

    def close(self):
        return None


class TPSRV2Bundle(RuntimeBundle):
    recorder = None
    branch = None
    out_root = None
    out_root_phys = None

    def configure(self, out_root, phys_dir, branch):
        """out_root: pilot output root (captures/ live here); phys_dir: production witness dir for this wave."""
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
        return snap

    def _install_recorders(self):
        rec, env = self.recorder, self.environment
        env.recorder = rec
        self.perception.recorder = rec
        self.verifier.__class__ = RecordingVerifier          # same instance/state (`prev`), only adds recording
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


def create_tp_sr_v2_runtime(manifest: dict):
    runtime = manifest["runtime"]
    root = Path(runtime["repository_path"])
    if runtime.get("active_task_id") != TASK_ID:
        raise BindingError("runtime.active_task_id must be T_P_SR")
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
            raise BindingError("T_P_SR rows must have an open container and the interferer role")
        extra = {}
        if row.get("second_half") is not None:
            extra = {"second_half": tuple(row["second_half"]), "second_density": float(row.get("second_density", 200.0)),
                     "second_friction": tuple(row.get("second_friction", (1.2, 0.005, 0.0001)))}
        cases[row["case_id"]] = CaseSpecV2(case_id=row["case_id"], split=row["split"], seed=int(row["seed"]), target_xy=tuple(row["target_xy"]),
                                           second_xy=tuple(row["second_xy"]), container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]),
                                           lid_closed=False, task_id=TASK_ID, second_role=SECOND_ROLE, deadline=deadline, **extra)
    return TPSRV2Bundle(environment=_NullEnv(), executor=None, observations=None, perception=None, verifier=None, evaluator=None, safety=None,
                        clock=None, snapshot_builder=None, task_id=TASK_ID, template=template, cases=cases, caches={},
                        env_factory=make_v2_env, perception_cls=TPSRNearestPalettePerceptionAdapter)


def create(manifest: dict):
    return create_tp_sr_v2_runtime(manifest)
