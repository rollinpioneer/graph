"""RuntimeFactory for T_P_SOFT_RELOCATION_V1 (S4).

Reuses the production RuntimeBundle, D0 environment, SkillExecutor, PerceptionAdapter, FactVerifier, TaskEvaluator,
SafetyManager, DurationProvider and SnapshotBuilder unchanged. This module only (a) grounds the T_P_SR contract
registry (no OPEN), (b) restricts the executable action set to the four task actions, and (c) reads the T_P_SR pool split.
No second execution semantics is defined here.
"""
from __future__ import annotations

from pathlib import Path

from cp_disr.common import BindingError
from cp_disr.contracts import Atom
from cp_disr.graph import Goal, build_template

from .clock import DurationProvider
from .d0_env import CaseSpec, make_env
from .observations import ObservationProvider
from . import perception as _perception
from .d0_env import COLORS
from .perception import PerceptionAdapter
from .runtime_factory import PREDICATES, RuntimeBundle, _Executor, _ground_contracts_for, _read
from .safety import SafetyManager
from .skill_executor import SkillExecutor
from .snapshot import SnapshotBuilder
from .task_evaluator import TaskEvaluator
from .verifier import FactVerifier

TASK_ID = "T_P_SR"
TASK_FAMILY_ID = "T_P_SOFT_RELOCATION_V1"
OBJECTS = {"target": "object", "interferer": "object", "container": "container", "buffer": "buffer"}
GOAL_FACTS = ("p:Inside:target:container",)
SECOND_ROLE = "interferer"
ALLOWED_ACTION_IDS = ("a:PICK:target:v1", "a:PICK:interferer:v1", "a:PLACE:target:container:v1", "a:PLACE_BUFFER:interferer:buffer:v1")
EXCLUDED_GROUNDINGS = ("a:PLACE:interferer:container:v1", "a:PLACE_BUFFER:target:buffer:v1")
FORBIDDEN_HARD_FACTS = ("ClearPath", "Occluded", "Blocked", "EasyGrasp", "SafeToPick")
# Facts produced by the production verifier for both cubes; kept as template propositions so FactStore and template align.
VERIFIER_FACT_IDS = ("p:GripperEmpty", "p:Held:target", "p:Held:interferer", "p:OnTable:target", "p:OnTable:interferer", "p:Open:container",
                     "p:Inside:target:container", "p:Inside:interferer:container", "p:AtBuffer:target:buffer", "p:AtBuffer:interferer:buffer")


PERCEPTION_VARIANT = "cp-disr-d0-rgbd-colorseg-v2+nearest-palette-assignment"


def _nearest_palette_mask(rgb, color, tol):
    """Production tolerance mask, additionally requiring that no other palette colour is strictly closer.

    Reason (S4 finding, static reset of T_P_SR_pool_01): with the cyan interferer in the scene the independent
    tolerance masks of the blue container and green buffer both fire on cyan cube pixels, so the container and buffer
    blobs are back-projected onto the interferer and Inside/AtBuffer(interferer) read TRUE at reset. Assigning each
    pixel to its nearest palette colour removes the cross-talk; the tolerance, palette and back-projection are unchanged.
    """
    import numpy as np
    img = rgb.astype(np.float32)
    if img.max() > 1.5:
        img = img / 255.0
    mine = np.linalg.norm(img - np.asarray(color[:3])[None, None, :], axis=2)
    ok = mine < tol
    for other in COLORS.values():
        if other is color or np.array_equal(np.asarray(other[:3]), np.asarray(color[:3])):
            continue
        ok &= mine <= np.linalg.norm(img - np.asarray(other[:3])[None, None, :], axis=2)
    return ok


def install_nearest_palette_mask():
    """Process-wide, idempotent; only called when a T_P_SR runtime is created, never on import."""
    _perception._mask = _nearest_palette_mask
    return PERCEPTION_VARIANT


def ground_task_contracts(contract_path, timeouts):
    """Production grounding helper, then restriction to ALLOWED_ACTION_IDS (the other groundings are absent, not merely masked)."""
    grounded = _ground_contracts_for(contract_path, timeouts, OBJECTS)
    ids = {c.id for c in grounded}
    if not set(ALLOWED_ACTION_IDS) <= ids or set(EXCLUDED_GROUNDINGS) - ids:
        raise BindingError("T_P_SR grounding does not contain the expected action set")
    if any(":OPEN:" in i for i in ids):
        raise BindingError("OPEN must not be registered for T_P_SR")
    kept = tuple(c for c in grounded if c.id in ALLOWED_ACTION_IDS)
    if tuple(sorted(c.id for c in kept)) != tuple(sorted(ALLOWED_ACTION_IDS)):
        raise BindingError("T_P_SR allowed action set mismatch")
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
        raise BindingError("T_P_SR template propositions differ from verifier facts")
    if any(any(h in p for h in FORBIDDEN_HARD_FACTS) for p in props):
        raise BindingError("forbidden hard fact registered")
    return template


def create_tp_sr_runtime(manifest: dict):
    runtime = manifest["runtime"]
    install_nearest_palette_mask()
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
    cases, caches = {}, {}
    for row in split["train"] + split["dev"]:
        if row.get("lid_closed") is not False or row.get("second_role") != SECOND_ROLE:
            raise BindingError("T_P_SR rows must have an open container and the interferer role")
        cases[row["case_id"]] = CaseSpec(case_id=row["case_id"], split=row["split"], seed=int(row["seed"]),
                                         target_xy=tuple(row["target_xy"]), second_xy=tuple(row["second_xy"]),
                                         container_xy=tuple(row["container_xy"]), buffer_xy=tuple(row["buffer_xy"]),
                                         lid_closed=False, task_id=TASK_ID, second_role=SECOND_ROLE, deadline=deadline)
    first = next(iter(cases.values()))
    env = make_env(first)
    env.last_perception = {}
    env.reset()
    clock = DurationProvider(env)
    safety = SafetyManager(env)
    bundle = RuntimeBundle(environment=env, executor=_Executor(SkillExecutor(env, safety, clock)), observations=ObservationProvider(env, clock),
                           perception=PerceptionAdapter(env), verifier=FactVerifier(env), evaluator=TaskEvaluator(env, deadline, task_id=TASK_ID),
                           safety=safety, clock=clock, snapshot_builder=SnapshotBuilder(template, (), env, clock, deadline),
                           task_id=TASK_ID, template=template, cases=cases, caches=caches)
    env.refresh_perception = lambda e=env, p=bundle.perception: p.infer(e.public_observation())
    return bundle


def create(manifest: dict):
    return create_tp_sr_runtime(manifest)
