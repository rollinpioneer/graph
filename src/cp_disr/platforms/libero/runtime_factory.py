"""Concrete D0 RuntimeFactory. Fail closed on missing bindings."""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
import hashlib, json, os

from cp_disr.common import BindingError
from cp_disr.contracts import from_dict, Registry
from cp_disr.facts import FactStore
from cp_disr.graph import build_template, Goal
from cp_disr.adapters import ExecutionResult

from .d0_env import CaseSpec, make_env
from .skill_executor import SkillExecutor
from .perception import PerceptionAdapter
from .verifier import FactVerifier
from .task_evaluator import TaskEvaluator
from .safety import SafetyManager
from .clock import DurationProvider
from .observations import ObservationProvider
from .snapshot import SnapshotBuilder, load_cache_edges


PREDICATES = {
    "GripperEmpty": [],
    "Held": ["object"],
    "OnTable": ["object"],
    "Open": ["container"],
    "Inside": ["object", "container"],
    "AtBuffer": ["object", "buffer"],
}
OBJECTS = {"target": "object", "second_object": "object", "container": "container", "buffer": "buffer"}


def _read(path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text)


def sha_file(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class _Executor:
    def __init__(self, inner):
        self.inner = inner
        self.last = None

    def execute(self, candidate_id, timeout_seconds):
        raw = self.inner.execute(candidate_id, timeout_seconds)
        self.last = raw
        return ExecutionResult(
            execution_id=raw["execution_id"],
            controller_exit=raw["controller_exit"],
            start_seconds=raw["start_seconds"],
            end_seconds=raw["end_seconds"],
            evidence_ids=tuple(raw.get("evidence_ids") or ()),
        )


@dataclass
class RuntimeBundle:
    environment: object
    executor: object
    observations: object
    perception: object
    verifier: object
    evaluator: object
    safety: object
    clock: object
    snapshot_builder: object
    task_id: str = "D0"
    current_snapshot: object = None
    original_prior_edges: tuple = ()
    episode_start_seconds: float = 0.0
    template: object = None
    cases: dict = field(default_factory=dict)
    caches: dict = field(default_factory=dict)
    _n: int = 0

    def start_case(self, case_id: str):
        spec = self.cases[case_id]
        self.environment.close()
        env = make_env(spec)
        env.last_perception = {}
        env.reset()
        env._apply_case_poses()
        env._open_gripper_reset()
        self.environment = env
        clock = DurationProvider(env)
        safety = SafetyManager(env)
        self.clock = clock
        self.safety = safety
        self.observations = ObservationProvider(env, clock)
        self.perception = PerceptionAdapter(env)
        env.refresh_perception = lambda e=env, p=self.perception: p.infer(e.public_observation())
        self.verifier = FactVerifier(env)
        deadline = float(spec.__dict__.get("deadline", 90.0))
        self.evaluator = TaskEvaluator(env, deadline)
        self.evaluator.reset_episode()
        self.executor = _Executor(SkillExecutor(env, safety, clock))
        cache = self.caches.get(case_id)
        edges = load_cache_edges(cache)
        self.original_prior_edges = edges
        self.snapshot_builder = SnapshotBuilder(self.template, edges, env, clock, deadline)
        obs = self.observations.observe()
        measured = self.perception.infer(obs)
        facts = FactStore(self.verifier.verify(measured, None))
        self._n += 1
        self.episode_start_seconds = clock.now_seconds()
        self.current_snapshot = self.snapshot_builder.initial(facts, obs, f"ep-{self._n}")
        return self.current_snapshot

    def next_case(self, task_cases, seed):
        return task_cases[self._n % len(task_cases)]


def _ground_contracts(contract_path, timeouts):
    doc = _read(contract_path)
    names = [c["name"] for c in doc["contracts"]]
    if "MOVE" in names:
        raise BindingError("D0 runtime registry must not include MOVE; use PICK+PLACE_BUFFER")
    for c in doc["contracts"]:
        c["timeout_seconds"] = float(timeouts[c["name"]])
        c["controller_ref"] = f"src/cp_disr/platforms/libero/skill_executor.py::{c['name']}"
        c["verifier_ref"] = "src/cp_disr/platforms/libero/verifier.py"
        c["verification"] = ["rgb_depth_postcondition"]
        if not isinstance(c.get("provenance"), str) or "MUST" in str(c.get("provenance")):
            c["provenance"] = "D0-runtime-bound-stage-1a-p0"
    reg = Registry(doc["predicate_types"])
    for c in doc["contracts"]:
        reg.register(from_dict(c))
    return reg.ground(OBJECTS)


def create_runtime(manifest: dict):
    root = Path(manifest["runtime"]["repository_path"])
    required = ["simulator_or_robot", "task_assets", "controller_manifest", "skill_timeouts", "task_deadlines", "task_splits"]
    runtime = manifest["runtime"]
    for k in required:
        if k not in runtime or runtime[k] in ("MUST_BIND", None, "", {}, []):
            raise BindingError("MUST_BIND: runtime." + k)
    timeouts = runtime["skill_timeouts"]["D0"]
    deadline = float(runtime["task_deadlines"]["D0"])
    contract_path = root / runtime.get("d0_contract_path", "configs/contracts/d0_runtime_skills.yaml")
    contracts = _ground_contracts(contract_path, timeouts)
    for c in contracts:
        object.__setattr__(c, "timeout_seconds", float(timeouts[c.name]))
    goals = (Goal("p:Inside:target:container", 1),)
    extra = []
    template = build_template(contracts, goals, PREDICATES, OBJECTS)
    split = _read(root / runtime["task_splits"]["D0"])
    cases = {}
    caches = {}
    for row in split["train"] + split["dev"]:
        spec = CaseSpec(
            case_id=row["case_id"],
            split=row["split"],
            seed=int(row["seed"]),
            target_xy=tuple(row["target_xy"]),
            second_xy=tuple(row["second_xy"]),
            container_xy=tuple(row["container_xy"]),
            buffer_xy=tuple(row["buffer_xy"]),
            lid_closed=bool(row.get("lid_closed", True)),
        )
        spec.deadline = deadline  # type: ignore
        cases[row["case_id"]] = spec
        if row.get("cache_dir"):
            caches[row["case_id"]] = root / row["cache_dir"]
    first = next(iter(cases.values()))
    env = make_env(first)
    env.last_perception = {}
    env.reset()
    clock = DurationProvider(env)
    safety = SafetyManager(env)
    bundle = RuntimeBundle(
        environment=env,
        executor=_Executor(SkillExecutor(env, safety, clock)),
        observations=ObservationProvider(env, clock),
        perception=PerceptionAdapter(env),
        verifier=FactVerifier(env),
        evaluator=TaskEvaluator(env, deadline),
        safety=safety,
        clock=clock,
        snapshot_builder=SnapshotBuilder(template, (), env, clock, deadline),
        template=template,
        cases=cases,
        caches=caches,
    )
    env.refresh_perception = lambda e=env, p=bundle.perception: p.infer(e.public_observation())
    return bundle


def create(manifest: dict):
    return create_runtime(manifest)
