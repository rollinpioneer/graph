"""B_PLAN: finite-cost symbolic search over registered skill contracts.

The planner is deliberately small and closed over public runtime state. It
receives only the measured FactStore, the registered grounded contracts, the
registered goals, and the public remaining deadline. Nominal successors are
read-only overlays; real execution, verification, and evaluation stay in the
shared RuntimeFactory bundle.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Mapping
import hashlib
import heapq
import json
import math
import time

from ..adapters import EvaluationInput
from ..common import BindingError, ContractError, canonical, digest
from ..contracts import nominal_overlay, precondition_value
from ..facts import FactStore, Truth
from ..runtime import load_runtime


SEARCH_STATUSES = frozenset(
    {"GOAL_ALREADY_SATISFIED", "PLAN_FOUND", "NO_PLAN", "SEARCH_TIMEOUT"}
)


@dataclass(frozen=True)
class SearchConfig:
    depth_limit: int = 6
    max_nodes: int = 4096
    cpu_time_limit_seconds: float = 2.0
    reference_skill_seconds: float = 4.2

    def __post_init__(self):
        if self.depth_limit <= 0:
            raise ValueError("depth_limit must be positive")
        if self.max_nodes <= 0:
            raise ValueError("max_nodes must be positive")
        if self.cpu_time_limit_seconds <= 0:
            raise ValueError("cpu_time_limit_seconds must be positive")
        if self.reference_skill_seconds <= 0:
            raise ValueError("reference_skill_seconds must be positive")


@dataclass(frozen=True)
class PlanResult:
    status: str
    plan: tuple[str, ...]
    cost: float
    expanded_nodes: int
    generated_nodes: int
    cpu_seconds: float
    depth: int
    unmet_goals: int
    relaxation_steps: int
    nominal_failures: int = 0
    reason: str = ""

    def as_dict(self):
        value = asdict(self)
        value["plan"] = list(self.plan)
        return value


def _truth(value) -> Truth:
    return value if isinstance(value, Truth) else Truth(value)


def _state_key(values: Mapping[str, Truth]) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), _truth(value).value) for key, value in values.items()))


def _state_values(state) -> dict[str, Truth]:
    return {key: Truth(value) for key, value in state}


def _goal_met(values: Mapping[str, Truth], goal) -> bool:
    current = _truth(values.get(goal.fact_id, Truth.UNKNOWN))
    return current == (Truth.TRUE if goal.sign == 1 else Truth.FALSE)


def _relaxed_goal_distance(values, goal, contracts) -> int:
    """Return a layered contract-count relaxation distance for one goal.

    This intentionally ignores destructive interference and treats each
    contract effect as a possible monotone support. It is a search heuristic,
    not a claim of admissibility.
    """
    target = goal.fact_id
    want = Truth.TRUE if goal.sign == 1 else Truth.FALSE
    current = {key: _truth(value) for key, value in values.items()}
    if current.get(target, Truth.UNKNOWN) == want:
        return 0
    positive = {key for key, value in current.items() if value == Truth.TRUE}
    negative = {key for key, value in current.items() if value == Truth.FALSE}
    for distance in range(1, len(contracts) + 2):
        add_now = set()
        delete_now = set()
        for contract in contracts:
            if any(atom.id not in positive for atom in contract.pre_pos):
                continue
            if any(atom.id not in negative for atom in contract.pre_neg):
                continue
            effects = [contract.effects]
            effects.extend(item.effects for item in contract.conditional)
            for effect in effects:
                add_now.update(atom.id for atom in effect.add)
                delete_now.update(atom.id for atom in effect.delete)
        if target in (add_now if want == Truth.TRUE else delete_now):
            return distance
        before = (len(positive), len(negative))
        positive.update(add_now)
        negative.update(delete_now)
        if (len(positive), len(negative)) == before:
            break
    return max(1, len(contracts) + 1)


def _goal_stats(values, goals, contracts) -> tuple[int, int]:
    unmet = 0
    relaxation = 0
    for goal in goals:
        if not _goal_met(values, goal):
            unmet += 1
            relaxation += _relaxed_goal_distance(values, goal, contracts)
    return unmet, relaxation


class BPlanPlanner:
    """Canonical finite-cost planner over the registered nominal contract graph."""

    def __init__(self, config: SearchConfig | None = None):
        self.config = config or SearchConfig()

    def plan(self, facts: FactStore, template, deadline_seconds: float) -> PlanResult:
        if not math.isfinite(float(deadline_seconds)) or float(deadline_seconds) < 0:
            raise ValueError("deadline_seconds must be finite and nonnegative")
        started = time.process_time()
        values = dict(facts.values)
        contracts = tuple(sorted(template.contracts, key=lambda item: item.id))
        unmet, relaxation = _goal_stats(values, template.goals, contracts)
        if unmet == 0:
            return PlanResult(
                "GOAL_ALREADY_SATISFIED",
                (),
                0.0,
                0,
                1,
                time.process_time() - started,
                0,
                0,
                0,
            )

        reference_cost = float(self.config.reference_skill_seconds)
        initial_state = _state_key(values)
        initial_plan: tuple[str, ...] = ()
        serial = 0
        generated = 1
        expanded = 0
        nominal_failures = 0
        initial_h = (unmet + relaxation) * reference_cost
        frontier = [(initial_h, 0.0, initial_plan, serial, initial_state)]
        best = {initial_state: (0.0, initial_plan)}
        timeout_reason = ""

        while frontier:
            elapsed = time.process_time() - started
            if elapsed >= self.config.cpu_time_limit_seconds:
                timeout_reason = "CPU_TIME_LIMIT"
                break
            if expanded >= self.config.max_nodes:
                timeout_reason = "MAX_NODES"
                break
            _, cost, plan_ids, _, state = heapq.heappop(frontier)
            previous = best.get(state)
            if previous is None or previous != (cost, plan_ids):
                continue
            values = _state_values(state)
            unmet, relaxation = _goal_stats(values, template.goals, contracts)
            if unmet == 0:
                return PlanResult(
                    "PLAN_FOUND",
                    plan_ids,
                    cost,
                    expanded,
                    generated,
                    time.process_time() - started,
                    len(plan_ids),
                    unmet,
                    relaxation,
                    nominal_failures,
                )
            if len(plan_ids) >= self.config.depth_limit:
                expanded += 1
                continue
            expanded += 1
            for contract in contracts:
                if time.process_time() - started >= self.config.cpu_time_limit_seconds:
                    timeout_reason = "CPU_TIME_LIMIT"
                    break
                if precondition_value(contract, values) != Truth.TRUE:
                    continue
                next_cost = cost + reference_cost
                if next_cost > float(deadline_seconds):
                    continue
                try:
                    overlay = nominal_overlay(
                        contract,
                        values,
                        derived=template.derived_rules,
                        exclusive_groups=template.exclusive_groups,
                    )
                    next_values = dict(overlay)
                except (ContractError, KeyError, ValueError):
                    nominal_failures += 1
                    continue
                next_state = _state_key(next_values)
                next_plan = plan_ids + (contract.id,)
                previous = best.get(next_state)
                if previous is not None and previous <= (next_cost, next_plan):
                    continue
                best[next_state] = (next_cost, next_plan)
                next_unmet, next_relaxation = _goal_stats(
                    next_values, template.goals, contracts
                )
                heuristic = (next_unmet + next_relaxation) * reference_cost
                serial += 1
                heapq.heappush(
                    frontier,
                    (
                        next_cost + heuristic,
                        next_cost,
                        next_plan,
                        serial,
                        next_state,
                    ),
                )
                generated += 1
            if timeout_reason:
                break

        status = "SEARCH_TIMEOUT" if timeout_reason else "NO_PLAN"
        return PlanResult(
            status,
            (),
            0.0,
            expanded,
            generated,
            time.process_time() - started,
            0,
            unmet,
            relaxation,
            nominal_failures,
            timeout_reason,
        )


def _read_document(path: Path):
    path = Path(path)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        return json.loads(text)
    import yaml

    return yaml.safe_load(text)


def _jsonable(value):
    if isinstance(value, Truth):
        return value.value
    if hasattr(value, "value") and not isinstance(value, (str, int, float, bool)):
        try:
            return value.value
        except Exception:
            pass
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if hasattr(value, "__dict__"):
        return _jsonable(vars(value))
    return value


def bind_t_b_manifest(root: Path, manifest_path: Path, split_path: Path, config=None):
    """Materialize a verified T_B RuntimeFactory binding for this checkout."""
    import yaml

    root = Path(root).resolve()
    manifest = _read_document(Path(manifest_path))
    runtime = manifest.setdefault("runtime", {})
    config = config or SearchConfig()
    split_path = Path(split_path).resolve()
    try:
        split_ref = str(split_path.relative_to(root))
    except ValueError as exc:
        raise BindingError("T_B development split must be inside repository") from exc
    factory_source = (root / "src/cp_disr/platforms/libero/runtime_factory.py").resolve()
    runtime["repository_path"] = str(root)
    runtime["experiment_root"] = str((root / "experiments").resolve())
    runtime["active_task_id"] = "T_B"
    runtime.setdefault("task_splits", {})["T_B"] = split_ref
    runtime.setdefault("reference_skill_seconds_by_task", {})["T_B"] = float(
        config.reference_skill_seconds
    )
    factory = manifest.setdefault("runtime_factory", {})
    factory["module"] = "cp_disr.platforms.libero.runtime_factory"
    factory["factory"] = "create_stage_2a_runtime"
    factory["source_path"] = str(factory_source)
    factory["sha256"] = hashlib.sha256(factory_source.read_bytes()).hexdigest()
    manifest["_b_plan_binding"] = {
        "task_id": "T_B",
        "algorithm": "finite_cost_symbolic_search_replanning",
        "search_depth": config.depth_limit,
        "max_nodes": config.max_nodes,
        "cpu_time_limit_seconds": config.cpu_time_limit_seconds,
        "reference_skill_seconds": config.reference_skill_seconds,
        "split": split_ref,
        "public_planner_inputs": [
            "FactStore",
            "goal",
            "complete_registered_SkillContract",
            "public_remaining_deadline",
        ],
        "prohibited_planner_inputs": [
            "hidden_simulator_truth",
            "future_results",
            "VLM_relations",
            "RL_Q",
        ],
    }
    return manifest, yaml.safe_dump(manifest, sort_keys=False)


def _evaluate(bundle, snapshot, execution, interval_start, interval_end):
    evidence = () if execution is None else tuple(execution.evidence_ids)
    value = EvaluationInput(
        task_id=bundle.task_id,
        env_id=snapshot.env_id,
        episode_id=snapshot.episode_id,
        evidence_refs=evidence,
        elapsed_seconds=float(interval_end - bundle.episode_start_seconds),
        interval_start_seconds=float(interval_start),
        interval_end_seconds=float(interval_end),
    )
    return bundle.evaluator.evaluate(value)


def _make_relation_free_runtime_split(source_path: Path, destination: Path):
    """Copy the authorized split while removing VLM relation cache bindings."""
    document = _read_document(Path(source_path))
    for split_name in ("train", "dev", "test"):
        for row in document.get(split_name) or []:
            for key in ("cache_dir", "cache_key", "cache_status"):
                row.pop(key, None)
    destination.write_text(canonical(document), encoding="utf-8")
    return document


def _load_reference_skill_seconds(root: Path) -> float:
    reference_path = Path(root) / "runs/stage_0a/calibration.json"
    reference = _read_document(reference_path)
    try:
        value = float(reference["families"]["T_B"]["d_ref_median"])
    except (KeyError, TypeError, ValueError) as exc:
        raise BindingError("T_B independent reference d_ref_median is unavailable") from exc
    if not math.isfinite(value) or value <= 0:
        raise BindingError("T_B independent reference d_ref_median must be positive")
    return value


def _contract_by_id(template):
    return {contract.id: contract for contract in template.contracts}


def run_development(
    root: Path,
    manifest_path: Path,
    split_path: Path,
    output_dir: Path,
    max_episodes: int = 10,
    gpu: int = 0,
):
    """Run only the authorized T_B development episodes."""
    del gpu
    root = Path(root).resolve()
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if not 1 <= int(max_episodes) <= 10:
        raise ValueError("max_episodes must be between 1 and 10")
    reference_skill_seconds = _load_reference_skill_seconds(root)
    config = SearchConfig(reference_skill_seconds=reference_skill_seconds)
    runtime_split_path = output_dir / "b_plan_runtime_split.json"
    split = _make_relation_free_runtime_split(split_path, runtime_split_path)
    manifest, manifest_text = bind_t_b_manifest(root, manifest_path, runtime_split_path, config)
    (output_dir / "b_plan_runtime_manifest.yaml").write_text(
        manifest_text, encoding="utf-8"
    )
    config_payload = {
        "method": "B_PLAN",
        "task_id": "T_B",
        "search": asdict(config),
        "public_inputs": [
            "FactStore from shared Verifier",
            "goal from shared GraphTemplate",
            "complete registered SkillContract",
            "public remaining deadline",
        ],
        "prohibited_inputs": [
            "hidden simulator truth",
            "future results",
            "VLM relation",
            "RL Q",
        ],
        "execution_policy": "execute only first action, verify, then replan",
        "no_rl": True,
        "no_provider": True,
        "relation_cache_loaded": False,
        "runtime_split": "b_plan_runtime_split.json",
    }
    (output_dir / "b_plan_config.json").write_text(
        canonical(config_payload), encoding="utf-8"
    )
    rows = list(split.get("dev") or [])[: int(max_episodes)]
    if not rows:
        raise BindingError("T_B development split has no dev rows")
    bundle = load_runtime(manifest)
    planner = BPlanPlanner(config)
    decisions = []
    episodes = []
    search_counts = {status: 0 for status in SEARCH_STATUSES}
    total_skill_failures = 0
    total_verifier_failures = 0
    try:
        for row in rows:
            case_id = row["case_id"]
            snapshot = bundle.start_case(case_id)
            deadline = float(bundle.cases[case_id].deadline)
            contract_map = _contract_by_id(snapshot.template)
            episode = {
                "case_id": case_id,
                "split": row.get("split", "dev"),
                "method": "B_PLAN",
                "status": "UNSET",
                "success": False,
                "decision_count": 0,
                "skill_failure_count": 0,
                "verifier_failure_count": 0,
                "search_statuses": [],
                "search_status_counts": {},
                "decisions": [],
            }
            max_decisions = max(
                1, int(math.ceil(deadline / config.reference_skill_seconds)) + 2
            )
            for decision_index in range(max_decisions):
                now = float(bundle.clock.now_seconds())
                remaining = max(0.0, deadline - now)
                result = planner.plan(snapshot.facts, snapshot.template, remaining)
                episode["decision_count"] += 1
                episode["search_statuses"].append(result.status)
                search_counts[result.status] = search_counts.get(result.status, 0) + 1
                decision = {
                    "case_id": case_id,
                    "decision_index": decision_index,
                    "facts_hash": digest(
                        {key: _truth(value).value for key, value in snapshot.facts.values.items()}
                    ),
                    "goal": _jsonable(snapshot.template.goals),
                    "remaining_deadline_seconds": remaining,
                    "search": result.as_dict(),
                    "first_action": result.plan[0] if result.plan else None,
                    "prior_edges_used": False,
                }
                episode["decisions"].append(decision)
                decisions.append(decision)
                if result.status in ("NO_PLAN", "SEARCH_TIMEOUT"):
                    episode["status"] = result.status
                    break
                if result.status == "GOAL_ALREADY_SATISFIED":
                    try:
                        task_result = _evaluate(
                            bundle,
                            snapshot,
                            None,
                            now,
                            float(bundle.clock.now_seconds()),
                        )
                        episode["evaluator"] = _jsonable(task_result)
                        episode["success"] = bool(task_result.success)
                        episode["status"] = (
                            "SUCCESS" if task_result.success else "GOAL_ALREADY_SATISFIED"
                        )
                    except Exception as exc:
                        episode["status"] = "EVALUATOR_FAILURE"
                        episode["evaluator_error"] = str(exc)
                    break
                action_id = result.plan[0]
                contract = contract_map[action_id]
                try:
                    execution = bundle.executor.execute(
                        action_id, float(contract.timeout_seconds)
                    )
                except Exception as exc:
                    episode["skill_failure_count"] += 1
                    total_skill_failures += 1
                    episode["status"] = "SKILL_FAILURE"
                    episode["skill_failure_error"] = str(exc)
                    break
                raw_execution = getattr(bundle.executor, "last", None)
                decision["execution"] = _jsonable(raw_execution or execution)
                if execution.controller_exit != "NORMAL_TERMINATION":
                    episode["skill_failure_count"] += 1
                    total_skill_failures += 1
                interval_end = float(bundle.clock.now_seconds())
                try:
                    observation = bundle.observations.observe()
                    measurement = bundle.perception.infer(observation)
                    records = bundle.verifier.verify(measurement, execution)
                    snapshot = bundle.snapshot_builder.build(
                        snapshot,
                        records,
                        observation,
                        execution,
                        interval_end,
                    )
                except Exception as exc:
                    episode["verifier_failure_count"] += 1
                    total_verifier_failures += 1
                    episode["status"] = "VERIFIER_FAILURE"
                    episode["verifier_failure_error"] = str(exc)
                    decision["verifier_error"] = str(exc)
                    break
                try:
                    task_result = _evaluate(
                        bundle,
                        snapshot,
                        execution,
                        execution.start_seconds,
                        interval_end,
                    )
                    decision["evaluator"] = _jsonable(task_result)
                except Exception as exc:
                    episode["status"] = "EVALUATOR_FAILURE"
                    episode["evaluator_error"] = str(exc)
                    break
                if execution.controller_exit != "NORMAL_TERMINATION":
                    episode["status"] = "SKILL_FAILURE"
                    break
                if task_result.success:
                    episode["success"] = True
                    episode["status"] = "SUCCESS"
                    break
                if task_result.terminated or task_result.truncated:
                    episode["status"] = str(task_result.reason)
                    break
            else:
                episode["status"] = "DECISION_CAP"
            episode["search_status_counts"] = {
                status: episode["search_statuses"].count(status)
                for status in sorted(set(episode["search_statuses"]))
            }
            episodes.append(episode)
    finally:
        bundle.environment.close()

    summary = {
        "status": "COMPLETE",
        "task_id": "T_B",
        "method": "B_PLAN",
        "episodes_requested": int(max_episodes),
        "episodes_completed": len(episodes),
        "success_count": sum(bool(item["success"]) for item in episodes),
        "episode_status_counts": {
            status: sum(item["status"] == status for item in episodes)
            for status in sorted({item["status"] for item in episodes})
        },
        "search_status_counts": search_counts,
        "skill_failure_count": total_skill_failures,
        "verifier_failure_count": total_verifier_failures,
        "search": asdict(config),
        "reference_skill_seconds": config.reference_skill_seconds,
        "no_rl": True,
        "no_provider": True,
        "no_hidden_truth_to_planner": True,
        "episodes": episodes,
    }
    (output_dir / "decisions.jsonl").write_text(
        "".join(canonical(item) + chr(10) for item in decisions), encoding="utf-8"
    )
    (output_dir / "episodes.json").write_text(canonical(episodes), encoding="utf-8")
    (output_dir / "b_plan_dev_results.json").write_text(
        canonical(summary), encoding="utf-8"
    )
    return summary
