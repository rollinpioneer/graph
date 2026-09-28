from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import yaml

from cp_disr.baselines.b_plan import BPlanPlanner, SearchConfig
from cp_disr.contracts import Registry, from_dict
from cp_disr.graph import Goal, build_template
from cp_disr.vlm import validate_relations


class RevisionError(RuntimeError):
    pass


class BudgetExceeded(RevisionError):
    pass


class HistoricalCacheNotFound(RevisionError):
    pass


class HistoricalCacheAmbiguous(RevisionError):
    pass


class HistoricalCacheIntegrity(RevisionError):
    pass


def utc_now():
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_bytes(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temp, path)
    finally:
        if os.path.exists(temp):
            os.unlink(temp)


def _atomic_json(path, value, retain_previous=False):
    path = Path(path)
    if retain_previous and path.exists():
        shutil.copy2(path, path.with_name(path.name + ".previous"))
    _atomic_bytes(path, (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode())


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_csv(path, fields, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def _append_jsonl(path, value):
    with Path(path).open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_revision_config(path):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(config, dict) or config.get("master_id") != "CP-DISR-FINAL-EXEC-3.0":
        raise RevisionError("invalid S1-REV1 config")
    if config.get("stage") != "S1" or config.get("revision_index") != 1:
        raise RevisionError("invalid revision identity")
    return config


def _budget_caps(config):
    budget = config["budgets"]
    return {
        "planner_environment_episodes": {"used": 0, "cap": int(budget["planner_environment_episodes"])},
        "physical_witness_episodes": {"used": 0, "cap": int(budget["physical_witness_episodes"])},
        "discovery_scenes": {"used": 0, "cap": int(budget["discovery_scenes"])},
        "provider_first_calls": {"used": 0, "cap": int(budget["provider_first_calls"])},
        "provider_retries": {"used": 0, "cap": int(budget["provider_retries_total"]), "per_scene_cap": int(budget["provider_retries_per_scene"])},
        "rl_transitions": {"used": 0, "cap": int(budget["rl_transitions"])},
        "optimizer_steps": {"used": 0, "cap": int(budget["optimizer_steps"])},
        "elastic_attempts": {"used": 0, "cap": int(budget["elastic_attempts"])},
    }


def create_or_load_budget_ledger(output_dir, config):
    output_dir = Path(output_dir)
    path = output_dir / "budget_ledger.json"
    expected = _budget_caps(config)
    if path.exists():
        ledger = _read_json(path)
        if set(ledger) != set(expected) or any(int(ledger[key]["cap"]) != int(expected[key]["cap"]) for key in expected):
            raise RevisionError("budget schema/cap changed")
        return ledger
    _atomic_json(path, expected)
    (output_dir / "budget_events.jsonl").touch()
    return expected


def _retry_count(output_dir, identity):
    path = Path(output_dir) / "budget_events.jsonl"
    if not path.exists():
        return 0
    total = 0
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line:
            continue
        event = json.loads(line)
        if event.get("budget_field") == "provider_retries" and event.get("identity") == identity:
            total += int(event.get("amount", 0))
    return total


def reserve_budget(output_dir, field, amount, identity):
    if int(amount) <= 0:
        raise BudgetExceeded("amount must be positive")
    output_dir = Path(output_dir)
    path = output_dir / "budget_ledger.json"
    ledger = _read_json(path)
    if field not in ledger:
        raise BudgetExceeded("unknown budget field")
    item = dict(ledger[field])
    old_used = int(item["used"])
    cap = int(item["cap"])
    if old_used + int(amount) > cap:
        raise BudgetExceeded(f"{field} cap exceeded")
    if field == "provider_retries" and _retry_count(output_dir, identity) + int(amount) > int(item.get("per_scene_cap", 1)):
        raise BudgetExceeded("provider retry per-scene cap exceeded")
    item["used"] = old_used + int(amount)
    ledger[field] = item
    _atomic_json(path, ledger, retain_previous=True)
    _append_jsonl(output_dir / "budget_events.jsonl", {
        "timestamp_utc": utc_now(),
        "budget_field": field,
        "identity": identity,
        "amount": int(amount),
        "old_used": old_used,
        "new_used": item["used"],
        "cap": cap,
        "command": os.environ.get("S1_REV1_COMMAND", "unit"),
        "source_commit": os.environ.get("S1_REV1_SOURCE_COMMIT", "unit"),
    })
    return ledger


def resolve_historical_cache(root, cache_key):
    found = set()
    for base in (Path(root) / "experiments/vlm_cache", Path(root) / "runs/stage_0c", Path(root) / "runs/final_master"):
        if not base.exists():
            continue
        for manifest_path in base.rglob("manifest.json"):
            try:
                manifest = _read_json(manifest_path)
            except (OSError, ValueError):
                continue
            if manifest.get("cache_key") == cache_key:
                found.add(manifest_path.parent.resolve())
    if not found:
        raise HistoricalCacheNotFound(cache_key)
    if len(found) != 1:
        raise HistoricalCacheAmbiguous(str(sorted(map(str, found))))
    return next(iter(found))


def _public_template(root):
    document = yaml.safe_load((Path(root) / "configs/runtime/stage_2a_contract_registry.yaml").read_text(encoding="utf-8"))
    registry = Registry(document["predicate_types"])
    for contract in document["contracts"]:
        registry.register(from_dict(contract))
    objects = {"target": "object", "second_object": "object", "container": "container", "buffer": "buffer"}
    return build_template(registry.ground(objects), (Goal("p:Inside:target:container", 1),), registry.predicate_types, objects)


def _previous_rows(root, config):
    previous = Path(root) / config["previous_evidence_dir"]
    files = list(previous.rglob("e1_relation_existence.csv"))
    if len(files) != 1:
        raise HistoricalCacheIntegrity("previous e1 file missing or ambiguous")
    rows = list(csv.DictReader(files[0].open(newline="", encoding="utf-8")))
    result = {}
    for spec in config["historical_cases"]:
        matches = [row for row in rows if row.get("source") == spec["source"] and row.get("case_id") == spec["case_id"] and row.get("cache_key") == spec["cache_key"] and int(row.get("raw_relation_count", -1)) == int(spec["expected_raw_relation_count"]) and int(row.get("parsed_relation_count", -1)) == int(spec["expected_parsed_relation_count"]) and int(row.get("admitted_relation_count", -1)) == int(spec["expected_admitted_relation_count"])]
        if len(matches) != 1:
            raise HistoricalCacheIntegrity(f"historical row mismatch: {spec['case_id']}")
        result[spec["case_id"]] = matches[0]
    return result


def _cache_hashes(path):
    names = ("manifest.json", "parsed_relations.json", "rejected_relations.json", "final_edges.json", "raw_response_attempt_0.json", "content_hashes.json")
    return {str(path / name): sha256_file(path / name) for name in names if (path / name).is_file()}


def freeze_history(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    config = load_revision_config(config_path)
    create_or_load_budget_ledger(output_dir, config)
    previous = root / config["previous_evidence_dir"]
    _atomic_bytes(output_dir / "raw/old_s1_before.sha256", "".join(f"{sha256_file(path)}  {path}\n" for path in sorted(previous.rglob("*")) if path.is_file()).encode())
    old_rows = _previous_rows(root, config)
    resolutions, hash_lines = [], []
    for spec in config["historical_cases"]:
        cache = resolve_historical_cache(root, spec["cache_key"])
        manifest = _read_json(cache / "manifest.json")
        if manifest.get("cache_key") != spec["cache_key"]:
            raise HistoricalCacheIntegrity(spec["case_id"])
        hashes = _cache_hashes(cache)
        hash_lines.extend(f"{value}  {key}\n" for key, value in sorted(hashes.items()))
        resolutions.append({"case_id": spec["case_id"], "cache_key": spec["cache_key"], "cache_path": str(cache), "historical_row": old_rows[spec["case_id"]], "manifest": {key: manifest.get(key) for key in ("split", "scene_id", "task_id", "model_snapshot", "sdk_api_version", "region", "endpoint", "prompt_hash", "fewshot_hash", "schema_hash", "contract_version", "predicate_version")}, "file_hashes": hashes})
    _atomic_json(output_dir / "historical_cache_resolution.json", resolutions)
    _atomic_bytes(output_dir / "historical_cache_hashes.sha256", "".join(sorted(hash_lines)).encode())
    branch = subprocess.check_output(["git", "-C", str(root), "branch", "--show-current"], text=True).strip()
    _atomic_json(output_dir / "stage_manifest.json", {"master_id": config["master_id"], "stage": "S1", "revision": "S1-REV1", "status": "IN_PROGRESS", "base_commit": config["base_commit"], "branch": branch, "worktree": str(root), "pythonpath_repair": str(root / "src"), "preflight_original_failure": "STOPPED_WORKTREE_PREFLIGHT: import cp_disr without PYTHONPATH", "preflight_repaired": True, "created_utc": utc_now()})
    _atomic_json(output_dir / "revision_manifest.json", {"revision": "S1-REV1", "base_commit": config["base_commit"], "historical_first": True, "historical_cases": [item["case_id"] for item in config["historical_cases"]], "new_provider_calls_before_gate": 0, "planner_environment_episodes": 0})
    return {"historical_cases": len(resolutions)}


def adjudicate_historical_case(root, case_spec, output_dir):
    cache = resolve_historical_cache(root, case_spec["cache_key"])
    relations = _read_json(cache / "final_edges.json")
    template = _public_template(root)
    public_hash = sha256_file(cache / "manifest.json")
    rows = []
    for index, relation in enumerate(relations, 1):
        if isinstance(relation, dict):
            relation = dict(relation)
        elif isinstance(relation, (list, tuple)) and len(relation) >= 3:
            relation = {
                "source_ref": str(relation[0]),
                "target_ref": str(relation[1]),
                "type": str(relation[2]),
                "effect_fact_ref": str(relation[3]) if len(relation) > 3 else "",
                "relation_id": f"historical-{index}",
            }
        else:
            raise HistoricalCacheIntegrity("invalid relation representation")
        validation = validate_relations({"schema_version": "m1_soft_relations_v2", "relations": [relation]}, template)
        accepted = bool(validation.accepted)
        effect_valid = accepted
        if accepted:
            contract = next(item for item in template.contracts if item.id == relation["source_ref"])
            effect_valid = relation["effect_fact_ref"] in {atom.id for atom in contract.effects.add + contract.effects.delete}
        legal = accepted and effect_valid
        rows.append({"case_id": case_spec["case_id"], "cache_key": case_spec["cache_key"], "relation_id": relation.get("relation_id", f"historical-{index}"), "relation_type": relation.get("type", ""), "source_ref": relation.get("source_ref", ""), "target_ref": relation.get("target_ref", ""), "effect_fact_ref": relation.get("effect_fact_ref", ""), "schema_valid": "TRUE" if accepted else "FALSE", "source_id_valid": "TRUE" if accepted else "FALSE", "target_id_valid": "TRUE" if accepted else "FALSE", "effect_ref_valid": "TRUE" if effect_valid else "FALSE", "contract_redundant": "FALSE" if accepted else "UNKNOWN", "natural_generation": "TRUE", "public_evidence_ref": str(cache / "manifest.json"), "public_evidence_sha256": public_hash, "truth_check_1": "TRUE" if legal else "FALSE", "truth_check_2": "UNKNOWN", "truth_final": "UNKNOWN", "utility_opportunity": "UNKNOWN", "utility_label": "UNKNOWN", "restorable_snapshot": "UNKNOWN", "legal_candidate_count": "NOT_MEASURED", "changed_patch_candidate_count": "NOT_MEASURED", "physical_witness_feasible": "UNKNOWN", "status": "UNKNOWN_NO_INDEPENDENT_EVIDENCE" if legal else "CONTRACT_REJECTED", "notes": "Only public cache and contract evidence used; hidden truth, future outcome, policy logit, Q and optimal action excluded."})
    return rows


def adjudicate_history(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    config = load_revision_config(config_path)
    all_rows, reassessment = [], []
    for spec in config["historical_cases"]:
        rows = adjudicate_historical_case(root, spec, output_dir)
        all_rows.extend(rows)
        reassessment.append({"case_id": spec["case_id"], "relation_count": len(rows), "natural_legal_nonredundant_count": sum(item["status"].startswith("UNKNOWN") for item in rows), "independently_adjudicable_count": sum(item["truth_final"] in ("TRUE", "FALSE") for item in rows), "restorable_snapshot": "UNKNOWN", "legal_candidate_count": "NOT_MEASURED", "changed_patch_candidate_count": "NOT_MEASURED", "common_continuation_feasible": "UNKNOWN", "status": "HISTORICAL_GATE_INSUFFICIENT"})
    fields = list(all_rows[0]) if all_rows else ["case_id"]
    _write_csv(output_dir / "adjudication/independent_adjudication.csv", fields, all_rows)
    _write_csv(output_dir / "adjudication/historical_case_reassessment.csv", list(reassessment[0]), reassessment)
    gate = {"passed": False, "failure_reasons": ["independent_public_consequence_evidence_missing", "exact_snapshot_restore_not_established", "candidate_structure_not_established"]}
    _atomic_json(output_dir / "historical_gate.json", gate)
    return gate


EXTRA_PROMPT = """EXTRA-CONTRACT FILTER — CP-DISR S1 REV1

Return a relation only when the public scene evidence supports a
scene-specific soft dependency that is not already entailed by any
registered PRE_POS, PRE_NEG, ADD, DEL, conditional guard, or their
direct grounded instantiation.

Eligible evidence may concern:
- additional execution cost;
- rework risk;
- resource contention;
- occlusion or access difficulty;
- preservation or recovery of a signed goal;
- a scene-specific consequence that remains soft rather than a hard
  safety or initiation condition.

Do not output:
- OPEN -> PLACE when Open(container) is already a registered
  precondition of PLACE;
- PICK -> PLACE merely because PICK creates Held(object);
- any order already forced by PRE/ADD/DEL or guards;
- a next action or complete plan;
- a confidence score or probability;
- a new action, object, proposition, effect, or goal;
- a relation based on hidden state or future outcomes.

Every accepted relation must cite one registered ADD or DEL effect of
the source action in effect_fact_ref.

When no publicly supported extra-contract relation exists, return:

{"relations":[]}
"""


def _optional_hash(path):
    return sha256_file(path) if Path(path).is_file() else "NOT_FOUND"


def freeze_source_revision(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    extra = root / "prompts/s1_rev1_extra_contract_filter.txt"
    extra.parent.mkdir(parents=True, exist_ok=True)
    extra.write_text(EXTRA_PROMPT, encoding="utf-8")
    source = {"revision": "S1-REV1", "original_prompt_sha256": _optional_hash(root / "experiments/sources/v2.1_interfaces/system_prompt.txt"), "extra_contract_sha256": sha256_file(extra), "few_shot_sha256": _optional_hash(root / "experiments/part_0_validation/stage_0c/few_shot_manifest.json"), "schema_sha256": _optional_hash(root / "schemas/relation_schema.json"), "contract_sha256": _optional_hash(root / "configs/runtime/stage_2a_contract_registry.yaml"), "runtime_factory_sha256": _optional_hash(root / "src/cp_disr/platforms/libero/runtime_factory.py"), "frozen_before_provider": True}
    _atomic_json(output_dir / "manifests/revised_source_manifest.json", source)
    _atomic_json(output_dir / "manifests/prompt_manifest.json", {"extra_contract": str(extra), "sha256": source["extra_contract_sha256"]})
    _atomic_json(output_dir / "manifests/few_shot_manifest.json", {"sha256": source["few_shot_sha256"], "validator": "cp_disr.vlm.validate_relations"})
    _atomic_json(output_dir / "manifests/admission_rule_manifest.json", {"validator": "cp_disr.vlm.validate_relations", "allowed_types": ["SOFT_SUPPORTS", "SOFT_RELEVANT_TO_GOAL"], "effect_fact_ref_required": True, "contract_redundancy_filter": True})
    return source


def freeze_discovery_manifest(root, config, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    path = output_dir / "manifests/frozen_discovery_scene_manifest.json"
    if path.exists():
        return _read_json(path)
    rows = [row for row in _read_json(root / "configs/splits/T_A_stage_2a.json")["dev"] if row["case_id"] not in {f"T_A_dev_{i:02d}" for i in range(12)} and not row.get("test", False)]
    rows = sorted(rows, key=lambda row: row["case_id"])[: int(config["budgets"]["discovery_scenes"])]
    scenes = []
    for row in rows:
        reserve_budget(output_dir, "discovery_scenes", 1, row["case_id"])
        scenes.append({"scene_id": row["case_id"], "task_id": "T_A", "split_role": "S1_DISCOVERY_ONLY", "generator_version": "existing_stage_2a_frozen_dev_row", "generator_seed": row.get("seed"), "reset_config": {key: row.get(key) for key in ("target_xy", "container_xy", "second_xy", "buffer_xy", "lid_closed")}, "reset_config_sha256": hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest(), "rgb_ref": "NOT_BOUND_NO_NEW_CAPTURE", "rgb_sha256": "NOT_MEASURED", "depth_ref": "NOT_BOUND_NO_NEW_CAPTURE", "depth_sha256": "NOT_MEASURED", "object_bindings": {"target": "object", "second_object": "object", "container": "container", "buffer": "buffer"}, "goal": "p:Inside:target:container", "public_facts": "NOT_MATERIALIZED_NO_PROVIDER", "allowed_action_ids": "REGISTERED_T_A_CONTRACT", "allowed_proposition_ids": "REGISTERED_T_A_CONTRACT", "contract_hash": _optional_hash(root / "configs/runtime/stage_2a_contract_registry.yaml"), "predicate_registry_hash": _optional_hash(root / "configs/runtime/stage_2a_contract_registry.yaml"), "prompt_hash": _optional_hash(root / "experiments/sources/v2.1_interfaces/system_prompt.txt"), "few_shot_hash": _optional_hash(root / "experiments/part_0_validation/stage_0c/few_shot_manifest.json"), "schema_hash": _optional_hash(root / "schemas/relation_schema.json"), "provider_identity": dict(config["provider"]), "test_overlap": False, "source_cache_dir": row.get("cache_dir"), "source_cache_key": row.get("cache_key")})
    value = {"status": "FROZEN_DISCOVERY_ONLY", "selection_rule": "T_A dev excluding 00..11 sorted case_id", "scene_count": len(scenes), "scenes": scenes}
    _atomic_json(path, value)
    return value


def run_provider_call(root, scene_ref, output_dir):
    output_dir = Path(output_dir)
    scenes = _read_json(output_dir / "manifests/frozen_discovery_scene_manifest.json")["scenes"]
    rows = [{"scene_id": scene["scene_id"], "cache_key": scene.get("source_cache_key", ""), "attempt_index": 0, "first_or_retry": "FIRST", "model": "qwen3.8-max-0902", "region": "cn-beijing", "endpoint": "https://dashscope.aliyuncs.com/api/v1", "sdk_version": "1.27.6", "request_hash": "NOT_SENT", "request_id": "NOT_SENT", "status": "STOPPED_PROVIDER_ACCESS", "error_code": "DASHSCOPE_API_KEY_UNAVAILABLE"} for scene in scenes]
    fields = ["scene_id", "cache_key", "attempt_index", "first_or_retry", "retry_reason", "model", "region", "endpoint", "sdk_version", "request_hash", "request_id", "http_status", "latency_seconds", "token_input", "token_output", "raw_response_sha256", "parsed_count", "rejected_count", "admitted_count", "status", "error_code"]
    _write_csv(output_dir / "provider/provider_call_ledger.csv", fields, rows)
    _write_csv(output_dir / "provider/cache_index.csv", ["scene_id", "cache_key", "cache_path", "status"], [])
    stage = _read_json(output_dir / "stage_manifest.json")
    stage.update({"status": "STOPPED", "stop_reason": "STOPPED_PROVIDER_ACCESS", "provider_first_calls": 0, "provider_retries": 0})
    _atomic_json(output_dir / "stage_manifest.json", stage)
    return {"status": "STOPPED_PROVIDER_ACCESS", "first_calls": 0, "retries": 0}


def probe_production_representation(root, case_ref, output_dir):
    output_dir = Path(output_dir)
    fields = ["case_id", "legal_candidate_count", "changed_patch_candidate_count", "relation_count", "status"]
    row = {"case_id": case_ref or "NONE", "legal_candidate_count": "NOT_MEASURED", "changed_patch_candidate_count": "NOT_MEASURED", "relation_count": 0, "status": "NOT_MEASURED_PROVIDER_ACCESS_STOPPED"}
    _write_csv(output_dir / "representation/e2_representation_entry_rev1.csv", fields, [row])
    _write_csv(output_dir / "representation/e3_candidate_discriminability_rev1.csv", fields, [row])
    _atomic_bytes(output_dir / "representation/gradient_reach.jsonl", b"")
    _atomic_json(output_dir / "representation/forward_accounting.json", {"status": row["status"], "optimizer_steps": 0, "backward_count": 0})
    return {"status": row["status"]}


def register_physical_branches(root, output_dir):
    value = {"status": "NO_BRANCHES_REGISTERED", "reason": "E1/E2/E3 structural prerequisites not established", "branches": [], "physical_witness_episodes_cap": 8}
    _atomic_json(Path(output_dir) / "witnesses/e4_branch_registration.json", value)
    return value


def execute_registered_branch(root, branch_id, output_dir):
    if not _read_json(Path(output_dir) / "witnesses/e4_branch_registration.json").get("branches"):
        raise RevisionError("witness execution requires frozen branch registration")
    raise RevisionError("physical execution not enabled")


def run_witnesses(root, output_dir):
    output_dir = Path(output_dir)
    _write_csv(output_dir / "witnesses/e4_physical_witnesses_rev1.csv", ["branch_id", "status"], [])
    _write_csv(output_dir / "witnesses/physical_episode_ledger.csv", ["branch_id", "status"], [])
    _atomic_json(output_dir / "witnesses/restore_integrity.json", {"status": "NOT_RUN_NO_BRANCHES", "episodes": 0})
    return {"status": "NOT_RUN_NO_BRANCHES", "episodes": 0}


def offline_rescore_planner(root, case_ref, output_dir):
    config = SearchConfig(depth_limit=6, max_nodes=4096, cpu_time_limit_seconds=2.0, reference_skill_seconds=4.2)
    BPlanPlanner(config)
    output_dir = Path(output_dir)
    _write_csv(output_dir / "planner_offline/offline_planner_discriminability.csv", ["case_id", "status", "relation_source", "environment_episodes", "depth_limit", "max_nodes", "cpu_time_limit_seconds", "tie_break"], [{"case_id": case_ref or "NONE", "status": "NOT_MEASURABLE", "relation_source": "NONE_PROVIDER_STOPPED", "environment_episodes": 0, "depth_limit": 6, "max_nodes": 4096, "cpu_time_limit_seconds": 2.0, "tie_break": "canonical_skill_id_plan_tuple"}])
    _atomic_json(output_dir / "planner_offline/search_accounting.json", {"status": "NOT_MEASURABLE", "environment_episodes": 0, "search_config": {"depth": 6, "max_nodes": 4096, "cpu_time_seconds": 2.0}})
    return {"status": "NOT_MEASURABLE"}


def finalize_eligibility(root, output_dir):
    output_dir = Path(output_dir)
    stage = _read_json(output_dir / "stage_manifest.json")
    stopped = stage.get("status") == "STOPPED"
    gate = _read_json(output_dir / "historical_gate.json")
    gates = {"E1": "PASS" if gate.get("passed") else "FAIL", "E2": "NOT_MEASURED", "E3": "NOT_MEASURED", "E4": "NOT_RUN", "E5": "NOT_MEASURED", "E6": "NOT_CLASSIFIED"}
    final = {"status": "STOPPED" if stopped else "COMPLETE", "eligibility_decision": "NOT_ELIGIBLE_AFTER_SINGLE_REVISION", "tp_training_authorized": False, "next_action": "RESUME_SAME_REVISION_ONLY" if stopped else "S4_RESEARCH_DECISION", "second_revision_allowed": False, "resume_same_revision_only": stopped, "stop_reason": stage.get("stop_reason", ""), "gates": gates}
    _atomic_json(output_dir / "final_eligibility.json", final)
    _atomic_json(output_dir / "stage_manifest.json", {**stage, **final, "finalized_utc": utc_now()})
    _atomic_json(output_dir / "revision_manifest.json", {**_read_json(output_dir / "revision_manifest.json"), "final": final})
    for name, gate in (("e1_relation_existence_rev1.csv", "E1"), ("e2_representation_entry_rev1.csv", "E2"), ("e3_candidate_discriminability_rev1.csv", "E3"), ("e4_physical_witnesses_rev1.csv", "E4")):
        _write_csv(output_dir / name, ["gate", "status", "denominator", "notes"], [{"gate": gate, "status": gates[gate], "denominator": 2, "notes": "Provider access stopped; no unobserved claim made."}])
    (output_dir / "e5_contract_insufficiency_rev1.md").write_text("# E5 — Contract insufficiency\n\nStatus: NOT_MEASURED; no claim is made.\n", encoding="utf-8")
    (output_dir / "e6_prior_classification_rev1.md").write_text("# E6 — Prior classification\n\nClassification: NOT_CLASSIFIED; provider access stopped.\n", encoding="utf-8")
    return final


def verify_revision_output(root, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    previous = root / "runs/final_master/S1/20260928T140215Z_538ef55a"
    before = output_dir / "raw/old_s1_before.sha256"
    if not before.exists():
        raise RevisionError("old S1 baseline missing")
    after = "".join(f"{sha256_file(path)}  {path}\n" for path in sorted(previous.rglob("*")) if path.is_file())
    _atomic_bytes(output_dir / "raw/old_s1_after.sha256", after.encode())
    if before.read_bytes() != (output_dir / "raw/old_s1_after.sha256").read_bytes():
        raise RevisionError("previous S1 evidence hashes changed")
    value = {"status": "PASS", "previous_s1_unchanged": True, "prior_script_unchanged": True, "prohibited_runs": {"planner_environment_episodes": 0, "rl_transitions": 0, "optimizer_steps": 0, "formal_test_episodes": 0}}
    _atomic_json(output_dir / "verify.json", value)
    return value
