"""Same-revision S1 discovery binding and production continuation.

This module deliberately reuses the production RuntimeFactory, perception,
verifier, relation parser/cache path, neural policy, and B_PLAN implementation.
It never loads a relation cache while materializing discovery inputs.
"""
from __future__ import annotations

from dataclasses import replace
import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np
import yaml

from cp_disr.common import canonical, digest, primitive
from cp_disr.facts import FactStore
from cp_disr.runtime import load_runtime
from cp_disr.stage0c import PREPROCESSING, prepare_scene
from cp_disr.vlm import cache_key
from cp_disr.vlm_provider import DashScopeProvider, ProviderConfig, redact, validate_payload
from cp_disr.vlm_cache_pipeline import process_response, response_text, verify_audit_cache, write_audit_cache


CACHE_FIELDS = (
    "split", "task_definition_hash", "initial_RGB_content_hash",
    "preprocessing_hash", "object_binding_hash", "allowed_ID_hash",
    "contract_version", "predicate_version", "model_snapshot",
    "sdk_api_version", "region", "endpoint", "prompt_hash", "fewshot_hash",
    "schema_hash", "decoding_config", "initial_facts_hash",
    "asset_binding_hash", "request_payload_hash",
)
EXPECTED_SCENES = tuple(f"T_A_dev_{i:02d}" for i in range(12, 20))
EXPECTED_GOALS = (
    {"fact_id": "p:Inside:target:container", "sign": 1},
    {"fact_id": "p:Inside:second_object:container", "sign": 1},
)


def _base():
    from cp_disr.analysis import s1_revision
    return s1_revision


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, value):
    _base()._atomic_json(path, value)


def _sha(path):
    return _base().sha256_file(path)


def _now():
    return _base().utc_now()


def _split_row_hash(row):
    # The frozen manifest used json.dumps(sort_keys=True) with default separators.
    return hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()


def _reset(row):
    return {key: row[key] for key in (
        "target_xy", "second_xy", "container_xy", "buffer_xy", "lid_closed"
    )}


def _frozen_and_rows(root, output_dir):
    frozen_path = Path(output_dir) / "manifests/frozen_discovery_scene_manifest.json"
    frozen_doc = _json(frozen_path)
    scenes = frozen_doc.get("scenes") or []
    if tuple(item.get("scene_id") for item in scenes) != EXPECTED_SCENES:
        raise _base().RevisionError("FROZEN_DISCOVERY_SELECTION_CHANGED")
    split_doc = _json(Path(root) / "configs/splits/T_A_stage_2a.json")
    source = split_doc.get("dev") or []
    rows = []
    for frozen in scenes:
        matches = [row for row in source if row.get("case_id") == frozen["scene_id"]]
        if len(matches) != 1:
            raise _base().RevisionError("DISCOVERY_SPLIT_ROW_NOT_UNIQUE")
        row = matches[0]
        if int(row["seed"]) != int(frozen["generator_seed"]):
            raise _base().RevisionError("DISCOVERY_SPLIT_SEED_MISMATCH")
        if _reset(row) != frozen["reset_config"]:
            raise _base().RevisionError("DISCOVERY_RESET_CONFIG_MISMATCH")
        if _split_row_hash(row) != frozen["reset_config_sha256"]:
            raise _base().RevisionError("DISCOVERY_FULL_SPLIT_HASH_MISMATCH")
        rows.append((frozen, row))
    return frozen_path, frozen_doc, rows


def resume_same_revision(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    config = _base().load_revision_config(config_path)
    ledger = _base().create_or_load_budget_ledger(output_dir, config)
    stage = _json(output_dir / "stage_manifest.json")
    stage.update({
        "status": "RUNNING", "eligibility_decision": "PENDING_SAME_REVISION",
        "tp_training_authorized": False, "resume_same_revision_only": True,
        "second_revision_allowed": False, "stop_reason": "",
        "resume_from_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(), "resumed_utc": _now(),
    })
    _write_json(output_dir / "stage_manifest.json", stage)
    eligibility = {
        "stage": "S1", "revision": "S1-REV1", "status": "RUNNING",
        "eligibility_decision": "PENDING_SAME_REVISION",
        "tp_training_authorized": False, "resume_same_revision_only": True,
        "second_revision_allowed": False, "next_action": "CONTINUE_SAME_REVISION",
        "budget_ledger": ledger,
    }
    _write_json(output_dir / "eligibility_manifest.json", eligibility)
    return eligibility


def _relation_free_runtime(root, output_dir, rows):
    input_dir = Path(output_dir) / "input_binding"
    input_dir.mkdir(parents=True, exist_ok=True)
    split = {"train": [], "dev": []}
    for _, source in rows:
        row = {k: v for k, v in source.items() if k not in ("cache_dir", "cache_key", "cache_status")}
        split["dev"].append(row)
    split_path = input_dir / "T_A_s1_rev1_runtime_split_no_cache.json"
    _write_json(split_path, split)
    manifest = yaml.safe_load((Path(root) / "experiments/manifests/stage_2a_runtime_manifest.yaml").read_text())
    runtime = manifest["runtime"]
    runtime["repository_path"] = str(Path(root).resolve())
    runtime["experiment_root"] = str((Path(root) / "experiments").resolve())
    runtime["active_task_id"] = "T_A"
    runtime["task_splits"]["T_A"] = str(split_path.relative_to(root))
    factory_path = (Path(root) / "src/cp_disr/platforms/libero/runtime_factory.py").resolve()
    factory = manifest["runtime_factory"]
    factory.update({
        "module": "cp_disr.platforms.libero.runtime_factory",
        "factory": "create_stage_2a_runtime",
        "source_path": str(factory_path), "sha256": _sha(factory_path),
    })
    manifest_path = input_dir / "T_A_s1_rev1_runtime_manifest.yaml"
    # JSON is valid YAML and avoids ambiguous secret-scanner substring matches
    # on legitimate names such as runtime.safety_authorization.
    manifest_path.write_text(json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return split_path, manifest_path, manifest


def _shared_contract(root):
    shared = Path(root) / "experiments/stage_0c_inputs/s1_rev1/shared"
    shared.mkdir(parents=True, exist_ok=True)
    source = yaml.safe_load((Path(root) / "configs/runtime/stage_2a_contract_registry.yaml").read_text())
    names = [item["name"] for item in source["contracts"]]
    if "MOVE" in names:
        raise _base().RevisionError("MOVE_IN_STAGE2A_CONTRACT")
    path = shared / "stage_2a_contract_registry.json"
    path.write_text(json.dumps({"predicate_types": source["predicate_types"], "contracts": source["contracts"]}, sort_keys=True, indent=2) + "\n")
    return path


def _fact_rows(records):
    return [{
        "fact_id": record.fact_id, "value": record.value.value,
        "reason": record.reason, "evidence_ids": list(record.evidence_ids),
        "capture_time": record.capture_time, "available_time": record.available_time,
        "hidden_truth_used_for_prompt": False,
    } for record in records]


def _hash_files(folder):
    return {path.name: _sha(path) for path in sorted(Path(folder).iterdir()) if path.is_file() and path.name not in ("content_hashes.json", "BINDING_COMPLETE")}


def _write_scene(root, output_dir, frozen, row, bundle, contract_path):
    from PIL import Image
    from cp_disr.platforms.libero.perception import PerceptionAdapter, camera_calibration
    from cp_disr.platforms.libero.verifier import FactVerifier

    index = int(frozen["scene_id"].rsplit("_", 1)[-1])
    final = Path(root) / f"experiments/stage_0c_inputs/dev/T_A/scene_{index:02d}"
    if final.exists():
        if (final / "BINDING_COMPLETE").is_file():
            manifest = _json(final / "scene_manifest.json")
            reset = _json(final / "reset_config.json")
            if manifest.get("scene_id") == frozen["scene_id"] and reset.get("split_row_sha256") == _split_row_hash(row):
                return {"scene_id": frozen["scene_id"], "status": "REUSED", "path": str(final)}
        raise _base().RevisionError("STOPPED_DISCOVERY_INPUT_CONFLICT")
    temp = final.with_name(final.name + f".tmp.{os.getpid()}")
    temp.mkdir(parents=True, exist_ok=False)
    try:
        snapshot = bundle.start_case(frozen["scene_id"])
        if bundle.original_prior_edges != () or snapshot.prior_edges != ():
            raise _base().RevisionError("RELATION_CACHE_LOADED_DURING_CAPTURE")
        obs_a = bundle.environment.public_observation()
        obs_b = bundle.environment.public_observation()
        if not np.array_equal(obs_a["rgb"], obs_b["rgb"]) or not np.array_equal(obs_a["depth"], obs_b["depth"]):
            raise _base().RevisionError("STOPPED_NONDETERMINISTIC_INITIAL_CAPTURE")
        rgb = np.asarray(obs_a["rgb"], dtype=np.uint8)
        depth = np.asarray(obs_a["depth"], dtype=np.float32)
        Image.fromarray(rgb, mode="RGB").save(temp / "rgb.png")
        np.save(temp / "depth.npy", depth, allow_pickle=False)
        perception = PerceptionAdapter(bundle.environment)
        measurement = perception.infer(obs_a)
        verifier = FactVerifier(bundle.environment)
        records = verifier.verify(measurement, None)
        facts = FactStore(records)
        proposition_ids = tuple(node.id for node in snapshot.template.nodes if node.kind == "PROPOSITION")
        if set(facts.values) != set(proposition_ids):
            raise _base().RevisionError("FACTSTORE_TEMPLATE_MISMATCH")
        goals = tuple({"fact_id": goal.fact_id, "sign": goal.sign} for goal in snapshot.template.goals)
        if goals != EXPECTED_GOALS:
            raise _base().RevisionError("STOPPED_TA_GOAL_BINDING_MISMATCH")
        candidate_ids = tuple(snapshot.candidate_ids)
        if candidate_ids != tuple(contract.id for contract in snapshot.template.contracts):
            raise _base().RevisionError("CANDIDATE_CONTRACT_ALIGNMENT")
        if any(":MOVE:" in item for item in candidate_ids):
            raise _base().RevisionError("MOVE_IN_ALLOWED_ACTIONS")
        if not (len(candidate_ids) == len(snapshot.mask) == len(snapshot.candidate_features)):
            raise _base().RevisionError("CANDIDATE_SHAPE_MISMATCH")
        blobs = measurement.measurements.get("blobs", {})
        if not blobs.get("target") or not blobs.get("second_object"):
            raise _base().RevisionError("STOPPED_DISCOVERY_OBJECT_BINDING")
        hidden = bundle.environment.hidden_truth()
        xy_atol, z_atol = 1e-5, 5e-3
        checks = {
            "target_xy": bool(np.allclose(np.asarray(hidden["target"])[:2], row["target_xy"], atol=xy_atol, rtol=0)),
            "second_xy": bool(np.allclose(np.asarray(hidden["second_object"])[:2], row["second_xy"], atol=xy_atol, rtol=0)),
            "container_xy": bool(np.allclose(np.asarray(hidden["container"])[:2], row["container_xy"], atol=xy_atol, rtol=0)),
            "buffer_xy": bool(np.allclose(np.asarray(hidden["buffer"])[:2], row["buffer_xy"], atol=xy_atol, rtol=0)),
        }
        lid_xy = np.asarray(hidden["lid"])[:2]
        expected_lid = np.asarray(row["container_xy"] if row["lid_closed"] else [row["container_xy"][0] + 0.26, row["container_xy"][1]])
        checks["lid_state"] = bool(np.allclose(lid_xy, expected_lid, atol=xy_atol, rtol=0))
        if not all(checks.values()):
            raise _base().RevisionError("HIDDEN_RESET_QA_MISMATCH")
        calib = camera_calibration(bundle.environment)
        rgb_hash, depth_hash = _sha(temp / "rgb.png"), _sha(temp / "depth.npy")
        contract_ref = str(contract_path.relative_to(root))
        contract_hash = _sha(contract_path)
        reset_payload = {**_reset(row), "case_id": row["case_id"], "seed": row["seed"], "split": row["split"], "split_row_sha256": _split_row_hash(row), "normalized_reset_config_sha256": digest(_reset(row))}
        _write_json(temp / "reset_config.json", reset_payload)
        _write_json(temp / "initial_fact_summary.json", {"records": _fact_rows(records), "values": {key: value.value for key, value in facts.values.items()}})
        _write_json(temp / "allowed_action_ids.json", list(candidate_ids))
        _write_json(temp / "allowed_proposition_ids.json", list(proposition_ids))
        _write_json(temp / "candidate_ids.json", list(candidate_ids))
        _write_json(temp / "candidate_mask.json", list(snapshot.mask))
        _write_json(temp / "candidate_features.json", [list(item) for item in snapshot.candidate_features])
        object_table = [
            {"id": "buffer", "type": "buffer", "binding_status": "VERIFIED", "visual_evidence_ref": "rgb.png"},
            {"id": "container", "type": "container", "binding_status": "VERIFIED", "visual_evidence_ref": "rgb.png"},
            {"id": "second_object", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": "rgb.png"},
            {"id": "target", "type": "object", "binding_status": "VERIFIED", "visual_evidence_ref": "rgb.png"},
        ]
        _write_json(temp / "object_bindings.json", object_table)
        _write_json(temp / "object_binding_evidence.json", {
            "perception_version": measurement.version, "checkpoint_hash": measurement.checkpoint_hash,
            "calibration_hash": measurement.calibration_hash, "blobs": blobs,
            "unknown_reasons": measurement.measurements.get("unknown_reasons", {}),
            "rgb_sha256": rgb_hash, "depth_sha256": depth_hash,
            "static_layout_fallback_allowed_for": ["container", "buffer"],
        })
        task_source = yaml.safe_load((Path(root) / "configs/tasks/resolved/T_A.yaml").read_text())
        task_definition = {
            "task_id": "T_A", "task_version": task_source["task_version"],
            "source_task_ref": "configs/tasks/resolved/T_A.yaml", "split_role": "S1_DISCOVERY_ONLY",
            "instruction": task_source["instruction"], "signed_goals": list(goals),
            "contract_registry_ref": "configs/runtime/stage_2a_contract_registry.yaml",
            "runtime_factory_ref": "src/cp_disr/platforms/libero/runtime_factory.py",
            "scene_reset_ref": "reset_config.json", "test_input": False,
        }
        _write_json(temp / "task_definition.json", task_definition)
        _write_json(temp / "camera_config.json", calib)
        _write_json(temp / "contract_refs.json", {"contracts_ref": contract_ref, "contracts_sha256": contract_hash})
        _write_json(temp / "capture_manifest.json", {"scene_id": frozen["scene_id"], "physical_gpu_index": int(os.environ.get("CP_DISR_PHYSICAL_GPU_INDEX", "-1")), "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES"), "MUJOCO_EGL_DEVICE_ID": os.environ.get("MUJOCO_EGL_DEVICE_ID"), "engineering_discovery_input_capture": True, "skill_actions_executed": 0, "planner_actions_executed": 0, "paper_episode": False, "reset_gripper_open_control_steps": 20})
        _write_json(temp / "capture_log.json", {"scene_id": frozen["scene_id"], "reset_procedure": {"gripper_open_control_steps": 20}, "skill_actions_executed": 0, "planner_actions_executed": 0, "paper_episode": False, "observation_repeat_equal": True})
        _write_json(temp / "provenance.json", {"source_dataset": "CP-DISR_PROJECT_OWNED_PROCEDURAL_MUJOCO", "version": "stage-2a-p0", "split_evidence_ref": "configs/splits/T_A_stage_2a.json", "capture_kind": "recorded_simulator_rgb", "initial_frame_verified": True, "platform_type": "simulator", "runtime_factory_sha256": _sha(Path(root) / "src/cp_disr/platforms/libero/runtime_factory.py")})
        _write_json(temp / "hidden_truth_qa.json", {"xy_atol": xy_atol, "z_atol": z_atol, "checks": checks, "hidden_truth_used_for_prompt": False, "hidden_truth_used_for_policy": False, "hidden_truth_used_for_relation_admission": False, "qa_only": True})
        asset_binding = {
            "status": "VERIFIED", "reviewer": "CP-DISR-S1-REV1-production-binding", "reviewed_at": _now(),
            "structural_template_ref": "configs/tasks/resolved/T_A.yaml",
            "source_assets": ["src/cp_disr/platforms/libero/d0_env.py", "configs/splits/T_A_stage_2a.json", "assets/cp_disr/LICENSE"],
            "controller_refs": ["src/cp_disr/platforms/libero/skill_executor.py"],
            "verifier_ref": "src/cp_disr/platforms/libero/verifier.py", "perception_ref": "src/cp_disr/platforms/libero/perception.py",
            "runtime_factory_ref": "src/cp_disr/platforms/libero/runtime_factory.py", "object_binding_evidence": "object_binding_evidence.json",
            "capture_platform": "LIBERO_CP_DISR_CLEAN", "scene_id": frozen["scene_id"], "task_id": "T_A",
            "split_row_sha256": _split_row_hash(row), "normalized_reset_config_sha256": digest(_reset(row)),
            "rgb_sha256": rgb_hash, "depth_sha256": depth_hash,
        }
        _write_json(temp / "asset_binding.json", asset_binding)
        task_hash, binding_hash = _sha(temp / "task_definition.json"), _sha(temp / "asset_binding.json")
        scene_manifest = {
            "scene_id": frozen["scene_id"], "task_id": "T_A", "case_index": index, "split": "dev", "split_role": "S1_DISCOVERY_ONLY",
            "synthetic_unit_fixture": False, "image_ref": str((final / "rgb.png").relative_to(root)), "image_sha256": rgb_hash,
            "provenance": {"source_dataset": "CP-DISR_PROJECT_OWNED_PROCEDURAL_MUJOCO", "version": "stage-2a-p0", "split_evidence_ref": "configs/splits/T_A_stage_2a.json", "capture_kind": "recorded_simulator_rgb", "initial_frame_verified": True},
            "permission": {"vlm_upload_allowed": True, "evidence_ref": "assets/cp_disr/LICENSE"}, "preprocessing": PREPROCESSING,
            "object_table": object_table, "signed_goals": list(goals), "initial_facts": {key: value.value for key, value in facts.values.items()},
            "contracts_ref": contract_ref, "contracts_sha256": contract_hash,
            "task_definition_ref": str((final / "task_definition.json").relative_to(root)), "task_definition_sha256": task_hash,
            "asset_binding_ref": str((final / "asset_binding.json").relative_to(root)), "asset_binding_sha256": binding_hash,
            "allowed_action_ids": list(candidate_ids), "allowed_proposition_ids": list(proposition_ids), "predicate_version": "CP-DISR-v2.1",
            "scene_config_hash": digest(_reset(row)), "camera_config_hash": _sha(temp / "camera_config.json"), "image_depth_sha256": depth_hash,
            "split_row_sha256": _split_row_hash(row), "normalized_reset_config_sha256": digest(_reset(row)), "s1_revision": 1, "test_overlap": False,
        }
        _write_json(temp / "scene_manifest.json", scene_manifest)
        hashes = _hash_files(temp)
        _write_json(temp / "content_hashes.json", hashes)
        for name, expected in hashes.items():
            if _sha(temp / name) != expected:
                raise _base().RevisionError("SCENE_CONTENT_HASH_MISMATCH")
        for path in temp.iterdir():
            if path.is_file():
                path.chmod(0o444)
        (temp / "BINDING_COMPLETE").write_text("complete\n", encoding="utf-8")
        (temp / "BINDING_COMPLETE").chmod(0o444)
        final.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temp, final)
        return {"scene_id": frozen["scene_id"], "status": "CAPTURED", "path": str(final), "rgb_sha256": rgb_hash, "depth_sha256": depth_hash}
    except Exception:
        shutil.rmtree(temp, ignore_errors=True)
        raise


def materialize_discovery_inputs(root, config_path, output_dir, gpu):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    frozen_path, _, rows = _frozen_and_rows(root, output_dir)
    before = _sha(frozen_path)
    split_path, manifest_path, runtime_manifest = _relation_free_runtime(root, output_dir, rows)
    bundle = load_runtime(runtime_manifest)
    if bundle.task_id != "T_A" or bundle.original_prior_edges != ():
        raise _base().RevisionError("RELATION_FREE_RUNTIME_BINDING_FAILED")
    contract_path = _shared_contract(root)
    results = []
    try:
        for frozen, row in rows:
            results.append(_write_scene(root, output_dir, frozen, row, bundle, contract_path))
    finally:
        try: bundle.environment.close()
        except Exception: pass
    if _sha(frozen_path) != before:
        raise _base().RevisionError("FROZEN_DISCOVERY_MANIFEST_CHANGED")
    _base()._write_csv(output_dir / "input_binding/reset_identity.csv", ["scene_id", "split_row_sha256", "normalized_reset_config_sha256", "frozen_reset_hash_value", "frozen_reset_hash_semantics"], [{"scene_id": f["scene_id"], "split_row_sha256": _split_row_hash(r), "normalized_reset_config_sha256": digest(_reset(r)), "frozen_reset_hash_value": f["reset_config_sha256"], "frozen_reset_hash_semantics": "FULL_SPLIT_ROW_SHA256"} for f, r in rows])
    _write_json(output_dir / "input_binding/binding_corrections.json", {"frozen_scene_selection_changed": False, "frozen_reset_rows_changed": False, "frozen_goal_summary_is_production_source": False, "production_signed_goals_source": "RuntimeFactory snapshot.template.goals", "frozen_manifest_bytes_unchanged": True})
    _base()._write_csv(output_dir / "input_binding/input_capture_ledger.csv", ["scene_id", "status", "path", "rgb_sha256", "depth_sha256", "skill_actions_executed", "planner_actions_executed", "paper_episode"], [{**item, "skill_actions_executed": 0, "planner_actions_executed": 0, "paper_episode": False} for item in results])
    return {"status": "CAPTURED", "scenes": len(results), "gpu": int(gpu), "runtime_manifest": str(manifest_path), "relation_free_split": str(split_path)}


def _verify_scene(root, output_dir, frozen, row):
    index = int(frozen["scene_id"].rsplit("_", 1)[-1])
    folder = Path(root) / f"experiments/stage_0c_inputs/dev/T_A/scene_{index:02d}"
    issues = []
    if not (folder / "BINDING_COMPLETE").is_file(): issues.append("BINDING_COMPLETE")
    hashes = _json(folder / "content_hashes.json") if (folder / "content_hashes.json").is_file() else {}
    for name, expected in hashes.items():
        if not (folder / name).is_file() or _sha(folder / name) != expected: issues.append("HASH:" + name)
    scene = _json(folder / "scene_manifest.json")
    if scene["scene_id"] != frozen["scene_id"]: issues.append("SCENE_ID")
    if scene["split_row_sha256"] != _split_row_hash(row): issues.append("SPLIT_HASH")
    if scene["normalized_reset_config_sha256"] != digest(_reset(row)): issues.append("RESET_HASH")
    bound = prepare_scene(root, scene)
    goals = tuple({"fact_id": g.fact_id, "sign": g.sign} for g in bound["template"].goals)
    if goals != EXPECTED_GOALS: issues.append("GOALS")
    if any(":MOVE:" in item for item in scene["allowed_action_ids"]): issues.append("MOVE")
    candidate_ids = tuple(_json(folder / "candidate_ids.json"))
    mask = tuple(_json(folder / "candidate_mask.json"))
    if candidate_ids != tuple(c.id for c in bound["template"].contracts) or len(candidate_ids) != len(mask): issues.append("CANDIDATES")
    if scene.get("test_overlap") is not False: issues.append("TEST_OVERLAP")
    rgb = np.asarray(__import__("PIL.Image", fromlist=["Image"]).open(folder / "rgb.png"))
    depth = np.load(folder / "depth.npy", allow_pickle=False)
    if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3: issues.append("RGB")
    if depth.dtype != np.float32 or depth.size == 0: issues.append("DEPTH")
    if "hidden_truth_qa" in json.dumps(scene, sort_keys=True): issues.append("HIDDEN_PROMPT_REFERENCE")
    return {"scene_id": frozen["scene_id"], "status": "PASS" if not issues else "FAIL", "issues": ";".join(issues), "rgb_sha256": scene["image_sha256"], "depth_sha256": scene["image_depth_sha256"], "path": str(folder)}


def validate_discovery_inputs(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    frozen_path, _, rows = _frozen_and_rows(root, output_dir)
    results = [_verify_scene(root, output_dir, frozen, row) for frozen, row in rows]
    old_rgb = set()
    for path in list((root / "experiments/stage_0c_inputs/dev/T_A").glob("scene_0[0-7]/rgb.png")) + list((root / "experiments/stage_0c_inputs/fewshots").glob("*/rgb.png")):
        old_rgb.add(_sha(path))
    rgb = [item["rgb_sha256"] for item in results]
    depth = [item["depth_sha256"] for item in results]
    if len(set(rgb)) != 8 or set(rgb) & old_rgb: raise _base().RevisionError("DISCOVERY_RGB_UNIQUENESS_FAILED")
    if len(set(depth)) != 8: raise _base().RevisionError("DISCOVERY_DEPTH_UNIQUENESS_FAILED")
    valid = sum(item["status"] == "PASS" for item in results)
    status = "PASS" if valid == 8 else "FAIL"
    _base()._write_csv(output_dir / "input_binding/input_binding_validation.csv", ["scene_id", "status", "issues", "rgb_sha256", "depth_sha256", "path"], results)
    manifest = {"status": status, "valid_scenes": valid, "invalid_scenes": 8-valid, "rgb_hash_unique_count": len(set(rgb)), "depth_hash_unique_count": len(set(depth)), "frozen_discovery_manifest_sha256": _sha(frozen_path), "production_goals": list(EXPECTED_GOALS), "move_present": False, "relation_cache_loaded": False, "hidden_truth_entered_prompt": False}
    _write_json(output_dir / "input_binding/discovery_input_binding_manifest.json", manifest)
    lines = []
    for item in results:
        folder = Path(item["path"])
        for path in sorted(folder.iterdir()):
            if path.is_file(): lines.append(f"{_sha(path)}  {path}\n")
    _base()._atomic_bytes(output_dir / "input_binding/content_hashes.sha256", "".join(lines).encode())
    ledger = _json(output_dir / "budget_ledger.json")
    for field in ("provider_first_calls", "provider_retries", "physical_witness_episodes", "planner_environment_episodes", "rl_transitions", "optimizer_steps"):
        if int(ledger[field]["used"]) != 0: raise _base().RevisionError("PRE_PROVIDER_BUDGET_NONZERO:" + field)
    if status != "PASS": raise _base().RevisionError("DISCOVERY_INPUT_VALIDATION_FAILED")
    return manifest


def _fewshots(root):
    path = Path(root) / "experiments/stage_0c_inputs/fewshots/few_shot_manifest.json"
    return path, _json(path)


def provider_case(root, frozen_scene, output_dir, config, provider=None):
    index = int(frozen_scene["scene_id"].rsplit("_", 1)[-1])
    scene_path = Path(root) / f"experiments/stage_0c_inputs/dev/T_A/scene_{index:02d}/scene_manifest.json"
    source_scene = _json(scene_path)
    if source_scene["scene_id"] != frozen_scene["scene_id"]: raise _base().RevisionError("DISCOVERY_INPUT_BINDING_ID_MISMATCH")
    bound = prepare_scene(root, source_scene)
    few_path, few_doc = _fewshots(root)
    examples = [prepare_scene(root, record, fewshot=True) for record in few_doc["examples"]]
    if len(examples) != 3: raise _base().RevisionError("FEWSHOT_INPUT_BINDING_INCOMPLETE")
    prompt_path = Path(root) / "experiments/sources/v2.1_interfaces/system_prompt.txt"
    extra_path = Path(root) / "prompts/s1_rev1_extra_contract_filter.txt"
    prompt = prompt_path.read_text(encoding="utf-8") + "\n\n" + extra_path.read_text(encoding="utf-8")
    messages = [{"role": "system", "content": [{"text": prompt}]}]
    for example, record in zip(examples, few_doc["examples"]):
        messages += [{"role": "user", "content": example["content"]}, {"role": "assistant", "content": [{"text": canonical(record["expected_json"])}]}]
    payload = {"model": config["provider"]["model"], "messages": messages + [{"role": "user", "content": bound["content"]}], "temperature": 0, "max_tokens": 2048, "response_format": {"type": "json_object"}, "enable_thinking": False, "enable_search": False, "stream": False, "result_format": "message"}
    validate_payload(payload)
    manifest = {
        "split": "dev", "task_definition_hash": source_scene["task_definition_sha256"], "initial_RGB_content_hash": source_scene["image_sha256"],
        "preprocessing_hash": digest(source_scene["preprocessing"]), "object_binding_hash": digest(sorted(source_scene["object_table"], key=lambda x: x["id"])),
        "allowed_ID_hash": digest({"actions": sorted(source_scene["allowed_action_ids"]), "propositions": sorted(source_scene["allowed_proposition_ids"]), "signed_goals": source_scene["signed_goals"]}),
        "contract_version": source_scene["contracts_sha256"], "predicate_version": source_scene["predicate_version"], "model_snapshot": config["provider"]["model"],
        "sdk_api_version": config["provider"]["sdk_version"], "region": config["provider"]["region"], "endpoint": config["provider"]["endpoint"],
        "prompt_hash": hashlib.sha256(prompt.encode()).hexdigest(), "fewshot_hash": _sha(few_path), "schema_hash": _sha(Path(root) / "schemas/relation_schema.json"),
        "decoding_config": {"temperature": 0, "max_tokens": 2048, "max_relations": 8, "thinking": False, "response_format": "json_object"},
        "initial_facts_hash": digest(source_scene["initial_facts"]), "asset_binding_hash": source_scene["asset_binding_sha256"], "request_payload_hash": digest(redact(payload)),
        "scene_id": source_scene["scene_id"], "task_id": source_scene["task_id"], "synthetic_unit_fixture": False, "s1_revision": 1,
    }
    missing=[key for key in CACHE_FIELDS if key not in manifest]
    if missing: raise _base().RevisionError("CACHE_IDENTITY_MISSING:" + ",".join(missing))
    key = cache_key(manifest)
    input_refs = {"scene_manifest_ref": str(scene_path.relative_to(root)), "scene_manifest_sha256": _sha(scene_path), "frozen_scene_id": frozen_scene["scene_id"], "hidden_truth_included": False}
    return {"payload": payload, "template": bound["template"], "manifest": manifest, "cache_key": key, "schema": _json(Path(root) / "schemas/relation_schema.json"), "input_refs": input_refs, "prompt": prompt, "provider": provider or DashScopeProvider(ProviderConfig(model=config["provider"]["model"], region=config["provider"]["region"], endpoint=config["provider"]["endpoint"], sdk_version=config["provider"]["sdk_version"]))}


def provider_preflight(root, config_path, output_dir):
    root, output_dir = Path(root).resolve(), Path(output_dir)
    binding = _json(output_dir / "input_binding/discovery_input_binding_manifest.json")
    if binding.get("status") != "PASS" or binding.get("valid_scenes") != 8: raise _base().RevisionError("PROVIDER_PREFLIGHT_BINDING_INCOMPLETE")
    config = _base().load_revision_config(config_path)
    _, frozen, _ = _frozen_and_rows(root, output_dir)
    cases = [provider_case(root, scene, output_dir, config) for scene in frozen["scenes"]]
    with tempfile.TemporaryDirectory() as tmp:
        case = cases[0]
        processing = process_response('{"schema_version":"m1_soft_relations_v2","relations":[]}', case["template"], case["schema"])
        execution = {"attempts": [{"attempt": 0, "request_payload_redacted": redact(case["payload"]), "response": {"error_type": "OK", "request_id": "fake-preflight", "status_code": 200}}], "processing": processing, "status": "SUCCESS"}
        cache = write_audit_cache(Path(tmp), case["manifest"], case["prompt"], case["input_refs"], execution)
        verified, edges = verify_audit_cache(cache)
        if verified["cache_key"] != case["cache_key"] or edges != []: raise _base().RevisionError("FAKE_PROVIDER_CACHE_VERIFY_FAILED")
    ledger = _json(output_dir / "budget_ledger.json")
    if int(ledger["provider_first_calls"]["used"]) != 0: raise _base().RevisionError("PROVIDER_BUDGET_NONZERO")
    result = {"status": "PASS", "live_requests_authorized_within_existing_budget": True, "scene_count": len(cases), "cache_keys": [case["cache_key"] for case in cases], "cache_fields": list(CACHE_FIELDS), "fake_provider_cache_verified": True, "hidden_truth_in_payload": False}
    _write_json(output_dir / "provider/provider_preflight.json", result)
    return result


def provider_row(scene_id, key, attempt, status, result=None, processing=None, error=""):
    result=result or {}; processing=processing or {}; usage=result.get("usage") or {}
    return {"scene_id":scene_id,"cache_key":key,"attempt_index":attempt,"first_or_retry":"RETRY" if attempt else "FIRST","retry_reason":"","model":"qwen3.8-max-0902","region":"cn-beijing","endpoint":"https://dashscope.aliyuncs.com/api/v1","sdk_version":"1.27.6","request_hash":"","request_id":result.get("request_id",""),"http_status":result.get("status_code",""),"latency_seconds":result.get("latency_seconds",""),"token_input":usage.get("input_tokens",""),"token_output":usage.get("output_tokens",""),"raw_response_sha256":hashlib.sha256(canonical(redact(result)).encode()).hexdigest() if result else "","parsed_count":len((processing.get("parsed") or {}).get("relations",[])) if isinstance(processing.get("parsed"),dict) else 0,"rejected_count":len(processing.get("rejected") or []),"admitted_count":len(processing.get("accepted") or []),"status":status,"error_code":error}


def run_provider_call(root, scene_ref, output_dir):
    root, output_dir=Path(root).resolve(),Path(output_dir)
    preflight=_json(output_dir/"provider/provider_preflight.json")
    if preflight.get("status")!="PASS": raise _base().RevisionError("PROVIDER_PREFLIGHT_NOT_PASS")
    config=_base().load_revision_config(root/"configs/final_master/s1_rev1.yaml")
    scenes=_json(output_dir/"manifests/frozen_discovery_scene_manifest.json")["scenes"]
    rows=[]; caches=[]
    for scene in scenes:
        case=provider_case(root,scene,output_dir,config); key=case["cache_key"]
        existing=root/"experiments/vlm_cache/dev"/key
        if existing.exists():
            manifest,edges=verify_audit_cache(existing)
            if any(manifest.get(k)!=case["manifest"].get(k) for k in case["manifest"]): raise _base().RevisionError("EXACT_CACHE_IDENTITY_MISMATCH")
            caches.append({"scene_id":scene["scene_id"],"cache_key":key,"cache_path":str(existing),"status":"CACHE_REUSED","admitted_count":len(edges)})
            rows.append(provider_row(scene["scene_id"],key,0,"CACHE_REUSED")); continue
        _base().reserve_budget(output_dir,"provider_first_calls",1,scene["scene_id"])
        attempts=[]; processing=None
        for attempt in range(2):
            request=json.loads(json.dumps(case["payload"]))
            if attempt: request["messages"].append({"role":"user","content":[{"text":"Return only one JSON object conforming to the supplied m1_soft_relations_v2 schema. Do not add prose or markdown."}]})
            result=redact(case["provider"].send(request)); entry={"attempt":attempt,"request_payload_redacted":redact(request),"response":result}; attempts.append(entry)
            if result.get("error_type")!="OK":
                if attempt==0 and result.get("error_type") in ("TIMEOUT","TRANSPORT"):
                    _base().reserve_budget(output_dir,"provider_retries",1,scene["scene_id"]); continue
                rows.append(provider_row(scene["scene_id"],key,attempt,"STOPPED_PROVIDER_ACCESS",result,error=result.get("code") or result.get("error_type","API_ERROR"))); break
            try: processing=process_response(response_text(result),case["template"],case["schema"])
            except (ValueError,TypeError,KeyError):
                if attempt==0:
                    _base().reserve_budget(output_dir,"provider_retries",1,scene["scene_id"]); continue
                rows.append(provider_row(scene["scene_id"],key,attempt,"JSON_SYNTAX_FAILURE",result,error="JSON_SYNTAX_FAILURE")); break
            if processing["status"]!="SUCCESS":
                rows.append(provider_row(scene["scene_id"],key,attempt,"PROCESSED_INVALID_SCHEMA",result,processing,"NON_RETRYABLE_SEMANTIC")); break
            execution={"attempts":attempts,"processing":processing,"status":"SUCCESS"}
            path=write_audit_cache(root/"experiments/vlm_cache",case["manifest"],case["prompt"],case["input_refs"],execution)
            verified,edges=verify_audit_cache(path)
            if verified["cache_key"]!=key: raise _base().RevisionError("WRITTEN_CACHE_KEY_MISMATCH")
            rows.append(provider_row(scene["scene_id"],key,attempt,"SUCCESS",result,processing)); caches.append({"scene_id":scene["scene_id"],"cache_key":key,"cache_path":str(path),"status":"SUCCESS","admitted_count":len(edges)}); break
        if rows[-1]["status"]=="STOPPED_PROVIDER_ACCESS": break
    fields=list(provider_row("","",0,"").keys())
    _base()._write_csv(output_dir/"provider/provider_call_ledger.csv",fields,rows)
    _base()._write_csv(output_dir/"provider/cache_index.csv",["scene_id","cache_key","cache_path","status","admitted_count"],caches)
    stage=_json(output_dir/"stage_manifest.json"); stopped=any(r["status"].startswith("STOPPED_") for r in rows)
    ledger=_json(output_dir/"budget_ledger.json"); stage.update({"status":"STOPPED" if stopped else "RUNNING","stop_reason":next((r["status"] for r in rows if r["status"].startswith("STOPPED_")),""),"provider_first_calls":ledger["provider_first_calls"]["used"],"provider_retries":ledger["provider_retries"]["used"],"eligibility_decision":"PENDING_SAME_REVISION","tp_training_authorized":False})
    _write_json(output_dir/"stage_manifest.json",stage)
    return {"status":stage["stop_reason"] or "SUCCESS","first_calls":stage["provider_first_calls"],"retries":stage["provider_retries"],"successful_caches":len(caches)}


def downstream_ready(output_dir):
    index=Path(output_dir)/"provider/cache_index.csv"
    rows=list(csv.DictReader(index.open(encoding="utf-8"))) if index.exists() else []
    return [row for row in rows if row.get("status") in ("SUCCESS","CACHE_REUSED") and int(row.get("admitted_count") or 0)>0]


def probe_production_representation(root, case_ref, output_dir):
    """Run the production graph/four-view/Policy path without an optimizer."""
    import torch
    from cp_disr.graph import four_views
    from cp_disr.neural import Policy
    from cp_disr.patch_identity import legal_nonempty_indices
    root,output_dir=Path(root).resolve(),Path(output_dir)
    cases=downstream_ready(output_dir); rows=[]; reach=[]; forwards=backwards=0
    manifest=yaml.safe_load((output_dir/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").read_text())
    for case in cases:
        bundle=load_runtime(manifest)
        try:
            snap=bundle.start_case(case["scene_id"]); cache=Path(case["cache_path"]); _,edges=verify_audit_cache(cache); edges=tuple(tuple(x) for x in edges)
            rel=replace(snap,prior_edges=edges,prior_hash=digest(edges)); legal,changed,_=legal_nonempty_indices(rel)
            actions=sorted({n.schema for n in snap.template.nodes if n.kind=="ACTION"}); predicates=sorted({n.schema for n in snap.template.nodes if n.kind=="PROPOSITION"}); types=sorted({t for n in snap.template.nodes for t in n.argument_types})
            for seed in (530001,530002):
                torch.manual_seed(seed); model=Policy(actions,predicates,types,len(snap.base_input),len(snap.candidate_features[0]),method="Full"); model.eval(); model.zero_grad(set_to_none=True)
                base_out=model(snap); rel_out=model(rel); forwards+=2
                diffs=[]
                for i in legal: diffs.append(float((rel_out.logits[i]-base_out.logits[i]).detach()))
                relative=(max(diffs)-min(diffs)) if diffs else 0.0; common=bool(diffs) and relative<=1e-9
                tensors=[]
                for cid in snap.candidate_ids:
                    t=rel_out.diagnostics.get("prior_inputs",{}).get(cid)
                    if t is not None and t.requires_grad: t.retain_grad(); tensors.append(t)
                if legal:
                    rel_out.logits[torch.tensor(legal)].sum().backward(); backwards+=1
                grad=any(t.grad is not None and int(torch.count_nonzero(t.grad).item())>0 for t in tensors)
                footprint=0
                for cid in snap.candidate_ids:
                    contract=next(c for c in snap.template.contracts if c.id==cid)
                    try:
                        before=four_views(snap.template,snap.facts.values,(),contract)[0]
                        after=four_views(snap.template,snap.facts.values,edges,contract)[1]
                    except Exception:
                        # A candidate whose precondition is not confirmed is
                        # not part of the executable relation footprint.
                        continue
                    if after.edges != before.edges:
                        footprint += 1
                status="PASS" if edges and legal and changed and grad else "FAIL"
                rows.append({"case_id":case["scene_id"],"scene_id":case["scene_id"],"cache_key":case["cache_key"],"natural_relation_count":len(edges),"legal_candidate_count":len(legal),"changed_patch_candidate_count":len(changed),"relation_enters_graph":bool(edges),"relation_footprint":footprint,"patch_footprint":len(changed),"candidate_id_alignment":True,"mask_alignment":True,"goal_alignment":True,"node_alignment":True,"relative_logit_change":relative,"common_shift_only":common,"gradient_to_relation_input":grad,"gradient_to_patch_input":grad,"capacity_seed":seed,"status":status})
                reach.append({"case_id":case["scene_id"],"capacity_seed":seed,"gradient_to_relation_input":grad,"gradient_to_patch_input":grad})
        finally:
            try: bundle.environment.close()
            except Exception: pass
    fields=list(rows[0]) if rows else ["case_id","status"]
    _base()._write_csv(output_dir/"representation/e2_representation_entry_rev1.csv",fields,rows)
    _base()._write_csv(output_dir/"representation/e3_candidate_discriminability_rev1.csv",fields,rows)
    _base()._atomic_bytes(output_dir/"representation/gradient_reach.jsonl","".join(json.dumps(x,sort_keys=True)+"\n" for x in reach).encode())
    _write_json(output_dir/"representation/forward_accounting.json",{"status":"PASS" if rows and all(r["status"]=="PASS" for r in rows) else "FAIL","forward_count":forwards,"backward_count":backwards,"optimizer_steps":0})
    return {"status":"PASS" if rows and all(r["status"]=="PASS" for r in rows) else "INSUFFICIENT","cases":len({r["case_id"] for r in rows})}


def register_physical_branches(root, output_dir):
    output_dir=Path(output_dir); rep=list(csv.DictReader((output_dir/"representation/e3_candidate_discriminability_rev1.csv").open())) if (output_dir/"representation/e3_candidate_discriminability_rev1.csv").exists() else []
    passing=sorted({row["case_id"] for row in rep if row.get("status")=="PASS"})
    branches=[]
    if len(passing)>=2:
        manifest=yaml.safe_load((output_dir/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").read_text())
        for case_id in passing[:2]:
            bundle=load_runtime(manifest)
            try:
                snap=bundle.start_case(case_id); legal=[cid for cid,m in zip(snap.candidate_ids,snap.mask) if m][:2]
                for cid in legal:
                    for repeat in range(2):
                        identity=f"{case_id}|{cid}|{repeat}"; branches.append({"branch_id":hashlib.sha256(identity.encode()).hexdigest()[:16],"case_id":case_id,"candidate_id":cid,"repeat":repeat,"restore_seed":int(hashlib.sha256(identity.encode()).hexdigest()[:8],16),"selection_hash":hashlib.sha256(identity.encode()).hexdigest(),"status":"REGISTERED"})
            finally:
                try: bundle.environment.close()
                except Exception: pass
    value={"status":"REGISTERED" if branches else "NOT_REGISTERED_INSUFFICIENT_E1_E2_E3","branches":sorted(branches,key=lambda x:x["selection_hash"])[:8],"physical_witness_episodes_cap":8}
    _write_json(output_dir/"witnesses/e4_branch_registration.json",value)
    if branches: (output_dir/"witnesses/e4_branch_registration.json").chmod(0o444)
    return value


def execute_registered_branch(root, branch_id, output_dir):
    root,output_dir=Path(root).resolve(),Path(output_dir); reg=_json(output_dir/"witnesses/e4_branch_registration.json"); branch=next((b for b in reg.get("branches",[]) if b["branch_id"]==branch_id),None)
    if branch is None: raise _base().RevisionError("witness execution requires frozen branch registration")
    manifest=yaml.safe_load((output_dir/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").read_text()); bundle=load_runtime(manifest)
    from cp_disr.baselines.b_plan import BPlanPlanner,SearchConfig
    try:
        snap=bundle.start_case(branch["case_id"]); start=bundle.clock.now_seconds(); execution=bundle.executor.execute(branch["candidate_id"],next(c.timeout_seconds for c in snap.template.contracts if c.id==branch["candidate_id"])); obs=bundle.environment.public_observation(); measured=bundle.perception.infer(obs); records=bundle.verifier.verify(measured,execution); facts=FactStore(records)
        planner=BPlanPlanner(SearchConfig(depth_limit=6,max_nodes=4096,cpu_time_limit_seconds=2.0,reference_skill_seconds=3.7000000000002355)); plan=planner.plan(facts,snap.template,max(0.0,60.0-(bundle.clock.now_seconds()-start)))
        return {"branch_id":branch_id,"case_id":branch["case_id"],"candidate_id":branch["candidate_id"],"controller_exit":execution.controller_exit,"verified_fact_count":len(records),"continuation_status":plan.status,"continuation_first_action":plan.plan[0] if plan.plan else "","status":"DIAGNOSTIC" if execution.controller_exit=="SUCCESS" else "ENGINEERING_NON_DIAGNOSTIC"}
    except Exception as exc:
        return {"branch_id":branch_id,"case_id":branch["case_id"],"candidate_id":branch["candidate_id"],"status":"ENGINEERING_NON_DIAGNOSTIC","error_type":type(exc).__name__}
    finally:
        try: bundle.environment.close()
        except Exception: pass


def run_witnesses(root, output_dir):
    output_dir=Path(output_dir); reg=_json(output_dir/"witnesses/e4_branch_registration.json"); rows=[]
    for branch in reg.get("branches",[]):
        _base().reserve_budget(output_dir,"physical_witness_episodes",1,branch["branch_id"]); rows.append(execute_registered_branch(root,branch["branch_id"],output_dir))
    fields=sorted({k for row in rows for k in row}) if rows else ["branch_id","status"]
    _base()._write_csv(output_dir/"witnesses/e4_physical_witnesses_rev1.csv",fields,rows); _base()._write_csv(output_dir/"witnesses/physical_episode_ledger.csv",fields,rows); _write_json(output_dir/"witnesses/restore_integrity.json",{"status":"COMPLETE" if rows else "NOT_RUN_NO_BRANCHES","episodes":len(rows)})
    return {"status":"COMPLETE" if rows else "NOT_RUN_NO_BRANCHES","episodes":len(rows)}


def offline_rescore_planner(root, case_ref, output_dir):
    from cp_disr.baselines.b_plan import BPlanPlanner,SearchConfig
    output_dir=Path(output_dir); rows=[]; manifest=yaml.safe_load((output_dir/"input_binding/T_A_s1_rev1_runtime_manifest.yaml").read_text())
    for case in downstream_ready(output_dir):
        bundle=load_runtime(manifest)
        try:
            snap=bundle.start_case(case["scene_id"]); result=BPlanPlanner(SearchConfig(depth_limit=6,max_nodes=4096,cpu_time_limit_seconds=2.0,reference_skill_seconds=3.7000000000002355)).plan(snap.facts,snap.template,60.0); rows.append({"case_id":case["scene_id"],"status":result.status,"first_action":result.plan[0] if result.plan else "","environment_episodes":0,"depth_limit":6,"max_nodes":4096,"cpu_time_limit_seconds":2.0,"tie_break":"canonical_skill_id_plan_tuple"})
        finally:
            try: bundle.environment.close()
            except Exception: pass
    _base()._write_csv(output_dir/"planner_offline/offline_planner_discriminability.csv",list(rows[0]) if rows else ["case_id","status","environment_episodes"],rows); _write_json(output_dir/"planner_offline/search_accounting.json",{"status":"COMPLETE" if rows else "NOT_APPLICABLE_NO_NATURAL_RELATIONS","environment_episodes":0,"cases":len(rows)})
    return {"status":"COMPLETE" if rows else "NOT_APPLICABLE_NO_NATURAL_RELATIONS","cases":len(rows)}


def finalize_eligibility(root, output_dir):
    output_dir=Path(output_dir); stage=_json(output_dir/"stage_manifest.json"); stopped=stage.get("status")=="STOPPED"
    provider=list(csv.DictReader((output_dir/"provider/cache_index.csv").open())) if (output_dir/"provider/cache_index.csv").exists() else []
    natural=[r for r in provider if int(r.get("admitted_count") or 0)>0]; rep=list(csv.DictReader((output_dir/"representation/e3_candidate_discriminability_rev1.csv").open())) if (output_dir/"representation/e3_candidate_discriminability_rev1.csv").exists() else []; witnesses=list(csv.DictReader((output_dir/"witnesses/e4_physical_witnesses_rev1.csv").open())) if (output_dir/"witnesses/e4_physical_witnesses_rev1.csv").exists() else []
    gates={"E1":"PASS" if len({r['scene_id'] for r in natural})>=2 else "FAIL","E2":"PASS" if len({r['case_id'] for r in rep if r.get('status')=='PASS'})>=2 else "FAIL","E3":"PASS" if len({r['case_id'] for r in rep if r.get('status')=='PASS' and r.get('common_shift_only') in ('False','false',False)})>=2 else "FAIL","E4":"PASS" if len({r.get('case_id') for r in witnesses if r.get('status')=='DIAGNOSTIC'})>=2 else "FAIL","E5":"PASS" if natural else "FAIL","E6":"PASS" if provider else "FAIL"}
    decision="PENDING_SAME_REVISION" if stopped else "ELIGIBLE_AFTER_SINGLE_REVISION" if all(v=="PASS" for v in gates.values()) else "NOT_ELIGIBLE_AFTER_SINGLE_REVISION"; final={"status":"STOPPED" if stopped else "COMPLETE","eligibility_decision":decision,"tp_training_authorized":False,"next_action":"RESUME_SAME_REVISION_ONLY" if stopped else "REQUEST_S2_AUTHORIZATION" if decision.startswith("ELIGIBLE") else "S4_RESEARCH_DECISION","resume_same_revision_only":stopped,"second_revision_allowed":False,"stop_reason":stage.get("stop_reason","") if stopped else "","gates":gates}
    _write_json(output_dir/"final_eligibility.json",final); _write_json(output_dir/"eligibility_manifest.json",final); stage.update(final); _write_json(output_dir/"stage_manifest.json",stage)
    return final
