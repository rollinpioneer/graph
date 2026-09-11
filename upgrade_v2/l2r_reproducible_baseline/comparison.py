from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from . import COMPARISON_VERSION
from .environment import MAIN_GATE_FIELDS
from .protocol import canonical_hash


def _jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = ["row", "passed", "left", "right"]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _state_mismatch(left: dict[str, Any], right: dict[str, Any], index: int) -> dict[str, Any] | None:
    for field in ("qpos", "qvel", "qacc", "qacc_warmstart", "mocap_pos", "mocap_quat", "eq_active", "model_eq_data", "xfrc_applied"):
        l_info, r_info = left.get("arrays", {}).get(field), right.get("arrays", {}).get(field)
        if l_info != r_info:
            l_values = np.asarray(left.get("array_values", {}).get(field, []), dtype=float)
            r_values = np.asarray(right.get("array_values", {}).get(field, []), dtype=float)
            max_abs = None
            l2 = None
            if l_values.shape == r_values.shape and l_values.size:
                delta = l_values - r_values
                max_abs = float(np.max(np.abs(delta)))
                l2 = float(np.linalg.norm(delta.ravel()))
            return {"row": index, "field": field, "left_shape": l_info.get("shape") if l_info else None, "right_shape": r_info.get("shape") if r_info else None, "left_dtype": l_info.get("dtype") if l_info else None, "right_dtype": r_info.get("dtype") if r_info else None, "left_byte_sha256": l_info.get("sha256") if l_info else None, "right_byte_sha256": r_info.get("sha256") if r_info else None, "max_abs_diff": max_abs, "l2_diff": l2}
    if left.get("state_sha256") != right.get("state_sha256"):
        return {"row": index, "field": "checkpoint_state", "left_state_sha256": left.get("state_sha256"), "right_state_sha256": right.get("state_sha256")}
    return None


def compare(left: Path, right: Path, output_root: Path, comparison_name: str) -> dict[str, Any]:
    output_root.mkdir(parents=False, exist_ok=False)
    left_result, right_result = _json(left / "result.json"), _json(right / "result.json")
    source_fields = ("runner_commit", "protocol_sha256", "environment_fingerprint_sha256", "model_fingerprint_sha256")
    source_mismatches = [{"field": key, "left": left_result.get(key), "right": right_result.get(key)} for key in source_fields if left_result.get(key) != right_result.get(key)]
    left_lock, right_lock = _json(left / "source_lock.json"), _json(right / "source_lock.json")
    for field in ("runner_commit", "protocol_sha256", "generation_runner_files", "program_sha256", "physical_spec_sha256", "generated_model_xml_sha256"):
        if left_lock.get(field) != right_lock.get(field):
            source_mismatches.append({"field": f"source_lock.{field}", "left": left_lock.get(field), "right": right_lock.get(field)})
    left_env, right_env = _json(left / "runtime_environment_fingerprint.json"), _json(right / "runtime_environment_fingerprint.json")
    for field in MAIN_GATE_FIELDS:
        if left_env.get(field) != right_env.get(field):
            source_mismatches.append({"field": f"environment.{field}", "left": left_env.get(field), "right": right_env.get(field)})
    left_states, right_states = _jsonl(left / "checkpoint_state_trace.jsonl"), _jsonl(right / "checkpoint_state_trace.jsonl")
    state_mismatches = []
    for index, (a, b) in enumerate(zip(left_states, right_states)):
        mismatch = _state_mismatch(a, b, index)
        if mismatch is not None:
            state_mismatches.append(mismatch)
            break
    if len(left_states) != len(right_states):
        state_mismatches.append({"field": "checkpoint_count", "left": len(left_states), "right": len(right_states)})
    left_caps, right_caps = _jsonl(left / "callback_capture_trace.jsonl"), _jsonl(right / "callback_capture_trace.jsonl")
    capture_fields = ("phase", "action", "action_index", "time", "raw_rgb_sha256", "jpeg_sha256", "callback_steps")
    capture_mismatches = []
    for index, (a, b) in enumerate(zip(left_caps, right_caps)):
        for field in capture_fields:
            if a.get(field) != b.get(field):
                capture_mismatches.append({"row": index, "field": field, "left": a.get(field), "right": b.get(field)}); break
        if capture_mismatches: break
    if len(left_caps) != len(right_caps):
        capture_mismatches.append({"field": "capture_count", "left": len(left_caps), "right": len(right_caps)})
    discrete_fields = ("actions.jsonl", "low_level_controls.jsonl", "events.jsonl", "vision_detections.jsonl")
    discrete = {}
    for name in discrete_fields:
        a, b = _jsonl(left / name), _jsonl(right / name)
        discrete[name] = {"passed": a == b, "left_count": len(a), "right_count": len(b), "left_sha256": canonical_hash(a), "right_sha256": canonical_hash(b)}
    source = {"passed": not source_mismatches, "mismatches": source_mismatches}
    physics = {"passed": not state_mismatches, "mismatches": state_mismatches, "left_count": len(left_states), "right_count": len(right_states)}
    render = {"passed": not capture_mismatches, "mismatches": capture_mismatches, "left_count": len(left_caps), "right_count": len(right_caps)}
    discrete_pass = all(row["passed"] for row in discrete.values())
    all_passed = bool(left_result.get("all_main_gates_passed") and right_result.get("all_main_gates_passed") and source["passed"] and physics["passed"] and render["passed"] and discrete_pass)
    first = (source_mismatches + state_mismatches + capture_mismatches)[:1]
    instrumentation = {"passed": True, "reason": "not_applicable"}
    if comparison_name.endswith("instrumented_C"):
        integrity_path = right / "instrumentation_integrity.json"
        instrumentation = _json(integrity_path) if integrity_path.is_file() else {"passed": False, "reason": "missing"}
        all_passed = all_passed and bool(instrumentation.get("passed")) and int(instrumentation.get("before_after_rows", -1)) == 290 and int(instrumentation.get("mutation_failures", -1)) == 0
    summary = {"schema": "l2rar2_r14b_comparison_summary_v1", "comparison_version": COMPARISON_VERSION, "comparison": comparison_name, "status": "PASS" if all_passed else "FAIL", "all_main_gates_passed": all_passed, "source": source, "physics": physics, "render": render, "discrete": discrete, "instrumentation": instrumentation, "first_mismatch": first[0] if first else None}
    _write(output_root / "comparison_summary.json", summary)
    _write(output_root / "first_mismatch.json", first[0] if first else {"first_mismatch": None})
    _write(output_root / "discrete_sequence_comparison.json", discrete)
    _write(output_root / "environment_comparison.json", {"passed": left_result.get("environment_fingerprint_sha256") == right_result.get("environment_fingerprint_sha256")})
    _write(output_root / "model_comparison.json", {"passed": left_result.get("model_fingerprint_sha256") == right_result.get("model_fingerprint_sha256")})
    _write_csv(output_root / "state_hash_comparison.csv", [{"row": i, "passed": a.get("state_sha256") == b.get("state_sha256"), "left": a.get("state_sha256"), "right": b.get("state_sha256")} for i, (a, b) in enumerate(zip(left_states, right_states))])
    _write_csv(output_root / "render_comparison.csv", [{"row": i, "passed": a.get("raw_rgb_sha256") == b.get("raw_rgb_sha256") and a.get("jpeg_sha256") == b.get("jpeg_sha256"), "left": a.get("raw_rgb_sha256"), "right": b.get("raw_rgb_sha256")} for i, (a, b) in enumerate(zip(left_caps, right_caps))])
    _write_csv(output_root / "detection_comparison.csv", [{"row": i, "passed": a == b, "left": canonical_hash(a), "right": canonical_hash(b)} for i, (a, b) in enumerate(zip(_jsonl(left / "vision_detections.jsonl"), _jsonl(right / "vision_detections.jsonl")))])
    if comparison_name.endswith("instrumented_C"):
        _write(output_root / "instrumentation_integrity.json", instrumentation)
    _write(output_root / "run_manifest.json", {"comparison": comparison_name, "left": str(left.resolve()), "right": str(right.resolve()), "summary_sha256": canonical_hash(summary)})
    return summary
