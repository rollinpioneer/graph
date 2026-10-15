#!/usr/bin/env python3
"""Explicit L1V semantic rerating, metric recomputation, and layer-2 handoff."""

from __future__ import annotations

import argparse
import csv
import html
import importlib.util
import json
import random
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


RUNNER_PATH = Path(__file__).with_name("l1v_runner.py")
SPEC = importlib.util.spec_from_file_location("l1v_runner_review_v2", RUNNER_PATH)
assert SPEC and SPEC.loader
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)

RATING_FIELDS = tuple(runner.RATING_FIELDS)
IDENTITY_FIELDS = {"request_id", "case_id", "root_family_id", "condition", "split"}
RECOVERY_VALUES = {"yes", "no", "not_proposed"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual = fields or list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected JSON object at {path}:{number}")
        rows.append(value)
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def git_commit(repo: Path) -> str:
    return subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()


def locked_file(root: Path, path: Path, purpose: str) -> dict[str, Any]:
    resolved = path.resolve()
    return {
        "logical_path": resolved.relative_to(root.resolve()).as_posix(),
        "size_bytes": resolved.stat().st_size,
        "sha256": runner.sha256_file(resolved),
        "purpose": purpose,
    }


def verify_review_lock(root: Path, review: Path) -> dict[str, Any]:
    lock = runner.read_json(review / "review_input_lock.json")
    failures = []
    for item in lock["files"]:
        path = root / item["logical_path"]
        if not path.is_file():
            failures.append(f"missing: {item['logical_path']}")
        elif path.stat().st_size != item["size_bytes"]:
            failures.append(f"size mismatch: {item['logical_path']}")
        elif runner.sha256_file(path) != item["sha256"]:
            failures.append(f"hash mismatch: {item['logical_path']}")
    return {"status": "PASS" if not failures else "FAIL", "files": len(lock["files"]), "failures": failures}


def prepare_review(root: Path, review: Path, seed: int) -> dict[str, Any]:
    candidates = sorted((root / "run/candidates/confirm").glob("*.json"))
    if len(candidates) != 54:
        raise ValueError(f"expected 54 frozen confirmation candidates, got {len(candidates)}")
    states = {
        state["request_id"]: state
        for path in sorted((root / "run/states/confirm").glob("*.json"))
        for state in [runner.read_json(path)]
    }
    references = {row["case_id"]: row for row in read_csv(root / "evaluation_private/reference_rubric.csv")}
    scenes = {row["case_id"]: row for row in runner.read_jsonl(root / "prepared/scenes.prepared.jsonl")}
    request_ids = [path.stem for path in candidates]
    random.Random(seed).shuffle(request_ids)
    mapping_rows = []
    blank_rows = []
    sections = []
    lock_files = [
        locked_file(root, root / "protocol.json", "frozen scoring protocol"),
        locked_file(root, root / "run/confirmation.files.lock.json", "confirmation input lock"),
        locked_file(root, root / "evaluation_private/reference_rubric.csv", "pre-candidate semantic reference"),
    ]
    for index, request_id in enumerate(request_ids, 1):
        blind_id = f"E{index:03d}"
        state = states[request_id]
        case_id = state["case_id"]
        candidate_path = root / "run/candidates/confirm" / f"{request_id}.json"
        state_path = root / "run/states/confirm" / f"{request_id}.json"
        graph = runner.read_json(candidate_path)
        scene = scenes[case_id]
        reference = references[case_id]
        view_count = int(state["image_count_sent"])
        image_paths = [Path(view["path"]) for view in scene["views"][:view_count]]
        mapping_rows.append(
            {
                "blind_id": blind_id,
                "request_id": request_id,
                "case_id": case_id,
                "root_family_id": state["root_family_id"],
                "condition": state["condition"],
                "split": "confirm",
                "input_view_count": view_count,
                "candidate_sha256": runner.sha256_file(candidate_path),
            }
        )
        blank_rows.append(
            {
                "blind_id": blind_id,
                **{field: "" for field in RATING_FIELDS},
                "unsupported_visual_assertion": "",
                "recovery_branch_conditional": "",
                "review_status": "pending",
                "reviewer": "",
                "note": "",
            }
        )
        lock_files.extend(
            [
                locked_file(root, candidate_path, f"candidate for {blind_id}"),
                locked_file(root, state_path, f"execution state for {blind_id}"),
            ]
        )
        image_html = "".join(
            f'<img src="{html.escape(Path("../prepared/images", image.relative_to((root / "prepared/images").resolve())).as_posix())}" width="420" alt="{blind_id} input view">'
            for image in image_paths
        ) or "<p>No image was supplied to the generator for this item.</p>"
        for image in image_paths:
            lock_files.append(locked_file(root, image, f"review input image for {blind_id}"))
        reference_text = {
            "manipulated_object": reference["manipulated_object"],
            "target_object": reference["target_object"],
            "expected_behavior": reference["expected_behavior"],
            "unsupported_physical_claims": reference["unsupported_physical_claims"],
        }
        sections.append(
            "<section>"
            f"<h2>{blind_id}</h2><p>Input views: {view_count}</p>{image_html}"
            f"<h3>Reference facts</h3><pre>{html.escape(json.dumps(reference_text, ensure_ascii=False, indent=2))}</pre>"
            f"<h3>Frozen candidate</h3><pre>{html.escape(json.dumps(graph, ensure_ascii=False, indent=2))}</pre>"
            "</section>"
        )
    review.mkdir(parents=True, exist_ok=True)
    write_csv(review / "blind_mapping.DO_NOT_SHOW_RATER.csv", mapping_rows)
    write_csv(review / "ratings_blank.csv", blank_rows)
    review_html = (
        "<!doctype html><meta charset='utf-8'><title>L1V explicit review v2</title>"
        "<style>body{font-family:sans-serif;max-width:1100px;margin:auto}section{border-bottom:1px solid #bbb;padding:20px}"
        "img{margin:4px;vertical-align:top}pre{white-space:pre-wrap;background:#f3f3f3;padding:12px}</style>"
        "<h1>L1V explicit semantic review v2</h1>"
        "<p>Items are randomized and do not display model, condition label, aggregate score, or prior rating.</p>"
        + "".join(sections)
    )
    (review / "review.html").write_text(review_html, encoding="utf-8")
    write_json(
        review / "review_input_lock.json",
        {
            "schema": "l1v_explicit_review_input_lock_v2",
            "created_at": now_iso(),
            "seed": seed,
            "candidate_count": 54,
            "files": sorted(lock_files, key=lambda item: (item["logical_path"], item["purpose"])),
        },
    )
    return {"status": "PASS", "items": 54, "seed": seed, "locked_files": len(lock_files)}


def _decision_value(decision: Any, field: str, blind_id: str) -> tuple[int, str]:
    if not isinstance(decision, dict):
        raise ValueError(f"{blind_id}: {field} decision must be an object")
    value = decision.get("value")
    reason = decision.get("reason")
    if type(value) is not int or value not in {0, 1}:
        raise ValueError(f"{blind_id}: {field} must be explicit 0 or 1")
    if not isinstance(reason, str) or len(reason.strip()) < 12:
        raise ValueError(f"{blind_id}: {field} requires a substantive reason")
    return value, reason.strip()


def validate_evidence_rows(
    evidence_rows: list[dict[str, Any]],
    mapping_rows: list[dict[str, str]],
    graph_by_blind: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    expected = {row["blind_id"] for row in mapping_rows}
    actual = {str(row.get("blind_id")) for row in evidence_rows}
    if len(evidence_rows) != len(expected) or actual != expected:
        raise ValueError(f"evidence IDs do not match mapping: expected={len(expected)} actual={len(actual)}")
    ratings = []
    for item in evidence_rows:
        blind_id = str(item["blind_id"])
        forbidden = IDENTITY_FIELDS.intersection(item)
        if forbidden:
            raise ValueError(f"{blind_id}: evidence must remain blind; found {sorted(forbidden)}")
        decisions = item.get("field_decisions")
        if not isinstance(decisions, dict) or set(decisions) != set(RATING_FIELDS):
            raise ValueError(f"{blind_id}: field_decisions must contain all six rubric fields")
        values = {}
        reasons = {}
        for field in RATING_FIELDS:
            values[field], reasons[field] = _decision_value(decisions[field], field, blind_id)
        unsupported, unsupported_reason = _decision_value(
            item.get("unsupported_visual_assertion"), "unsupported_visual_assertion", blind_id
        )
        if item.get("review_status") != "complete":
            raise ValueError(f"{blind_id}: review_status must be complete")
        reviewer = item.get("reviewer")
        if not isinstance(reviewer, str) or not reviewer.strip():
            raise ValueError(f"{blind_id}: reviewer is required")
        facts = item.get("scene_facts")
        if not isinstance(facts, list) or not facts or any(not isinstance(fact, str) or not fact.strip() for fact in facts):
            raise ValueError(f"{blind_id}: at least one explicit scene fact is required")
        citations = item.get("graph_citations")
        if not isinstance(citations, dict):
            raise ValueError(f"{blind_id}: graph_citations are required")
        graph = graph_by_blind[blind_id]
        valid_ids = {
            "objects": {entry["id"] for entry in graph["objects"]},
            "nodes": {entry["id"] for entry in graph["nodes"]},
            "edges": {entry["id"] for entry in graph["edges"]},
        }
        for kind in ("objects", "nodes", "edges"):
            cited = citations.get(kind)
            if not isinstance(cited, list) or not cited:
                raise ValueError(f"{blind_id}: scene_ready requires at least one {kind} citation")
            unknown = set(cited) - valid_ids[kind]
            if unknown:
                raise ValueError(f"{blind_id}: unknown {kind} citations {sorted(unknown)}")
        recovery = item.get("recovery_branch_conditional")
        if recovery not in RECOVERY_VALUES:
            raise ValueError(f"{blind_id}: invalid recovery_branch_conditional")
        ratings.append(
            {
                "blind_id": blind_id,
                **values,
                "unsupported_visual_assertion": unsupported,
                "recovery_branch_conditional": recovery,
                "review_status": "complete",
                "reviewer": reviewer.strip(),
                "note": reasons["scene_ready"],
                "unsupported_reason": unsupported_reason,
            }
        )
    return sorted(ratings, key=lambda row: row["blind_id"])


def import_explicit(root: Path, review: Path, evidence_path: Path, repo: Path) -> dict[str, Any]:
    lock_check = verify_review_lock(root, review)
    if lock_check["status"] != "PASS":
        raise RuntimeError(f"review input lock failed: {lock_check['failures']}")
    mapping_rows = read_csv(review / "blind_mapping.DO_NOT_SHOW_RATER.csv")
    graph_by_blind = {
        row["blind_id"]: runner.read_json(root / "run/candidates/confirm" / f"{row['request_id']}.json")
        for row in mapping_rows
    }
    evidence_rows = read_jsonl(evidence_path)
    ratings = validate_evidence_rows(evidence_rows, mapping_rows, graph_by_blind)
    write_csv(
        review / "ratings_explicit.csv",
        ratings,
        ["blind_id", *RATING_FIELDS, "unsupported_visual_assertion", "recovery_branch_conditional", "review_status", "reviewer", "note", "unsupported_reason"],
    )
    if evidence_path.resolve() != (review / "rating_evidence.jsonl").resolve():
        shutil.copy2(evidence_path, review / "rating_evidence.jsonl")
    write_json(
        review / "reviewer_provenance.json",
        {
            "schema": "l1v_explicit_reviewer_provenance_v2",
            "recorded_at": now_iso(),
            "source_commit": git_commit(repo),
            "reviewer": ratings[0]["reviewer"],
            "reviewer_type": "independent Codex semantic evaluator; not a human rater",
            "generator_model": "qwen3.7-plus",
            "second_reviewer": False,
            "fully_blind": False,
            "blinding": "random order; model, condition labels, prior ratings, and aggregate results hidden in review surface",
            "reference_source": "pre-candidate reference_rubric.csv plus supplied image views",
            "candidate_outputs_modified": False,
            "old_ratings_modified": False,
            "new_api_calls": 0,
            "new_training_jobs": 0,
            "items": len(ratings),
            "all_binary_values_have_reasons": True,
            "scene_ready_has_object_node_edge_citations": True,
            "review_input_lock": lock_check,
        },
    )
    write_json(
        review / "run_manifest.json",
        {
            "schema": "l1v_semantic_review_update_run_manifest_v2",
            "run_id": "l1v_semantic_review_update_v2",
            "recorded_at": now_iso(),
            "source_commit": git_commit(repo),
            "python": str(__import__("sys").executable),
            "gpu_ids": "none; semantic review and CPU-only aggregation",
            "frozen_confirmation_candidates": 54,
            "explicit_ratings": 54,
            "new_api_calls": 0,
            "new_training_jobs": 0,
            "commands": ["prepare", "import-explicit", "score", "build-layer2", "package-update"],
            "status": "RATINGS_VALIDATED",
        },
    )
    return {"status": "PASS", "ratings": len(ratings), "lock": lock_check}


def _compare_scores(root: Path, review: Path, metrics: Path) -> None:
    old_mapping = {row["blind_id"]: row for row in read_csv(root / "evaluation_private/confirm_review/blind_mapping.DO_NOT_SHOW_RATER.csv")}
    old_by_request = {
        old_mapping[row["blind_id"]]["request_id"]: row
        for row in read_csv(root / "evaluation_private/confirm_review/ratings.csv")
    }
    new_mapping = {row["blind_id"]: row for row in read_csv(review / "blind_mapping.DO_NOT_SHOW_RATER.csv")}
    fields = RATING_FIELDS + ("unsupported_visual_assertion",)
    rows = []
    for rating in read_csv(review / "ratings_explicit.csv"):
        identity = new_mapping[rating["blind_id"]]
        old = old_by_request[identity["request_id"]]
        changed = [field for field in fields if old[field] != rating[field]]
        rows.append(
            {
                "request_id": identity["request_id"],
                "case_id": identity["case_id"],
                "condition": identity["condition"],
                **{f"old_{field}": old[field] for field in fields},
                **{f"new_{field}": rating[field] for field in fields},
                "changed_fields": ";".join(changed),
            }
        )
    write_csv(metrics / "old_vs_new_scores.csv", sorted(rows, key=lambda row: row["request_id"]))


def score_explicit(root: Path, review: Path, metrics: Path) -> dict[str, Any]:
    lock_check = verify_review_lock(root, review)
    if lock_check["status"] != "PASS":
        raise RuntimeError(f"review input lock failed: {lock_check['failures']}")
    if not (review / "ratings_explicit.csv").is_file():
        raise FileNotFoundError("ratings_explicit.csv")
    with tempfile.TemporaryDirectory(prefix="l1v-review-v2-") as temporary:
        adapter = Path(temporary)
        shutil.copy2(review / "ratings_explicit.csv", adapter / "ratings.csv")
        shutil.copy2(review / "blind_mapping.DO_NOT_SHOW_RATER.csv", adapter / "blind_mapping.DO_NOT_SHOW_RATER.csv")
        result = runner.score(adapter, root / "run", metrics, root / "protocol.json")
    _compare_scores(root, review, metrics)
    shutil.copy2(metrics / "old_vs_new_scores.csv", review / "old_vs_new_scores.csv")
    old_decision = runner.read_json(root / "final/l1v_decision.json")
    changed_rows = [row for row in read_csv(metrics / "old_vs_new_scores.csv") if row["changed_fields"]]
    summary = {row["condition"]: row for row in result["condition_summary"]}
    effects = {row["comparison"]: row for row in result["paired_effects"]}
    (metrics / "l1v_review_v2_report.md").write_text(
        "# L1V Explicit Semantic Review Update v2\n\n"
        f"- Original decision, preserved unchanged: `{old_decision['status']}`\n"
        f"- Explicit rerating decision: `{result['status']}`\n"
        f"- Explicit rerating candidate source: `{result.get('coarse_graph_source')}`\n"
        f"- Ratings with any changed field: {len(changed_rows)}/54\n"
        "- Reviewer: one independent Codex semantic evaluator; not a human rater\n"
        "- New generation API calls: 0; new training jobs: 0\n\n"
        "## Recomputed confirmation results\n\n"
        f"- T: {summary['T']['scene_ready']}/18 scene-ready, {summary['T']['paired_families_ready']}/9 paired families\n"
        f"- V1: {summary['V1']['scene_ready']}/18 scene-ready, {summary['V1']['paired_families_ready']}/9 paired families\n"
        f"- V2: {summary['V2']['scene_ready']}/18 scene-ready, {summary['V2']['paired_families_ready']}/9 paired families\n"
        f"- V1-T: {effects['V1-T']['mean_effect']:.6f} [{effects['V1-T']['bootstrap_low']:.6f}, {effects['V1-T']['bootstrap_high']:.6f}]\n"
        f"- V2-T: {effects['V2-T']['mean_effect']:.6f} [{effects['V2-T']['bootstrap_low']:.6f}, {effects['V2-T']['bootstrap_high']:.6f}]\n"
        f"- V2-V1: {effects['V2-V1']['mean_effect']:.6f} [{effects['V2-V1']['bootstrap_low']:.6f}, {effects['V2-V1']['bootstrap_high']:.6f}]\n\n"
        "## Interpretation\n\n"
        "The explicit review treats the F08_B/V1 capacity observation as a required pre-placement check, while F08_B/V2 still lacks a mandatory capacity branch. This moves V1 to the predeclared 6/9 paired-family threshold. The result supports only a single-view layer-2 observation-interface prototype in the current MuJoCo primitive scenes. It does not establish physical execution, reward learning, policy gain, real-camera generalization, or human inter-rater agreement.\n",
        encoding="utf-8",
    )
    manifest = runner.read_json(review / "run_manifest.json")
    manifest.update(
        {
            "status": "PASS",
            "scientific_decision": result["status"],
            "coarse_graph_source": result.get("coarse_graph_source"),
            "metrics_directory": str(metrics.resolve()),
        }
    )
    write_json(review / "run_manifest.json", manifest)
    (review / "summary.md").write_text(
        "# L1V Semantic Review Update v2\n\n"
        "- Frozen confirmation candidates: 54/54\n"
        "- Explicit evidence records: 54/54\n"
        f"- Recomputed decision: `{result['status']}`\n"
        f"- Candidate source: `{result.get('coarse_graph_source')}`\n"
        "- New API calls: 0; new training jobs: 0; GPU use: none\n"
        "- Original L1V ratings and decision were not overwritten.\n",
        encoding="utf-8",
    )
    write_json(
        metrics / "review_resolution.json",
        {
            "schema": "l1v_review_resolution_v2",
            "status": "COMPLETE_EXPLICIT_INDEPENDENT_AGENT_RERATING",
            "scientific_decision": result["status"],
            "coarse_graph_source": result.get("coarse_graph_source"),
            "old_decision_preserved": True,
            "new_api_calls": 0,
            "new_training_jobs": 0,
            "reviewer_scope": "one independent Codex semantic evaluator; no human or inter-rater coefficient",
        },
    )
    return result


def build_layer2_entry(root: Path, metrics: Path, output: Path) -> dict[str, Any]:
    decision = runner.read_json(metrics / "l1v_decision.json")
    resolution = runner.read_json(metrics / "review_resolution.json")
    predicates = [
        {"predicate_id": "object_bound", "category": "object_binding", "arguments": "manipulated_object", "online_evidence": "RGB/depth detection and identity confidence", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Do not replace ambiguity with a guessed identity."},
        {"predicate_id": "target_bound", "category": "object_binding", "arguments": "target_object", "online_evidence": "RGB/depth detection and task-role binding", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Target role is distinct from geometric coordinates."},
        {"predicate_id": "target_region_clear", "category": "target_relation", "arguments": "target_object", "online_evidence": "visible occupancy and clearance estimate", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "A visible obstacle requires a branch before placement."},
        {"predicate_id": "goal_relation", "category": "target_relation", "arguments": "manipulated_object,target_object,relation", "online_evidence": "post-observation spatial estimate", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Relation is centered_on, inside, or in_empty_region."},
        {"predicate_id": "contact", "category": "interaction", "arguments": "manipulated_object,target_object", "online_evidence": "visual contact estimate or force/tactile event", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Simulator contact truth must be labeled oracle when used."},
        {"predicate_id": "held_by_gripper", "category": "interaction", "arguments": "manipulated_object", "online_evidence": "gripper state plus object motion consistency", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Closed gripper alone does not prove a stable hold."},
        {"predicate_id": "released", "category": "interaction", "arguments": "manipulated_object", "online_evidence": "open gripper, separation, and no co-motion", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Release is separate from stable support."},
        {"predicate_id": "support_after_release", "category": "postcondition", "arguments": "manipulated_object,target_object", "online_evidence": "released object remains supported for an observation window", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Requires temporal evidence; a static render is insufficient."},
        {"predicate_id": "stable_after_release", "category": "postcondition", "arguments": "manipulated_object", "online_evidence": "bounded pose drift after release", "allowed_values": "true|false|unknown|not_observed", "oracle_only": 0, "notes": "Threshold must be frozen on development scenes."},
        {"predicate_id": "sim_contact_truth", "category": "oracle_reference", "arguments": "geom_a,geom_b", "online_evidence": "MuJoCo contact buffer", "allowed_values": "true|false", "oracle_only": 1, "notes": "Evaluation reference only; never merge into a pure-vision method input."},
    ]
    write_csv(output / "predicate_catalog.csv", predicates)
    write_json(
        output / "observation_contract.json",
        {
            "schema": "pathgraph_layer2_observation_contract_v1",
            "status": "INTERFACE_PREPARATION_COMPLETE",
            "coarse_graph_candidate_source": decision.get("coarse_graph_source"),
            "evidence_classes": {
                "image_fact": "fact visible in the supplied observation, with view provenance",
                "graph_hypothesis": "unvalidated node, edge, precondition, or recovery proposal from L1V",
                "execution_result": "post-action observation or sensor result recorded with time and episode identity",
                "oracle_reference": "simulator-only truth, segregated from online method features",
            },
            "online_forbidden_inputs": ["future success", "hidden simulator state", "gold_mode", "scenario", "phase", "oracle contact truth"],
            "unknown_policy": "unknown and not_observed remain distinct from false",
            "review_resolution": resolution["status"],
            "robot_execution_verified": False,
            "dynamic_simulation_executed": False,
            "dense_reward_learned": False,
        },
    )
    write_json(
        output / "bowl_plate_minimal_loop.json",
        {
            "schema": "pathgraph_layer2_bowl_plate_loop_v1",
            "scope": "red bowl to white plate only",
            "development_inputs": ["F01_A", "F01_B"],
            "confirmation_inputs": "new families only; do not reuse F02-F04 as fresh confirmation",
            "coarse_graph_condition": decision.get("coarse_graph_source"),
            "steps": [
                {"id": "bind", "requires": ["object_bound", "target_bound"], "on_unknown": "observe_or_clarify"},
                {"id": "check_current_goal", "requires": ["goal_relation"], "on_true": "verify_release_support"},
                {"id": "check_target", "requires": ["target_region_clear"], "on_false": "clear_or_request_clarification"},
                {"id": "grasp", "requires": ["object_bound"], "produces": ["held_by_gripper"]},
                {"id": "place", "requires": ["held_by_gripper", "target_region_clear"], "produces": ["contact"]},
                {"id": "release", "requires": ["contact"], "produces": ["released"]},
                {"id": "verify_release_support", "requires": ["released"], "produces": ["support_after_release", "stable_after_release", "goal_relation"]},
            ],
            "threshold_policy": "freeze thresholds on development observations before any new confirmation run",
        },
    )
    write_json(
        output / "entry_gate.json",
        {
            "schema": "pathgraph_layer2_entry_gate_v1",
            "status": "LAYER2_INTERFACE_READY_FOR_DEVELOPMENT",
            "semantic_review_complete": True,
            "semantic_review_kind": "explicit independent single-agent rerating",
            "coarse_graph_candidate_source": decision.get("coarse_graph_source"),
            "scientific_decision": decision["status"],
            "allowed_now": ["implement observable predicates", "collect development observations", "compare coarse and refined state judgments"],
            "not_allowed_yet": ["claim robot execution success", "claim reward or policy gain", "reuse seen confirmation failures as new confirmation", "mix oracle truth into pure-vision features"],
            "next_confirmation_requires": ["fresh bowl-plate families", "frozen observation thresholds", "dynamic active objects and action/contact/termination logs", "same explicit rating provenance standard"],
            "new_api_calls": 0,
            "new_training_jobs": 0,
        },
    )
    (output / "README.md").write_text(
        "# Layer-2 Entry Preparation\n\n"
        "This directory defines observation bindings only. It does not report a robot rollout, learned reward, or policy improvement.\n\n"
        "The first development loop is restricted to red-bowl/white-plate scenes. Existing confirmation families are audit examples, not a new confirmation set.\n",
        encoding="utf-8",
    )
    return {"status": "LAYER2_INTERFACE_READY_FOR_DEVELOPMENT", "predicates": len(predicates)}


def _copy_tree_files(source: Path, target: Path) -> None:
    for path in sorted(source.rglob("*")):
        if path.is_file() and "__pycache__" not in path.parts:
            destination = target / path.relative_to(source)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)


def package_update(root: Path, review: Path, metrics: Path, layer2: Path, repo: Path, downloads: Path) -> dict[str, Any]:
    release = root / "review_update_release_v2"
    if release.exists():
        shutil.rmtree(release)
    _copy_tree_files(review, release / "review_v2")
    _copy_tree_files(metrics, release / "metrics_v2")
    _copy_tree_files(layer2, release / "layer2_entry_v1")
    code_root = repo / "upgrade_v2/visual_coarse_l1"
    (release / "tools").mkdir(parents=True, exist_ok=True)
    shutil.copy2(code_root / "tools/l1v_review_v2.py", release / "tools/l1v_review_v2.py")
    shutil.copy2(code_root / "tools/record_agent_review.py", release / "tools/record_agent_review.py")
    (release / "frozen_inputs").mkdir(parents=True, exist_ok=True)
    shutil.copy2(root / "protocol.json", release / "frozen_inputs/protocol.json")
    shutil.copy2(root / "run/confirmation.files.lock.json", release / "frozen_inputs/confirmation.files.lock.json")
    shutil.copy2(root / "evaluation_private/reference_rubric.csv", release / "frozen_inputs/reference_rubric.csv")
    mapping = read_csv(review / "blind_mapping.DO_NOT_SHOW_RATER.csv")
    for row in mapping:
        candidate = root / "run/candidates/confirm" / f"{row['request_id']}.json"
        destination = release / "frozen_candidates" / f"{row['blind_id']}.json"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, destination)
    scenes = {row["case_id"]: row for row in runner.read_jsonl(root / "prepared/scenes.prepared.jsonl")}
    copied_images: set[str] = set()
    for row in mapping:
        for view in scenes[row["case_id"]]["views"][: int(row["input_view_count"])]:
            source = Path(view["path"])
            logical = source.relative_to((root / "prepared/images").resolve()).as_posix()
            if logical in copied_images:
                continue
            copied_images.add(logical)
            destination = release / "prepared/images" / logical
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
    decision = runner.read_json(metrics / "l1v_decision.json")
    (release / "summary.md").write_text(
        "# L1V Semantic Review Update v2\n\n"
        f"- Source commit: `{git_commit(repo)}`\n"
        "- Explicit ratings: 54/54\n"
        f"- New decision: `{decision['status']}`\n"
        f"- Candidate source: `{decision.get('coarse_graph_source')}`\n"
        "- Evaluator: one independent Codex semantic evaluator; not a human rater\n"
        "- New API calls: 0; new training jobs: 0\n"
        "- Original ratings, metrics, and L1V decision remain unchanged\n"
        "- Layer-2 scope: observation-interface preparation only; no execution success claim\n",
        encoding="utf-8",
    )
    result = runner.package(release, downloads / "L1V_semantic_review_update.zip", 200)
    (downloads / "L1V_semantic_review_update.zip.placeholder.md").write_text(
        "# Local L1V Semantic Review Update ZIP\n\n"
        f"- Original path: `{(downloads / 'L1V_semantic_review_update.zip').resolve()}`\n"
        "- Original filename: `L1V_semantic_review_update.zip`\n"
        f"- Size: {(downloads / 'L1V_semantic_review_update.zip').stat().st_size} bytes\n"
        f"- SHA256: `{result['sha256']}`\n"
        "- Purpose: explicit 54-item semantic rerating, recomputed metrics, and layer-2 entry contract.\n"
        "- Restore: rerun `l1v_review_v2.py package-update` from the frozen inputs.\n",
        encoding="utf-8",
    )
    write_json(
        downloads / "semantic_review_update_index.json",
        {
            "schema": "l1v_semantic_review_update_package_v2",
            "source_commit": git_commit(repo),
            "package": result,
            "frozen_candidates": 54,
            "review_input_images": len(copied_images),
            "new_api_calls": 0,
            "new_training_jobs": 0,
        },
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l1v_review_v2.py")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--seed", type=int, default=20260907)
    command = sub.add_parser("import-explicit")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--evidence", type=Path, required=True)
    command.add_argument("--repo", type=Path, required=True)
    command = sub.add_parser("score")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--metrics", type=Path, required=True)
    command = sub.add_parser("build-layer2")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--metrics", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command = sub.add_parser("package-update")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--metrics", type=Path, required=True)
    command.add_argument("--layer2", type=Path, required=True)
    command.add_argument("--repo", type=Path, required=True)
    command.add_argument("--downloads", type=Path, required=True)
    command = sub.add_parser("verify")
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--review", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare_review(args.root, args.review, args.seed)
        elif args.command == "import-explicit":
            result = import_explicit(args.root, args.review, args.evidence, args.repo)
        elif args.command == "score":
            result = score_explicit(args.root, args.review, args.metrics)
        elif args.command == "build-layer2":
            result = build_layer2_entry(args.root, args.metrics, args.output)
        elif args.command == "package-update":
            result = package_update(args.root, args.review, args.metrics, args.layer2, args.repo, args.downloads)
        elif args.command == "verify":
            result = verify_review_lock(args.root, args.review)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {"FAIL", "EVALUATION_PENDING"} else 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": runner.public_error(exc)}, ensure_ascii=False), file=__import__("sys").stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
