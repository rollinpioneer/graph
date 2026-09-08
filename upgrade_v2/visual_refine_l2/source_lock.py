"""L1V entry validation and frozen V1 candidate provenance."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import lock_record, now_iso, read_csv, read_json, read_jsonl, write_csv, write_json, write_report


ENTRY_COMMIT = "5a3aafd6ac6d139ba324ee57110bbac508e17570"


def check_entry(decision_path: Path, review_report: Path, resolution_path: Path, entry_gate_path: Path, output: Path, report: Path) -> dict[str, Any]:
    decision = read_json(decision_path)
    resolution = read_json(resolution_path)
    gate = read_json(entry_gate_path)
    failures = []
    expected = {
        "scientific_decision": (decision.get("status"), "L1V_READY_FOR_REFINEMENT_SINGLE_VIEW"),
        "coarse_graph_source": (decision.get("coarse_graph_source"), "V1"),
        "review_resolution": (resolution.get("status"), "COMPLETE_EXPLICIT_INDEPENDENT_AGENT_RERATING"),
        "entry_status": (gate.get("status"), "LAYER2_INTERFACE_READY_FOR_DEVELOPMENT"),
        "entry_source": (gate.get("coarse_graph_candidate_source"), "V1"),
        "semantic_review_complete": (gate.get("semantic_review_complete"), True),
        "new_api_calls": (gate.get("new_api_calls"), 0),
        "new_training_jobs": (gate.get("new_training_jobs"), 0),
    }
    for name, (actual, wanted) in expected.items():
        if actual != wanted:
            failures.append(f"{name}: expected {wanted!r}, got {actual!r}")
    if not review_report.is_file():
        failures.append(f"missing review report: {review_report}")
    result = {
        "schema": "pathgraph_l2r_entry_gate_v1",
        "status": "L2R_ENTRY_AND_SOURCE_LOCKED" if not failures else "L1V_REVIEW_NOT_COMPLETE",
        "entry_commit": ENTRY_COMMIT,
        "scientific_decision": decision.get("status"),
        "coarse_graph_candidate_source": decision.get("coarse_graph_source"),
        "semantic_review_complete": gate.get("semantic_review_complete"),
        "new_api_calls": 0,
        "new_training_jobs": 0,
        "failures": failures,
    }
    write_json(output, result)
    write_report(report, "L2R Entry Gate", [("status", result["status"]), ("failures", len(failures)), ("coarse graph source", result["coarse_graph_candidate_source"])])
    return result


def freeze_source(candidate_paths: list[Path], states_dir: Path, review_evidence: Path, review_ratings: Path, mapping_path: Path, entry_gate: Path, output: Path, hash_table: Path, report: Path, repo_root: Path) -> dict[str, Any]:
    if len(candidate_paths) != 18 or len({path.name for path in candidate_paths}) != 18:
        raise ValueError(f"expected 18 unique V1 candidates, got {len(candidate_paths)}")
    if any(not path.name.endswith("_V1.json") for path in candidate_paths):
        raise ValueError("source list contains a non-V1 candidate")
    gate = read_json(entry_gate)
    if gate.get("status") != "LAYER2_INTERFACE_READY_FOR_DEVELOPMENT":
        raise ValueError("L1V layer-2 entry gate is not open")
    evidence_rows = read_jsonl(review_evidence)
    ratings = {row["blind_id"]: row for row in read_csv(review_ratings)}
    mapping = read_csv(mapping_path)
    by_request = {row["request_id"]: row for row in mapping}
    records = []
    source_rows = []
    files = [
        lock_record(review_evidence, "explicit 54-item semantic evidence", repo_root),
        lock_record(review_ratings, "explicit 54-item semantic ratings", repo_root),
        lock_record(mapping_path, "blind ID to frozen request mapping", repo_root),
        lock_record(entry_gate, "L1V layer-2 entry permission", repo_root),
    ]
    evidence_ids = {row["blind_id"] for row in evidence_rows}
    for candidate in sorted(candidate_paths):
        request_id = candidate.stem
        identity = by_request.get(request_id)
        if identity is None:
            raise ValueError(f"candidate missing from review mapping: {request_id}")
        blind_id = identity["blind_id"]
        if blind_id not in evidence_ids or blind_id not in ratings:
            raise ValueError(f"candidate lacks explicit evidence/rating: {request_id}")
        state = states_dir / f"{request_id}.json"
        if not state.is_file() or not read_json(state).get("structure_valid"):
            raise ValueError(f"candidate state missing or structurally invalid: {request_id}")
        candidate_record = lock_record(candidate, f"frozen V1 coarse graph {request_id}", repo_root)
        state_record = lock_record(state, f"frozen V1 request state {request_id}", repo_root)
        files.extend([candidate_record, state_record])
        source_rows.append({
            "request_id": request_id,
            "case_id": identity["case_id"],
            "root_family_id": identity["root_family_id"],
            "blind_id": blind_id,
            "condition": identity["condition"],
            "scene_ready": ratings[blind_id]["scene_ready"],
            "candidate_sha256": candidate_record["sha256"],
            "state_sha256": state_record["sha256"],
        })
        records.append(request_id)
    payload = {
        "schema": "pathgraph_l2r_l1v_source_lock_v1",
        "status": "L2R_ENTRY_AND_SOURCE_LOCKED",
        "created_at": now_iso(),
        "entry_commit": ENTRY_COMMIT,
        "coarse_graph_source": "V1",
        "candidate_count": 18,
        "request_ids": records,
        "manual_path_correction": {
            "specified_evidence": "review_v2/evidence.jsonl",
            "actual_evidence": "review_v2/rating_evidence.jsonl",
            "specified_ratings": "review_v2/ratings.csv",
            "actual_ratings": "review_v2/ratings_explicit.csv",
            "reason": "use the auditable files produced by the fixed L1V entry commit",
        },
        "new_api_calls": 0,
        "new_training_jobs": 0,
        "files": sorted(files, key=lambda row: row["logical_path"]),
    }
    write_json(output, payload)
    write_csv(hash_table, source_rows, ["request_id", "case_id", "root_family_id", "blind_id", "condition", "scene_ready", "candidate_sha256", "state_sha256"], delimiter="\t")
    write_report(report, "L1V Source Lock", [("status", payload["status"]), ("V1 candidates", 18), ("locked files", len(files)), ("API calls", 0), ("training jobs", 0)])
    return payload
