"""Freeze the existing U3 negative result without rewriting it."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from .common import read_csv, read_json, sha256_file, write_csv, write_json


def _failure_class(row: dict[str, str]) -> tuple[str, str]:
    errors = int(row.get("error_count", "0") or 0)
    if errors == 0 and row.get("hard_valid", "").lower() in {"true", "1"}:
        return "NONE", "hard-valid semantic hypothesis; no evidence arrays are used"
    if row.get("schema_valid", "").lower() not in {"true", "1"}:
        return "OUTPUT_TRUNCATED", "schema-invalid parsed output"
    if int(row.get("hallucinated_evidence_id_count", "0") or 0) > 0:
        return "HALLUCINATED_SEGMENT_ID", "raw evidence identifiers were not in the locked registry"
    if int(row.get("transition_pair_count", "0") or 0) > 0 and row.get("evidence_valid", "").lower() != "true":
        return "UNSUPPORTED_TRANSITION_PAIR", "transition pair was absent from compact train evidence"
    if row.get("condition") == "instruction_only" and row.get("instruction_only_evidence_valid", "").lower() != "true":
        return "INSTRUCTION_ONLY_EVIDENCE_VIOLATION", "instruction-only candidate cited evidence"
    return "GENERIC_PLACEHOLDER_EVIDENCE", "candidate contains unsupported or unresolved evidence references"


def freeze_u3_result(*, execution_summary: Path, handoff: Path, hard_checks: Path, scores: Path, output: Path, lock: Path, report: Path) -> dict[str, Any]:
    checks = read_csv(hard_checks)
    score_rows = read_csv(scores)
    score_by_id = {row.get("candidate_id", ""): row for row in score_rows}
    rows: list[dict[str, Any]] = []
    for row in checks:
        cls, detail = _failure_class(row)
        cid = row.get("candidate_id", "")
        rows.append({
            "candidate_id": cid,
            "provider": row.get("provider", ""),
            "condition": row.get("condition", ""),
            "parse_valid": row.get("schema_valid", "False"),
            "schema_valid": row.get("schema_valid", "False"),
            "topology_valid": row.get("topology_valid", "False"),
            "predicate_valid": row.get("predicate_valid", "False"),
            "evidence_valid": row.get("evidence_valid", "False"),
            "failure_class": cls,
            "failure_detail": detail,
            "reusable_as_semantic_hypothesis": cid in {"qwen:instruction_only_r01", "qwen:instruction_only_r02"},
            "preliminary_score": score_by_id.get(cid, {}).get("preliminary_score", "not_applicable"),
        })
    write_csv(output, rows)
    source = {
        "schema": "u3_v1_result_lock_v1",
        "source_files": {name: {"path": str(path), "sha256": sha256_file(path)} for name, path in (("execution_summary", execution_summary), ("handoff", handoff), ("hard_checks", hard_checks), ("scores", scores))},
        "decision": "U3_INCONCLUSIVE",
        "qwen_hard_valid": sum(row.get("provider") == "qwen" and row.get("hard_valid", "").lower() in {"true", "1"} for row in checks),
        "deepseek_hard_valid": sum(row.get("provider") == "deepseek" and row.get("hard_valid", "").lower() in {"true", "1"} for row in checks),
        "selected_for_u4": 0,
        "negative_result_frozen": True,
    }
    write_json(lock, source)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# U3 v1 negative result frozen\n\n" + "\n".join([
        "- status: `U3_NEGATIVE_RESULT_FROZEN`",
        "- decision: `U3_INCONCLUSIVE`",
        f"- Qwen hard-valid: `{source['qwen_hard_valid']}`",
        f"- DeepSeek hard-valid: `{source['deepseek_hard_valid']}`",
        "- selected for U4: `0`",
        "- reusable semantic hypotheses: `qwen:instruction_only_r01`, `qwen:instruction_only_r02`",
    ]) + "\n", encoding="utf-8")
    return source | {"status": "U3_NEGATIVE_RESULT_FROZEN", "failure_count": len(rows)}
