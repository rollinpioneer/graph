"""Provider request tables, bounded execution, and completion verification."""

from __future__ import annotations

import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .common import read_csv, read_json, read_jsonl, sha256_file, write_csv, write_json
from .providers import ModelCallResult, call_deepseek, call_qwen, is_qwen_schema_format_error, parse_json_content


def _messages(row: dict[str, str]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": Path(row["system_prompt_path"]).read_text(encoding="utf-8")},
        {"role": "user", "content": Path(row["user_prompt_path"]).read_text(encoding="utf-8")},
    ]


def build_provider_request_table(
    *, provider: str, requests: Path, model: str, output: Path,
    all_replicates: bool = False, replicate: int | None = None, expected_count: int | None = None,
) -> dict[str, Any]:
    records = read_jsonl(requests)
    if provider == "qwen":
        if not all_replicates:
            raise ValueError("Qwen must use all three replicates")
        selected = records
    elif provider == "deepseek":
        if replicate != 1:
            raise ValueError("DeepSeek cross-check must use only replicate 1")
        selected = [row for row in records if int(row["replicate"]) == 1]
    else:
        raise ValueError(f"unsupported provider: {provider}")
    selected = sorted(selected, key=lambda row: (row["condition"], int(row["replicate"])))
    expected = expected_count if expected_count is not None else (9 if provider == "qwen" else 3)
    if len(selected) != expected:
        raise ValueError(f"{provider} expected {expected} requests; got {len(selected)}")
    rows = [{**row, "provider": provider, "model": model} for row in selected]
    write_csv(output, rows, delimiter="\t")
    return {"provider": provider, "model": model, "request_count": len(rows), "status": "PASS"}


def _schema_errors(value: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(str(x) for x in error.absolute_path) or '$'}: {error.message}" for error in validator.iter_errors(value)]


def _candidate_paths(root: Path, request_id: str) -> dict[str, Path]:
    return {
        "raw": root / "raw_sanitized" / f"{request_id}.json",
        "content": root / "content" / f"{request_id}.json",
        "parsed": root / "parsed" / f"{request_id}.json",
        "state": root / "state" / f"{request_id}.json",
        "repair": root / "repair" / f"{request_id}.json",
    }


def _repair_messages(content: str, errors: list[str], schema: dict[str, Any]) -> list[dict[str, str]]:
    payload = {
        "original_content": content,
        "local_validation_errors": errors,
        "strict_schema": schema,
        "instruction": "Return only one corrected JSON object conforming to strict_schema. Do not add task evidence or commentary.",
    }
    return [
        {"role": "system", "content": "Return a non-empty JSON object only. Do not explain your reasoning."},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, sort_keys=True)},
    ]


def _result_state(result: ModelCallResult, *, row: dict[str, str], response_mode: str, parsed: bool, schema_errors: list[str], repair_attempted: bool, repair_valid: bool | None) -> dict[str, Any]:
    return {
        "request_id": row["request_id"], "condition": row["condition"], "replicate": int(row["replicate"]),
        "provider": result.provider, "requested_model": result.requested_model, "returned_model": result.returned_model,
        "response_id": result.response_id, "http_status": result.http_status, "http_success": result.error_type is None,
        "content_nonempty": bool(result.content.strip()), "json_parse_success": parsed,
        "schema_errors": schema_errors, "schema_valid": parsed and not schema_errors,
        "response_mode": response_mode, "prompt_sha256": row["prompt_sha256"],
        "network_retries": result.network_retries, "latency_seconds": result.latency_seconds,
        "input_tokens": result.input_tokens, "output_tokens": result.output_tokens,
        "reasoning_tokens": result.reasoning_tokens, "finish_reason": result.finish_reason,
        "schema_repair_attempted": repair_attempted, "schema_repair_valid": repair_valid,
        "error_type": result.error_type, "error_message": result.error_message,
    }


def _run_one(
    *, row: dict[str, str], provider: str, model: str, schema: dict[str, Any], output_root: Path,
    max_output_tokens: int, timeout: int, network_retries: int, response_mode: str, repair_limit: int,
) -> dict[str, Any]:
    paths = _candidate_paths(output_root, row["request_id"])
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    if paths["state"].is_file() and paths["parsed"].is_file():
        prior = read_json(paths["state"])
        if prior.get("prompt_sha256") == row["prompt_sha256"] and prior.get("schema_valid"):
            return {**prior, "resumed": True}
    messages = _messages(row)
    if provider == "qwen":
        result, sanitized = call_qwen(
            request_id=row["request_id"], model=model, messages=messages, schema=schema,
            response_mode=response_mode, max_output_tokens=max_output_tokens, timeout=timeout,
            network_retries=network_retries,
        )
    else:
        result, sanitized = call_deepseek(
            request_id=row["request_id"], model=model, messages=messages, max_output_tokens=max_output_tokens,
            timeout=timeout, network_retries=network_retries,
        )
    if sanitized is not None:
        write_json(paths["raw"], sanitized)
    parsed: dict[str, Any] | None = None
    errors: list[str] = []
    if result.content.strip():
        try:
            parsed = parse_json_content(result.content)
            errors = _schema_errors(parsed, schema)
        except (ValueError, json.JSONDecodeError) as exc:
            errors = [str(exc)]
    else:
        errors = ["empty content"]
    repair_attempted = False
    repair_valid: bool | None = None
    final_result = result
    final_sanitized = sanitized
    if provider == "deepseek" and result.error_type is None and errors and repair_limit >= 1:
        repair_attempted = True
        repair_result, repair_sanitized = call_deepseek(
            request_id=row["request_id"], model=model, messages=_repair_messages(result.content, errors, schema),
            max_output_tokens=max_output_tokens, timeout=timeout, network_retries=network_retries,
        )
        repair_result.schema_repair_attempted = True
        if repair_sanitized is not None:
            write_json(paths["repair"], repair_sanitized)
        final_result = repair_result
        final_sanitized = repair_sanitized
        parsed = None
        errors = []
        if repair_result.content.strip():
            try:
                parsed = parse_json_content(repair_result.content)
                errors = _schema_errors(parsed, schema)
            except (ValueError, json.JSONDecodeError) as exc:
                errors = [str(exc)]
        else:
            errors = ["empty repair content"]
        repair_valid = not errors
    # Persist only the final model text and sanitized provider envelope.
    write_json(paths["content"], {"request_id": row["request_id"], "provider": provider, "content": final_result.content})
    if parsed is not None and not errors:
        write_json(paths["parsed"], parsed)
    state = _result_state(
        final_result, row=row, response_mode=response_mode, parsed=parsed is not None,
        schema_errors=errors, repair_attempted=repair_attempted, repair_valid=repair_valid,
    )
    # Raw is primary response by design.  Record that a repair envelope exists
    # without overwriting the original response.
    state["raw_sanitized_present"] = sanitized is not None
    state["repair_sanitized_present"] = final_sanitized is not None if repair_attempted else False
    write_json(paths["state"], state)
    return state


def _locked_qwen_mode() -> str:
    protocol = os.environ.get("U3_PROTOCOL")
    if protocol:
        lock_path = Path(protocol) / "model_execution_lock.json"
        if lock_path.is_file():
            mode = read_json(lock_path).get("qwen", {}).get("response_mode")
            if mode in {"strict_json_schema", "json_object_local_validation"}:
                return str(mode)
    return "strict_json_schema"


def run_requests(
    *, provider: str, request_table: Path, model: str, concurrency: int, max_output_tokens: int,
    timeout: int, network_retries: int, output_root: Path, status_table: Path, usage_table: Path,
    schema: Path | None = None, repair_limit: int = 0, expected_count: int | None = None,
) -> dict[str, Any]:
    rows = read_csv(request_table, delimiter="\t")
    expected = expected_count if expected_count is not None else (9 if provider == "qwen" else 3)
    if len(rows) != expected:
        raise ValueError(f"{provider} table must contain {expected} rows")
    actual_schema = read_json(schema) if schema else None
    if provider == "deepseek" and actual_schema is None:
        raise ValueError("DeepSeek execution requires local strict schema")
    if provider == "qwen" and actual_schema is None:
        protocol = os.environ.get("U3_REPAIR")
        actual_schema = read_json(Path(protocol) / "configs" / "schema_strict.json") if protocol else None
    if actual_schema is None:
        raise ValueError("strict schema is required")
    response_mode = _locked_qwen_mode() if provider == "qwen" else "json_object_local_validation"
    states: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(int(concurrency), expected))) as executor:
        futures = [executor.submit(
            _run_one, row=row, provider=provider, model=model, schema=actual_schema, output_root=output_root,
            max_output_tokens=max_output_tokens, timeout=timeout, network_retries=network_retries,
            response_mode=response_mode, repair_limit=repair_limit,
        ) for row in rows]
        for future in as_completed(futures):
            states.append(future.result())
    states.sort(key=lambda row: row["request_id"])
    write_csv(status_table, states, delimiter="\t")
    usage = [{key: row.get(key) for key in ("request_id", "condition", "replicate", "input_tokens", "output_tokens", "reasoning_tokens", "latency_seconds", "returned_model", "response_id", "network_retries")} for row in states]
    write_csv(usage_table, usage)
    return {"provider": provider, "response_mode": response_mode, "expected": expected, "completed": len(states), "http_success": sum(bool(row["http_success"]) for row in states), "schema_valid": sum(bool(row["schema_valid"]) for row in states)}


def verify_provider_run(*, provider: str, status: Path, response_root: Path, expected: int, schema: Path, output: Path, report: Path) -> dict[str, Any]:
    rows = read_csv(status, delimiter="\t")
    actual_schema = read_json(schema)
    failures: list[str] = []
    for row in rows:
        candidate = response_root / "parsed" / f"{row['request_id']}.json"
        if not candidate.is_file():
            failures.append(f"{row['request_id']}: missing parsed candidate")
            continue
        try:
            parsed = read_json(candidate)
            errors = _schema_errors(parsed, actual_schema)
            if errors:
                failures.append(f"{row['request_id']}: schema invalid: {errors[0]}")
        except Exception as exc:
            failures.append(f"{row['request_id']}: {exc}")
        if str(row.get("http_success", "")).lower() not in {"true", "1"}:
            failures.append(f"{row['request_id']}: HTTP failure")
    summary = {
        "provider": provider, "expected": expected, "status_rows": len(rows),
        "http_success": sum(str(row.get("http_success", "")).lower() in {"true", "1"} for row in rows),
        "content_nonempty": sum(str(row.get("content_nonempty", "")).lower() in {"true", "1"} for row in rows),
        "schema_valid": sum(str(row.get("schema_valid", "")).lower() in {"true", "1"} for row in rows),
        "parsed_files": sum((response_root / "parsed" / f"{row['request_id']}.json").is_file() for row in rows),
        "status": "PASS" if len(rows) == expected and not failures else "FAIL", "failures": failures,
    }
    write_json(output, summary)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Provider run verification\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in summary.items() if key != "failures") + ("\n\n## Failures\n\n" + "\n".join(f"- {item}" for item in failures) if failures else "\n"), encoding="utf-8")
    return summary


def provider_smoke(*, provider: str, model: str, schema: Path, output: Path, log: Path) -> dict[str, Any]:
    actual_schema = read_json(schema)
    minimal = {
        "schema_version": "u3_candidate_graph_v1", "graph_id": "smoke_candidate", "scope": "stochastic_simulator_only",
        "nodes": [
            {"id": "start", "description": "observable start", "role": "start", "observable_predicates": ["contact_absent"], "unknown_conditions": [], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"},
            {"id": "middle", "description": "observable contact", "role": "intermediate", "observable_predicates": ["contact_present"], "unknown_conditions": [], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"},
            {"id": "success", "description": "observable goal", "role": "success_terminal", "observable_predicates": ["object_inside_goal"], "unknown_conditions": [], "source_cluster_ids": [], "evidence_segment_ids": [], "status": "hypothesized"},
        ],
        "edges": [
            {"id": "e1", "src": "start", "dst": "middle", "preconditions": ["contact_absent"], "effects": ["contact_present"], "hypothesized_type": "forward", "source_transition_pairs": [], "evidence_segment_ids": [], "unknown_conditions": [], "cost_measurement_needed": True, "status": "hypothesized"},
            {"id": "e2", "src": "middle", "dst": "success", "preconditions": ["contact_present"], "effects": ["object_inside_goal"], "hypothesized_type": "forward", "source_transition_pairs": [], "evidence_segment_ids": [], "unknown_conditions": [], "cost_measurement_needed": True, "status": "hypothesized"},
        ],
        "unresolved_questions": [],
    }
    messages = [{"role": "system", "content": "Return only one non-empty JSON object conforming to the supplied schema. Do not explain reasoning."}, {"role": "user", "content": json.dumps({"schema": actual_schema, "minimal_shape_example": minimal}, ensure_ascii=False)}]
    if provider == "qwen":
        result, sanitized = call_qwen(request_id="qwen_smoke", model=model, messages=messages, schema=actual_schema, response_mode="strict_json_schema", max_output_tokens=5000, timeout=300, network_retries=0)
        mode = "strict_json_schema"
        strict_fallback_reason: str | None = None
        if result.error_type is not None and is_qwen_schema_format_error(result):
            result, sanitized = call_qwen(request_id="qwen_smoke", model=model, messages=messages, schema=actual_schema, response_mode="json_object_local_validation", max_output_tokens=5000, timeout=300, network_retries=0)
            mode = "json_object_local_validation"
            strict_fallback_reason = "endpoint_rejected_json_schema"
        elif result.error_type is None:
            try:
                strict_errors = _schema_errors(parse_json_content(result.content), actual_schema)
            except Exception as exc:
                strict_errors = [str(exc)]
            # An endpoint which accepts a strict-format request but returns an
            # invalid object cannot satisfy the strict-mode gate.  Select the
            # documented uniform json_object fallback before any main request.
            if strict_errors:
                result, sanitized = call_qwen(request_id="qwen_smoke", model=model, messages=messages, schema=actual_schema, response_mode="json_object_local_validation", max_output_tokens=5000, timeout=300, network_retries=0)
                mode = "json_object_local_validation"
                strict_fallback_reason = "strict_response_not_locally_schema_valid"
    elif provider == "deepseek":
        result, sanitized = call_deepseek(request_id="deepseek_smoke", model=model, messages=messages, max_output_tokens=5000, timeout=300, network_retries=0)
        mode = "json_object_local_validation"
    else:
        raise ValueError(f"unsupported provider: {provider}")
    errors: list[str] = []
    if result.error_type is None:
        try:
            errors = _schema_errors(parse_json_content(result.content), actual_schema)
        except Exception as exc:
            errors = [str(exc)]
    else:
        errors = [result.error_message or result.error_type]
    value = {
        "provider": provider, "requested_model": model, "returned_model": result.returned_model,
        "response_mode": mode, "http_success": result.error_type is None, "content_nonempty": bool(result.content.strip()),
        "schema_valid": not errors, "errors": errors, "sanitized_response": sanitized,
    }
    if provider == "qwen":
        value["strict_fallback_reason"] = strict_fallback_reason
    write_json(output, value)
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text("provider=%s\nresponse_mode=%s\nhttp_success=%s\nschema_valid=%s\n" % (provider, mode, value["http_success"], value["schema_valid"]), encoding="utf-8")
    if not value["http_success"] or errors:
        raise RuntimeError(f"{provider} smoke failed: {errors[0] if errors else 'unknown error'}")
    return value
