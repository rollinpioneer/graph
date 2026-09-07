#!/usr/bin/env python3
"""L1V preparation, bounded Qwen execution, review, scoring, and delivery."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import html
import io
import json
import os
import random
import re
import shutil
import socket
import sqlite3
import statistics
import sys
import time
import urllib.error
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from PIL import Image
from jsonschema import Draft202012Validator


PROTOCOL = "l1v_visual_coarse_v1"
MODEL = "qwen3.7-plus"
CONDITIONS = ("T", "V1", "V2")
FORBIDDEN_REQUEST_KEYS = {
    "state_label_local_only",
    "paired_change",
    "expected_behavior",
    "goal_appears_satisfied",
    "object_identity_ambiguous",
    "main_view_occluded",
    "target_blocked",
    "reference_rubric",
    "gold_mode",
    "scenario",
    "phase",
}
SECRET_RE = re.compile(rb"(?:sk-[A-Za-z0-9._-]{16,}|Authorization\s*:\s*Bearer\s+\S+)", re.I)
EXTERNAL_SUFFIXES = {".sqlite", ".db", ".npz", ".npy", ".parquet", ".h5", ".hdf5", ".pt", ".pth", ".ckpt", ".zip"}


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"expected object at {path}:{number}")
        rows.append(value)
    return rows


def write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    actual_fields = fields or (list(rows[0]) if rows else [])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=actual_fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def public_error(exc: BaseException) -> str:
    text = str(exc)
    text = re.sub(r"sk-[A-Za-z0-9._-]{8,}", "[REDACTED]", text)
    text = re.sub(r"Bearer\s+\S+", "Bearer [REDACTED]", text, flags=re.I)
    return text[:500]


def lock_files(root: Path, paths: Iterable[Path], output: Path, extra: dict[str, Any] | None = None) -> dict[str, Any]:
    files = []
    for path in sorted({path.resolve() for path in paths}):
        files.append({"path": str(path), "relative_path": path.relative_to(root.resolve()).as_posix() if root.resolve() in path.parents else path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    payload = {"schema": "l1v_file_lock_v1", "created_at": now_iso(), "files": files, **(extra or {})}
    write_json(output, payload)
    return payload


def verify_lock(path: Path) -> dict[str, Any]:
    lock = read_json(path)
    failures = []
    for item in lock.get("files", []):
        source = Path(item["path"])
        if not source.is_file():
            failures.append(f"missing: {source}")
        elif sha256_file(source) != item["sha256"]:
            failures.append(f"hash mismatch: {source}")
    return {"status": "PASS" if not failures else "FAIL", "lock": str(path), "failures": failures, "file_count": len(lock.get("files", []))}


def normalize_image(source: Path, target: Path) -> dict[str, Any]:
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((1024, 1024), Image.Resampling.LANCZOS)
        target.parent.mkdir(parents=True, exist_ok=True)
        image.save(target, format="JPEG", quality=92, optimize=True, exif=b"")
        width, height = image.size
    return {"path": str(target.resolve()), "sha256": sha256_file(target), "width": width, "height": height, "size_bytes": target.stat().st_size}


def _validate_manifest(rows: list[dict[str, Any]]) -> None:
    if len(rows) != 24:
        raise ValueError(f"expected 24 scene cases, got {len(rows)}")
    families: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        families.setdefault(str(row["root_family_id"]), []).append(row)
        if row.get("source_kind") not in {"photo", "simulator_rgb"}:
            raise ValueError(f"invalid source kind for {row.get('case_id')}")
        for required in ("scene_verified", "upload_allowed", "same_initial_state_views"):
            if row.get(required) is not True:
                raise ValueError(f"{row.get('case_id')} requires {required}=true")
        if len(row.get("views", [])) != 2:
            raise ValueError(f"{row.get('case_id')} must have two views")
    if len(families) != 12 or any(len(items) != 2 for items in families.values()):
        raise ValueError("expected 12 root families with two cases each")
    dev = {key for key, items in families.items() if items[0].get("split") == "dev"}
    confirm = set(families) - dev
    if len(dev) != 3 or len(confirm) != 9:
        raise ValueError(f"expected 3 dev and 9 confirm families, got {len(dev)} and {len(confirm)}")
    for family, items in families.items():
        if len({item["task_instruction"] for item in items}) != 1:
            raise ValueError(f"task changed within family {family}")
        if len({item["split"] for item in items}) != 1:
            raise ValueError(f"split changed within family {family}")


def _request_text(task: str, capabilities: dict[str, Any], schema: dict[str, Any]) -> str:
    public = {
        "task_instruction": task,
        "robot_capabilities": capabilities,
        "output_schema": schema,
        "request": "请根据你实际收到的信息生成当前初始场景的语义粗任务图。未知信息必须显式保留。",
    }
    return json.dumps(public, ensure_ascii=False, sort_keys=True)


def _request_row(case: dict[str, Any], condition: str, images: list[dict[str, Any]], capabilities: dict[str, Any], schema: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    selected = [] if condition == "T" else images[:1] if condition == "V1" else images[:2]
    request_id = f"{prefix}{case['split']}_{case['case_id']}_{condition}"
    row = {
        "request_id": request_id,
        "case_id": case["case_id"],
        "root_family_id": case["root_family_id"],
        "split": case["split"],
        "condition": condition,
        "task_instruction": case["task_instruction"],
        "image_count": len(selected),
        "images": selected,
        "user_text": _request_text(case["task_instruction"], capabilities, schema),
        "requested_model": MODEL,
    }
    leaked = FORBIDDEN_REQUEST_KEYS.intersection(row)
    if leaked:
        raise ValueError(f"forbidden request keys: {sorted(leaked)}")
    row["request_metadata_sha256"] = sha256_bytes(canonical(row))
    return row


def prepare(manifest: Path, images_root: Path, output: Path, code_root: Path) -> dict[str, Any]:
    scenes = read_jsonl(manifest)
    _validate_manifest(scenes)
    schema = read_json(code_root / "configs" / "coarse_graph.schema.json")
    capabilities = read_json(code_root / "configs" / "robot_capabilities.json")
    prepared_scenes = []
    seen_hashes: dict[str, str] = {}
    for scene in scenes:
        normalized_views = []
        for view in scene["views"]:
            source = images_root / view["path"]
            if not source.is_file():
                raise FileNotFoundError(source)
            target = output / "images" / scene["root_family_id"] / f"{scene['case_id']}_{view['view_id']}.jpg"
            normalized = normalize_image(source, target)
            prior = seen_hashes.get(normalized["sha256"])
            if prior:
                raise ValueError(f"duplicate normalized image: {prior} and {target}")
            seen_hashes[normalized["sha256"]] = str(target)
            normalized_views.append({"view_id": view["view_id"], **normalized})
        prepared_scenes.append({**scene, "views": normalized_views})
    write_jsonl(output / "scenes.prepared.jsonl", prepared_scenes)
    requests = [_request_row(scene, condition, scene["views"], capabilities, schema) for scene in prepared_scenes for condition in CONDITIONS]
    dev = [row for row in requests if row["split"] == "dev"]
    confirm = [row for row in requests if row["split"] == "confirm"]
    if len(dev) != 18 or len(confirm) != 54:
        raise ValueError("request denominator mismatch")
    smoke_cases = [scene for scene in prepared_scenes if scene["split"] == "dev" and scene["root_family_id"] == sorted({x["root_family_id"] for x in prepared_scenes if x["split"] == "dev"})[0]]
    smoke = [_request_row(scene, "V1", scene["views"], capabilities, schema, prefix="smoke_") for scene in smoke_cases]
    random.Random(20260907).shuffle(dev)
    random.Random(20260908).shuffle(confirm)
    write_jsonl(output / "requests" / "smoke.jsonl", smoke)
    write_jsonl(output / "requests" / "dev.jsonl", dev)
    write_jsonl(output / "requests" / "confirm.jsonl", confirm)
    audit = {
        "schema": "l1v_request_audit_v1",
        "request_counts": {"smoke": len(smoke), "dev": len(dev), "confirm": len(confirm)},
        "condition_image_counts": {condition: sorted({row["image_count"] for row in requests if row["condition"] == condition}) for condition in CONDITIONS},
        "forbidden_request_keys": sorted(FORBIDDEN_REQUEST_KEYS),
        "reference_path_in_request": False,
        "status": "PASS",
    }
    write_json(output / "request_audit.json", audit)
    tracked = [output / "scenes.prepared.jsonl", output / "request_audit.json", *(output / "requests" / f"{name}.jsonl" for name in ("smoke", "dev", "confirm"))]
    tracked.extend(path for path in (output / "images").rglob("*.jpg"))
    lock = lock_files(output, tracked, output / "prepared.lock.json", {"protocol": PROTOCOL, "model": MODEL, "cases": 24, "families": 12, "views": 48})
    return {"status": "L1V_VISUAL_INPUTS_READY", "cases": 24, "families": 12, "views": 48, "requests": audit["request_counts"], "lock_files": len(lock["files"])}


def seal(prepared: Path, protocol_path: Path, output: Path, confirm: bool, reference: Path | None, pilot_review: Path | None) -> dict[str, Any]:
    verification = verify_lock(prepared / "prepared.lock.json")
    if verification["status"] != "PASS":
        raise RuntimeError(f"prepared lock failed: {verification['failures']}")
    protocol = read_json(protocol_path)
    required = {
        "protocol": PROTOCOL,
        "model": MODEL,
        "call_cap": 80,
        "planned_smoke_calls": 2,
        "planned_dev_calls": 18,
        "planned_confirm_calls": 54,
    }
    for key, value in required.items():
        if protocol.get(key) != value:
            raise ValueError(f"protocol {key} must be {value!r}")
    if not protocol.get("data_upload_approved") or not protocol.get("api_execution_approved"):
        raise ValueError("data upload and API execution approvals are required")
    output.mkdir(parents=True, exist_ok=True)
    files = [prepared / "prepared.lock.json", protocol_path]
    payload: dict[str, Any] = {"schema": "l1v_execution_lock_v1", "protocol": PROTOCOL, "created_at": now_iso(), "prepared_lock_sha256": sha256_file(prepared / "prepared.lock.json"), "protocol_sha256": sha256_file(protocol_path), "confirmation": confirm}
    if confirm:
        if reference is None or pilot_review is None:
            raise ValueError("confirmation requires reference and pilot review")
        pilot = read_json(pilot_review)
        gates = ("allow_confirmation", "actual_image_transport_checked", "reference_complete")
        if not all(pilot.get(key) is True for key in gates) or pilot.get("prompt_changed_after_pilot") is not False:
            raise ValueError("pilot confirmation gate is not satisfied")
        reference_rows = read_csv(reference)
        if len(reference_rows) != 24 or any(row.get("ready_reference") != "1" for row in reference_rows):
            raise ValueError("reference rubric is incomplete")
        files.extend([reference, pilot_review])
        payload.update({"reference_sha256": sha256_file(reference), "pilot_review_sha256": sha256_file(pilot_review)})
    write_json(output / ("confirmation.lock.json" if confirm else "execution.lock.json"), payload)
    lock_files(output, files, output / ("confirmation.files.lock.json" if confirm else "execution.files.lock.json"), {"confirmation": confirm})
    return {"status": "PASS", "confirmation": confirm, "output": str(output)}


def graph_errors(value: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    errors = [f"{'/'.join(str(item) for item in error.absolute_path) or '$'}: {error.message}" for error in Draft202012Validator(schema).iter_errors(value)]
    node_ids = [node.get("id") for node in value.get("nodes", []) if isinstance(node, dict)]
    if len(node_ids) != len(set(node_ids)):
        errors.append("duplicate node id")
    edge_ids = [edge.get("id") for edge in value.get("edges", []) if isinstance(edge, dict)]
    if len(edge_ids) != len(set(edge_ids)):
        errors.append("duplicate edge id")
    valid_nodes = set(node_ids)
    for edge in value.get("edges", []):
        if isinstance(edge, dict) and (edge.get("src") not in valid_nodes or edge.get("dst") not in valid_nodes):
            errors.append(f"dangling edge reference: {edge.get('id')}")
    for group in ("objects", "nodes", "edges"):
        for item in value.get(group, []):
            if isinstance(item, dict) and item.get("status") != "hypothesized":
                errors.append(f"{group} element is not hypothesized")
    return errors


def ledger_connect(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=60)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        "CREATE TABLE IF NOT EXISTS attempts (id INTEGER PRIMARY KEY, request_id TEXT, batch TEXT, attempt INTEGER, started_at TEXT, ended_at TEXT, status TEXT, http_status INTEGER, response_id TEXT, error_type TEXT)"
    )
    connection.commit()
    return connection


def reserve_attempt(path: Path, request_id: str, batch: str, attempt: int, cap: int) -> int:
    connection = ledger_connect(path)
    try:
        connection.execute("BEGIN IMMEDIATE")
        count = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        if count >= cap:
            connection.rollback()
            raise RuntimeError(f"global call cap {cap} reached")
        cursor = connection.execute(
            "INSERT INTO attempts(request_id,batch,attempt,started_at,status) VALUES(?,?,?,?,?)",
            (request_id, batch, attempt, now_iso(), "STARTED"),
        )
        connection.commit()
        return int(cursor.lastrowid)
    finally:
        connection.close()


def finish_attempt(path: Path, row_id: int, status: str, http_status: int | None, response_id: str | None, error_type: str | None) -> None:
    connection = ledger_connect(path)
    try:
        connection.execute(
            "UPDATE attempts SET ended_at=?, status=?, http_status=?, response_id=?, error_type=? WHERE id=?",
            (now_iso(), status, http_status, response_id, error_type, row_id),
        )
        connection.commit()
    finally:
        connection.close()


def ledger_summary(path: Path) -> dict[str, Any]:
    if not path.is_file():
        return {"attempts": 0, "statuses": {}}
    connection = ledger_connect(path)
    try:
        attempts = int(connection.execute("SELECT COUNT(*) FROM attempts").fetchone()[0])
        statuses = {row[0]: int(row[1]) for row in connection.execute("SELECT status, COUNT(*) FROM attempts GROUP BY status")}
        return {"attempts": attempts, "statuses": statuses}
    finally:
        connection.close()


def image_data_url(path: Path) -> str:
    return "data:image/jpeg;base64," + base64.b64encode(path.read_bytes()).decode("ascii")


def endpoint(base_url: str) -> str:
    value = base_url.rstrip("/")
    return value if value.endswith("/chat/completions") else value + "/chat/completions"


def call_http(row: dict[str, Any], system_prompt: str, protocol: dict[str, Any], ledger: Path, batch: str) -> dict[str, Any]:
    key = os.environ.get("QWEN_API_KEY", "")
    base_url = os.environ.get("QWEN_BASE_URL", "")
    if len(key) < 20 or not base_url:
        raise RuntimeError("QWEN_API_KEY and QWEN_BASE_URL must be injected from repository-external secrets")
    content: list[dict[str, Any]] = [{"type": "text", "text": row["user_text"]}]
    for image in row["images"]:
        content.append({"type": "image_url", "image_url": {"url": image_data_url(Path(image["path"]))}})
    payload = {
        "model": MODEL,
        "messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": content}],
        "response_format": {"type": "json_object"},
        "temperature": 0.0,
        "max_tokens": int(protocol["max_tokens"]),
        "enable_thinking": False,
        "stream": False,
    }
    payload_bytes = canonical(payload)
    payload_sha = sha256_bytes(payload_bytes)
    retries = int(protocol.get("network_retry_limit", 1))
    timeout = int(protocol.get("timeout_seconds", 300))
    last: dict[str, Any] | None = None
    for attempt in range(retries + 1):
        claim = reserve_attempt(ledger, row["request_id"], batch, attempt, int(protocol["call_cap"]))
        request = urllib.request.Request(
            endpoint(base_url),
            data=payload_bytes,
            method="POST",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        )
        started = time.monotonic()
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                http_status = int(response.status)
                raw = json.loads(response.read().decode("utf-8"))
            choices = raw.get("choices") or []
            message = (choices[0].get("message") or {}) if choices else {}
            content_text = message.get("content") or ""
            response_id = raw.get("id")
            finish_attempt(ledger, claim, "HTTP_SUCCESS", http_status, response_id, None)
            usage = raw.get("usage") or {}
            return {
                "request_id": row["request_id"],
                "http_success": True,
                "http_status": http_status,
                "requested_model": MODEL,
                "returned_model": raw.get("model"),
                "response_id": response_id,
                "finish_reason": choices[0].get("finish_reason") if choices else None,
                "content": content_text,
                "usage": {"prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens")},
                "latency_seconds": round(time.monotonic() - started, 6),
                "attempt": attempt,
                "payload_sha256": payload_sha,
                "image_count_sent": row["image_count"],
                "image_sha256": [image["sha256"] for image in row["images"]],
            }
        except urllib.error.HTTPError as exc:
            status = int(exc.code)
            finish_attempt(ledger, claim, "PROVIDER_ERROR", status, None, type(exc).__name__)
            last = {"error_type": type(exc).__name__, "error_message": public_error(exc), "http_status": status, "transport_uncertain": False}
            if status not in {429, 500, 502, 503, 504} or attempt >= retries:
                break
            time.sleep(2 ** (attempt + 1))
        except (urllib.error.URLError, TimeoutError, socket.timeout) as exc:
            uncertain = isinstance(exc, (TimeoutError, socket.timeout)) or "timed out" in str(exc).lower()
            finish_attempt(ledger, claim, "TRANSPORT_UNCERTAIN" if uncertain else "PROVIDER_ERROR", None, None, type(exc).__name__)
            last = {"error_type": type(exc).__name__, "error_message": public_error(exc), "http_status": None, "transport_uncertain": uncertain}
            if uncertain or attempt >= retries:
                break
            time.sleep(2 ** (attempt + 1))
    assert last is not None
    return {
        "request_id": row["request_id"],
        "http_success": False,
        "requested_model": MODEL,
        "returned_model": None,
        "response_id": None,
        "content": "",
        "usage": {},
        "latency_seconds": None,
        "attempt": attempt,
        "payload_sha256": payload_sha,
        "image_count_sent": row["image_count"],
        "image_sha256": [image["sha256"] for image in row["images"]],
        **last,
    }


def _execute_one(row: dict[str, Any], batch: str, run_root: Path, prepared: Path, protocol: dict[str, Any], system_prompt: str, schema: dict[str, Any]) -> dict[str, Any]:
    state_path = run_root / "states" / batch / f"{row['request_id']}.json"
    candidate_path = run_root / "candidates" / batch / f"{row['request_id']}.json"
    if state_path.is_file():
        previous = read_json(state_path)
        if previous.get("terminal") is True and previous.get("request_metadata_sha256") == row["request_metadata_sha256"]:
            return {**previous, "resumed": True}
    result = call_http(row, system_prompt, protocol, run_root / "api_ledger.sqlite", batch)
    parsed = None
    errors: list[str] = []
    if result.get("content"):
        try:
            parsed = json.loads(result["content"])
            if not isinstance(parsed, dict):
                errors = ["response is not a JSON object"]
            else:
                errors = graph_errors(parsed, schema)
        except json.JSONDecodeError as exc:
            errors = [f"JSON decode: {exc}"]
    elif result.get("http_success"):
        errors = ["empty response content"]
    if parsed is not None:
        write_json(run_root / "raw_parsed" / batch / f"{row['request_id']}.json", parsed)
        if not errors:
            write_json(candidate_path, parsed)
    state = {
        "schema": "l1v_request_state_v1",
        "request_id": row["request_id"],
        "case_id": row["case_id"],
        "root_family_id": row["root_family_id"],
        "split": row["split"],
        "condition": row["condition"],
        "request_metadata_sha256": row["request_metadata_sha256"],
        "image_count_planned": row["image_count"],
        "image_count_sent": result.get("image_count_sent"),
        "image_sha256": result.get("image_sha256", []),
        "requested_model": result.get("requested_model"),
        "returned_model": result.get("returned_model"),
        "response_id": result.get("response_id"),
        "finish_reason": result.get("finish_reason"),
        "payload_sha256": result.get("payload_sha256"),
        "http_success": result.get("http_success", False),
        "http_status": result.get("http_status"),
        "transport_uncertain": result.get("transport_uncertain", False),
        "content_nonempty": bool(result.get("content")),
        "json_parse_success": parsed is not None,
        "structure_errors": errors,
        "structure_valid": parsed is not None and not errors,
        "usage": result.get("usage", {}),
        "latency_seconds": result.get("latency_seconds"),
        "attempt": result.get("attempt"),
        "error_type": result.get("error_type"),
        "error_message": result.get("error_message"),
        "terminal": True,
    }
    write_json(state_path, state)
    write_json(run_root / "responses_sanitized" / batch / f"{row['request_id']}.json", {key: result.get(key) for key in ("request_id", "requested_model", "returned_model", "response_id", "finish_reason", "usage", "latency_seconds", "http_success", "http_status", "payload_sha256", "image_count_sent", "image_sha256", "error_type", "error_message")})
    return state


def run_batch(prepared: Path, run_root: Path, batch: str, execute: bool, code_root: Path) -> dict[str, Any]:
    if batch not in {"smoke", "dev", "confirm"}:
        raise ValueError(batch)
    required_lock = run_root / ("confirmation.files.lock.json" if batch == "confirm" else "execution.files.lock.json")
    if not required_lock.is_file():
        raise FileNotFoundError(required_lock)
    verified = verify_lock(required_lock)
    if verified["status"] != "PASS":
        raise RuntimeError(f"execution lock failed: {verified['failures']}")
    requests = read_jsonl(prepared / "requests" / f"{batch}.jsonl")
    expected = {"smoke": 2, "dev": 18, "confirm": 54}[batch]
    if len(requests) != expected:
        raise ValueError(f"{batch} expected {expected} requests")
    if not execute:
        return {"status": "PLAN_ONLY", "batch": batch, "request_count": len(requests), "image_counts": {condition: sum(row["image_count"] for row in requests if row["condition"] == condition) for condition in CONDITIONS}}
    if os.environ.get("L1V_API_APPROVED") != "1":
        raise RuntimeError("L1V_API_APPROVED=1 is required for live execution")
    protocol_path = next(Path(item["path"]) for item in read_json(required_lock)["files"] if Path(item["path"]).name == "protocol.json")
    protocol = read_json(protocol_path)
    schema = read_json(code_root / "configs" / "coarse_graph.schema.json")
    system_prompt = (code_root / "prompts" / "system_zh.txt").read_text(encoding="utf-8")
    states = []
    concurrency = max(1, min(int(protocol.get("concurrency", 3)), 3))
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_execute_one, row, batch, run_root, prepared, protocol, system_prompt, schema) for row in requests]
        for future in as_completed(futures):
            states.append(future.result())
    states.sort(key=lambda row: row["request_id"])
    write_csv(run_root / "summaries" / f"{batch}_status.csv", states)
    usage_rows = []
    for state in states:
        usage = state.get("usage") or {}
        usage_rows.append({"request_id": state["request_id"], "condition": state["condition"], "prompt_tokens": usage.get("prompt_tokens"), "completion_tokens": usage.get("completion_tokens"), "total_tokens": usage.get("total_tokens"), "latency_seconds": state.get("latency_seconds"), "response_id": state.get("response_id")})
    write_csv(run_root / "summaries" / f"{batch}_usage.csv", usage_rows)
    summary = {
        "status": "PASS" if all(row["http_success"] for row in states) else "PARTIAL",
        "batch": batch,
        "requests": len(states),
        "http_success": sum(bool(row["http_success"]) for row in states),
        "structure_valid": sum(bool(row["structure_valid"]) for row in states),
        "transport_uncertain": sum(bool(row["transport_uncertain"]) for row in states),
        "ledger": ledger_summary(run_root / "api_ledger.sqlite"),
    }
    write_json(run_root / "summaries" / f"{batch}_summary.json", summary)
    return summary


RATING_FIELDS = ["goal_correct", "objects_correct", "scene_specific", "order_sensible", "checks_observable", "scene_ready"]


def review(prepared: Path, run_root: Path, split: str, review_root: Path) -> dict[str, Any]:
    if split not in {"dev", "confirm"}:
        raise ValueError(split)
    requests = read_jsonl(prepared / "requests" / f"{split}.jsonl")
    random.Random(20260919 if split == "confirm" else 20260918).shuffle(requests)
    mapping = []
    rating_rows = []
    sections = []
    for index, row in enumerate(requests, 1):
        blind_id = f"R{index:03d}"
        candidate_path = run_root / "raw_parsed" / split / f"{row['request_id']}.json"
        state_path = run_root / "states" / split / f"{row['request_id']}.json"
        candidate = candidate_path.read_text(encoding="utf-8") if candidate_path.is_file() else "{}"
        state = read_json(state_path) if state_path.is_file() else {}
        image_tags = "".join(f'<img src="file://{html.escape(image["path"])}" width="320" alt="input view">' for image in row["images"])
        sections.append(f"<section><h2>{blind_id}</h2><p>{html.escape(row['task_instruction'])}</p><div>{image_tags}</div><pre>{html.escape(candidate)}</pre><p>structure_valid={state.get('structure_valid')}</p></section>")
        mapping.append({"blind_id": blind_id, "request_id": row["request_id"], "case_id": row["case_id"], "root_family_id": row["root_family_id"], "condition": row["condition"], "split": split})
        rating_rows.append({"blind_id": blind_id, **{field: "" for field in RATING_FIELDS}, "unsupported_visual_assertion": "", "recovery_branch_conditional": "", "review_status": "", "reviewer": "", "note": ""})
    review_root.mkdir(parents=True, exist_ok=True)
    write_csv(review_root / "blind_mapping.DO_NOT_SHOW_RATER.csv", mapping)
    write_csv(review_root / "ratings.csv", rating_rows)
    (review_root / "review.html").write_text("<!doctype html><meta charset='utf-8'><style>body{font-family:sans-serif;max-width:1100px;margin:auto}section{border-bottom:1px solid #ccc;padding:20px}img{margin:4px}pre{white-space:pre-wrap;background:#eee;padding:12px}</style>" + "".join(sections), encoding="utf-8")
    return {"status": "PASS", "split": split, "items": len(requests), "review_root": str(review_root)}


def _bootstrap(values: list[float], seed: int, repeats: int = 5000) -> tuple[float, float]:
    if not values:
        return float("nan"), float("nan")
    rng = random.Random(seed)
    samples = sorted(statistics.mean(rng.choice(values) for _ in values) for _ in range(repeats))
    return samples[int(.025 * repeats)], samples[min(repeats - 1, int(.975 * repeats))]


def score(review_root: Path, run_root: Path, output: Path, protocol_path: Path) -> dict[str, Any]:
    ratings = read_csv(review_root / "ratings.csv")
    mapping = {row["blind_id"]: row for row in read_csv(review_root / "blind_mapping.DO_NOT_SHOW_RATER.csv")}
    if len(ratings) != 54:
        result = {"status": "EVALUATION_PENDING", "reason": f"expected 54 ratings, got {len(ratings)}"}
        write_json(output / "execution_note.json", result)
        return result
    rows = []
    missing = []
    for rating in ratings:
        identity = mapping.get(rating["blind_id"])
        if identity is None:
            missing.append(rating["blind_id"])
            continue
        if rating.get("review_status") != "complete" or any(rating.get(field) not in {"0", "1"} for field in RATING_FIELDS + ["unsupported_visual_assertion"]):
            missing.append(rating["blind_id"])
            continue
        state_path = run_root / "states" / "confirm" / f"{identity['request_id']}.json"
        state = read_json(state_path) if state_path.is_file() else {}
        row: dict[str, Any] = {**identity}
        for field in RATING_FIELDS + ["unsupported_visual_assertion"]:
            row[field] = int(rating[field])
        row["structure_valid"] = int(bool(state.get("structure_valid")))
        row["endpoint_scene_ready"] = row["scene_ready"] if row["structure_valid"] else 0
        row["reviewer"] = rating.get("reviewer")
        row["note"] = rating.get("note")
        rows.append(row)
    if missing:
        result = {"status": "EVALUATION_PENDING", "missing_or_incomplete": missing}
        write_json(output / "execution_note.json", result)
        return result
    write_csv(output / "case_scores.csv", rows)
    by_family: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        by_family.setdefault((row["root_family_id"], row["condition"]), []).append(row)
    family_rows = []
    for (family, condition), items in sorted(by_family.items()):
        family_rows.append({"root_family_id": family, "condition": condition, "case_count": len(items), "scene_ready_mean": statistics.mean(item["endpoint_scene_ready"] for item in items), "both_scene_ready": int(all(item["endpoint_scene_ready"] for item in items)), "structure_valid_count": sum(item["structure_valid"] for item in items), "unsupported_assertion_count": sum(item["unsupported_visual_assertion"] for item in items)})
    write_csv(output / "family_scores.csv", family_rows)
    summaries = []
    for condition in CONDITIONS:
        items = [row for row in rows if row["condition"] == condition]
        summaries.append({"condition": condition, "cases": len(items), "structure_valid": sum(row["structure_valid"] for row in items), "scene_ready": sum(row["endpoint_scene_ready"] for row in items), "scene_ready_rate": statistics.mean(row["endpoint_scene_ready"] for row in items), "paired_families_ready": sum(row["both_scene_ready"] for row in family_rows if row["condition"] == condition), "unsupported_assertions": sum(row["unsupported_visual_assertion"] for row in items)})
    write_csv(output / "condition_summary.csv", summaries)
    family_map = {(row["root_family_id"], row["condition"]): float(row["scene_ready_mean"]) for row in family_rows}
    families = sorted({row["root_family_id"] for row in family_rows})
    effects = []
    for left, right in (("V1", "T"), ("V2", "T"), ("V2", "V1")):
        values = [family_map[(family, left)] - family_map[(family, right)] for family in families]
        low, high = _bootstrap(values, 20260931 + len(effects))
        effects.append({"comparison": f"{left}-{right}", "families": len(values), "mean_effect": statistics.mean(values), "bootstrap_low": low, "bootstrap_high": high, "bootstrap_resamples": 5000})
    write_csv(output / "paired_effects.csv", effects)
    protocol = read_json(protocol_path)
    threshold = protocol["confirmation_thresholds"]
    summary_map = {row["condition"]: row for row in summaries}
    effect_map = {row["comparison"]: row["mean_effect"] for row in effects}

    def qualifies(condition: str, gain_name: str) -> bool:
        item = summary_map[condition]
        return (
            item["structure_valid"] >= threshold["structure_valid_min"]
            and item["scene_ready"] >= threshold["scene_ready_min"]
            and item["paired_families_ready"] >= threshold["paired_family_ready_min"]
            and item["unsupported_assertions"] <= threshold["unsupported_assertions_max"]
            and effect_map[gain_name] >= threshold["visual_gain_min"]
        )

    if qualifies("V1", "V1-T"):
        decision = "L1V_READY_FOR_REFINEMENT_SINGLE_VIEW"
        source = "V1"
    elif qualifies("V2", "V2-T"):
        decision = "L1V_READY_FOR_REFINEMENT_MULTIVIEW"
        source = "V2"
    elif max(summary_map["V1"]["scene_ready"], summary_map["V2"]["scene_ready"]) >= threshold["scene_ready_min"]:
        decision = "L1V_COARSE_GRAPHS_ONLY_NO_VISUAL_GAIN"
        source = "V1" if summary_map["V1"]["scene_ready"] >= summary_map["V2"]["scene_ready"] else "V2"
    else:
        decision = "L1V_NOT_SUPPORTED_THIS_SETUP"
        source = None
    result = {"status": decision, "coarse_graph_source": source, "condition_summary": summaries, "paired_effects": effects, "reviewer_count": len({row["reviewer"] for row in rows if row.get("reviewer")})}
    write_json(output / "l1v_decision.json", result)
    write_json(output / "execution_note.json", {"status": "PASS", "invalid_outputs_count_as_zero": True, "family_is_statistical_unit": True})
    return result


def zip_info(name: str) -> zipfile.ZipInfo:
    info = zipfile.ZipInfo(name, date_time=(1980, 1, 1, 0, 0, 0))
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o100644 << 16
    return info


def package(source: Path, output: Path, max_file_mb: float) -> dict[str, Any]:
    source = source.resolve()
    output = output.resolve()
    if not source.is_dir() or source in output.parents:
        raise ValueError("invalid source/output")
    limit = int(max_file_mb * 1024 * 1024)
    entries: list[tuple[str, Path]] = []
    omitted = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts or ".git" in path.parts:
            continue
        relative = path.relative_to(source).as_posix()
        lowered = {part.lower() for part in path.relative_to(source).parts}
        if path.name.startswith(".env") or lowered.intersection({"secrets", "secret"}):
            raise RuntimeError(f"secret-like path rejected: {relative}")
        reason = None
        if path.suffix.lower() in EXTERNAL_SUFFIXES:
            reason = "externalized_binary_or_nested_archive"
        elif path.stat().st_size > limit:
            reason = "over_size_threshold"
        if reason:
            omitted.append({"logical_path": relative, "original_path": str(path), "original_filename": path.name, "size_bytes": path.stat().st_size, "sha256": sha256_file(path), "purpose": "L1V runtime or heavy artifact", "recovery_method": "restore the exact original path or rerun the locked command", "reason": reason})
        else:
            content = path.read_bytes()
            if SECRET_RE.search(content):
                raise RuntimeError(f"possible credential in {relative}")
            entries.append((relative, path))
    if not entries:
        raise ValueError("empty package")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".partial")
    sums = []
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative, path in entries:
            content = path.read_bytes()
            archive.writestr(zip_info(relative), content)
            sums.append(f"{sha256_bytes(content)}  {relative}")
        table = io.StringIO()
        fields = ["logical_path", "original_path", "original_filename", "size_bytes", "sha256", "purpose", "recovery_method", "reason"]
        writer = csv.DictWriter(table, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(omitted)
        manifest = table.getvalue().encode("utf-8")
        archive.writestr(zip_info("external_artifacts.tsv"), manifest)
        sums.append(f"{sha256_bytes(manifest)}  external_artifacts.tsv")
        archive.writestr(zip_info("PACKAGE_SHA256SUMS.txt"), "\n".join(sums) + "\n")
    with zipfile.ZipFile(temporary) as archive:
        if archive.testzip() is not None:
            raise RuntimeError("ZIP CRC validation failed")
        for line in archive.read("PACKAGE_SHA256SUMS.txt").decode("utf-8").splitlines():
            digest, relative = line.split("  ", 1)
            if sha256_bytes(archive.read(relative)) != digest:
                raise RuntimeError(f"internal SHA mismatch: {relative}")
    temporary.replace(output)
    digest = sha256_file(output)
    output.with_name(output.name + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    return {"status": "PASS", "zip": str(output), "sha256": digest, "files": len(entries), "externalized": len(omitted)}


def verify_run(prepared: Path, run_root: Path) -> dict[str, Any]:
    failures = []
    prepared_check = verify_lock(prepared / "prepared.lock.json")
    if prepared_check["status"] != "PASS":
        failures.extend(prepared_check["failures"])
    for batch, expected in (("smoke", 2), ("dev", 18), ("confirm", 54)):
        states = list((run_root / "states" / batch).glob("*.json")) if (run_root / "states" / batch).is_dir() else []
        if states and len(states) != expected:
            failures.append(f"{batch}: expected {expected} states, got {len(states)}")
        for path in states:
            state = read_json(path)
            expected_images = {"T": 0, "V1": 1, "V2": 2}[state["condition"]]
            if state.get("image_count_sent") != expected_images:
                failures.append(f"{state['request_id']}: image count mismatch")
    result = {"status": "PASS" if not failures else "FAIL", "failures": failures, "ledger": ledger_summary(run_root / "api_ledger.sqlite")}
    write_json(run_root / "summaries" / "verification.json", result)
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l1v_runner.py")
    sub = parser.add_subparsers(dest="command", required=True)
    command = sub.add_parser("prepare")
    command.add_argument("--manifest", type=Path, required=True)
    command.add_argument("--images", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    command = sub.add_parser("seal")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--confirm", action="store_true")
    command.add_argument("--reference", type=Path)
    command.add_argument("--pilot-review", type=Path)
    command = sub.add_parser("run")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--batch", choices=("smoke", "dev", "confirm"), required=True)
    command.add_argument("--execute", action="store_true")
    command.add_argument("--code-root", type=Path, default=Path(__file__).resolve().parents[1])
    command = sub.add_parser("review")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    command.add_argument("--split", choices=("dev", "confirm"), required=True)
    command.add_argument("--review", type=Path, required=True)
    command = sub.add_parser("score")
    command.add_argument("--review", type=Path, required=True)
    command.add_argument("--run", type=Path, required=True)
    command.add_argument("--metrics", type=Path, required=True)
    command.add_argument("--config", type=Path, required=True)
    command = sub.add_parser("package")
    command.add_argument("--source", type=Path, required=True)
    command.add_argument("--zip", type=Path, required=True)
    command.add_argument("--max-file-mb", type=float, default=200)
    command = sub.add_parser("verify")
    command.add_argument("--prepared", type=Path, required=True)
    command.add_argument("--out", type=Path, required=True)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        if args.command == "prepare":
            result = prepare(args.manifest, args.images, args.out, args.code_root)
        elif args.command == "seal":
            result = seal(args.prepared, args.config, args.out, args.confirm, args.reference, args.pilot_review)
        elif args.command == "run":
            result = run_batch(args.prepared, args.out, args.batch, args.execute, args.code_root)
        elif args.command == "review":
            result = review(args.prepared, args.out, args.split, args.review)
        elif args.command == "score":
            result = score(args.review, args.run, args.metrics, args.config)
        elif args.command == "package":
            result = package(args.source, args.zip, args.max_file_mb)
        elif args.command == "verify":
            result = verify_run(args.prepared, args.out)
        else:  # pragma: no cover
            raise AssertionError(args.command)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if result.get("status") not in {"FAIL", "BLOCKED", "VISION_ENDPOINT_BLOCKED", "PILOT_INTERFACE_BLOCKED"} else 2
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": public_error(exc)}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
