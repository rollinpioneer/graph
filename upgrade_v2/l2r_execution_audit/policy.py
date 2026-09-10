"""Permanent zero-physics policy for the R12 maintenance audit.

The policy validates the pinned R11 accounting record and records denials in
the shared Git worktree control directory.  It intentionally has no simulator
or scientific dependency and never contains an allow path.
"""
from __future__ import annotations

import fcntl
import hashlib
import json
import os
import re
import subprocess
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SOURCE_COMMIT = "dc51582b99759a6e5a7427e1965a5110c5944ed4"
SOURCE_PATH = "artifacts/pathgraph_sarm/upgrade_v2/loss_observability_l2rar2_r11_v1/physical_execution_accounting.json"
EXPECTED_GIT_BLOB = "c16f62af82b20b5e2e2e6313ec095965ae5cb756"
POLICY_ID = "L2RAR2_R12_ZERO_PHYSICS_AUDIT_V1"
NEW_PHYSICAL_BUDGET = 0


class SafetyError(RuntimeError):
    """The pinned control state cannot be trusted."""


class PhysicalExecutionDenied(SafetyError):
    """The request was refused before a physical dependency is reached."""


def canonical(value: Any) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False).encode("utf-8")


def strict_json(text: str) -> Any:
    def reject_constant(value: str) -> None:
        raise SafetyError(f"non-finite JSON constant: {value}")

    def reject_duplicate(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise SafetyError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    return json.loads(text, parse_constant=reject_constant, object_pairs_hook=reject_duplicate)


def validate_accounting(record: dict[str, Any]) -> None:
    expected = {"instances_used": 40, "maximum_instances": 8, "probe_invocations": 5, "instances_per_invocation": 8}
    for key, value in expected.items():
        if type(record.get(key)) is not int or record[key] != value:
            raise SafetyError(f"R11 accounting mismatch: {key}")
    if record.get("budget_status") != "BUDGET_EXCEEDED_BLOCKED":
        raise SafetyError("R11 blocked state was not preserved")
    for key in ("additional_physical_replay_authorized", "mechanism_claims_allowed", "probe_cached_equivalence"):
        if record.get(key) is not False:
            raise SafetyError(f"R11 restriction mismatch: {key}")


def _git_read(repo: Path, *args: str) -> bytes:
    result = subprocess.run(["git", "-C", str(repo), *args], check=False, capture_output=True, timeout=30)
    if result.returncode:
        raise SafetyError(f"Git read failed: {args[0]}")
    return result.stdout


def verified_accounting(repo: Path) -> tuple[dict[str, Any], bytes]:
    content = _git_read(repo, "show", f"{SOURCE_COMMIT}:{SOURCE_PATH}")
    blob = hashlib.sha1(f"blob {len(content)}\0".encode() + content).hexdigest()
    if blob != EXPECTED_GIT_BLOB:
        raise SafetyError("pinned R11 accounting Git blob differs")
    record = strict_json(content.decode("utf-8"))
    if not isinstance(record, dict):
        raise SafetyError("pinned accounting is not an object")
    validate_accounting(record)
    return record, content


def control_dir(repo: Path) -> Path:
    common = Path(_git_read(repo, "rev-parse", "--git-common-dir").decode().strip())
    if not common.is_absolute():
        common = repo.resolve() / common
    common = common.resolve(strict=True)
    for candidate in (common / "research_controls", common / "research_controls" / POLICY_ID):
        if candidate.is_symlink():
            raise SafetyError("symlink is not accepted for safety registry")
    return common / "research_controls" / POLICY_ID


@contextmanager
def _locked(directory: Path) -> Iterator[None]:
    if not directory.is_dir() or directory.is_symlink():
        raise SafetyError("safety registry is not initialized")
    lock_path = directory / "registry.lock"
    fd = os.open(lock_path, os.O_RDWR | getattr(os, "O_NOFOLLOW", 0))
    try:
        fcntl.flock(fd, fcntl.LOCK_EX)
        yield
    finally:
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)


def _validate_journal(text: str) -> list[dict[str, Any]]:
    if not text or not text.endswith("\n"):
        raise SafetyError("empty or interrupted denial journal")
    rows: list[dict[str, Any]] = []
    previous = "0" * 64
    for line in text.splitlines():
        row = strict_json(line)
        if not isinstance(row, dict):
            raise SafetyError("denial journal record is not an object")
        base = {key: row.get(key) for key in ("sequence", "previous_record_sha256", "payload")}
        if base["sequence"] != len(rows) or base["previous_record_sha256"] != previous:
            raise SafetyError("denial journal chain mismatch")
        digest = hashlib.sha256(canonical(base)).hexdigest()
        if digest != row.get("record_sha256"):
            raise SafetyError("denial journal digest mismatch")
        payload = row.get("payload")
        if not isinstance(payload, dict) or payload.get("policy_id") != POLICY_ID:
            raise SafetyError("denial journal policy mismatch")
        if not rows:
            if payload.get("event") != "legacy_aggregate_import" or payload.get("r11_instances_used") != 40 or payload.get("r11_maximum_instances") != 8 or payload.get("source_git_blob") != EXPECTED_GIT_BLOB or payload.get("r12_new_physical_budget") != 0:
                raise SafetyError("legacy aggregate import mismatch")
        elif payload.get("event") != "physical_request_denied" or payload.get("decision") != "DENY" or payload.get("actual_physical_executions") != 0:
            raise SafetyError("non-denial journal record")
        rows.append(row)
        previous = digest
    return rows


def _envelope(payload: dict[str, Any], sequence: int, previous: str) -> dict[str, Any]:
    base = {"sequence": sequence, "previous_record_sha256": previous, "payload": payload}
    return {**base, "record_sha256": hashlib.sha256(canonical(base)).hexdigest()}


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def reject_physics_now(reason: str = "R12_ZERO_PHYSICS") -> None:
    """Unconditional in-process stop, before any runner/factory/import."""
    raise PhysicalExecutionDenied(reason)


def deny_physical_request(repo: Path, request_id: str, requested_instances: int) -> None:
    """Validate the pinned aggregate, append a denial, and always raise."""
    if not re.fullmatch(r"[A-Za-z0-9_.:-]{1,96}", request_id):
        raise SafetyError("request_id must be a short non-sensitive identifier")
    if type(requested_instances) is not int or requested_instances <= 0:
        raise SafetyError("requested_instances must be positive")
    _, accounting_bytes = verified_accounting(repo)
    if hashlib.sha1(f"blob {len(accounting_bytes)}\0".encode() + accounting_bytes).hexdigest() != EXPECTED_GIT_BLOB:
        raise SafetyError("accounting verification failed")
    directory = control_dir(repo)
    with _locked(directory):
        path = directory / "denial_journal.jsonl"
        if not path.is_file() or path.is_symlink():
            raise SafetyError("denial journal missing; refusing reset")
        rows = _validate_journal(path.read_text(encoding="utf-8"))
        payload = {"event": "physical_request_denied", "policy_id": POLICY_ID, "created_utc": _utc(), "request_id": request_id, "requested_instances": requested_instances, "decision": "DENY", "actual_physical_executions": 0, "reason": "R11_BLOCKED_40_OF_8_AND_R12_ZERO_PHYSICS"}
        row = _envelope(payload, len(rows), rows[-1]["record_sha256"])
        fd = os.open(path, os.O_WRONLY | os.O_APPEND | getattr(os, "O_NOFOLLOW", 0))
        with os.fdopen(fd, "a", encoding="utf-8") as stream:
            stream.write(canonical(row).decode() + "\n")
            stream.flush()
            os.fsync(stream.fileno())
    raise PhysicalExecutionDenied("R12 forbids physical execution; request logged, nothing launched")
