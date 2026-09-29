"""S1-REV1 integration primitives for budget receipts and evidence loading."""
from __future__ import annotations

import csv
import fcntl
import hashlib
import json
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Any


RECEIPT_VERSION = "s1_branch_receipt_v1"
GATE_TABLES = (
    "relation_adjudications",
    "alignment_records",
    "gradient_records",
    "witness_records",
    "consequence_comparisons",
    "utility_records",
)


class IntegrationError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, sort_keys=True, indent=2, ensure_ascii=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
        try:
            fd = os.open(path.parent, os.O_DIRECTORY)
            os.fsync(fd)
            os.close(fd)
        except OSError:
            pass
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _append_jsonl(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def _transaction_lock(output_dir: Path):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / ".s1_branch_transaction.lock"
    with path.open("a+") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def _read_json(path: Path) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise IntegrationError(f"MISSING_JSON:{path}") from exc


def _registration(output_dir: Path, branch_id: str) -> tuple[Path, dict, dict]:
    path = Path(output_dir) / "witnesses/e4_branch_registration.json"
    doc = _read_json(path)
    branch = next((item for item in doc.get("branches", []) if item.get("branch_id") == branch_id), None)
    if not isinstance(branch, dict):
        raise IntegrationError("BRANCH_NOT_REGISTERED")
    return path, doc, branch


def _manifest_path(output_dir: Path, branch: dict) -> Path:
    return Path(branch.get("manifest_path") or Path(output_dir) / "input_binding/T_A_s1_rev1_runtime_manifest.yaml")


def _receipt_path(output_dir: Path, branch_id: str) -> Path:
    return Path(output_dir) / "witnesses/branch_receipts" / f"{branch_id}.json"


def _attempt_path(output_dir: Path) -> Path:
    return Path(output_dir) / "attempt_registry.json"


def _load_attempts(output_dir: Path) -> dict:
    path = _attempt_path(output_dir)
    return _read_json(path) if path.is_file() else {}


def _save_attempts(output_dir: Path, attempts: dict) -> None:
    _atomic_json(_attempt_path(output_dir), attempts)


def _event(output_dir: Path, branch_id: str, status: str, **fields) -> None:
    _append_jsonl(
        Path(output_dir) / "budget_events.jsonl",
        {
            "event_type": "branch_attempt",
            "branch_id": branch_id,
            "status": status,
            "timestamp_utc": __import__("time").strftime("%Y-%m-%dT%H:%M:%SZ", __import__("time").gmtime()),
            **fields,
        },
    )


def _receipt_identity(output_dir: Path, branch_id: str, branch: dict, registration_path: Path, manifest_path: Path) -> dict:
    return {
        "receipt_version": RECEIPT_VERSION,
        "plan_id": str(branch.get("plan_id") or branch_id),
        "attempt_id": str(branch.get("attempt_id") or branch_id),
        "branch_id": branch_id,
        "registration_sha256": sha256_file(registration_path),
        "source_manifest_sha256": sha256_file(manifest_path),
        "budget_key": "physical_witness_episodes",
    }


def reserve_branch_attempt(output_dir: Path, branch_id: str) -> dict:
    """Reserve exactly one physical slot and create an immutable-bound receipt."""
    output_dir = Path(output_dir)
    with _transaction_lock(output_dir):
        registration_path, _, branch = _registration(output_dir, branch_id)
        if branch.get("authorized") is not True or branch.get("execute_now") is not True:
            raise IntegrationError("EXPLICIT_BRANCH_AUTHORIZATION_REQUIRED")
        manifest_path = _manifest_path(output_dir, branch)
        if not manifest_path.is_file():
            raise IntegrationError("SOURCE_MANIFEST_MISSING")
        ledger_path = output_dir / "budget_ledger.json"
        if not ledger_path.is_file():
            raise IntegrationError("BUDGET_LEDGER_MISSING")
        ledger = _read_json(ledger_path)
        item = ledger.get("physical_witness_episodes")
        if not isinstance(item, dict):
            raise IntegrationError("BUDGET_KEY_MISSING")
        attempts = _load_attempts(output_dir)
        prior = attempts.get(branch_id)
        if prior in {"RESERVED", "STARTED", "COMPLETED", "FAILED", "UNKNOWN"}:
            raise IntegrationError(f"DUPLICATE_ATTEMPT:{prior}")
        receipt_path = _receipt_path(output_dir, branch_id)
        if receipt_path.exists():
            raise IntegrationError("RECEIPT_ALREADY_EXISTS")
        used = int(item.get("used", 0))
        cap = int(item.get("cap", 0))
        if used >= cap:
            raise IntegrationError("BUDGET_EXHAUSTED")
        item = dict(item)
        item["used"] = used + 1
        ledger["physical_witness_episodes"] = item
        _atomic_json(ledger_path, ledger)
        identity = _receipt_identity(output_dir, branch_id, branch, registration_path, manifest_path)
        receipt = {
            **identity,
            "reservation_id": canonical_hash({**identity, "old_used": used, "new_used": item["used"]}),
            "state": "RESERVED",
            "old_used": used,
            "new_used": item["used"],
            "cap": cap,
            "source_manifest_path": str(manifest_path),
        }
        _atomic_json(receipt_path, receipt)
        attempts[branch_id] = "RESERVED"
        _save_attempts(output_dir, attempts)
        _event(output_dir, branch_id, "RESERVED", reservation_id=receipt["reservation_id"], old_used=used, new_used=item["used"], cap=cap)
        return receipt


def claim_branch_attempt(output_dir, branch_id) -> dict:
    """Atomically consume a RESERVED receipt exactly once before runtime construction."""
    output_dir = Path(output_dir)
    with _transaction_lock(output_dir):
        registration_path, _, branch = _registration(output_dir, branch_id)
        receipt_path = _receipt_path(output_dir, branch_id)
        if not receipt_path.is_file():
            raise IntegrationError("BRANCH_RECEIPT_MISSING")
        receipt = _read_json(receipt_path)
        if receipt.get("state") != "RESERVED":
            raise IntegrationError(f"RECEIPT_NOT_CLAIMABLE:{receipt.get('state')}")
        manifest_path = _manifest_path(output_dir, branch)
        expected = _receipt_identity(output_dir, branch_id, branch, registration_path, manifest_path)
        for key, value in expected.items():
            if receipt.get(key) != value:
                raise IntegrationError(f"RECEIPT_BINDING_MISMATCH:{key}")
        attempts = _load_attempts(output_dir)
        if attempts.get(branch_id) != "RESERVED":
            raise IntegrationError("ATTEMPT_NOT_RESERVED")
        receipt = {**receipt, "state": "STARTED"}
        _atomic_json(receipt_path, receipt)
        attempts[branch_id] = "STARTED"
        _save_attempts(output_dir, attempts)
        _event(output_dir, branch_id, "STARTED", reservation_id=receipt["reservation_id"])
        return receipt


def finish_branch_attempt(output_dir, branch_id, state: str, **fields) -> dict:
    output_dir = Path(output_dir)
    if state not in {"COMPLETED", "FAILED", "UNKNOWN"}:
        raise IntegrationError("INVALID_ATTEMPT_FINAL_STATE")
    with _transaction_lock(output_dir):
        path = _receipt_path(output_dir, branch_id)
        if not path.is_file():
            raise IntegrationError("BRANCH_RECEIPT_MISSING")
        receipt = _read_json(path)
        if receipt.get("state") not in {"STARTED", "RESERVED"}:
            return receipt
        receipt = {**receipt, "state": state, **fields}
        _atomic_json(path, receipt)
        attempts = _load_attempts(output_dir)
        attempts[branch_id] = state
        _save_attempts(output_dir, attempts)
        _event(output_dir, branch_id, state, reservation_id=receipt.get("reservation_id"), **fields)
        return receipt


def restore_receipt_valid(bundle: object, branch: dict) -> tuple[bool, str]:
    receipt = getattr(bundle, "restore_receipt", None)
    if not isinstance(receipt, dict):
        return False, "RESTORE_RECEIPT_MISSING"
    required = (
        "requested_case_id", "applied_case_id", "requested_restore_seed", "applied_restore_seed",
        "normalized_reset_config_sha256", "runtime_source_sha256", "task_contract_identity",
        "state_identity_kind", "state_identity_sha256", "public_facts_sha256",
        "candidate_ids_sha256", "candidate_mask_sha256", "restore_checks",
    )
    if any(not receipt.get(key) and receipt.get(key) != 0 for key in required):
        return False, "RESTORE_RECEIPT_INCOMPLETE"
    if receipt["requested_case_id"] != branch.get("case_id") or receipt["applied_case_id"] != branch.get("case_id"):
        return False, "RESTORE_CASE_MISMATCH"
    if int(receipt["requested_restore_seed"]) != int(branch.get("restore_seed")):
        return False, "RESTORE_SEED_REQUEST_MISMATCH"
    if int(receipt["applied_restore_seed"]) != int(branch.get("restore_seed")):
        return False, "RESTORE_SEED_APPLIED_MISMATCH"
    checks = receipt["restore_checks"]
    if not isinstance(checks, dict) or not checks or not all(value is True for value in checks.values()):
        return False, "RESTORE_CHECKS_INCOMPLETE"
    if getattr(bundle, "restore_verified", False) is not True:
        return False, "RESTORE_NOT_VERIFIED"
    if getattr(bundle, "snapshot_identity", None) != receipt["state_identity_sha256"]:
        return False, "RESTORE_STATE_IDENTITY_MISMATCH"
    return True, "PASS"


def _load_table(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return [dict(row) for row in csv.DictReader(handle)]
    value = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict) and isinstance(value.get("rows"), list):
        return [row for row in value["rows"] if isinstance(row, dict)]
    raise IntegrationError(f"EVIDENCE_TABLE_NOT_ROWS:{path}")


def load_gate_evidence_bundle(root: Path, evidence_manifest: Path | dict) -> dict:
    root = Path(root).resolve()
    manifest_path = None
    if isinstance(evidence_manifest, (str, Path)):
        manifest_path = Path(evidence_manifest).resolve()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = dict(evidence_manifest)
    if manifest.get("schema_version") != "s1_gate_evidence_bundle_v1":
        raise IntegrationError("EVIDENCE_SCHEMA_MISMATCH")
    tables = manifest.get("tables", {})
    bundle = {
        "source_hashes": {},
        "evidence_context": manifest.get("evidence_context", "RESEARCH"),
        "scientific_admissible": manifest.get("evidence_context") == "RESEARCH",
        "missing_evidence_tables": [],
        "hash_mismatches": [],
        "evidence_manifest_sha256": sha256_file(manifest_path) if manifest_path and manifest_path.is_file() else "",
    }
    for name in GATE_TABLES:
        entry = tables.get(name)
        if not isinstance(entry, dict) or not entry.get("path"):
            bundle[name] = []
            bundle["missing_evidence_tables"].append(name)
            continue
        path = Path(entry["path"])
        if not path.is_absolute():
            path = (manifest_path.parent / path if manifest_path else root / path).resolve()
        if not path.is_file():
            bundle[name] = []
            bundle["missing_evidence_tables"].append(name)
            continue
        actual = sha256_file(path)
        bundle["source_hashes"][name] = actual
        if entry.get("sha256") != actual:
            bundle["hash_mismatches"].append(name)
            bundle[name] = []
            continue
        bundle[name] = _load_table(path)
    return bundle
