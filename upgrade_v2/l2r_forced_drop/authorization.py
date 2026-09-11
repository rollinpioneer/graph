from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class AuthorizationDenied(RuntimeError):
    pass


@dataclass(frozen=True)
class Authorization:
    stage: str
    authorized_instances: int
    protocol_sha256: str
    runner_commit: str
    runner_file_hashes: dict[str, str]
    single_use_nonce: str
    output_root: str
    expires_at_utc: str


def _load(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AuthorizationDenied(f"authorization unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise AuthorizationDenied("authorization must be an object")
    return value


def validate_authorization(path: Path, *, expected_stage: str, expected_protocol_sha256: str, current_commit: str | None = None) -> Authorization:
    data = _load(path)
    errors: list[str] = []
    if data.get("status") != "AUTHORIZED": errors.append("status")
    if data.get("stage") != expected_stage: errors.append("stage")
    if data.get("protocol_sha256") != expected_protocol_sha256: errors.append("protocol_sha256")
    if current_commit is not None and data.get("runner_commit") != current_commit: errors.append("runner_commit")
    if data.get("agent_self_authorization_prohibited") is not True: errors.append("self_authorization_guard")
    if type(data.get("authorized_instances")) is not int or data["authorized_instances"] <= 0: errors.append("authorized_instances")
    for key in ("runner_commit", "runner_file_hashes", "single_use_nonce", "output_root", "expires_at_utc"):
        if not data.get(key): errors.append(key)
    try:
        expires = datetime.fromisoformat(str(data.get("expires_at_utc")).replace("Z", "+00:00"))
        if expires <= datetime.now(timezone.utc): errors.append("expired")
    except (TypeError, ValueError):
        errors.append("expiry_format")
    if errors:
        raise AuthorizationDenied("authorization denied: " + ", ".join(errors))
    return Authorization(data["stage"], data["authorized_instances"], data["protocol_sha256"], data["runner_commit"], data["runner_file_hashes"], data["single_use_nonce"], data["output_root"], data["expires_at_utc"])


def protocol_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
