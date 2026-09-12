from __future__ import annotations
from pathlib import Path
import subprocess
from .environment_preflight import frozen_environment, preflight


def classify_attempt(output_root: Path) -> dict[str, object]:
    # model_construction_started is the conservative boundary; a runner may also
    # emit physics_started/mj_step_started when available.
    started = any((output_root / marker).is_file() for marker in (
        "model_construction_started.json", "physics_started.json", "mj_step_started.json"))
    return {
        "setup_attempt": True,
        "grant_consumed": True,
        "physical_instance_started": started,
        "physical_budget_delta": 1 if started else 0,
        "physics_steps_started": started,
    }


def preflight_or_block() -> dict[str, object]:
    result = preflight()
    return {"status": "PASS" if result.ok else "BLOCKED_PREPHYSICS_ENVIRONMENT", "errors": list(result.errors), "environment": result.environment}


def invoke_frozen_runner(command: list[str], *, cwd: Path, grant: dict) -> subprocess.CompletedProcess:
    """Invoke only a caller-supplied frozen runner after grant validation."""
    if not grant.get("authorization_id") or not grant.get("single_use_nonce"):
        raise PermissionError("valid stage grant required")
    check = preflight_or_block()
    if check["status"] != "PASS":
        raise RuntimeError(str(check["status"]) + ":" + ",".join(check["errors"]))
    return subprocess.run(command, cwd=cwd, env=frozen_environment(), check=False, text=True, capture_output=True)
