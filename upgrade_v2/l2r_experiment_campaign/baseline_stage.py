from __future__ import annotations
from pathlib import Path
import subprocess
def invoke_frozen_runner(command: list[str], *, cwd: Path, grant: dict) -> subprocess.CompletedProcess:
    """Invoke only a caller-supplied frozen runner after grant validation."""
    if not grant.get("authorization_id") or not grant.get("single_use_nonce"):
        raise PermissionError("valid stage grant required")
    return subprocess.run(command, cwd=cwd, check=False, text=True, capture_output=True)
