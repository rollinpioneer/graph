from __future__ import annotations

import hashlib
import importlib
import os
import platform
import sys
from pathlib import Path
from typing import Any

from .protocol import REQUIRED_ENVIRONMENT, canonical_hash

MAIN_GATE_FIELDS = (
    "python", "python_executable", "platform_system", "platform_release", "platform_machine",
    "numpy_version", "numpy_native_hashes", "mujoco_version", "mujoco_native_hashes",
    "opencv_version", "opencv_native_hashes", "critical_environment", "renderer_backend",
)


def validate_process_environment() -> None:
    missing = [key for key, expected in REQUIRED_ENVIRONMENT.items() if os.environ.get(key) != expected and key.isupper()]
    if missing:
        raise RuntimeError("ENVIRONMENT_CONTRACT_MISMATCH: " + ", ".join(missing))
    if sys.version_info[:3] != tuple(int(part) for part in REQUIRED_ENVIRONMENT["python"].split(".")):
        raise RuntimeError("ENVIRONMENT_CONTRACT_MISMATCH: python")


def _native_hashes(module: Any) -> dict[str, str | None]:
    paths: list[Path] = []
    module_file = getattr(module, "__file__", None)
    if module_file:
        paths.append(Path(module_file))
    package_path = getattr(module, "__path__", ())
    for root in package_path:
        paths.extend(sorted(Path(root).glob("*.so")))
        paths.extend(sorted(Path(root).glob("*.cpython-*.so")))
    result: dict[str, str | None] = {}
    for path in sorted({p.resolve() for p in paths if p.is_file()}):
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        result[str(path)] = digest
    return result


def runtime_fingerprint(*, mujoco: Any, cv2: Any, numpy: Any) -> dict[str, Any]:
    critical = {key: os.environ.get(key) for key in REQUIRED_ENVIRONMENT if key.isupper()}
    renderer_backend = os.environ.get("MUJOCO_GL", "UNSET")
    return {
        "schema": "l2rar2_r14b_runtime_environment_fingerprint_v1",
        "python": platform.python_version(),
        "python_executable": str(Path(sys.executable).resolve()),
        "platform_system": platform.system(),
        "platform_release": platform.release(),
        "platform_machine": platform.machine(),
        "numpy_version": str(numpy.__version__),
        "numpy_native_hashes": _native_hashes(numpy),
        "mujoco_version": str(mujoco.__version__),
        "mujoco_native_hashes": _native_hashes(mujoco),
        "opencv_version": str(cv2.__version__),
        "opencv_native_hashes": _native_hashes(cv2),
        "critical_environment": critical,
        "renderer_backend": renderer_backend,
        "main_gate_fields": list(MAIN_GATE_FIELDS),
        "fingerprint_sha256": None,
    }


def finalize_fingerprint(value: dict[str, Any]) -> dict[str, Any]:
    result = dict(value)
    result["fingerprint_sha256"] = canonical_hash({key: result[key] for key in result if key != "fingerprint_sha256"})
    return result


def compare_main_gate_fields(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    mismatches = []
    for field in MAIN_GATE_FIELDS:
        if left.get(field) != right.get(field):
            mismatches.append({"field": field, "left": left.get(field), "right": right.get(field)})
    return {"passed": not mismatches, "mismatches": mismatches}
