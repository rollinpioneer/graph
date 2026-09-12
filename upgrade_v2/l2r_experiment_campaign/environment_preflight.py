from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass

FROZEN_ENVIRONMENT = {
    "PYTHONHASHSEED": "0",
    "PYTHONNOUSERSITE": "1",
    "MUJOCO_GL": "egl",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "MKL_NUM_THREADS": "1",
    "NUMEXPR_NUM_THREADS": "1",
}


@dataclass(frozen=True)
class Preflight:
    ok: bool
    errors: tuple[str, ...]
    environment: dict[str, str]


def frozen_environment() -> dict[str, str]:
    env = os.environ.copy()
    env.update(FROZEN_ENVIRONMENT)
    return env


def preflight(*, python_executable: str | None = None) -> Preflight:
    executable = python_executable or sys.executable
    errors: list[str] = []
    version = subprocess.run(
        [executable, "-c", "import sys; print('.'.join(map(str, sys.version_info[:3])))"],
        check=False, capture_output=True, text=True, env=frozen_environment(),
    )
    if version.returncode != 0 or version.stdout.strip() != "3.10.19":
        errors.append("python_version")
    for key, value in FROZEN_ENVIRONMENT.items():
        if frozen_environment().get(key) != value:
            errors.append(key)
    # Make the contract active for subsequent setup operations in this process.
    os.environ.update(FROZEN_ENVIRONMENT)
    return Preflight(not errors, tuple(sorted(set(errors))), frozen_environment())
