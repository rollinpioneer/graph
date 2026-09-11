from __future__ import annotations

from pathlib import Path
from typing import Any

from .ordinary import run_ordinary


def run_instrumented(*, repo: Path, protocol: dict[str, Any], output_root: Path) -> dict[str, Any]:
    return run_ordinary(repo=repo, protocol=protocol, output_root=output_root, stage="R14B_INSTRUMENTED_C", instrumented=True)
