"""Small dependency-free validation helpers. Never imports a simulator or model."""
from __future__ import annotations
import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

BASE_COMMIT = "fdc5acd1bbec7906fb1b2dffb091d2c82979d033"

def number(x: Any, label: str = "value") -> float:
    if isinstance(x, bool):
        raise ValueError(f"{label}: boolean is not a number")
    try:
        out = float(x)
    except (TypeError, ValueError) as e:
        raise ValueError(f"{label}: expected finite number") from e
    if not math.isfinite(out):
        raise ValueError(f"{label}: non-finite")
    return out

def integer(x: Any, label: str = "value") -> int:
    # Nanosecond clocks must never round-trip through a float.
    if isinstance(x, bool):
        raise ValueError(f"{label}: boolean is not an integer")
    if isinstance(x, int):
        return x
    if isinstance(x, str) and re.fullmatch(r"[+-]?[0-9]+", x):
        return int(x)
    raise ValueError(f"{label}: exact integer or integer string required")

def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1024 * 1024), b""):
            h.update(b)
    return h.hexdigest()

def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> list[dict]:
    out = []
    with path.open(encoding="utf-8") as f:
        for line_no, text in enumerate(f, 1):
            if not text.strip():
                continue
            value = json.loads(text)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_no}: object required")
            out.append(value)
    return out

def write_new(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, allow_nan=False)
        f.write("\n")

def ratio(k: int, n: int) -> float | None:
    return k / n if n else None
