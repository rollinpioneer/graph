"""Strict local IO. No training imports."""
from __future__ import annotations
import hashlib, json, math, os
from pathlib import Path

def _pairs(items):
    out = {}
    for k, v in items:
        if k in out:
            raise ValueError(f"duplicate JSON key: {k}")
        out[k] = v
    return out

def _parse_float(s):
    value = float(s)
    if not math.isfinite(value):
        raise ValueError("nonfinite float literal")
    return value

def load_json(path):
    return json.loads(
        Path(path).read_text(encoding="utf-8"),
        object_pairs_hook=_pairs,
        parse_float=_parse_float,
        parse_constant=lambda x: (_ for _ in ()).throw(ValueError(f"nonfinite: {x}")),
    )

def canonical(x):
    return json.dumps(x, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)

def hash_json(x):
    return hashlib.sha256(canonical(x).encode("utf-8")).hexdigest()

def sha256_file(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()

def write_new(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())

def write_text_new(path, text):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    raw = text.encode("utf-8")
    if not raw.endswith(b"\n"):
        raw += b"\n"
    with path.open("xb") as f:
        f.write(raw)
        f.flush()
        os.fsync(f.fileno())

def git_blob_sha1(data: bytes):
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()
