from __future__ import annotations
import csv, hashlib, json
from pathlib import Path

def sha256_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()

def require_new(p: Path) -> Path:
    if p.exists():
        raise FileExistsError(p)
    p.mkdir(parents=True, exist_ok=False)
    return p

def write_json(p: Path, v) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(json.dumps(v, ensure_ascii=False, indent=2, allow_nan=False)+"\n", encoding="utf-8")

def write_text(p: Path, t: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(p)
    p.write_text(t if t.endswith("\n") else t+"\n", encoding="utf-8")

def write_csv(p: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError("empty table")
    if p.exists():
        raise FileExistsError(p)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open("x", encoding="utf-8", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)

def write_jsonl(p: Path, rows: list) -> None:
    if p.exists():
        raise FileExistsError(p)
    with p.open("x", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False, allow_nan=False)+"\n")