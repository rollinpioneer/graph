from __future__ import annotations
import csv, hashlib, json, os, shutil, time
from pathlib import Path

def digest(path: Path) -> str:
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1<<20), b''): h.update(b)
    return h.hexdigest()

def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True)+'\n')

def write_csv(path: Path, rows, fields=None, delimiter=','):
    rows=list(rows); path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None: fields=list(rows[0]) if rows else []
    with path.open('w', newline='') as f:
        w=csv.DictWriter(f, fieldnames=fields, delimiter=delimiter); w.writeheader(); w.writerows(rows)

def safe_rel(path: Path, root: Path) -> str:
    return str(path.resolve().relative_to(root.resolve()))
