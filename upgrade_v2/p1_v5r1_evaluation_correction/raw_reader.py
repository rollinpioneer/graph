from __future__ import annotations
import csv, json
from pathlib import Path
from typing import Any

def parse_flag(x):
    if x is None or x=="":
        return None
    if type(x) is bool:
        return x
    if type(x) is int and x in (0,1):
        return bool(x)
    if type(x) is str:
        if x in ("1","true","True"): return True
        if x in ("0","false","False"): return False
    raise ValueError(f"invalid flag {x!r}")

def load_identity(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def load_timeseries(path: Path) -> list[dict[str, Any]]:
    rows=[]
    with path.open(encoding="utf-8", newline="") as f:
        for i,r in enumerate(csv.DictReader(f)):
            r["_row"]=i
            rows.append(r)
    ticks=[int(float(r["tick"])) for r in rows]
    if ticks != list(range(ticks[0], ticks[0]+len(ticks))):
        # still require strictly increasing
        if any(ticks[i] >= ticks[i+1] for i in range(len(ticks)-1)):
            raise ValueError(f"nonmonotonic tick in {path}")
    return rows

def objects_of(row: dict) -> list[str]:
    if "A_x" in row: return ["A","B"]
    if "obj_x" in row: return ["obj"]
    raise ValueError("unknown schema")

def fnum(row, key):
    if key not in row or row[key] in ("", None):
        return None
    return float(row[key])