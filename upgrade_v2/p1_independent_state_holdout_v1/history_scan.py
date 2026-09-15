"""Refuse to generate if proposed family/seed IDs already exist in historical artifacts."""
from __future__ import annotations
import json, re
from pathlib import Path

PROPOSED_FAMILIES = ["990100", "990101", "990102", "990103"]
PROPOSED_SEEDS = [99110000, 99110100, 99110200, 99110300]
PROPOSED_PREFIXES = ["P1HOLD_F990100", "P1HOLD_F990101", "P1HOLD_F990102", "P1HOLD_F990103"]
TOKENS = PROPOSED_FAMILIES + [str(s) for s in PROPOSED_SEEDS] + PROPOSED_PREFIXES
PATTERNS = [re.compile(rf"\b{re.escape(x)}\b") for x in TOKENS]


def scan(repo: Path) -> dict:
    repo = Path(repo)
    hits = []
    roots = [repo / "artifacts"]
    for extra in (
        Path("/home/__compress_data/xushijie/graph_pathgraph_p1_v5_data_v6"),
        Path("/home/__compress_data/xushijie/graph_pathgraph_p1_v6gm1_data"),
    ):
        if extra.exists():
            roots.append(extra)
    for root in roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            if "independent_state_holdout_v1" in path.as_posix():
                continue
            if path.suffix.lower() not in {".csv", ".json", ".jsonl", ".md", ".txt", ".tsv"}:
                continue
            if path.stat().st_size > 8_000_000:
                continue
            try:
                text = path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat, label in zip(PATTERNS, TOKENS):
                if pat.search(text):
                    hits.append({"path": str(path), "token": label})
                    break
    return {
        "proposed_families": PROPOSED_FAMILIES,
        "proposed_seeds": PROPOSED_SEEDS,
        "hits": hits,
        "collision": bool(hits),
    }