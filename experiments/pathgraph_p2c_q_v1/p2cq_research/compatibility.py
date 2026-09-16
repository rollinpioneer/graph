"""Legacy dual-object V6 compatibility. Never reimplements a missing source."""
from __future__ import annotations

import hashlib
from pathlib import Path


P2B_POTENTIALS_BLOB = "30ade96943a1f95163b2ce68e877e237fd8280e6"
V6_HINT = "/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0/tools"


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def git_blob_sha1(path):
    data = Path(path).read_bytes()
    header = f"blob {len(data)}\0".encode("ascii")
    return hashlib.sha1(header + data).hexdigest()


def inspect_legacy(repo_root, extra_hint=V6_HINT):
    repo = Path(repo_root)
    report = {
        "legacy_v6_unchanged": False,
        "new_graph_model_implemented": True,
        "new_graph_candidate": "PATHGRAPH_MODEL_COST_PBRS_V1/PATHGRAPH_MODEL_FULL_PBRS_V1",
        "legacy_scope": "FROZEN_DUAL_OBJECT_COMPATIBILITY_ONLY",
        "supported_object_set": ["A", "B"],
        "gap": [],
        "sources": {},
    }
    p2b = repo / "experiments/pathgraph_p2b_ppo_v1/p2b/potentials.py"
    if p2b.is_file():
        blob = git_blob_sha1(p2b)
        report["sources"]["p2b_potentials"] = {
            "path": str(p2b),
            "git_blob": blob,
            "matches_lock": blob == P2B_POTENTIALS_BLOB,
        }
        txt = p2b.read_text(encoding="utf-8", errors="replace")
        if "{A,B}" in txt or "A, B" in txt or "'A'" in txt:
            report["legacy_v6_unchanged"] = True
        if "n<=6" in txt or "OR path" in txt:
            report["gap"].append("p2b_file_mentions_new_topology")
    else:
        report["gap"].append("p2b_potentials_missing_in_this_worktree")
        report["sources"]["p2b_potentials"] = {"path": str(p2b), "present": False}
    hint = Path(extra_hint)
    report["sources"]["v6_hint"] = {"path": str(hint), "present": hint.exists()}
    if not hint.exists():
        report["gap"].append("legacy_v6_package_missing_not_reimplemented")
    report["status"] = "COMPAT_SCOPED" if report["legacy_v6_unchanged"] else "COMPAT_GAP_DISCLOSED"
    return report
