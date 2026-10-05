#!/usr/bin/env python
"""CP-DISR-C1-MECH-CONFIRM-V1 Stage 3: build and freeze the Fresh Confirm suite (pure symbolic; no simulator, no policy, no results exist yet).

    python scripts/c1_fresh_confirm_build.py --root .

Construction attempts (deterministic, namespace -rN) are a design-time device: the first attempt whose dedupe, depth, binding-novelty and geometry audits all pass is
frozen. No model result exists at this point, so nothing can be selected on outcome. Writes the two split files and prep/split_manifest.json,
prep/fresh_confirm_audits.json, prep/fresh_confirm_symbolic_manifest.json. Refuses to overwrite a frozen suite.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from cp_disr import c1_fresh_confirm as F  # noqa: E402
from cp_disr import struct_gen as G  # noqa: E402

PREP = Path("runs/final_master/c1_route_b/mech_confirm_v1/prep")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    a = ap.parse_args()
    root = Path(a.root).resolve()
    if (root / F.QUAL_REL).exists() or (root / F.TEST_REL).exists():
        raise SystemExit("the Fresh Confirm suite is already frozen; it is never regenerated")
    attempts = []
    chosen = None
    for attempt in range(F.MAX_CONSTRUCTION_ATTEMPTS):
        rows, dedupe, audits = F.build(root, attempt)
        geometry = F.geometry_by_level(rows)
        means = [v["mean_target_second_distance"] for v in geometry.values()]
        ok = {"dedupe": dedupe["verdict"] == "PASS", "final_depth": audits["final"]["depth"]["verdict"] == "PASS", "final_novelty": audits["final"]["novelty"]["verdict"] == "PASS",
              "qualification_depth": audits["qualification"]["depth"]["verdict"] == "PASS", "qualification_novelty": audits["qualification"]["novelty"]["verdict"] == "PASS",
              "fresh_geometry_balanced_across_levels": (max(means) - min(means)) < 0.04}
        attempts.append({"attempt": attempt, "checks": ok, "geometry": geometry})
        if all(ok.values()):
            chosen = (attempt, rows, dedupe, audits, geometry)
            break
    if chosen is None:
        raise SystemExit("no construction attempt passed the design audits: %s" % json.dumps(attempts))
    attempt, rows, dedupe, audits, geometry = chosen
    docs = F.split_documents(rows)
    hashes = F.write_documents(root, docs)
    final_rows = [r for r in rows if r["split"] == "test"]
    manifest, contracts, cache = G.build_manifest(root, [dict(r, split="test") for r in final_rows])
    prep = root / PREP
    prep.mkdir(parents=True, exist_ok=True)
    G.write_json(prep / "fresh_confirm_symbolic_manifest.json", manifest)
    counts = {c: {s: len([r for r in rows if r["split"] == s and r["cell"] == c]) for s in ("qualification", "test")} for c in F.CELLS}
    split_manifest = {"card": F.CARD, "suite": F.SUITE, "namespace": F.NAMESPACE, "construction_attempt_used": attempt, "construction_attempts": attempts, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                      "cells": list(F.CELLS), "per_cell": F.PER_CELL, "counts": {"qualification": 8, "test": 32}, "counts_by_cell": counts,
                      "file_sha256": {str(k): v for k, v in hashes.items()}, "case_hash": F.case_hash_table(rows), "suite_sha256": G.digest({"q": [r for r in rows if r["split"] == "qualification"], "t": final_rows}),
                      "expected_depth": {c: G.EXPECTED_DEPTH[c] for c in F.CELLS}, "fresh_geometry_by_level": geometry, "dedupe": dedupe["verdict"],
                      "frozen_rule": "after generation no case, weight or count may change; qualification cases never enter the final score; the final file is opened only by the evaluation script"}
    G.write_json(prep / "split_manifest.json", split_manifest)
    G.write_json(prep / "fresh_confirm_audits.json", {"card": F.CARD, "dedupe": dedupe, "final": audits["final"], "qualification": audits["qualification"], "fresh_geometry_by_level": geometry,
                                                       "construction_attempts": attempts, "verdict": "PASS"})
    print(json.dumps({"attempt": attempt, "counts_by_cell": counts, "files": hashes}, indent=1))


if __name__ == "__main__":
    main()
