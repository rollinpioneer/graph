#!/usr/bin/env python
"""CLI for CP-DISR-S4-FAMILY-B-OBS-V2-VALIDATION-1 (no provider/RL/optimizer; 4 attempts max)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cp_disr.analysis import family_b_obs_v2 as m


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["freeze-profile", "register", "inventory", "run", "worker", "gate",
                                        "timeline", "summarize", "verify", "receipt"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--output")
    ap.add_argument("--branch-id")
    ap.add_argument("--gpus", default="3,4")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--junit")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    if a.command == "freeze-profile":
        res = m.freeze_profile(root)
        print(json.dumps({"profile_sha256": res["profile_sha256"]}))
        return 0
    out = Path(a.output).resolve()
    if a.command == "register":
        print(json.dumps(m.register(root, out)))
    elif a.command == "inventory":
        inv = m.inventory(m.protected_paths())
        m.write(out / "protected_before.json", {
            "sha256": inv, "capture_count": sum("/captures/" in k for k in inv), "files": len(inv),
            "old_branch_commit": m.git(root, "rev-parse", "codex/cp-disr-s4-family-b-staging-v1")})
        print(json.dumps({"files": len(inv)}))
    elif a.command == "run":
        gpus = [int(x) for x in a.gpus.split(",") if x != ""]
        print(json.dumps(m.run_wave(root, out, gpus, a.max_workers)))
    elif a.command == "worker":
        r = m.worker(root, out, a.branch_id)
        print(json.dumps({"branch_id": r["branch_id"], "status": r["status"]}))
    elif a.command == "gate":
        g = m.technical_gate(root, out)
        m.branch_results_csv(out)
        print(json.dumps({"status": g["status"], "label": g["label"], "problems": g["problems"]}))
    elif a.command == "timeline":
        print(json.dumps({"rows": m.timeline(out)}))
    elif a.command == "summarize":
        m.summarize(root, out)
        print("ok")
    elif a.command == "verify":
        print(json.dumps(m.verify(root, out)))
    elif a.command == "receipt":
        print(json.dumps(m.junit_receipt(a.junit, out)["groups"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
