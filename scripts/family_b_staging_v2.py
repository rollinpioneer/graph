#!/usr/bin/env python
"""CLI for CP-DISR-S4-FAMILY-B-STAGING-V2-CONTINUATION-1 (staged; provider/representation/RL/optimizer = 0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cp_disr.analysis import family_b_staging_v2 as s

COMMANDS = ["register", "inventory", "compat", "gate-m", "run-complementary", "check-complementary", "run-remaining16",
            "worker", "analyze", "summarize", "verify", "receipt"]


def junit_receipt(junit_path, out):
    import xml.etree.ElementTree as ET
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = [{"name": c.attrib["name"], "classname": c.attrib["classname"],
              "failed": c.find("failure") is not None or c.find("error") is not None} for c in suite.iter("testcase")]
    receipt = {"tests": len(cases), "failures": sum(c["failed"] for c in cases),
               "online_attempts_before_tests_passed": 0, "cases": cases}
    s.write(Path(out) / "test_receipt.json", receipt)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output")
    ap.add_argument("--branch-id")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--junit")
    a = ap.parse_args(argv)
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    if a.command == "register":
        print(json.dumps(s.register(root, out)))
    elif a.command == "inventory":
        print(json.dumps({"files": s.inventory_before(root, out)}))
    elif a.command == "compat":
        print(json.dumps({"status": s.compatibility_manifest(root, out)["status"]}))
    elif a.command == "gate-m":
        print(json.dumps({"status": s.gate_m(root, out)["status"]}))
    elif a.command == "run-complementary":
        print(json.dumps(s.run_complementary(root, out, gpus)))
    elif a.command == "check-complementary":
        g = s.complementary_gate(out)
        print(json.dumps({"status": g["status"], "failed_step": g["failed_step"]}))
    elif a.command == "run-remaining16":
        print(json.dumps(s.run_remaining16(root, out, gpus, workers=a.workers)))
    elif a.command == "worker":
        res = s.worker(root, out, a.branch_id)
        print(json.dumps({"branch_id": res["branch_id"], "status": res["status"]}))
    elif a.command == "analyze":
        print(json.dumps(s.analyze(root, out)))
    elif a.command == "summarize":
        s.summarize(root, out)
        print("ok")
    elif a.command == "verify":
        print(json.dumps(s.verify(root, out)))
    elif a.command == "receipt":
        print(json.dumps({"tests": junit_receipt(a.junit, out)["tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
