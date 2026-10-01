#!/usr/bin/env python
"""CLI for CP-DISR-S4-FAMILY-B-V3-R1-REACH-SAFE-MIRROR-PILOT-1 (staged; provider/representation/RL/optimizer = 0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cp_disr.analysis import family_b_v3r1_reach_safe_mirror as v

COMMANDS = ["gate0", "register", "inventory", "run-waves", "worker", "analyze", "summarize", "verify", "receipt"]


def junit_receipt(junit_path, out):
    import xml.etree.ElementTree as ET
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = [{"name": c.attrib["name"], "classname": c.attrib["classname"],
              "failed": c.find("failure") is not None or c.find("error") is not None} for c in suite.iter("testcase")]
    receipt = {"tests": len(cases), "failures": sum(c["failed"] for c in cases),
               "online_attempts_before_tests_passed": 0, "cases": cases}
    v.write(Path(out) / "test_receipt.json", receipt)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output")
    ap.add_argument("--branch-id")
    ap.add_argument("--gpus", default="")
    ap.add_argument("--junit")
    a = ap.parse_args(argv)
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    if a.command == "gate0":
        print(json.dumps({"status": v.gate0(root, out)["status"]}))
    elif a.command == "register":
        print(json.dumps(v.register(root, out)))
    elif a.command == "inventory":
        print(json.dumps({"files": v.inventory_before(root, out)}))
    elif a.command == "run-waves":
        print(json.dumps(v.run_waves(root, out, gpus)))
    elif a.command == "worker":
        res = v.worker(root, out, a.branch_id)
        print(json.dumps({"branch_id": res["branch_id"], "status": res["status"]}))
    elif a.command == "analyze":
        print(json.dumps(v.analyze(root, out)))
    elif a.command == "summarize":
        v.summarize(root, out)
        print("ok")
    elif a.command == "verify":
        print(json.dumps(v.verify(root, out)))
    elif a.command == "receipt":
        print(json.dumps({"tests": junit_receipt(a.junit, out)["tests"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
