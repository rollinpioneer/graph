#!/usr/bin/env python
"""CLI for CP-DISR-S4-FAMILY-B-OBS-V2-R2-VALIDATION-1 (staged: canary first; provider/RL/optimizer = 0)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cp_disr.analysis import family_b_obs_v2 as m
from cp_disr.analysis import family_b_obs_v2_r2 as r2

COMMANDS = ["register-r2", "inventory", "run-canary", "check-canary", "run-remainder", "worker", "gate-r2",
            "timeline", "summarize-r2", "verify-r2", "receipt"]


def junit_receipt(junit_path, out):
    import xml.etree.ElementTree as ET
    root = ET.parse(junit_path).getroot()
    suite = root if root.tag == "testsuite" else root.find("testsuite")
    cases = [{"name": c.attrib["name"], "classname": c.attrib["classname"],
              "failed": c.find("failure") is not None or c.find("error") is not None} for c in suite.iter("testcase")]
    groups = {"R2_dispatch_and_identity": "test_R2_", "RFC_binding_repair": "test_RFC", "A_B_C_D_v2": "test_"}
    receipt = {"tests": len(cases), "failures": sum(c["failed"] for c in cases),
               "groups": {k: {"tests": sum(p in c["name"] for c in cases),
                              "failed": sum(p in c["name"] and c["failed"] for c in cases)}
                          for k, p in groups.items()},
               "online_attempts_before_tests_passed": 0, "cases": cases}
    m.write(Path(out) / "test_receipt.json", receipt)
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output")
    ap.add_argument("--branch-id")
    ap.add_argument("--gpus", default="3,4")
    ap.add_argument("--junit")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.output).resolve()
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    if a.command == "register-r2":
        print(json.dumps(r2.register(root, out)))
    elif a.command == "inventory":
        print(json.dumps({"files": r2.inventory_before(root, out)}))
    elif a.command == "run-canary":
        print(json.dumps(r2.run_canary(root, out, gpus)))
    elif a.command == "check-canary":
        g = r2.check_canary(root, out)
        print(json.dumps({"status": g["status"], "problems": g["problems"]}))
    elif a.command == "run-remainder":
        print(json.dumps(r2.run_remainder(root, out, gpus)))
    elif a.command == "worker":
        res = r2.worker(root, out, a.branch_id)
        print(json.dumps({"branch_id": res["branch_id"], "status": res["status"]}))
    elif a.command == "gate-r2":
        g = r2.gate_r2(root, out)
        print(json.dumps({"status": g["status"], "label": g["label"], "problems": g["problems"]}))
    elif a.command == "timeline":
        print(json.dumps({"rows": m.timeline(out)}))
    elif a.command == "summarize-r2":
        r2.summarize(root, out)
        print("ok")
    elif a.command == "verify-r2":
        print(json.dumps(r2.verify(root, out)))
    elif a.command == "receipt":
        print(json.dumps(junit_receipt(a.junit, out)["groups"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
