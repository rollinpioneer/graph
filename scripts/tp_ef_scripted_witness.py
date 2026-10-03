#!/usr/bin/env python
"""CP-DISR-TP-EF-SCRIPTED-WITNESS-1 runner."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_scripted_witness as w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["register", "branch", "run", "verify"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    ap.add_argument("--branch-id")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    if a.command == "register":
        r = w.register(root, out)
        print(json.dumps({"status": r["status"], "problems": r["problems"]}))
    elif a.command == "branch":
        r = w.run_branch(root, out, a.branch_id)
        print(json.dumps({"branch": a.branch_id, "status": r["status"], "anomalies": r["anomalies"]}))
    elif a.command == "run":
        g = w.gate0(root, out)
        if g["problems"]:
            print(json.dumps({"stopped": "GATE0", "problems": g["problems"]}))
            return
        summary = w.run_all(root, out)
        gate = w.finalize(root, out, summary)
        v = w.verify(root, out)
        print(json.dumps({"stopped": summary["stopped"], "sequence": summary["sequence"], "mechanism_gate": gate["status"], "verify": v["status"]}))
    elif a.command == "verify":
        print(json.dumps({"verify": w.verify(root, out)["status"]}))


if __name__ == "__main__":
    main()
