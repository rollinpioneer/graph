#!/usr/bin/env python
"""CP-DISR-TP-EF-POST-OPEN-CAPTURE-1 runner."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_post_open_capture as c
from cp_disr.analysis import tp_ef_protocol_review as pr

COMMANDS = ["gate0", "setup", "restore", "run", "verify"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    ap.add_argument("--case")
    ap.add_argument("--snapshot-dir")
    ap.add_argument("--snapshot-root")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    if a.command == "gate0":
        r = c.gate0(root, out)
        pr.write_json(out / "protected_hashes_before.json", c.protected_hashes(root))
        print(json.dumps({"status": r["status"], "problems": r["problems"], "gpu": r["gpu"]}))
    elif a.command == "setup":
        r = c.setup_episode(root, out, a.case, a.snapshot_dir, out / "budget_ledger.json")
        print(json.dumps({"case": a.case, "status": r["status"], "failures": r["failures"]}))
    elif a.command == "restore":
        r = c.restore_validation(root, out, a.case, a.snapshot_dir)
        print(json.dumps({"case": a.case, "status": r["status"], "failures": r["failures"]}))
    elif a.command == "run":
        summary = c.run_all(root, out, a.snapshot_root)
        fin = c.finalize(root, out, a.snapshot_root, summary)
        before = pr.read_json(out / "protected_hashes_before.json")
        v = c.verify(root, out, a.snapshot_root, before)
        print(json.dumps({"summary": summary, "all_pass": fin["all_pass"], "verify": v["status"]}))
    elif a.command == "verify":
        before = pr.read_json(out / "protected_hashes_before.json")
        print(json.dumps({"verify": c.verify(root, out, a.snapshot_root, before)["status"]}))


if __name__ == "__main__":
    main()
