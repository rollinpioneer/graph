#!/usr/bin/env python
"""CP-DISR-TP-EF-POST-OPEN-RESTORE-R2 runner."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_post_open_capture as cap
from cp_disr.analysis import tp_ef_post_open_restore_r2 as r2
from cp_disr.analysis import tp_ef_protocol_review as pr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["gate0", "setup", "restore", "run", "verify"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    ap.add_argument("--case")
    ap.add_argument("--snapshot-dir")
    ap.add_argument("--snapshot-root")
    ap.add_argument("--arrays-root")
    ap.add_argument("--arrays-dir")
    ap.add_argument("--tag")
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    if a.command == "gate0":
        pr.write_json(out / "protected_hashes_before.json", r2.protected_hashes(root))
        r = r2.gate0(root, out, a.arrays_root)
        print(json.dumps({"status": r["status"], "problems": r["problems"]}))
    elif a.command == "setup":
        r = cap.setup_episode(root, out, a.case, a.snapshot_dir, out / "budget_ledger.json")
        print(json.dumps({"case": a.case, "status": r["status"], "failures": r["failures"]}))
    elif a.command == "restore":
        r = r2.restore_r2(root, out, a.case, a.snapshot_dir, a.tag, a.arrays_dir)
        print(json.dumps({"case": a.case, "status": r["status"], "failures": r["failures"]}))
    elif a.command == "run":
        summary = r2.run_all(root, out, a.arrays_root, a.snapshot_root)
        fin = r2.finalize(root, out, summary)
        v = r2.verify(root, out, pr.read_json(out / "protected_hashes_before.json"))
        print(json.dumps({"summary": summary, "all_pass": fin["all_pass"], "verify": v["status"]}))
    elif a.command == "verify":
        print(json.dumps({"verify": r2.verify(root, out, pr.read_json(out / "protected_hashes_before.json"))["status"]}))


if __name__ == "__main__":
    main()
