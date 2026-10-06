#!/usr/bin/env python
"""CLI for CP-DISR-C1-CROSS-TASK-BLOCKSWORLD-V1: register | train | status. There is no default 'all' and no supervisor; every training is started explicitly.

    python scripts/c1_bw_launch.py register --root . --prep-commit <sha> --authorization-text-file <file>
    python scripts/c1_bw_launch.py train    --root . --run-root <dir> --run-id R-C1-BW-B2-0 --gpu 2
    python scripts/c1_bw_launch.py status   --run-root <dir>
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("register")
    r.add_argument("--root", default=".")
    r.add_argument("--prep-commit", required=True)
    r.add_argument("--authorization-text-file", required=True)
    r.add_argument("--stamp")
    t = sub.add_parser("train")
    t.add_argument("--root", default=".")
    t.add_argument("--run-root", required=True)
    t.add_argument("--run-id", required=True)
    t.add_argument("--gpu", type=int, required=True)
    s = sub.add_parser("status")
    s.add_argument("--run-root", required=True)
    a = ap.parse_args()
    if a.cmd == "train":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)            # must precede any torch import
    from cp_disr.blocksworld import launch as L
    if a.cmd == "register":
        spec = json.loads((Path(a.root).resolve() / L.PREP_REL / "task_spec.json").read_text())
        run_root = L.register(a.root, a.prep_commit, Path(a.authorization_text_file).read_text(encoding="utf-8"), spec["runs"], spec["ppo"], spec["half_life_H"], a.stamp)
        print(json.dumps({"registered": True, "run_root": str(run_root)}))
    elif a.cmd == "train":
        try:
            print(json.dumps(L.train(a.root, a.run_root, a.run_id, a.gpu), default=str))
        except L.LaunchError as exc:
            print(json.dumps({"status": "BLOCKED", "error": str(exc)}))
            sys.exit(2)
    else:
        print(json.dumps(L.status(a.run_root)))


if __name__ == "__main__":
    main()
