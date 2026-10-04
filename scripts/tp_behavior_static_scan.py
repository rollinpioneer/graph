#!/usr/bin/env python
"""CP-DISR-TP-BEHAVIOR-STATIC-1 scanner (static, no simulator)."""
import argparse
import json

from cp_disr.analysis import tp_behavior_static as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, help="pinned BEHAVIOR-1K checkout")
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    x = ap.parse_args()
    v = m.run(x.repo, x.root, x.output)
    print(json.dumps({k: v.get(k) for k in ("verdict", "engineering_status", "blocker", "denominator")}))


if __name__ == "__main__":
    main()
