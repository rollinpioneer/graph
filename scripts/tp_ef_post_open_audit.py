#!/usr/bin/env python
"""CP-DISR-TP-EF-POST-OPEN-AUDIT-1 runner (zero environment / provider / RL)."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_post_open_audit as a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    x = ap.parse_args()
    root, out = Path(x.root).resolve(), Path(x.output).resolve()
    res = a.run(root, out)
    print(json.dumps(res, sort_keys=True))
    print(json.dumps({"verify": a.verify(root, out)["status"]}))


if __name__ == "__main__":
    main()
