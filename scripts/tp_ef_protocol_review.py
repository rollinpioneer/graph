#!/usr/bin/env python
"""CP-DISR-TP-EF-PROTOCOL-REVIEW-1 runner (zero environment / provider / RL)."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_protocol_review as r


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    out.mkdir(parents=True, exist_ok=True)
    r.snapshot_discovery(root, out)
    res = r.review(root, out)
    r.write_prose(out, res)
    r.request_md(out, res)
    print(json.dumps(res, sort_keys=True))
    print(json.dumps({"verify": r.verify(root, out)["status"]}))


if __name__ == "__main__":
    main()
