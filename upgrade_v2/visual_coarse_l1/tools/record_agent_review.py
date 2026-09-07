#!/usr/bin/env python3
"""Compatibility entry point for importing explicit L1V semantic evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from l1v_review_v2 import import_explicit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    result = import_explicit(args.root, args.review, args.evidence, args.repo)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
