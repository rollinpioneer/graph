#!/usr/bin/env python
"""CP-DISR-TP-LIBERO-OPPORTUNITY-AUDIT-1 runner (zero environment, static)."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_libero_audit as a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--pip-root", default=None)
    ap.add_argument("--output", required=True)
    x = ap.parse_args()
    root, out = Path(x.root).resolve(), Path(x.output).resolve()
    verdict = a.run(root, x.libero_root, out, x.pip_root)
    a.final_summary(out)
    r = a.verify(root, out)
    print(json.dumps({"verdict": verdict["verdict"], "cards": verdict["cards"], "tasks": verdict["tasks"], "verify": r["status"]}))


if __name__ == "__main__":
    main()
