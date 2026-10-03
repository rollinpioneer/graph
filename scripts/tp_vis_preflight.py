#!/usr/bin/env python
"""CP-DISR-TP-VIS-PREFLIGHT-1 runner (zero environment)."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_vis_preflight as v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--output", required=True)
    a = ap.parse_args()
    root, out = Path(a.root).resolve(), Path(a.output).resolve()
    verdict = v.run(root, out)
    v.final_summary(out, verdict)
    r = v.verify(root, out)
    print(json.dumps({"verdict": verdict["verdict"], "stop_phase": verdict["stop_phase"], "phases": verdict["phases"], "verify": r["status"]}))


if __name__ == "__main__":
    main()
