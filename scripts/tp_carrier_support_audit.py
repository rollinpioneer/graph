#!/usr/bin/env python
"""CP-DISR-TP-CARRIER-SUPPORT-DESIGN-AUDIT-1 runner (zero environment, static). stage1 freezes the universes; stage2 reads geometry."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_carrier_support_audit as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["stage1", "stage2"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--output", required=True)
    x = ap.parse_args()
    root, out = Path(x.root).resolve(), Path(x.output).resolve()
    if x.stage == "stage1":
        print(json.dumps(m.stage1(root, x.libero_root, out)))
    else:
        v = m.run_stage2(root, x.libero_root, out)
        m.final_summary(out)
        r = m.verify(root, out)
        print(json.dumps({"verdict": v["verdict"], "failing": v["failing"], "verify": r["status"]}))


if __name__ == "__main__":
    main()
