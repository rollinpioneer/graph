#!/usr/bin/env python
"""CP-DISR-TP-DV1-BINDING-PREFLIGHT-1 runner (zero environment, static)."""
import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_dv1_binding_preflight as m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--libero-root", required=True)
    ap.add_argument("--output", required=True)
    x = ap.parse_args()
    root, out = Path(x.root).resolve(), Path(x.output).resolve()
    v = m.run(root, x.libero_root, out)
    m.final_summary(out)
    r = m.verify(root, out)
    print(json.dumps({"verdict": v["verdict"], "failing": v["failing"], "verify": r["status"]}))


if __name__ == "__main__":
    main()