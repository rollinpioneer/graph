#!/usr/bin/env python
"""CP-DISR-TB-STRUCT-GEN-V1 results builder (files only).

    python scripts/struct_gen_results.py build --root . --out <registration dir>
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cp_disr import struct_gen_analysis as A  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=["build"])
    ap.add_argument("--root", default=".")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    out = Path(a.out).resolve()
    ledger = json.loads((out / "launch_state.json").read_text())
    classification, verify = A.build(a.root, out, ledger)
    print(json.dumps({"verify": verify["verdict"], "problems": verify["problems"], "outcome": classification.get("outcome"), "name": classification.get("name"),
                      "dev_counts": classification["dev_family_counts"], "T": classification.get("T_mean_over_seeds"), "effect": classification.get("effect_on_paper")}, indent=1, default=str))


if __name__ == "__main__":
    main()
