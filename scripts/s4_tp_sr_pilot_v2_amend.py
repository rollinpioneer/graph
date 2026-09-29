#!/usr/bin/env python3
"""Derived-only commands for WAVE_D_D7_CHECKER_AMENDMENT_1: amended classification, summary and verification of saved Wave D evidence.
Never prepares, dispatches, constructs an environment, resets, executes a skill, or calls a provider."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m  # noqa: E402
from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as d  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=("classify-diagnostics", "summarize", "verify"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"))
    ap.add_argument("--output", required=True)
    a = ap.parse_args(argv)
    out = Path(a.output).resolve()
    try:
        fn = {"classify-diagnostics": d.amend_gate, "summarize": d.summarize_wave_d, "verify": d.verify_wave_d}[a.command]
        res = fn(a.root, a.config, out)
    except m.StopRun as exc:
        print(json.dumps({"status": exc.code, "detail": exc.detail}, ensure_ascii=False))
        return 2
    print(json.dumps(m.jsonable(res), ensure_ascii=False, default=str)[:3000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
