#!/usr/bin/env python3
"""Plan v1.1 Stage 1A entry. Does not call historical stage-1a-run."""
from __future__ import annotations
import argparse, os, sys
from pathlib import Path

ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)

from cp_disr.stage1a_v11 import cmd_stage_1a_v11_run


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only-startup-gates", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-updates", type=int, default=16)
    p.add_argument("--stop-after-updates", type=int, default=None)
    p.add_argument("--method", choices=["B2", "Full"], default=None)
    p.add_argument("--stamp", default=None)
    p.add_argument("--configsha", default=None)
    p.add_argument("--skip-startup-gates", action="store_true")
    a = p.parse_args()
    payload = cmd_stage_1a_v11_run(
        ROOT,
        gpu=a.gpu,
        only_startup_gates=a.only_startup_gates,
        resume=a.resume,
        max_updates=a.max_updates,
        stop_after_updates=a.stop_after_updates,
        method=a.method,
        stamp=a.stamp,
        configsha=a.configsha,
        skip_startup_gates=a.skip_startup_gates,
    )
    print(payload.get("status"), payload.get("execution_reason"))
    return 0 if payload.get("status") in ("PASS", "STARTUP_GATES_PASS", "PASS_WITH_NOTES") else 2


if __name__ == "__main__":
    raise SystemExit(main())
