#!/usr/bin/env python3
from __future__ import annotations
import argparse, os, sys
from pathlib import Path
ROOT = Path("/home/__compress_data/xushijie/graph_cp_disr_v2_1")
sys.path.insert(0, str(ROOT / "src"))
os.chdir(ROOT)
from cp_disr.common import BindingError, canonical
from cp_disr.stage1a_v11_final_eval import cmd_stage_1a_v11_final_eval

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--phase", choices=["freeze", "register", "materialize", "eval", "report", "all"], default="all")
    p.add_argument("--method", choices=["B2", "Full"], default=None)
    p.add_argument("--gpu", type=int, default=0)
    a = p.parse_args()
    try:
        payload = cmd_stage_1a_v11_final_eval(ROOT, phase=a.phase, method=a.method, gpu=a.gpu)
        print(canonical(payload))
        return 0 if payload.get("status") in ("PASS", "RUNNING") else 2
    except BindingError as exc:
        print(canonical({"status": "BLOCKED", "reason": str(exc)}))
        return 2

if __name__ == "__main__":
    raise SystemExit(main())
