#!/usr/bin/env python3
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from cp_disr.stage1a_smoke import cmd_stage_1a_run
if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=ROOT)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--only-startup-gates", action="store_true")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--max-updates", type=int, default=16)
    a = p.parse_args()
    print(cmd_stage_1a_run(a.root, gpu=a.gpu, only_startup_gates=a.only_startup_gates, resume=a.resume, max_updates=a.max_updates))
