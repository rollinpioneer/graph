from __future__ import annotations
import argparse, json
from pathlib import Path
from .approval import validate_approval
from .stage_grants import grants
def main() -> int:
    ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest="cmd",required=True)
    p=sub.add_parser("validate"); p.add_argument("--lock",type=Path,required=True); p.add_argument("--approval",type=Path)
    g=sub.add_parser("grants"); g.add_argument("--approval",type=Path,required=True); g.add_argument("--lock",type=Path,required=True)
    a=ap.parse_args(); lock=json.loads(a.lock.read_text());
    if a.cmd=="validate":
        errors=[] if not a.approval else validate_approval(json.loads(a.approval.read_text()),lock)
        print(json.dumps({"status":"PASS" if not errors else "FAIL","errors":errors,"physical_executions":0,"mujoco_imported":False},indent=2)); return 0 if not errors else 2
    approval=json.loads(a.approval.read_text()); errors=validate_approval(approval,lock)
    if errors: print(json.dumps({"status":"FAIL","errors":errors})); return 2
    print(json.dumps({"status":"PASS","grants":grants(approval)},indent=2)); return 0
if __name__ == "__main__": raise SystemExit(main())
