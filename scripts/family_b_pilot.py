#!/usr/bin/env python3
"""Family B staging qualification CLI."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))
from cp_disr.analysis import family_b_pilot as p  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=(
        "inspect-bindings", "provider-preflight", "freeze", "run-wave", "worker",
        "check-technical", "analyze-physical", "call-provider", "probe-representation", "evaluate-reference-bank", "offline-preflight",
        "protected-before", "assemble", "verify"))
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--config", default="configs/final_master/s4_family_b_staging.yaml")
    ap.add_argument("--output", required=True)
    ap.add_argument("--wave", choices=("technical", "remaining"))
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--gpu-policy", default="immediate")
    ap.add_argument("--branch-id")
    ap.add_argument("--group", choices=("relations", "vlm-action"))
    ap.add_argument("--scope", choices=("artifacts", "protocol", "all"), default="all")
    args = ap.parse_args()
    root, out = Path(args.root).resolve(), Path(args.output).resolve()
    cfg = p.config(root, args.config)
    if args.command == "inspect-bindings":
        result = p.inspect_bindings(root, cfg, out)
    elif args.command == "provider-preflight":
        result = p.provider_preflight(root, cfg, out)
    elif args.command == "offline-preflight":
        result = p.offline_preflight(root, cfg, out)
    elif args.command == "freeze":
        result = p.freeze(root, cfg, out)
    elif args.command == "worker":
        if not args.branch_id:
            raise ValueError("--branch-id required")
        result = p.worker(root, out, args.branch_id)
    elif args.command == "run-wave":
        if not args.wave or args.gpu_policy != "immediate":
            raise ValueError("wave and immediate GPU policy required")
        result = p.run_wave(root, cfg, out, args.wave,
                            tuple(int(x) for x in args.gpus.split(",")),
                            args.max_workers)
    elif args.command == "check-technical":
        result = p.check_technical(root, cfg, out)
    elif args.command == "analyze-physical":
        result = p.analyze_physical(root, cfg, out)
    elif args.command == "call-provider":
        if not args.group:
            raise ValueError("--group required")
        from cp_disr.analysis.family_b_provider import call_group
        result = call_group(root, cfg, out, args.group)
    elif args.command == "probe-representation":
        from cp_disr.analysis.family_b_representation import probe
        result = probe(root, cfg, out)
    elif args.command == "evaluate-reference-bank":
        from cp_disr.analysis.family_b_reference_bank import evaluate
        result = evaluate(root, cfg, out)
    elif args.command == "protected-before":
        from cp_disr.analysis.family_b_closeout import protected_inventory
        result = protected_inventory(root, out, "before")
    elif args.command == "assemble":
        from cp_disr.analysis.family_b_closeout import assemble
        result = assemble(root, cfg, out)
    elif args.command == "verify":
        from cp_disr.analysis.family_b_closeout import verify
        result = verify(root, cfg, out, args.scope)
    print(json.dumps(result, sort_keys=True, default=str)[:4000])


if __name__ == "__main__":
    main()
