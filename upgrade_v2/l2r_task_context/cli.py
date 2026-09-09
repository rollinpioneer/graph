from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .collector import collect, plan_families
from .confirm import confirm
from .contract import freeze_contract
from .evaluate import evaluate, lock_candidate, replay_development
from .handoff import build_handoff
from .inputs import prepare
from .trace import trace_historical
from .cache_fault_split import run_diagnostic


def _methods(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--methods", nargs="+", required=True)


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser(description="L2RA-R2 bounded task-context research pipeline")
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--anchor", type=Path, required=True)
    p.add_argument("--formal-ref", required=True)
    p.add_argument("--maintenance-ref", required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("trace")
    p.add_argument("--inputs", type=Path, required=True)
    _methods(p)
    p.add_argument("--data-role", required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("freeze-contract")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--trace-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("replay-development")
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--contract-root", type=Path, required=True)
    _methods(p)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("plan-families")
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--inputs", type=Path, required=True)
    p.add_argument("--contract-root", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("collect")
    p.add_argument("--generation-lock", type=Path, required=True)
    p.add_argument("--partition", choices=("dev_fit", "dev_select", "confirmation"), required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("evaluate")
    p.add_argument("--data-root", type=Path, required=True)
    _methods(p)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--contract-root", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("cache-fault-split")
    p.add_argument("--data-root", type=Path, required=True)
    p.add_argument("--unresolved", type=Path, required=True)
    p.add_argument("--reference-contract", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("lock-candidate")
    p.add_argument("--development-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--generation-lock", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p = commands.add_parser("confirm")
    p.add_argument("--selection-lock", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--output-root", type=Path, required=True)
    p = commands.add_parser("handoff")
    p.add_argument("--run-root", type=Path, required=True)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--output-root", type=Path, required=True)
    return root


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    if args.command == "prepare":
        result = prepare(args.anchor, args.formal_ref, args.maintenance_ref, args.protocol, args.output_root)
    elif args.command == "trace":
        result = trace_historical(args.inputs, args.methods, args.data_role, args.workers, args.output_root)
    elif args.command == "freeze-contract":
        result = freeze_contract(args.inputs, args.trace_root, args.protocol, args.output_root)
    elif args.command == "replay-development":
        result = replay_development(args.inputs, args.contract_root, args.methods, args.output_root)
    elif args.command == "plan-families":
        result = plan_families(args.protocol, args.inputs, args.contract_root, args.output)
    elif args.command == "collect":
        result = collect(args.generation_lock, args.partition, args.workers, args.output_root)
    elif args.command == "evaluate":
        result = evaluate(args.data_root, args.methods, args.protocol, args.contract_root, args.output_root)
    elif args.command == "cache-fault-split":
        result = run_diagnostic(args.data_root, args.unresolved, args.reference_contract, args.output_root)
    elif args.command == "lock-candidate":
        result = lock_candidate(args.development_root, args.protocol, args.generation_lock, args.output)
    elif args.command == "confirm":
        result = confirm(args.selection_lock, args.protocol, args.workers, args.output_root)
    else:
        result = build_handoff(args.run_root, args.protocol, args.output_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
