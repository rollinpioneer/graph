from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import sys
from pathlib import Path

from . import confirmation, handoff, inputs, selection, trace_errors
from .evaluate_v2 import build_features_for_data, evaluate_development
from .recorder import collect_development


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_command(run_root: Path, argv: list[str], status: str) -> None:
    path = run_root / "rounds/actual_commands.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"status={status}\npython={' '.join(argv)}\npython_version={platform.python_version()}\n\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PathGraph-SARM L2RA-R1 hold evidence protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-repo", type=Path, required=True); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--old-l2ra", type=Path, required=True); p.add_argument("--old-l2r", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("inspect-errors")
    p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("collect-development")
    p.add_argument("--partition", choices=["dev_fit", "dev_select"], required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--workers", type=int, default=1); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("build-features")
    p.add_argument("--partition", required=True); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("freeze-development")
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate-development")
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--development-lock", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("select-and-lock")
    p.add_argument("--development-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--workers", type=int, default=1); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("handoff")
    p.add_argument("--run-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "prepare":
            result = inputs.prepare(args.source_repo, args.repo_root, args.old_l2ra, args.old_l2r, args.protocol, args.output_root)
        elif args.command == "inspect-errors":
            result = trace_errors.inspect_errors(_json(args.inputs), _json(args.protocol), args.output_root)
        elif args.command == "collect-development":
            result = collect_development(args.output_root, _json(args.protocol), args.partition, max(1, args.workers))
        elif args.command == "build-features":
            result = build_features_for_data(args.data_root, args.output_root)
        elif args.command == "freeze-development":
            protocol = _json(args.protocol)
            from .inputs import sha256_file, write_json
            data = []
            for path in sorted((args.run_root / "data/new_development").rglob("metadata.json")):
                data.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
            result = {"schema": "pathgraph_l2rar1_development_lock_v1", "status": "LOCKED_BEFORE_SELECT", "protocol_sha256": sha256_file(args.protocol), "metadata_files": data, "sampling": "control_tick_20hz", "reference_version": "timestamp_aligned_proxy_hold_v2", "metric_version": "prefix_event_v2", "api_calls": 0, "training_jobs": 0, "api_key_read": False}
            write_json(args.output, result)
        elif args.command == "evaluate-development":
            result = evaluate_development(args.data_root, _json(args.protocol), args.output_root)
        elif args.command == "select-and-lock":
            result = selection.select_and_lock(args.development_root, args.protocol, args.output, args.output.parent.parent)
        elif args.command == "confirm":
            result = confirmation.confirm(args.selection_lock, _json(args.protocol), _json(args.inputs), args.output_root)
        else:
            result = handoff.handoff(args.run_root, args.protocol, args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0
    except Exception as exc:
        print(json.dumps({"status": "EXECUTION_ERROR", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
raise SystemExit(main())
