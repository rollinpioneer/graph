"""Command line entry points for the L2RA protocol."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import inputs, replay, diagnose, probes, evaluate, select, confirm, handoff, delivery


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser(description="PathGraph-SARM L2RA offline ambiguity diagnosis")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--frozen-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("trace"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--split", required=True); p.add_argument("--graph", type=Path); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("diagnose"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--trace-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("collect-probes"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--partition", default="development"); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("evaluate-development"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--probe-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--frozen-graph", type=Path, required=True)
    p = sub.add_parser("lock-candidate"); p.add_argument("--development-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--frozen-root", type=Path, required=True)
    p = sub.add_parser("confirm"); p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--frozen-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("handoff"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("package-round"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-file-mb", type=float, default=200)
    p = sub.add_parser("deliver"); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--downloads", type=Path, required=True); p.add_argument("--code-root", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        args.output_root.mkdir(parents=True, exist_ok=True)
        target_protocol = args.output_root / "configs/l2ra_protocol.json"
        target_protocol.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(args.protocol, target_protocol)
        result = inputs.prepare(args.repo_root, args.frozen_root, target_protocol, args.output_root)
    elif args.command == "trace":
        resolved = _json(args.inputs); graph = args.graph or (Path(resolved["repository"]["path"]) / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/final_v1/graphs/G2_evidence_refined.json")
        result = replay.trace_split(resolved, args.split, graph, args.output_root)
    elif args.command == "diagnose": result = diagnose.diagnose(_json(args.inputs), args.trace_root, args.output_root)
    elif args.command == "collect-probes": result = probes.collect_probes(_json(args.inputs), _json(args.protocol), args.output_root, args.partition, args.workers)
    elif args.command == "evaluate-development":
        result = evaluate.evaluate_development(_json(args.inputs), args.probe_root, args.output_root, args.frozen_graph)
    elif args.command == "lock-candidate": result = select.lock_candidate(args.development_root, args.protocol, args.run_root, args.frozen_root)
    elif args.command == "confirm": result = confirm.confirm(args.selection_lock, _json(args.protocol), _json(args.inputs), args.output_root.parents[1], args.frozen_root, args.workers)
    elif args.command == "handoff": result = handoff.handoff(args.run_root, _json(args.inputs), args.protocol, args.output_root)
    elif args.command == "package-round":
        from .tools.l2ra_support import package_round
        package_round(args.source, args.output, args.max_file_mb); result = {"status": "PASS"}
    else: result = delivery.deliver(args.repo_root, args.run_root, args.downloads, args.code_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
