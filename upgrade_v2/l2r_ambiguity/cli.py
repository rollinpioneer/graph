"""Command line entry points for the L2RA protocol."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from . import inputs, replay, diagnose, probes, evaluate, select, confirm, handoff, delivery


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _resolved_frozen_root(resolved: dict) -> Path:
    source = next(item for item in resolved["frozen_sources"] if item["logical_id"] == "frozen:locks/predicate_thresholds.json")
    return Path(source["resolved_path"]).parent.parent


def _repo_root(path: Path) -> Path:
    for candidate in (path.resolve(), *path.resolve().parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(f"cannot resolve repository root from {path}")


def _confirmation_run_root(output_root: Path) -> Path:
    """Resolve the L2RA run root from either the run root or its final_v1 dir."""
    output_root = output_root.resolve()
    if output_root.name == "final_v1" and output_root.parent.name == "edge_ambiguity_l2ra_v1":
        return output_root.parent
    if (output_root / "rounds").is_dir() and (output_root / "locks").is_dir():
        return output_root
    candidate = output_root.parent
    if (candidate / "rounds").is_dir() and (candidate / "locks").is_dir():
        return candidate
    raise FileNotFoundError(f"cannot resolve L2RA run root from {output_root}")


def main() -> int:
    parser = argparse.ArgumentParser(description="PathGraph-SARM L2RA offline ambiguity diagnosis")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare"); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--frozen-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("trace"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--split", required=True); p.add_argument("--graph", type=Path); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("diagnose"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--trace-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("collect-probes"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--partition", default="development"); p.add_argument("--workers", type=int, default=1)
    p = sub.add_parser("evaluate-development"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--probe-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--frozen-graph", type=Path)
    p = sub.add_parser("lock-candidate"); p.add_argument("--development-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--frozen-root", type=Path)
    p = sub.add_parser("confirm"); p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--frozen-root", type=Path); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--workers", type=int, default=4)
    p = sub.add_parser("handoff"); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("package-round"); p.add_argument("--source", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-file-mb", type=float, default=200)
    p = sub.add_parser("deliver"); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--downloads", type=Path, required=True); p.add_argument("--code-root", type=Path, required=True); p.add_argument("--staging-root", type=Path)
    args = parser.parse_args()
    if args.command == "prepare":
        args.output_root.mkdir(parents=True, exist_ok=True)
        target_protocol = args.output_root / "configs/l2ra_protocol.json"
        target_protocol.parent.mkdir(parents=True, exist_ok=True)
        if args.protocol.resolve() != target_protocol.resolve():
            shutil.copy2(args.protocol, target_protocol)
        result = inputs.prepare(args.repo_root, args.frozen_root, target_protocol, args.output_root)
    elif args.command == "trace":
        resolved = _json(args.inputs); graph = args.graph or (Path(resolved["repository"]["path"]) / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/final_v1/graphs/G2_evidence_refined.json")
        result = replay.trace_split(resolved, args.split, graph, args.output_root)
    elif args.command == "diagnose": result = diagnose.diagnose(_json(args.inputs), args.trace_root, args.output_root)
    elif args.command == "collect-probes": result = probes.collect_probes(_json(args.inputs), _json(args.protocol), args.output_root, args.partition, args.workers)
    elif args.command == "evaluate-development":
        resolved = _json(args.inputs)
        graph = args.frozen_graph or (_resolved_frozen_root(resolved) / "graphs/G2_evidence_refined.json")
        result = evaluate.evaluate_development(resolved, _json(args.protocol), args.probe_root, args.output_root, graph)
    elif args.command == "lock-candidate":
        frozen = args.frozen_root or (_repo_root(args.protocol) / "artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/final_v1")
        result = select.lock_candidate(args.development_root, args.protocol, args.output, frozen)
    elif args.command == "confirm":
        resolved = _json(args.inputs)
        result = confirm.confirm(args.selection_lock, _json(args.protocol), resolved, _confirmation_run_root(args.output_root), args.frozen_root or _resolved_frozen_root(resolved), args.workers)
    elif args.command == "handoff": result = handoff.handoff(args.run_root, _json(args.inputs), args.protocol, args.output_root)
    elif args.command == "package-round":
        from .tools.l2ra_support import package_round
        package_round(args.source, args.output, args.max_file_mb); result = {"status": "PASS"}
    else: result = delivery.deliver(args.repo_root, args.run_root, args.downloads, args.code_root, args.staging_root)
    print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
