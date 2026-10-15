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
from .recorder import audit_callback_equivalence, collect_development, normalize_development_streams


def _json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_command(run_root: Path, argv: list[str], status: str) -> None:
    path = run_root / "rounds/actual_commands.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"status={status}\npython={' '.join(argv)}\npython_version={platform.python_version()}\n\n")


def _verify_development_lock(path: Path, protocol_path: Path, run_root: Path, *, require_pre_select: bool = False) -> None:
    from .inputs import sha256_file

    lock = _json(path)
    allowed = {"LOCKED_BEFORE_SELECT", "LOCKED_AFTER_COLLECTION_BEFORE_EVALUATION"}
    if lock.get("status") not in allowed or (require_pre_select and lock.get("status") != "LOCKED_BEFORE_SELECT"):
        raise ValueError("development lock state is invalid for this operation")
    checks = [(protocol_path, lock.get("protocol_sha256"))]
    checks.extend((Path(item["path"]), item.get("sha256")) for item in lock.get("code_files", []))
    checks.extend((Path(item["path"]), item.get("sha256")) for item in lock.get("runtime_files", []))
    checks.extend((Path(item["path"]), item.get("sha256")) for item in lock.get("dataset_files", []))
    checks.extend([
        (run_root / "data/new_development/generator_lock.json", lock.get("generator_lock_sha256")),
        (run_root / "data/new_development/reference_contract.json", lock.get("reference_contract_sha256")),
    ])
    callback = lock.get("callback_equivalence", {})
    if callback:
        checks.append((Path(callback.get("path", "")), callback.get("sha256")))
    normalization = lock.get("dense_stream_normalization", {})
    if normalization:
        checks.append((Path(normalization.get("path", "")), normalization.get("sha256")))
    normalization_details = lock.get("dense_stream_normalization_details", {})
    if normalization_details:
        checks.append((Path(normalization_details.get("path", "")), normalization_details.get("sha256")))
    collection = lock.get("collection_lock", {})
    if collection:
        checks.append((Path(collection.get("path", "")), collection.get("sha256")))
    mismatches = []
    for dependency, expected in checks:
        actual = sha256_file(dependency) if dependency.is_file() else None
        if not expected or actual != expected:
            mismatches.append({"path": str(dependency), "expected": expected, "actual": actual})
    if mismatches:
        raise ValueError(f"development lock mismatch: {json.dumps(mismatches, sort_keys=True)}")


def _run_root_for(args: argparse.Namespace) -> Path:
    for name in ("output_root", "run_root"):
        value = getattr(args, name, None)
        if value is not None:
            path = Path(value)
            if name == "run_root":
                return path
            if path.name == "new_development":
                return path.parent.parent
            if path.parent.name == "rounds":
                return path.parent.parent
            if path.parent.name == "locks":
                return path.parent.parent
            if path.parent.parent.name == "rounds":
                return path.parent.parent.parent
            if path.name == "final_v1":
                return path.parent
            return path.parent
    development_root = getattr(args, "development_root", None)
    if development_root is not None:
        return Path(development_root).parent.parent
    return Path.cwd()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="PathGraph-SARM L2RA-R1 hold evidence protocol")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare")
    p.add_argument("--source-repo", type=Path, required=True); p.add_argument("--repo-root", type=Path, required=True); p.add_argument("--old-l2ra", type=Path, required=True); p.add_argument("--old-l2r", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("inspect-errors")
    p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("collect-development")
    p.add_argument("--partition", choices=["dev_fit", "dev_select"], required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--development-lock", type=Path); p.add_argument("--workers", type=int, default=1); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("build-features")
    p.add_argument("--partition", required=True); p.add_argument("--data-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("freeze-development")
    p.add_argument("--protocol", type=Path, required=True); p.add_argument("--run-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("audit-callback")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("normalize-development-streams")
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("evaluate-development")
    p.add_argument("--data-root", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--development-lock", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("select-and-lock")
    p.add_argument("--development-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("confirm")
    p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--inputs", type=Path, required=True); p.add_argument("--workers", type=int, default=1); p.add_argument("--output-root", type=Path, required=True)
    p = sub.add_parser("handoff")
    p.add_argument("--run-root", type=Path, required=True); p.add_argument("--protocol", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args(argv)
    run_root = _run_root_for(args)
    _write_command(run_root, ["python", *sys.argv[1:]] if argv is None else ["python", *argv], "STARTED")
    try:
        if args.command == "prepare":
            result = inputs.prepare(args.source_repo, args.repo_root, args.old_l2ra, args.old_l2r, args.protocol, args.output_root)
        elif args.command == "inspect-errors":
            result = trace_errors.inspect_errors(_json(args.inputs), _json(args.protocol), args.output_root)
        elif args.command == "collect-development":
            if args.partition == "dev_select":
                if args.development_lock is None:
                    raise ValueError("dev_select requires --development-lock")
                _verify_development_lock(args.development_lock, args.protocol, _run_root_for(args), require_pre_select=True)
            result = collect_development(args.output_root, _json(args.protocol), args.partition, max(1, args.workers))
        elif args.command == "build-features":
            result = build_features_for_data(args.data_root, args.output_root)
        elif args.command == "audit-callback":
            result = audit_callback_equivalence(args.output)
        elif args.command == "normalize-development-streams":
            result = normalize_development_streams(args.data_root, args.output)
        elif args.command == "freeze-development":
            protocol = _json(args.protocol)
            from .inputs import sha256_file, write_json
            data = []
            for path in sorted((args.run_root / "data/new_development").rglob("metadata.json")):
                data.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
            dataset_files = []
            dataset_names = {
                "metadata.json", "observations_dense.jsonl", "observations_action_end.jsonl",
                "events.jsonl", "oracle_timeline.csv", "actions.csv", "low_level_controls.jsonl",
            }
            for path in sorted((args.run_root / "data/new_development").rglob("*")):
                if path.is_file() and path.name in dataset_names:
                    dataset_files.append({"path": str(path.resolve()), "sha256": sha256_file(path), "size_bytes": path.stat().st_size})
            code_files = sorted(Path(__file__).resolve().parent.glob("*.py"))
            runtime_relatives = [
                "upgrade_v2/visual_refine_l2/dynamic_simulator.py",
                "upgrade_v2/visual_refine_l2/renderer.py",
                "upgrade_v2/visual_refine_l2/vision.py",
                "upgrade_v2/visual_refine_l2/tracking.py",
                "upgrade_v2/visual_refine_l2/predicates.py",
                "upgrade_v2/l2r_ambiguity/event_memory.py",
                "upgrade_v2/l2r_ambiguity/probes.py",
                "tools/hold_features_reference.py",
            ]
            runtime_files = [args.run_root.parents[3] / relative for relative in runtime_relatives]
            callback_audit = args.run_root / "rounds/l2rar1_2_observation_and_reference/callback_equivalence.json"
            normalization_audit = args.run_root / "rounds/l2rar1_2_observation_and_reference/dense_stream_normalization.json"
            normalization_details = normalization_audit.with_suffix(".csv")
            select_manifest = args.run_root / "data/new_development/dev_select_rollout_manifest.csv"
            collection_lock = args.run_root / "locks/development_collection_lock.json"
            result = {
                "schema": "pathgraph_l2rar1_development_lock_v2",
                "status": "LOCKED_AFTER_COLLECTION_BEFORE_EVALUATION" if select_manifest.is_file() else "LOCKED_BEFORE_SELECT",
                "collection_lock": {"path": str(collection_lock.resolve()), "sha256": sha256_file(collection_lock) if collection_lock.is_file() else None},
                "protocol_sha256": sha256_file(args.protocol),
                "metadata_files": data,
                "dataset_files": dataset_files,
                "generator_lock_sha256": sha256_file(args.run_root / "data/new_development/generator_lock.json"),
                "reference_contract_sha256": sha256_file(args.run_root / "data/new_development/reference_contract.json"),
                "code_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in code_files],
                "runtime_files": [{"path": str(path.resolve()), "sha256": sha256_file(path)} for path in runtime_files if path.is_file()],
                "statistics_rule": {
                    "path": str((Path(__file__).resolve().parent / "evaluate_v2.py").resolve()),
                    "sha256": sha256_file(Path(__file__).resolve().parent / "evaluate_v2.py"),
                    "unit": protocol.get("statistics", {}).get("unit"),
                    "bootstrap_resamples": protocol.get("statistics", {}).get("bootstrap_resamples"),
                    "bootstrap_seed": protocol.get("statistics", {}).get("bootstrap_seed"),
                    "thresholds": protocol.get("thresholds"),
                },
                "callback_equivalence": {"path": str(callback_audit.resolve()), "sha256": sha256_file(callback_audit) if callback_audit.is_file() else None},
                "dense_stream_normalization": {"path": str(normalization_audit.resolve()), "sha256": sha256_file(normalization_audit) if normalization_audit.is_file() else None},
                "dense_stream_normalization_details": {"path": str(normalization_details.resolve()), "sha256": sha256_file(normalization_details) if normalization_details.is_file() else None},
                "sampling": "control_tick_20hz",
                "comparison_sampling": "action_end",
                "reference_version": "timestamp_aligned_proxy_hold_v2",
                "metric_version": "prefix_event_v2",
                "candidate_grid": protocol.get("candidate_grid"),
                "new_seeds": protocol.get("new_seeds"),
                "api_calls": 0,
                "training_jobs": 0,
                "api_key_read": False,
            }
            write_json(args.output, result)
        elif args.command == "evaluate-development":
            _verify_development_lock(args.development_lock, args.protocol, _run_root_for(args))
            result = evaluate_development(args.data_root, _json(args.protocol), args.output_root)
        elif args.command == "select-and-lock":
            result = selection.select_and_lock(args.development_root, args.protocol, args.output, args.output.parent.parent)
        elif args.command == "confirm":
            result = confirmation.confirm(args.selection_lock, _json(args.protocol), _json(args.inputs), args.output_root)
        else:
            result = handoff.handoff(args.run_root, args.protocol, args.output_root)
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        _write_command(run_root, ["python", *sys.argv[1:]] if argv is None else ["python", *argv], "SUCCEEDED")
        return 0
    except Exception as exc:
        _write_command(run_root, ["python", *sys.argv[1:]] if argv is None else ["python", *argv], "FAILED")
        print(json.dumps({"status": "EXECUTION_ERROR", "error": str(exc)}, ensure_ascii=False, indent=2), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
