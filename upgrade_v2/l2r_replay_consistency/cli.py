from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .authorization import AuthorizationDenied, consume_authorization, runner_file_hashes, validate_authorization
from .protocol import load_json, load_protocol, sha256, verify_file_hashes


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="l2r_replay_consistency")
    subparsers = parser.add_subparsers(dest="command", required=True)
    preflight = subparsers.add_parser("preflight")
    ordinary = subparsers.add_parser("ordinary")
    for command in (preflight, ordinary):
        command.add_argument("--repo", type=Path, default=Path.cwd())
        command.add_argument("--protocol", type=Path, required=True)
        command.add_argument("--cache-root", type=Path, required=True)
    ordinary.add_argument("--authorization", type=Path, required=True)
    ordinary.add_argument("--output-root", type=Path, required=True)
    return parser


def _static_preflight(repo: Path, protocol_path: Path, cache_root: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    protocol = load_protocol(protocol_path)
    verify_file_hashes(repo, protocol["source_file_hashes"], "source files")
    verify_file_hashes(cache_root, protocol["cache_file_hashes"], "cache files")
    metadata = load_json(cache_root / "metadata.json")
    exact = {
        "root_family_id": protocol["root_family_id"],
        "case_id": protocol["case_id"],
        "family_seed": protocol["family_seed"],
        "rollout_seed": protocol["rollout_seed"],
        "program_sha256": protocol["program_sha256"],
        "repair_version": protocol["repair_version"],
        "collection_version": protocol["collection_version"],
    }
    mismatches = [key for key, expected in exact.items() if metadata.get(key) != expected]
    if metadata.get("control_variant", {}).get("variant_id") != protocol["control_variant"]:
        mismatches.append("control_variant")
    if int(metadata.get("action_end_observation_count", -1)) != int(protocol["expected_action_count"]):
        mismatches.append("expected_action_count")
    if int(metadata.get("dense_observation_count", -1)) != int(protocol["expected_control_callback_count"]):
        mismatches.append("expected_control_callback_count")
    expected_renderer_count = int(protocol["expected_control_callback_count"]) + int(protocol["expected_action_end_callback_count"])
    if expected_renderer_count != int(protocol["expected_renderer_callback_count"]):
        mismatches.append("expected_renderer_callback_count")
    if mismatches:
        raise ValueError("cache metadata mismatch: " + ", ".join(mismatches))
    info = {
        "status": "PASS_STATIC_ONLY",
        "protocol_sha256": sha256(protocol_path),
        "runner_file_hashes": runner_file_hashes(repo),
        "cache_root": str(cache_root.resolve()),
        "case_id": protocol["case_id"],
        "rollout_seed": protocol["rollout_seed"],
        "physics_executions": 0,
    }
    return protocol, info


def _write_failure(output_root: Path, reason: str) -> None:
    if not output_root.is_dir():
        return
    payload = {
        "schema": "l2rar2_r14_ordinary_replay_result_v1",
        "status": "STOP_AFTER_EXECUTION_1",
        "reason": reason,
        "executions_used": 1 if (output_root / "model_construction_started.json").is_file() else 0,
        "automatic_retry": False,
        "instrumented_replay_executions": 0,
        "r16_calibration_executions": 0,
        "r16_development_executions": 0,
        "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
        "runtime": {"python": platform.python_version(), "python_executable": sys.executable},
    }
    (output_root / "result.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output_root / "STOP_AFTER_EXECUTION_1").write_text(reason + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    repo = args.repo.resolve()
    try:
        protocol, info = _static_preflight(repo, args.protocol, args.cache_root)
        if args.command == "preflight":
            print(json.dumps(info, sort_keys=True))
            return 0
        authorization = validate_authorization(
            args.authorization,
            repo=repo,
            protocol_sha256=info["protocol_sha256"],
            requested_output_root=args.output_root,
        )
        consume_authorization(args.output_root, authorization, args.authorization)
    except (AuthorizationDenied, OSError, ValueError) as exc:
        print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": str(exc)}, sort_keys=True))
        return 3

    try:
        from .ordinary import run_ordinary

        result = run_ordinary(protocol, args.cache_root, args.output_root)
    except Exception as exc:
        _write_failure(args.output_root, f"ordinary replay failed after authorization consumption: {type(exc).__name__}: {exc}")
        print(json.dumps({"status": "STOP_AFTER_EXECUTION_1", "reason": str(exc)}, sort_keys=True))
        return 4
    print(json.dumps(result, sort_keys=True))
    return 0 if result["all_main_gates_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
