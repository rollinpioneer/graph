from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .authorization import AuthorizationDenied, consume_authorization, validate_authorization
from .comparison import compare
from .instrumented import run_instrumented
from .ordinary import run_ordinary
from .protocol import make_protocol, load_protocol, sha256


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True, allow_nan=False) + "\n", encoding="utf-8")


def _failure(path: Path, reason: str) -> None:
    if path.is_dir():
        _write_json(path / "result.json", {"schema": "l2rar2_r14b_execution_result_v1", "status": "STOP_AFTER_EXECUTION_1", "reason": reason, "executions_used": 1, "automatic_retry": False, "recorded_at_utc": datetime.now(timezone.utc).isoformat()})
        (path / "STOP_AFTER_EXECUTION_1").write_text(reason + "\n", encoding="utf-8")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="upgrade_v2.l2r_reproducible_baseline.cli")
    subs = parser.add_subparsers(dest="command", required=True)
    plan = subs.add_parser("plan-protocol")
    plan.add_argument("--repo", type=Path, required=True)
    plan.add_argument("--root-family-id", required=True); plan.add_argument("--family-index", type=int, required=True)
    plan.add_argument("--family-seed", type=int, required=True); plan.add_argument("--rollout-seed", type=int, required=True)
    plan.add_argument("--case-id", required=True); plan.add_argument("--output", type=Path, required=True)
    for name in ("run-ordinary", "run-instrumented"):
        run = subs.add_parser(name)
        run.add_argument("--repo", type=Path, required=True); run.add_argument("--stage", required=True)
        run.add_argument("--protocol", type=Path, required=True); run.add_argument("--authorization", type=Path, required=True)
        run.add_argument("--output-root", type=Path, required=True)
        run.add_argument("--baseline-A", type=Path); run.add_argument("--ordinary-B", type=Path)
        run.add_argument("--ordinary-A-B-comparison", type=Path)
    cmp = subs.add_parser("compare")
    cmp.add_argument("--left", type=Path, required=True); cmp.add_argument("--right", type=Path, required=True)
    cmp.add_argument("--comparison", required=True); cmp.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "plan-protocol":
        protocol = make_protocol()
        requested = {"root_family_id": args.root_family_id, "family_index": args.family_index, "family_seed": args.family_seed, "rollout_seed": args.rollout_seed, "case_id": args.case_id}
        actual = {key: protocol[key] for key in requested}
        if actual != requested:
            raise SystemExit(f"fixed protocol mismatch: {actual!r}")
        _write_json(args.output, protocol)
        print(json.dumps({"status": "DRAFT_NOT_AUTHORIZED", "protocol_sha256": sha256(args.output)}, sort_keys=True))
        return 0
    if args.command == "compare":
        summary = compare(args.left.resolve(), args.right.resolve(), args.output_root.resolve(), args.comparison)
        print(json.dumps(summary, sort_keys=True)); return 0 if summary["all_main_gates_passed"] else 5

    repo = args.repo.resolve(); protocol_path = args.protocol.resolve(); protocol = load_protocol(protocol_path)
    protocol["protocol_sha256"] = sha256(protocol_path)
    stage = args.stage
    ordinary_stages = {"R14B_ORDINARY_BASELINE_A", "R14B_ORDINARY_REPEAT_B"}
    if (args.command == "run-instrumented" and stage != "R14B_INSTRUMENTED_C") or (args.command == "run-ordinary" and stage not in ordinary_stages):
        print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": "command and stage mismatch"})); return 3
    baseline: dict[str, Any] | None = None
    if stage == "R14B_ORDINARY_REPEAT_B":
        if not args.baseline_A or not (args.baseline_A / "result.json").is_file():
            print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": "baseline A required"})); return 3
        result = json.loads((args.baseline_A / "result.json").read_text(encoding="utf-8"))
        manifest = json.loads((args.baseline_A / "artifact_manifest.json").read_text(encoding="utf-8"))
        baseline = {"baseline_A_artifact_manifest_sha256": manifest["artifact_manifest_sha256"], "baseline_A_result_sha256": sha256(args.baseline_A / "result.json"), "baseline_A_environment_fingerprint_sha256": result.get("environment_fingerprint_sha256"), "baseline_A_model_fingerprint_sha256": result.get("model_fingerprint_sha256"), "baseline_A_review_status": "PASS"}
    if stage == "R14B_INSTRUMENTED_C":
        if not args.ordinary_B or not args.ordinary_A_B_comparison or not (args.ordinary_A_B_comparison).is_file():
            print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": "ordinary B and A/B PASS comparison required"})); return 3
        b_result = json.loads((args.ordinary_B / "result.json").read_text(encoding="utf-8")); b_manifest = json.loads((args.ordinary_B / "artifact_manifest.json").read_text(encoding="utf-8")); ab = json.loads(args.ordinary_A_B_comparison.read_text(encoding="utf-8"))
        if not ab.get("all_main_gates_passed"):
            print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": "A/B comparison is not PASS"})); return 3
        baseline = {"ordinary_A_vs_B_comparison_sha256": sha256(args.ordinary_A_B_comparison), "ordinary_A_vs_B_status": "PASS", "repeat_B_artifact_manifest_sha256": b_manifest["artifact_manifest_sha256"], "repeat_B_result_sha256": sha256(args.ordinary_B / "result.json")}
    try:
        auth = validate_authorization(args.authorization, repo=repo, protocol=protocol, protocol_sha256=protocol["protocol_sha256"], requested_output_root=args.output_root, stage=stage, baseline=baseline)
        consume_authorization(args.output_root, auth, args.authorization, repo=repo)
    except (AuthorizationDenied, OSError, ValueError) as exc:
        print(json.dumps({"status": "DENIED_BEFORE_MODEL_CONSTRUCTION", "reason": str(exc)}, sort_keys=True)); return 3
    try:
        result = run_instrumented(repo=repo, protocol=protocol, output_root=args.output_root) if args.command == "run-instrumented" else run_ordinary(repo=repo, protocol=protocol, output_root=args.output_root, stage=stage)
    except Exception as exc:
        _failure(args.output_root, f"execution failed after authorization consumption: {type(exc).__name__}: {exc}")
        print(json.dumps({"status": "STOP_AFTER_EXECUTION_1", "reason": str(exc)}, sort_keys=True)); return 4
    print(json.dumps(result, sort_keys=True)); return 0 if result["all_main_gates_passed"] else 5


if __name__ == "__main__":
    raise SystemExit(main())
