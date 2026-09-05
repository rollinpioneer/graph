"""CLI for the U2 audit correction and U3 train-only handoff."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from .export import export_u3_train
from .metrics import recompute_boundaries, recompute_reward
from .primitives import write_json


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _repo_root(path: Path) -> Path:
    path = path.resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _freeze_handoff(args: argparse.Namespace) -> int:
    u2_root = args.u2_root.resolve()
    output = args.output.resolve()
    repo_root = _repo_root(u2_root)
    boundary_status = _read_json_if_exists(args.boundary_status)
    reward_source = _read_json_if_exists(args.reward_source)
    prompt_manifest = _read_json_if_exists(args.prompt_manifest)
    old_handoff_path = u2_root / "final_v1" / "u3_u4_handoff.json"
    old_handoff = _read_json_if_exists(old_handoff_path)
    source_lock = _read_json_if_exists(u2_root / "segment_representation_v1" / "configs" / "boundary_source_lock.json")
    fallback_policy = _read_json_if_exists(args.fallback_policy)
    locked_source = source_lock.get("source_method", old_handoff.get("boundary_source", "unknown"))
    corrected_metrics = _locked_boundary_metrics(args.boundary_status.parent, locked_source)
    historical_metrics = _historical_locked_boundary_metrics(args.boundary_status.parent, locked_source)
    handoff = {
        "schema": "u3_minimal_handoff_v1",
        "status": "U3_ENTRY_READY_WITH_BOUNDARY_FALLBACK",
        "source_commit": _git_commit(repo_root),
        "original_archive_decision": old_handoff.get("u2_decision", "GO_U3_WITH_BOUNDARY_FALLBACK"),
        "scope": "same stochastic simulator family only",
        "physical_generalization_eligible": False,
        "original_task_generalization_eligible": False,
        "boundary": {
            "automatic_boundary_supported": False,
            "fallback_required": True,
            "locked_source": locked_source,
            "original_metrics": {
                "source": "u2_final_report_and_frozen_tables",
                "status": "historical_preserved",
                "locked_source_test_tol2": historical_metrics,
            },
            "corrected_metrics": {
                "status": boundary_status.get("status", "NOT_RUN"),
                "path": _relative(boundary_status.get("output_root", ""), repo_root),
                "models_evaluated": boundary_status.get("models_evaluated", []),
                "missing_prediction_count": boundary_status.get("missing_prediction_count", 0),
                "locked_source_test_tol2": corrected_metrics,
                "old_vs_corrected_path": _relative(args.boundary_status.parent / "old_vs_corrected_metrics.csv", repo_root),
            },
        },
        "reward": {
            "status": reward_source.get("status", "NOT_RUN"),
            "split": reward_source.get("split", "test"),
            "path": _relative(reward_source.get("output_root", ""), repo_root),
            "closed_full_input_cycle_residual": "not_measured",
        },
        "u3_prompt_input": {
            "status": prompt_manifest.get("status", "NOT_READY"),
            "train_only_verified": prompt_manifest.get("input_split") == "train" and prompt_manifest.get("excluded_split_counts", {}).get("test", 0) > 0,
            "path": _relative(args.prompt_manifest, repo_root),
            "test_gold_in_prompts": False,
            "train_root_family_count": prompt_manifest.get("train_root_family_count"),
            "train_segment_count": prompt_manifest.get("train_segment_count"),
        },
        "shared_gold_calibration": fallback_policy.get("shared_calibration_supervision", {}),
        "additional_fallback": fallback_policy.get("additional_fallback", {}),
        "llm_candidates": {
            "count": 0,
            "request_count": prompt_manifest.get("candidate_request_count", 0),
            "status": "MODEL_EXECUTION_PENDING",
            "raw_responses_present": False,
            "parsed_graphs_present": False,
        },
        "allowed_next": ["U3 simulator-scoped candidate graph proposal", "U4 simulator-scoped data validation"],
    }
    write_json(output, handoff)
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "superseded_notes.md").write_text(
        "# Superseded U2 claims\n\n"
        "- Historical boundary/reward tables remain preserved for provenance.\n"
        "- Boundary values in `boundaries_v2/` use episode-local dynamic-program matching.\n"
        "- Reward values in `reward_v2/` use incoming-transition attribution and test-only family bootstrap.\n"
        "- The historical current-frame/history formula comparison is `not_computed`; no gain claim is carried forward.\n"
        "- `weak-only` and `budget0` retain shared small-gold calibration supervision and are not zero-gold claims.\n",
        encoding="utf-8",
    )
    (output.parent / "u2_u3_handoff_report.md").write_text(
        "# U2/U3 minimal handoff\n\n"
        f"- Original archive decision: `{handoff['original_archive_decision']}`\n"
        f"- Corrected boundary status: `{handoff['boundary']['corrected_metrics']['status']}`\n"
        f"- Reward status: `{handoff['reward']['status']}`\n"
        f"- U3 prompt input: `{handoff['u3_prompt_input']['status']}`\n"
        f"- LLM candidates completed: `{handoff['llm_candidates']['count']}`; status `{handoff['llm_candidates']['status']}`\n"
        "- Scope: same stochastic simulator family only; automatic boundary fallback remains required.\n",
        encoding="utf-8",
    )
    print(json.dumps(handoff, ensure_ascii=False, indent=2))
    return 0


def _read_json_if_exists(path: str | Path | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.is_file():
        return {}
    try:
        value = json.loads(p.read_text(encoding="utf-8"))
        return value if isinstance(value, dict) else {}
    except json.JSONDecodeError:
        return {}


def _locked_boundary_metrics(boundary_root: Path, locked_source: str) -> dict:
    """Return the actual corrected locked-source test result when available."""

    metrics_path = boundary_root / "boundary_metrics_by_model.csv"
    if not metrics_path.is_file():
        return {"status": "not_estimable", "reason": "corrected metric table missing"}
    with metrics_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("split") == "test" and row.get("tolerance") == "2" and (row.get("model") == locked_source or row.get("model", "").endswith(f":{locked_source}")):
                fields = (
                    "boundary_f1",
                    "boundary_precision",
                    "boundary_recall",
                    "boundary_mae",
                    "boundary_root_family_macro",
                    "boundary_root_family_macro_ci_low",
                    "boundary_root_family_macro_ci_high",
                    "unknown_frame_rate",
                    "boundary_estimability",
                )
                return {"status": row.get("boundary_estimability", "not_estimable"), **{name: row.get(name, "not_estimable") or "not_estimable" for name in fields}}
    return {"status": "not_estimable", "reason": f"no corrected test tolerance=2 row for locked source {locked_source}"}


def _historical_locked_boundary_metrics(boundary_root: Path, locked_source: str) -> dict:
    """Read, but never overwrite, the frozen pre-correction comparison row."""

    comparison_path = boundary_root / "old_vs_corrected_metrics.csv"
    if not comparison_path.is_file():
        return {"status": "not_estimable", "reason": "historical comparison table missing"}
    with comparison_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row.get("model") == locked_source and row.get("old_split") == "test":
                return {
                    "status": row.get("corrected_estimability", "not_estimable"),
                    "boundary_f1": row.get("old_boundary_f1_tol2", "not_estimable") or "not_estimable",
                    "correction_delta": row.get("corrected_minus_old_f1_tol2", "not_estimable") or "not_estimable",
                    "source": row.get("old_source", "historical_preserved"),
                }
    return {"status": "not_estimable", "reason": f"no historical test row for locked source {locked_source}"}


def _relative(value: str | Path, repo_root: Path) -> str:
    if not value:
        return ""
    path = Path(value)
    try:
        return str(path.resolve().relative_to(repo_root))
    except (ValueError, OSError):
        return str(value)


def _git_commit(repo_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m upgrade_v2.u2_handoff_patch.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    command = sub.add_parser("recompute-boundaries")
    command.add_argument("--u2-root", type=_path, required=True)
    command.add_argument("--splits", default="val,test")
    command.add_argument("--tolerances", default="1,2")
    command.add_argument("--group-key", default="root_family_id")
    command.add_argument("--output", type=_path, required=True)
    command.set_defaults(func=lambda args: _run_boundaries(args))

    command = sub.add_parser("recompute-reward")
    command.add_argument("--u2-root", type=_path, required=True)
    command.add_argument("--split", default="test")
    command.add_argument("--potential-lock", type=_path, required=False)
    command.add_argument("--boundary-lock", type=_path, required=False)
    command.add_argument("--bootstrap", type=int, default=5000)
    command.add_argument("--seed", type=int, default=20260957)
    command.add_argument("--output", type=_path, required=True)
    command.set_defaults(func=lambda args: _run_reward(args))

    command = sub.add_parser("export-u3-train")
    command.add_argument("--u2-root", type=_path, required=True)
    command.add_argument("--split", default="train")
    command.add_argument("--include-unknown", default="true")
    command.add_argument("--fallback-max-clips", type=int, default=30)
    command.add_argument("--output", type=_path, required=True)
    command.set_defaults(func=lambda args: _run_export(args))

    command = sub.add_parser("freeze-handoff")
    command.add_argument("--u2-root", type=_path, required=True)
    command.add_argument("--boundary-status", type=_path, required=True)
    command.add_argument("--reward-source", type=_path, required=True)
    command.add_argument("--prompt-manifest", type=_path, required=True)
    command.add_argument("--fallback-policy", type=_path, required=True)
    command.add_argument("--output", type=_path, required=True)
    command.set_defaults(func=_freeze_handoff)
    return parser


def _run_boundaries(args: argparse.Namespace) -> int:
    status = recompute_boundaries(args.u2_root, args.output, tuple(x for x in args.splits.split(",") if x), tuple(int(x) for x in args.tolerances.split(",") if x), args.group_key)
    status["output_root"] = str(args.output.resolve())
    write_json(args.output / "recompute_status.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "BOUNDARY_CACHE_RECOMPUTED" else 2


def _run_reward(args: argparse.Namespace) -> int:
    status = recompute_reward(
        args.u2_root,
        args.output,
        args.split,
        args.bootstrap,
        args.seed,
        args.potential_lock,
        args.boundary_lock,
    )
    status["output_root"] = str(args.output.resolve())
    write_json(args.output / "reward_source_map.json", status)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0 if status["status"] == "REWARD_CACHE_RECOMPUTED" else 2


def _run_export(args: argparse.Namespace) -> int:
    status = export_u3_train(args.u2_root, args.output, args.split, str(args.include_unknown).lower() in {"1", "true", "yes"}, args.fallback_max_clips)
    print(json.dumps(status, ensure_ascii=False, indent=2))
    return 0


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
