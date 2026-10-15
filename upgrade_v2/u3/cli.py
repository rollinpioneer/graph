"""CLI for the bounded, reproducible U3 hypothesized-graph workflow."""

from __future__ import annotations

import argparse
import importlib
import json
from pathlib import Path
from typing import Any

from .common import git_commit, read_csv, read_json, read_jsonl, repo_root, scan_paths, sha256_file, write_csv, write_json
from .evidence import build_compact_evidence, build_task_contract, supervision_ledger
from .execute import build_provider_request_table, provider_smoke, run_requests, verify_provider_run
from .handoff import build_u4_handoff, select_candidates
from .package import package_round, package_u3_complete
from .prompts import build_requests, refine_condition_once
from .schema import write_schema, write_vocabulary
from .score import score_candidates
from .stability import analyze_stability, compare_providers
from .validate import validate_candidates


def _path(value: str) -> Path:
    return Path(value).expanduser()


def _report(path: Path, title: str, values: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n" + "\n".join(f"- {key}: `{value}`" for key, value in values.items()) + "\n", encoding="utf-8")


def _print(value: dict[str, Any]) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status") in {"PASS", "U3_ENTRY_SECURITY_LOCKED", "U3_PROMPT_PACKAGE_REPAIRED", "U3_MODEL_EXECUTION_LOCKED", "REFINE_U3_PROMPT_ONCE", "U3_INCONCLUSIVE"} else 2


def command_check_entry(args: argparse.Namespace) -> int:
    manifest, handoff, fallback = read_json(args.prompt_manifest), read_json(args.handoff), read_json(args.fallback)
    expected = {
        "train_episode_count": 504, "train_root_family_count": 84, "train_segment_count": 2651,
        "train_transition_count": 2147, "test_gold_in_prompts": False, "unknown_retained": True,
        "llm_candidate_count": 0, "llm_status": "MODEL_EXECUTION_PENDING",
    }
    errors = [f"{key}={manifest.get(key)!r}, expected {wanted!r}" for key, wanted in expected.items() if manifest.get(key) != wanted]
    if handoff.get("status") != "U3_ENTRY_READY_WITH_BOUNDARY_FALLBACK": errors.append("handoff status is not U3_ENTRY_READY_WITH_BOUNDARY_FALLBACK")
    if fallback.get("automatic_boundary_supported") is not False: errors.append("automatic boundary support must remain false")
    if handoff.get("boundary", {}).get("fallback_required") is not True: errors.append("boundary fallback must remain required")
    value = {"status": "U3_ENTRY_SECURITY_LOCKED" if not errors else "REPAIR_INPUT_PATHS", "errors": errors, **{key: manifest.get(key) for key in expected}, "automatic_boundary_supported": fallback.get("automatic_boundary_supported"), "fallback_required": handoff.get("boundary", {}).get("fallback_required")}
    write_json(args.output, value); _report(args.report, "U3 entry check", value)
    return _print(value)


def command_hash_files(args: argparse.Namespace) -> int:
    manifest_rows = read_csv(args.manifest, delimiter="\t")
    root = repo_root(args.manifest)
    rows = []
    missing = []
    for row in manifest_rows:
        path = Path(row["path"])
        if not path.is_absolute(): path = root / path
        if not path.is_file():
            missing.append(str(path)); continue
        rows.append({"name": row["name"], "path": str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    write_csv(args.output, rows, delimiter="\t")
    value = {"status": "PASS" if not missing else "FAIL", "source_commit": git_commit(root), "file_count": len(rows), "missing": missing, "files": rows}
    write_json(args.lock, value)
    return _print(value)


def command_build_predicates(args: argparse.Namespace) -> int:
    importlib.import_module(args.simulator_module); importlib.import_module(args.event_schema_module)
    vocabulary = write_vocabulary(args.output)
    value = {"status": "PASS", "predicate_count": len(vocabulary["allowed_predicates"]), "forbidden_predicate_count": len(vocabulary["forbidden_predicates"]), "output": str(args.output)}
    _report(args.report, "U3 predicate vocabulary", value); return _print(value)


def command_build_contract(args: argparse.Namespace) -> int:
    importlib.import_module(args.simulator_module); importlib.import_module(args.event_schema_module)
    contract = build_task_contract(args.fallback_policy); write_json(args.output, contract)
    value = {"status": "PASS", "scope": contract["scope"], "sensor_field_count": len(contract["sensor_fields"]), "fallback_required": contract["boundary_fallback"]["fallback_required"]}
    _report(args.report, "U3 task contract", value); return _print(value)


def command_build_schema(args: argparse.Namespace) -> int:
    value = write_schema(args.predicate_vocabulary, args.output, args.example)
    result = {"status": "PASS", "schema": value.get("$id"), "output": str(args.output), "example": str(args.example)}
    return _print(result)


def command_build_evidence(args: argparse.Namespace) -> int:
    value = build_compact_evidence(args.train_segments, args.train_prototypes, args.train_transitions, args.fallback_clips, args.dataset_root, args.output_dir, args.max_representatives_per_cluster, args.max_transition_pairs)
    _report(args.report, "U3 compact evidence", value); return _print(value)


def command_build_requests(args: argparse.Namespace) -> int:
    value = build_requests(task_contract_path=args.task_contract, vocabulary_path=args.predicate_vocabulary, schema_path=args.schema, cluster_path=args.cluster_evidence, transition_path=args.transition_evidence, fallback_path=args.fallback_confirmations, ledger_path=args.supervision_ledger, conditions=list(args.conditions), replicates=args.replicates, ordering_seeds=[int(item) for item in args.ordering_seeds.split(",")], max_prompt_chars=args.max_prompt_chars, output=args.output, prompt_dir=args.prompt_dir, manifest=args.manifest)
    return _print(value)


def command_validate_prompt(args: argparse.Namespace) -> int:
    records = read_jsonl(args.requests); excluded = read_csv(args.excluded_splits); registry = read_json(args.evidence_registry); schema = read_json(args.schema)
    excluded_episode_ids = {row["episode_id"] for row in excluded}; errors: list[str] = []
    expected_count = args.expected_count
    if len(records) != expected_count: errors.append(f"expected {expected_count} requests, got {len(records)}")
    pairs = {(row.get("condition"), int(row.get("replicate", 0))) for row in records}
    if len(pairs) != expected_count: errors.append("condition/replicate grid incomplete")
    max_chars = 0
    for row in records:
        system = Path(row["system_prompt_path"]).read_text(encoding="utf-8"); user = Path(row["user_prompt_path"]).read_text(encoding="utf-8")
        prompt = system + user; max_chars = max(max_chars, len(prompt))
        if any(episode in prompt for episode in excluded_episode_ids): errors.append(f"{row['request_id']}: excluded episode ID in prompt")
        if row.get("test_gold_in_prompt") is not False: errors.append(f"{row['request_id']}: test gold flag is not false")
        if "test_gold" in prompt.lower(): errors.append(f"{row['request_id']}: forbidden test_gold field")
        if not row.get("prompt_sha256") or not row.get("schema_sha256"): errors.append(f"{row['request_id']}: missing SHA")
    result = {"status": "PASS" if not errors else "FAIL", "request_count": len(records), "condition_replicate_grid": len(pairs), "val_test_episode_ids_in_prompt": sum(any(episode in (Path(row["system_prompt_path"]).read_text() + Path(row["user_prompt_path"]).read_text()) for episode in excluded_episode_ids) for row in records), "evidence_registry_segment_count": len(registry["segment_ids"]), "schema_id": schema.get("$id"), "max_prompt_chars_observed": max_chars, "errors": errors}
    write_json(args.output, result); _report(args.report, "U3 prompt package gate", result); return _print(result)


def command_refine_prompt(args: argparse.Namespace) -> int:
    return _print(refine_condition_once(requests=args.requests, condition=args.condition, prompt_dir=args.prompt_dir, output=args.output, manifest=args.manifest))


def command_secret_scan(args: argparse.Namespace) -> int:
    root = repo_root(Path.cwd())
    result = scan_paths([Path(item) for item in args.paths], root, include_git_diff=args.git_diff)
    write_json(args.output, result); return _print(result)


def command_smoke(args: argparse.Namespace) -> int:
    value = provider_smoke(provider=args.provider, model=args.model, schema=args.schema, output=args.output, log=args.log)
    value["status"] = "PASS"; write_json(args.output, value); return _print(value)


def command_freeze(args: argparse.Namespace) -> int:
    qwen, deepseek = read_json(args.qwen_smoke), read_json(args.deepseek_smoke)
    if not qwen.get("schema_valid") or not deepseek.get("schema_valid"): raise RuntimeError("both provider smokes must pass before execution lock")
    root = repo_root(args.output)
    lock = {"locked_before_main_requests": True, "source_commit": git_commit(root), "input_modality": "text_only", "qwen": {"model": args.qwen_model, "role": "primary", "request_count": 9, "concurrency": 3, "response_mode": qwen["response_mode"]}, "deepseek": {"model": args.deepseek_model, "role": "cross_model_check", "request_count": 3, "concurrency": 3, "response_mode": "json_object_local_validation", "schema_repair_limit_per_request": 1}, "max_output_tokens": 5000, "candidate_call_cap": 15, "temperature": "provider_default", "thinking": "enabled", "web_search": False, "tools": False, "test_gold_in_prompt": False, "model_change_after_output": False}
    write_json(args.output, lock)
    rows = [{"path": str(path), "sha256": sha256_file(path)} for path in (args.output, args.schema, args.requests)]
    args.checksums.parent.mkdir(parents=True, exist_ok=True); args.checksums.write_text("".join(f"{row['sha256']}  {row['path']}\n" for row in rows), encoding="utf-8")
    return _print({"status": "U3_MODEL_EXECUTION_LOCKED", "qwen_response_mode": qwen["response_mode"], "checksums": rows})


def command_build_provider_table(args: argparse.Namespace) -> int:
    value = build_provider_request_table(provider=args.provider, requests=args.requests, model=args.model, output=args.output, all_replicates=args.all_replicates, replicate=args.replicate, expected_count=args.expected_count)
    return _print(value)


def command_run_requests(args: argparse.Namespace) -> int:
    value = run_requests(provider=args.provider, request_table=args.request_table, model=args.model, concurrency=args.concurrency, max_output_tokens=args.max_output_tokens, timeout=args.timeout, network_retries=args.network_retries, output_root=args.output_root, status_table=args.status_table, usage_table=args.usage_table, schema=args.schema, repair_limit=args.schema_repair_limit, expected_count=args.expected_count)
    value["status"] = "PASS" if value["http_success"] == value["expected"] else "FAIL"; return _print(value)


def command_verify(args: argparse.Namespace) -> int:
    return _print(verify_provider_run(provider=args.provider, status=args.status, response_root=args.response_root, expected=args.expected, schema=args.schema, output=args.output, report=args.report))


def command_validate_candidates(args: argparse.Namespace) -> int:
    return _print(validate_candidates(qwen_root=args.qwen_root, deepseek_root=args.deepseek_root, schema_path=args.schema, vocabulary_path=args.predicate_vocabulary, evidence_registry_path=args.evidence_registry, cluster_evidence_path=args.cluster_evidence, transition_evidence_path=args.transition_evidence, excluded_splits=args.excluded_splits, output=args.output, details=args.details, report=args.report))


def command_score(args: argparse.Namespace) -> int:
    return _print(score_candidates(hard_checks=args.hard_checks, candidate_roots=list(args.candidate_roots), cluster_evidence_path=args.cluster_evidence, transition_evidence_path=args.transition_evidence, ledger_path=args.supervision_ledger, output=args.output, condition_summary=args.condition_summary, report=args.report))


def command_stability(args: argparse.Namespace) -> int:
    return _print(analyze_stability(qwen_root=args.qwen_root, hard_checks=args.hard_checks, conditions=list(args.conditions), output=args.output, pairwise=args.pairwise, report=args.report))


def command_compare(args: argparse.Namespace) -> int:
    return _print(compare_providers(qwen_root=args.qwen_root, deepseek_root=args.deepseek_root, replicate=args.replicate, output=args.output, report=args.report))


def command_select(args: argparse.Namespace) -> int:
    return _print(select_candidates(scores=args.scores, stability=args.stability, pairwise_distance=args.pairwise_distance, cross_provider=args.cross_provider, candidate_roots=list(args.candidate_roots), max_extra_diverse=args.max_extra_diverse, diversity_min_distance=args.diversity_min_distance, diversity_max_score_drop=args.diversity_max_score_drop, output=args.output, copy_dir=args.copy_dir, report=args.report))


def command_handoff(args: argparse.Namespace) -> int:
    value = build_u4_handoff(selected=args.selected, candidate_dir=args.candidate_dir, task_contract=args.task_contract, predicate_vocabulary=args.predicate_vocabulary, supervision_ledger=args.supervision_ledger, fallback_policy=args.fallback_policy, contradiction_queue=args.contradiction_queue, output=args.output, report=args.report)
    return _print({"status": "PASS", "decision": value["u3_decision"], "selected_count": len(value["selected_candidates"])})


def command_package_round(args: argparse.Namespace) -> int:
    return _print({"status": "PASS", **package_round(round_dir=args.round_dir, output=args.output, max_file_mb=args.max_file_mb)})


def command_package_complete(args: argparse.Namespace) -> int:
    return _print({"status": "PASS", **package_u3_complete(u3_root=args.u3_root, final_root=args.final_root, output=args.output, max_file_mb=args.max_file_mb)})


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m upgrade_v2.u3.cli")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("check-entry"); p.add_argument("--prompt-manifest", type=_path, required=True); p.add_argument("--handoff", type=_path, required=True); p.add_argument("--fallback", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_check_entry)
    p = sub.add_parser("hash-files"); p.add_argument("--manifest", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--lock", type=_path, required=True); p.set_defaults(func=command_hash_files)
    p = sub.add_parser("build-predicate-vocabulary"); p.add_argument("--simulator-module", required=True); p.add_argument("--event-schema-module", required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_build_predicates)
    p = sub.add_parser("build-task-contract"); p.add_argument("--simulator-module", required=True); p.add_argument("--event-schema-module", required=True); p.add_argument("--fallback-policy", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_build_contract)
    p = sub.add_parser("build-strict-schema"); p.add_argument("--predicate-vocabulary", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--example", type=_path, required=True); p.set_defaults(func=command_build_schema)
    p = sub.add_parser("build-compact-evidence"); p.add_argument("--train-segments", type=_path, required=True); p.add_argument("--train-prototypes", type=_path, required=True); p.add_argument("--train-transitions", type=_path, required=True); p.add_argument("--fallback-clips", type=_path, required=True); p.add_argument("--dataset-root", type=_path, required=True); p.add_argument("--max-representatives-per-cluster", type=int, default=4); p.add_argument("--max-transition-pairs", type=int, default=60); p.add_argument("--output-dir", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_build_evidence)
    p = sub.add_parser("build-requests"); p.add_argument("--task-contract", type=_path, required=True); p.add_argument("--predicate-vocabulary", type=_path, required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--cluster-evidence", type=_path, required=True); p.add_argument("--transition-evidence", type=_path, required=True); p.add_argument("--fallback-confirmations", type=_path, required=True); p.add_argument("--supervision-ledger", type=_path, required=True); p.add_argument("--conditions", nargs="+", required=True); p.add_argument("--replicates", type=int, required=True); p.add_argument("--ordering-seeds", required=True); p.add_argument("--max-prompt-chars", type=int, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--prompt-dir", type=_path, required=True); p.add_argument("--manifest", type=_path, required=True); p.set_defaults(func=command_build_requests)
    p = sub.add_parser("validate-prompt-package"); p.add_argument("--requests", type=_path, required=True); p.add_argument("--excluded-splits", type=_path, required=True); p.add_argument("--evidence-registry", type=_path, required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--expected-count", type=int, default=9); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_validate_prompt)
    p = sub.add_parser("refine-prompt-once"); p.add_argument("--requests", type=_path, required=True); p.add_argument("--condition", required=True); p.add_argument("--prompt-dir", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--manifest", type=_path, required=True); p.set_defaults(func=command_refine_prompt)
    p = sub.add_parser("secret-scan"); p.add_argument("--paths", nargs="+", required=True); p.add_argument("--git-diff", action="store_true"); p.add_argument("--output", type=_path, required=True); p.set_defaults(func=command_secret_scan)
    p = sub.add_parser("provider-smoke"); p.add_argument("--provider", choices=("qwen", "deepseek"), required=True); p.add_argument("--model", required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--log", type=_path, required=True); p.set_defaults(func=command_smoke)
    p = sub.add_parser("freeze-execution-lock"); p.add_argument("--qwen-smoke", type=_path, required=True); p.add_argument("--deepseek-smoke", type=_path, required=True); p.add_argument("--qwen-model", required=True); p.add_argument("--deepseek-model", required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--requests", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--checksums", type=_path, required=True); p.set_defaults(func=command_freeze)
    p = sub.add_parser("build-provider-request-table"); p.add_argument("--provider", choices=("qwen", "deepseek"), required=True); p.add_argument("--requests", type=_path, required=True); p.add_argument("--model", required=True); p.add_argument("--all-replicates", action="store_true"); p.add_argument("--replicate", type=int); p.add_argument("--expected-count", type=int); p.add_argument("--output", type=_path, required=True); p.set_defaults(func=command_build_provider_table)
    p = sub.add_parser("run-requests"); p.add_argument("--provider", choices=("qwen", "deepseek"), required=True); p.add_argument("--request-table", type=_path, required=True); p.add_argument("--model", required=True); p.add_argument("--concurrency", type=int, required=True); p.add_argument("--max-output-tokens", type=int, required=True); p.add_argument("--timeout", type=int, required=True); p.add_argument("--network-retries", type=int, required=True); p.add_argument("--schema-repair-limit", type=int, default=0); p.add_argument("--schema", type=_path); p.add_argument("--expected-count", type=int); p.add_argument("--output-root", type=_path, required=True); p.add_argument("--status-table", type=_path, required=True); p.add_argument("--usage-table", type=_path, required=True); p.add_argument("--resume", action="store_true"); p.set_defaults(func=command_run_requests)
    p = sub.add_parser("verify-provider-run"); p.add_argument("--provider", choices=("qwen", "deepseek"), required=True); p.add_argument("--status", type=_path, required=True); p.add_argument("--response-root", type=_path, required=True); p.add_argument("--expected", type=int, required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_verify)
    p = sub.add_parser("validate-candidates"); p.add_argument("--qwen-root", type=_path, required=True); p.add_argument("--deepseek-root", type=_path, required=True); p.add_argument("--schema", type=_path, required=True); p.add_argument("--predicate-vocabulary", type=_path, required=True); p.add_argument("--evidence-registry", type=_path, required=True); p.add_argument("--cluster-evidence", type=_path, required=True); p.add_argument("--transition-evidence", type=_path, required=True); p.add_argument("--excluded-splits", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--details", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_validate_candidates)
    p = sub.add_parser("score-candidates"); p.add_argument("--hard-checks", type=_path, required=True); p.add_argument("--candidate-roots", type=_path, nargs="+", required=True); p.add_argument("--cluster-evidence", type=_path, required=True); p.add_argument("--transition-evidence", type=_path, required=True); p.add_argument("--supervision-ledger", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--condition-summary", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_score)
    p = sub.add_parser("analyze-stability"); p.add_argument("--qwen-root", type=_path, required=True); p.add_argument("--hard-checks", type=_path, required=True); p.add_argument("--conditions", nargs="+", required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--pairwise", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_stability)
    p = sub.add_parser("compare-providers"); p.add_argument("--qwen-root", type=_path, required=True); p.add_argument("--deepseek-root", type=_path, required=True); p.add_argument("--replicate", type=int, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_compare)
    p = sub.add_parser("select-candidates"); p.add_argument("--scores", type=_path, required=True); p.add_argument("--stability", type=_path, required=True); p.add_argument("--pairwise-distance", type=_path, required=True); p.add_argument("--cross-provider", type=_path, required=True); p.add_argument("--candidate-roots", type=_path, nargs="+", required=True); p.add_argument("--one-qwen-per-condition", action="store_true"); p.add_argument("--max-extra-diverse", type=int, required=True); p.add_argument("--diversity-min-distance", type=float, required=True); p.add_argument("--diversity-max-score-drop", type=float, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--copy-dir", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_select)
    p = sub.add_parser("build-u4-handoff"); p.add_argument("--selected", type=_path, required=True); p.add_argument("--candidate-dir", type=_path, required=True); p.add_argument("--task-contract", type=_path, required=True); p.add_argument("--predicate-vocabulary", type=_path, required=True); p.add_argument("--supervision-ledger", type=_path, required=True); p.add_argument("--fallback-policy", type=_path, required=True); p.add_argument("--contradiction-queue", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--report", type=_path, required=True); p.set_defaults(func=command_handoff)
    p = sub.add_parser("package-round"); p.add_argument("--round-dir", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--max-file-mb", type=int, default=200); p.set_defaults(func=command_package_round)
    p = sub.add_parser("package-u3-complete"); p.add_argument("--u3-root", type=_path, required=True); p.add_argument("--round-zip-dir", type=_path); p.add_argument("--final-root", type=_path, required=True); p.add_argument("--output", type=_path, required=True); p.add_argument("--max-file-mb", type=int, default=200); p.set_defaults(func=command_package_complete)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
