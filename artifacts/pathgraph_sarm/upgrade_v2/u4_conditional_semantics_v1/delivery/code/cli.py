"""U4-R1 command line with explicit, resumable compatibility arguments."""
from __future__ import annotations

import argparse
import json
import shutil
import unittest
from collections import defaultdict
from pathlib import Path

from .confirm import build_lock, build_occurrences_confirm, evaluate as evaluate_confirmation
from .evaluator_v2 import compare_old_new, compare_contract_sources, graph_metrics, rescore
from .fresh_families import generate
from .historical_lock import freeze_history, verify_history
from .io import read_csv, read_json, read_jsonl, sha256_file, write_csv, write_json, write_jsonl
from .multigraph import build_graphs
from .occurrence_table import build_occurrences, build_u2_train_occurrences, write_occurrence_summary
from .package import package_complete, package_round
from .select_graph import evaluate_graphs, select
from .separability import fit_baselines, summarize_mixed


def _status(value: dict) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status", "PASS") not in {"FAIL", "BLOCKED", "ERROR"} else 2


def _path(value: str | None) -> Path | None:
    return Path(value) if value else None


def _first_path(values: list[Path] | None, fallback: Path | None = None) -> Path | None:
    return values[0] if values else fallback


def _csv_values(value: str | None, default: tuple[str, ...]) -> list[str]:
    if not value:
        return list(default)
    return [item.strip() for item in value.split(",") if item.strip()]


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upgrade_v2.u4_conditional.cli")
    s = p.add_subparsers(dest="command", required=True)
    x = s.add_parser("freeze-history"); x.add_argument("--repo", type=Path); x.add_argument("--historical-root", type=Path); x.add_argument("--checkpoint", type=Path); x.add_argument("--protocol", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--hash-table", type=Path); x.add_argument("--report", type=Path); x.add_argument("--paths", type=Path, nargs="*")
    x = s.add_parser("validate-inputs"); x.add_argument("--lock", type=Path, required=True)
    s.add_parser("run-evaluator-tests")
    x = s.add_parser("compare-evaluator-contracts"); x.add_argument("--old", type=Path); x.add_argument("--old-evaluator-source", type=Path, dest="old_source"); x.add_argument("--new", type=Path); x.add_argument("--new-evaluator-module"); x.add_argument("--output", type=Path, required=True); x.add_argument("--report", type=Path)
    x = s.add_parser("rescore-frozen-confirmation"); x.add_argument("--graph", type=Path); x.add_argument("--graphs", type=Path, nargs="+"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--continuations", type=Path); x.add_argument("--family-lock", type=Path); x.add_argument("--evaluator-version"); x.add_argument("--statistics-unit"); x.add_argument("--bootstrap", type=int, default=5000); x.add_argument("--bootstrap-seed", type=int, default=910201); x.add_argument("--output", type=Path, required=True); x.add_argument("--per-family", type=Path); x.add_argument("--paired", type=Path); x.add_argument("--report", type=Path)
    x = s.add_parser("compare-old-new-metrics"); x.add_argument("--old", type=Path, required=True); x.add_argument("--new", type=Path, required=True); x.add_argument("--old-paired", type=Path); x.add_argument("--new-paired", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--report", type=Path)
    x = s.add_parser("build-semantic-occurrence-table"); x.add_argument("--rollouts", type=Path); x.add_argument("--u2-dataset", type=Path); x.add_argument("--train-segments", type=Path); x.add_argument("--train-boundaries", type=Path); x.add_argument("--u4-dev-occurrences", type=Path); x.add_argument("--history-steps", type=int, default=8); x.add_argument("--repo", type=Path); x.add_argument("--split", default="development"); x.add_argument("--fit-splits"); x.add_argument("--output", type=Path, required=True); x.add_argument("--manifest", type=Path); x.add_argument("--summary", type=Path); x.add_argument("--report", type=Path)
    x = s.add_parser("summarize-mixed-pairs"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--fit-splits"); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("fit-separability-baselines"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--train-splits"); x.add_argument("--validation-split"); x.add_argument("--models"); x.add_argument("--feature-groups"); x.add_argument("--class-balance"); x.add_argument("--model-dir", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--table", type=Path); x.add_argument("--per-pair", type=Path); x.add_argument("--report", type=Path)
    x = s.add_parser("build-conditional-graphs"); x.add_argument("--raw-graph", type=Path); x.add_argument("--base-graph", type=Path, dest="base_graph"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--separability", type=Path); x.add_argument("--models", type=Path); x.add_argument("--fit-splits"); x.add_argument("--output-dir", type=Path, required=True); x.add_argument("--edit-log", type=Path); x.add_argument("--manifest", type=Path)
    x = s.add_parser("evaluate-conditional-graphs"); x.add_argument("--graphs", type=Path, nargs="+"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--split"); x.add_argument("--evaluator-version"); x.add_argument("--statistics-unit"); x.add_argument("--output", type=Path, required=True); x.add_argument("--table", type=Path); x.add_argument("--per-family", type=Path); x.add_argument("--report", type=Path)
    x = s.add_parser("select-conditional-graph"); x.add_argument("--metrics", type=Path, required=True); x.add_argument("--require-transition-drop-max", type=float, default=.05); x.add_argument("--max-ambiguous-guard-rate", type=float, default=.10); x.add_argument("--output", type=Path, required=True); x.add_argument("--lock", type=Path); x.add_argument("--report", type=Path)
    x = s.add_parser("generate-fresh-families"); x.add_argument("--output", type=Path); x.add_argument("--output-root", type=Path); x.add_argument("--lock", type=Path); x.add_argument("--family-lock", type=Path, dest="family_lock"); x.add_argument("--seed", "--generator-seed", type=int, default=910500); x.add_argument("--count", "--family-count", type=int, default=36); x.add_argument("--rollout-seed-base", type=int, default=9110000); x.add_argument("--rollouts-per-family", type=int, default=4); x.add_argument("--family-prefix", default="u4r1_confirm"); x.add_argument("--manifest", type=Path)
    x = s.add_parser("infer-fresh-boundaries"); x.add_argument("--checkpoint", type=Path, required=True); x.add_argument("--expected-sha256"); x.add_argument("--rollouts", type=Path); x.add_argument("--rollout-root", type=Path, dest="rollout_root"); x.add_argument("--threshold", type=float, default=.5); x.add_argument("--device", default="auto"); x.add_argument("--workers", type=int); x.add_argument("--output", type=Path); x.add_argument("--output-root", type=Path, dest="output_root"); x.add_argument("--manifest", type=Path); x.add_argument("--inference-manifest", type=Path, dest="inference_manifest")
    x = s.add_parser("build-confirmation-occurrences"); x.add_argument("--rollouts", type=Path); x.add_argument("--rollout-root", type=Path, dest="rollout_root"); x.add_argument("--predictions", type=Path); x.add_argument("--boundary-root", type=Path); x.add_argument("--cluster-reference", type=Path); x.add_argument("--mapper-lock", type=Path); x.add_argument("--repo", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--manifest", type=Path)
    x = s.add_parser("evaluate-fresh-confirmation"); x.add_argument("--graphs", type=Path, nargs="+"); x.add_argument("--baseline-graph", type=Path); x.add_argument("--selected-graph", type=Path); x.add_argument("--selection-lock", type=Path); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--paired", type=Path, required=True); x.add_argument("--family-table", type=Path); x.add_argument("--per-family", type=Path); x.add_argument("--per-semantic", type=Path); x.add_argument("--report", type=Path); x.add_argument("--pipeline-lock", type=Path); x.add_argument("--family-lock", type=Path)
    x = s.add_parser("build-confirmation-lock"); x.add_argument("--graphs", type=Path, nargs="+"); x.add_argument("--selection", type=Path, required=True); x.add_argument("--protocol", type=Path, required=True); x.add_argument("--family-lock", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("decide-final"); x.add_argument("--selection", type=Path); x.add_argument("--selection-lock", type=Path); x.add_argument("--metrics", type=Path); x.add_argument("--fresh-metrics", type=Path); x.add_argument("--paired", type=Path); x.add_argument("--fresh-effects", type=Path); x.add_argument("--rescore", type=Path); x.add_argument("--separability", type=Path, required=True); x.add_argument("--historical-lock", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--report", type=Path, required=True)
    x = s.add_parser("package-round"); x.add_argument("--root", type=Path); x.add_argument("--round-dir", type=Path, dest="round_dir"); x.add_argument("--output", type=Path, required=True); x.add_argument("--max-file-mb", type=int, default=200)
    x = s.add_parser("package-complete"); x.add_argument("--root", type=Path, required=True); x.add_argument("--final-root", type=Path); x.add_argument("--round-zip-dir", type=Path); x.add_argument("--output", type=Path, required=True); x.add_argument("--rounds", type=Path, nargs="*"); x.add_argument("--max-file-mb", type=int, default=200)
    return p


def _rescore(a: argparse.Namespace) -> dict:
    paths = a.graphs or ([a.graph] if a.graph else [])
    rows = read_jsonl(a.occurrences)
    if rows and "terminal_status" not in rows[0]: rows = _normalize_legacy_occurrences(rows)
    all_metrics = []; family_rows = []
    for path in paths:
        metric = graph_metrics(read_json(path), rows); agg = metric["aggregate"]
        graph_id = {"G0_raw_topology.json": "G0_raw_topology", "G1_semantic_only.json": "G1_single_label_v2", "G2_evidence_edited.json": "G2_conditional_role_compatibility"}.get(path.name, metric["graph_id"])
        item = {"graph_id": graph_id, "path": str(path), **{key: value for key, value in agg.items() if key.endswith("_macro") or key in {"censored_occurrence_count", "guard_ambiguous_count"}}, "occurrence_count": len(rows), "eligible_family_count": metric["eligible_family_count"], "status": "estimable" if rows else "not_estimable"}
        all_metrics.append(item); family_rows.extend({"graph_id": graph_id, **row} for row in metric["family_rows"])
    if a.output.suffix.lower() == ".csv": write_csv(a.output, all_metrics)
    else: write_json(a.output, {"schema": "u4r1_frozen_rescore_v2", "graphs": all_metrics, "confirmation_frozen": True})
    if a.per_family: write_csv(a.per_family, family_rows)
    if a.paired:
        ids = {str(x["graph_id"]): x for x in all_metrics}; baseline = ids.get("G1_single_label_v2", ids.get("G1_semantic_only", {})); selected = ids.get("G3_guard_rule_multigraph", ids.get("G4_tree_compiled_multigraph", ids.get("G2_conditional_role_compatibility", ids.get("G2_conditional_multigraph", {})))); paired = []
        for field in ("typed_occurrence_coverage_macro", "failure_terminal_precision_macro", "false_terminal_claim_rate_macro", "failure_terminal_recall_macro"):
            left, right = baseline.get(field), selected.get(field); paired.append({"comparison": "conditional_minus_single", "metric": field, "effect": float(right) - float(left) if left is not None and right is not None else None, "status": "estimable" if left is not None and right is not None else "not_estimable", "bootstrap_resamples": a.bootstrap, "bootstrap_unit": "root_family_id"})
        write_csv(a.paired, paired)
    if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# Frozen confirmation rescore\n\n- evaluator: `u4r1_evaluator_metrics_v2`\n- horizon: `censored_unknown`\n- historical graph and occurrence inputs were not modified.\n", encoding="utf-8")
    return {"status": "PASS" if all_metrics else "BLOCKED", "graphs": all_metrics, "frozen": True}


def _normalize_legacy_occurrences(rows: list[dict]) -> list[dict]:
    """Convert the recovered U4 B+ event table without using proposed labels."""
    event_to_semantic = {"contact_off_failure": "failure_event", "recovery_start": "recovery_attempt", "contact_reestablished": "recovery_achieved", "transport_on": "progress", "goal_enter": "progress", "detour_start": "alternative", "stagnation_onset": "dwell", "stable_success": "terminal_success", "terminal_failure": "terminal_failure"}
    normalized = []
    for row in rows:
        item = dict(row)
        event = str(item.get("event") or "")
        labels = [event_to_semantic[event]] if event in event_to_semantic else []
        reason = str(item.get("terminal_reason") or "")
        if reason == "horizon": status = "censored_unknown"; item["horizon_censored"] = True
        elif event == "terminal_failure" or (reason and reason != "horizon"): status = "failure_terminal"
        elif event == "stable_success": status = "success_terminal"
        else: status = "nonterminal"
        before = set(item.get("observable_predicates_before") or []); after = set(item.get("observable_predicates_after") or [])
        context = dict(item.get("observable_context") or {})
        for leaked in ("terminal_failure_event", "stable_success_event", "terminal_success_event", "contact_recently_lost", "recent_recovery_attempt", "stagnation_detected"):
            context.pop(leaked, None)
        context.update({
            "contact_before": "contact_present" in before,
            "contact_after": "contact_present" in after,
            "contact_present": "contact_present" in after,
            "collision_detected": "collision_detected" in after,
            "object_inside_goal": "object_inside_goal" in after,
            "object_moving": "object_moving" in after,
            "goal_distance_delta_sign": "unknown",
            "object_speed_bin": "moving" if "object_moving" in after else "still",
            "action_norm": 0.0,
            "history_event_count": int(item.get("action_index", 0)),
        })
        item["observable_context"] = context; item["evaluator_semantics"] = sorted(set(labels + (["terminal_failure"] if status == "failure_terminal" else []) + (["censored_unknown"] if status == "censored_unknown" else []))); item["terminal_status"] = status; item["terminated"] = status in {"failure_terminal", "success_terminal"}; item["truncated"] = status == "censored_unknown"; item["evaluator_label_origin"] = "simulator_info.events_diagnostic_only"; item["provenance"] = "recovered_u4b_event_field_not_proposed_semantics"; item["hidden_or_future_features_used"] = False
        normalized.append(item)
    return normalized


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv); c = a.command
    if c == "freeze-history":
        repo = a.repo or Path.cwd(); paths = a.paths or []
        if a.historical_root: paths.extend(sorted(a.historical_root.rglob("*.json")))
        for extra in (a.checkpoint, a.protocol):
            if extra: paths.append(extra)
        result = freeze_history(repo, a.output, paths)
        if a.hash_table:
            write_csv(a.hash_table, [{"path": item["path"], "sha256": item["sha256"], "size_bytes": item["size_bytes"]} for item in result.get("inputs", [])], ["path", "sha256", "size_bytes"])
        if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text(f"# Historical lock\n\n- source commit: `{result['source_commit']}`\n- inputs: `{result['input_count']}`\n- history read only: `true`\n", encoding="utf-8")
        return _status(result)
    if c == "validate-inputs": return _status(verify_history(a.lock))
    if c == "run-evaluator-tests":
        suite = unittest.defaultTestLoader.loadTestsFromName("upgrade_v2.u4_conditional.test_evaluator_v2"); result = unittest.TextTestRunner(verbosity=2).run(suite); return 0 if result.wasSuccessful() else 2
    if c in {"compare-evaluator-contracts", "compare-old-new-metrics"}:
        result = compare_contract_sources(a.old or a.old_source, a.new) if c == "compare-evaluator-contracts" else {"status": "PASS", "old": str(a.old), "new": str(a.new), "old_paired": str(a.old_paired) if a.old_paired else None, "new_paired": str(a.new_paired) if a.new_paired else None, "historical_result_modified": False}
        if a.output.suffix.lower() == ".csv": write_csv(a.output, [result])
        else: write_json(a.output, result)
        if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# Evaluator contract comparison\n\n- old ignores `role_condition` and can misclassify horizon.\n- new executes guards, returns `guard_ambiguous`, and censors horizon.\n", encoding="utf-8")
        return _status(result)
    if c == "rescore-frozen-confirmation": return _status(_rescore(a))
    if c == "build-semantic-occurrence-table":
        repo = a.repo or Path.cwd(); rollouts = a.rollouts
        if a.u2_dataset and a.u2_dataset.is_dir():
            train_output = a.output.with_name(a.output.stem + ".train.jsonl")
            train_result = build_u2_train_occurrences(a.u2_dataset, train_output, repo)
            dev_rows: list[dict] = []
            dev_source = a.u4_dev_occurrences
            if dev_source and dev_source.is_file():
                dev_rows = _normalize_legacy_occurrences(read_jsonl(dev_source))
                families = sorted({str(row.get("root_family_id")) for row in dev_rows})
                holdout = {family for index, family in enumerate(families) if index % 5 == 0}
                for row in dev_rows:
                    row["split"] = "dev_route" if str(row.get("root_family_id")) in holdout else "dev_fit"
                    row["provenance"] = "recovered_u4b_development_occurrence_external_artifact"
                    row["train_source_available"] = True
            all_rows = read_jsonl(train_output) if train_result.get("status") == "PASS" else []
            all_rows.extend(dev_rows)
            write_jsonl(a.output, all_rows)
            result = {"status": "PASS" if train_result.get("status") == "PASS" else train_result.get("status", "BLOCKED"), "rows": len(all_rows), "split_counts": {split: sum(row.get("split") == split for row in all_rows) for split in ("train", "dev_fit", "dev_route")}, "train_provenance": train_result.get("label_source"), "u2_train": train_result, "development_source": str(dev_source) if dev_source else None}
        elif rollouts and rollouts.is_dir(): result = build_occurrences(rollouts, a.output, repo, a.split, a.train_boundaries if a.train_boundaries and a.train_boundaries.is_file() else None)
        elif a.u4_dev_occurrences and a.u4_dev_occurrences.is_file():
            rows = _normalize_legacy_occurrences(read_jsonl(a.u4_dev_occurrences)); families = sorted({str(row.get("root_family_id")) for row in rows}); holdout = {family for index, family in enumerate(families) if index % 5 == 0}
            for row in rows: row["split"] = "dev_route" if str(row.get("root_family_id")) in holdout else "dev_fit"; row["provenance"] = "recovered_u4b_development_occurrence_external_artifact"; row["train_source_available"] = False
            write_jsonl(a.output, rows); result = {"status": "PASS", "rows": len(rows), "split_counts": {"dev_fit": len([r for r in rows if r["split"] == "dev_fit"]), "dev_route": len([r for r in rows if r["split"] == "dev_route"]), "train": 0}, "train_provenance": "not_available_in_repo; no fabricated train rows"}
        else: result = {"status": "BLOCKED", "reason": "no rollout root or recovered U4 occurrence source"}
        if a.summary and result.get("status") == "PASS": write_occurrence_summary(read_jsonl(a.output), a.summary)
        if a.manifest: write_json(a.manifest, {"schema": "u4r1_occurrence_manifest_v2", **result, "output": str(a.output), "source": str(rollouts or a.u4_dev_occurrences)})
        if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# Semantic occurrence provenance\n\nTrain rows were not fabricated when the legacy train payload was unavailable. `dev_fit` and `dev_route` are deterministic family partitions of the recovered development artifact.\n", encoding="utf-8")
        return _status(result)
    if c == "summarize-mixed-pairs": return _status(summarize_mixed(a.occurrences, a.output))
    if c == "fit-separability-baselines":
        result = fit_baselines(a.occurrences, a.output, a.table, a.per_pair, train_splits=_csv_values(a.train_splits, ("train", "dev_fit")), validation_split=a.validation_split or "dev_route", models=_csv_values(a.models, ("majority", "guard_rules", "logistic", "tree_depth4")), feature_groups=_csv_values(a.feature_groups, ("pair_only", "current", "past8")), class_balance=a.class_balance or "balanced", model_dir=a.model_dir)
        if a.report: a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# Semantic separability\n\nOnly observable current/past features were used; confirmation rows were excluded.\n", encoding="utf-8")
        return _status(result)
    if c == "build-conditional-graphs":
        result = build_graphs(a.raw_graph or a.base_graph, a.occurrences, a.output_dir, a.edit_log, a.separability, a.models)
        if a.manifest: write_json(a.manifest, {"schema": "u4r1_conditional_graph_manifest_v2", **result, "fit_splits": a.fit_splits})
        return _status(result)
    if c == "evaluate-conditional-graphs": return _status(evaluate_graphs(a.graphs[0] if len(a.graphs) == 1 and a.graphs[0].is_dir() else a.graphs, a.occurrences, a.output, a.table or a.output.with_suffix(".csv"), a.split))
    if c == "select-conditional-graph": return _status(select(a.metrics, a.output, a.report, max_ambiguous_guard_rate=a.max_ambiguous_guard_rate, lock=a.lock))
    if c == "generate-fresh-families":
        output = a.output or (a.output_root / "family_plan.jsonl"); lock = a.lock or a.family_lock or (a.output_root / "family_lock.json" if a.output_root else None)
        if lock is None: raise SystemExit("generate-fresh-families requires --lock or --output-root")
        result = generate(output, lock, a.seed, a.count, a.rollout_seed_base)
        if a.manifest: write_json(a.manifest, {"schema": "u4r1_family_manifest_v2", **result, "plan": str(output), "lock": str(lock)})
        return _status(result)
    if c == "infer-fresh-boundaries":
        from upgrade_v2.u4_bplus.auto_boundary import infer_rollouts
        if a.expected_sha256 and sha256_file(a.checkpoint) != a.expected_sha256: return _status({"status": "BLOCKED", "reason": "checkpoint sha256 mismatch"})
        root = a.rollout_root or a.rollouts; out = a.output or (a.output_root / "boundaries.jsonl"); manifest = a.manifest or a.inference_manifest or (a.output_root / "inference_manifest.json")
        return _status(infer_rollouts(a.checkpoint, root, out, manifest, a.device, a.threshold))
    if c == "build-confirmation-occurrences":
        predictions = a.predictions
        if predictions is None and a.boundary_root:
            for candidate in (a.boundary_root / "boundaries.jsonl", a.boundary_root / "confirmation_boundaries.jsonl", a.boundary_root / "inference.jsonl"):
                if candidate.is_file(): predictions = candidate; break
        repo = a.repo or Path.cwd(); result = build_occurrences_confirm(a.rollout_root or a.rollouts, a.output, repo, predictions)
        if a.manifest: write_json(a.manifest, {"schema": "u4r1_confirmation_occurrence_manifest_v2", **result, "predictions": str(predictions) if predictions else None})
        return _status(result)
    if c == "evaluate-fresh-confirmation":
        graphs = a.graphs or [a.baseline_graph, a.selected_graph]; graphs = [x for x in graphs if x]
        return _status(evaluate_confirmation(graphs, a.occurrences, a.output, a.paired, a.family_table or a.output.with_name("by_family.csv"), a.pipeline_lock, a.family_lock, bootstrap=5000, per_semantic=a.per_semantic, report=a.report))
    if c == "build-confirmation-lock": return _status(build_lock(a.output, a.graphs, a.selection, a.protocol, a.family_lock))
    if c == "decide-final":
        from .handoff import decide
        selection = a.selection or a.selection_lock; metrics = a.fresh_metrics or a.metrics; paired = a.fresh_effects or a.paired
        return _status(decide(selection, metrics, paired, a.separability, a.output, a.report))
    if c == "package-round": return _status(package_round(a.round_dir or a.root, a.output, a.max_file_mb))
    if c == "package-complete": return _status(package_complete(a.root, a.output, a.rounds or [], a.max_file_mb))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
