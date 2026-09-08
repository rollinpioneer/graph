"""Command-line entry point for all L2R.0-L2R.6 operations."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .canonicalize import canonicalize, review_canonicalization
from .compile_graph import compile_graphs
from .confirm import evaluate_fresh
from .dataset import SCENARIOS, collect_dataset, generate_fresh, validate_dynamic_dataset, verify_dynamic_simulator
from .execute_graph import execute_graphs
from .handoff import decide_final
from .io import bool_value, read_csv, read_json, write_csv
from .package import package_complete, package_round
from .predicates import evaluate_predicates, fit_thresholds, infer_dataset
from .refine_graph import build_g3, propose_refinements, select_refined_graph
from .source_lock import check_entry, freeze_source


def _paths_file(path: Path) -> list[Path]:
    return [Path(line.strip()) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _dataset_manifest(dataset: Path, output: Path) -> Path:
    rows = []
    for metadata_path in sorted(dataset.rglob("metadata.json")):
        metadata = read_json(metadata_path)
        termination = read_json(metadata_path.parent / "termination.json")
        rows.append({**metadata, "path": str(metadata_path.parent.resolve()), "termination_type": termination["termination_type"]})
    write_csv(output, rows)
    return output


def _prediction_manifest(predictions: Path, output: Path) -> Path:
    rows = []
    for path in sorted(predictions.glob("*.jsonl")):
        rollout_id = path.stem
        family_id = rollout_id.rsplit("_r", 1)[0]
        frames = sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        first = json.loads(next(line for line in path.read_text(encoding="utf-8").splitlines() if line.strip()))
        queried = any(json.loads(line).get("side_view_used") for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
        rows.append({"rollout_id": rollout_id, "root_family_id": family_id, "prediction_path": str(path.resolve()), "frames": frames, "camera": first.get("camera", "front"), "second_view_queried": int(queried), "thresholds_sha256": "reconstructed"})
    write_csv(output, rows)
    return output


def _emit(value: Any) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    status = str(value.get("status", "PASS")) if isinstance(value, dict) else "PASS"
    return 2 if status in {"FAIL", "BLOCKED", "L1V_REVIEW_NOT_COMPLETE", "CANONICALIZATION_REVIEW_FAILED", "DYNAMIC_SIMULATOR_FAILED", "DYNAMIC_DATASET_FAILED", "OBSERVABILITY_INSUFFICIENT"} else 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m upgrade_v2.visual_refine_l2.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("check-entry")
    p.add_argument("--decision", type=Path, required=True); p.add_argument("--review-report", type=Path, required=True)
    p.add_argument("--review-resolution", type=Path, required=True); p.add_argument("--entry-gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("freeze-source")
    p.add_argument("--candidate-list", type=Path, required=True); p.add_argument("--states", type=Path, required=True)
    p.add_argument("--review-evidence", type=Path, required=True); p.add_argument("--review-ratings", type=Path, required=True)
    p.add_argument("--review-mapping", type=Path); p.add_argument("--entry-gate", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--hash-table", type=Path, required=True)
    p.add_argument("--report", type=Path, required=True); p.add_argument("--repo-root", type=Path, default=Path.cwd())

    p = sub.add_parser("canonicalize-v1-graphs")
    p.add_argument("--candidate-list", type=Path, required=True); p.add_argument("--rules", type=Path, required=True)
    p.add_argument("--action-vocabulary", type=Path, required=True); p.add_argument("--state-vocabulary", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True); p.add_argument("--canonical-graph", type=Path, required=True)
    p.add_argument("--case-mapping", type=Path, required=True); p.add_argument("--unmapped", type=Path, required=True)
    p.add_argument("--evidence", type=Path); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("review-canonicalization")
    p.add_argument("--canonical-graph", type=Path, required=True); p.add_argument("--mapping", type=Path, required=True)
    p.add_argument("--evidence", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("verify-dynamic-simulator")
    p.add_argument("--scenarios", nargs="+", default=list(SCENARIOS)); p.add_argument("--families-per-scenario", type=int, default=1)
    p.add_argument("--rollouts-per-family", type=int, default=2); p.add_argument("--seed", type=int, required=True)
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("collect-dynamic-dataset")
    p.add_argument("--split", required=True); p.add_argument("--family-count", type=int, required=True); p.add_argument("--rollouts-per-family", type=int, required=True)
    p.add_argument("--scenarios", nargs="+", required=True); p.add_argument("--family-seed", type=int, required=True); p.add_argument("--rollout-seed-base", type=int, required=True)
    p.add_argument("--camera-jitter"); p.add_argument("--object-size-jitter"); p.add_argument("--friction-jitter")
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--family-split", type=Path)
    p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("validate-dynamic-dataset")
    p.add_argument("--root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--family-split", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--event-counts", type=Path, required=True); p.add_argument("--report", type=Path, required=True)
    p.add_argument("--expected-rollouts", type=int, default=192); p.add_argument("--expected-families", type=int, default=48)

    p = sub.add_parser("fit-vision-thresholds")
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--dataset-manifest", type=Path); p.add_argument("--family-split", type=Path, required=True)
    p.add_argument("--fit-split", required=True); p.add_argument("--grid", type=Path, required=True); p.add_argument("--online-inputs")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--grid-results", type=Path, required=True); p.add_argument("--feature-cache", type=Path)

    p = sub.add_parser("infer-observable-predicates")
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--family-split", type=Path); p.add_argument("--split")
    p.add_argument("--thresholds", type=Path, required=True); p.add_argument("--camera", default="front"); p.add_argument("--selected-graph-lock", type=Path)
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True)

    p = sub.add_parser("evaluate-predicates")
    p.add_argument("--predictions", type=Path, required=True); p.add_argument("--prediction-manifest", type=Path); p.add_argument("--oracle-diagnostic", type=Path, required=True)
    p.add_argument("--dataset-manifest", type=Path); p.add_argument("--statistics-unit", default="root_family_id")
    p.add_argument("--output", type=Path, required=True); p.add_argument("--per-predicate", type=Path, required=True); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("compile-coarse-graphs")
    p.add_argument("--canonical-graph", type=Path, required=True); p.add_argument("--thresholds", type=Path, required=True); p.add_argument("--predicate-schema", type=Path, required=True)
    p.add_argument("--output-g0", type=Path, required=True); p.add_argument("--output-g1", type=Path, required=True); p.add_argument("--binding-table", type=Path, required=True); p.add_argument("--unbound", type=Path, required=True)

    p = sub.add_parser("execute-graphs")
    p.add_argument("--graphs", nargs="+", type=Path, required=True); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--dataset-manifest", type=Path)
    p.add_argument("--predicate-root", type=Path, required=True); p.add_argument("--prediction-manifest", type=Path); p.add_argument("--family-split", type=Path); p.add_argument("--split")
    p.add_argument("--output-root", type=Path, required=True); p.add_argument("--metrics", type=Path, required=True); p.add_argument("--errors", type=Path); p.add_argument("--per-family", type=Path)

    p = sub.add_parser("propose-refinements")
    p.add_argument("--base-graph", type=Path, required=True); p.add_argument("--errors", type=Path, required=True); p.add_argument("--dataset", type=Path, required=True); p.add_argument("--split", required=True)
    p.add_argument("--allowed-edits", required=True); p.add_argument("--max-edits", type=int, required=True); p.add_argument("--minimum-families", type=int, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--edit-log", type=Path, required=True); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("select-refined-graph")
    p.add_argument("--metrics", type=Path, required=True); p.add_argument("--per-family", type=Path, required=True); p.add_argument("--edit-log", type=Path, required=True)
    p.add_argument("--g0", type=Path, required=True); p.add_argument("--g1", type=Path, required=True); p.add_argument("--g2", type=Path, required=True); p.add_argument("--g3", type=Path, required=True)
    p.add_argument("--predicate-thresholds", type=Path, required=True); p.add_argument("--camera-config", type=Path, required=True); p.add_argument("--family-generation-lock", type=Path, required=True)
    p.add_argument("--single-view-preferred", default="true"); p.add_argument("--active-view-query-max", type=float, default=.35)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--lock", type=Path, required=True); p.add_argument("--report", type=Path, required=True)

    p = sub.add_parser("generate-fresh-families")
    p.add_argument("--generation-lock", type=Path, required=True); p.add_argument("--output-root", type=Path, required=True); p.add_argument("--manifest", type=Path, required=True); p.add_argument("--family-lock", type=Path, required=True); p.add_argument("--workers", type=int, default=1)

    p = sub.add_parser("evaluate-fresh-confirmation")
    p.add_argument("--graphs", nargs="+", type=Path, required=True); p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--family-lock", type=Path, required=True)
    p.add_argument("--dataset", type=Path, required=True); p.add_argument("--dataset-manifest", type=Path); p.add_argument("--predicate-root", type=Path, required=True); p.add_argument("--prediction-manifest", type=Path)
    p.add_argument("--statistics-unit", default="root_family_id"); p.add_argument("--bootstrap", type=int, default=5000); p.add_argument("--bootstrap-seed", type=int, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--paired", type=Path, required=True); p.add_argument("--per-family", type=Path, required=True); p.add_argument("--per-scenario", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.add_argument("--execution-root", type=Path)

    p = sub.add_parser("decide-final")
    p.add_argument("--l1v-decision", type=Path, required=True); p.add_argument("--source-lock", type=Path, required=True); p.add_argument("--predicate-metrics", type=Path, required=True)
    p.add_argument("--selection-lock", type=Path, required=True); p.add_argument("--confirmation", type=Path, required=True); p.add_argument("--paired-effects", type=Path, required=True); p.add_argument("--per-scenario", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True); p.add_argument("--report", type=Path, required=True); p.add_argument("--final-root", type=Path)

    p = sub.add_parser("package-round")
    p.add_argument("--round-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-file-mb", type=float, default=200)
    p = sub.add_parser("package-complete")
    p.add_argument("--root", type=Path, required=True); p.add_argument("--final-root", type=Path, required=True); p.add_argument("--round-zip-dir", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.add_argument("--max-file-mb", type=float, default=200)
    return parser


def main() -> int:
    args = build_parser().parse_args(); command = args.command
    try:
        if command == "check-entry": result = check_entry(args.decision, args.review_report, args.review_resolution, args.entry_gate, args.output, args.report)
        elif command == "freeze-source":
            mapping = args.review_mapping or args.review_ratings.parent / "blind_mapping.DO_NOT_SHOW_RATER.csv"
            result = freeze_source(_paths_file(args.candidate_list), args.states, args.review_evidence, args.review_ratings, mapping, args.entry_gate, args.output, args.hash_table, args.report, args.repo_root)
        elif command == "canonicalize-v1-graphs":
            evidence = args.evidence or args.canonical_graph.parent / "canonicalization_evidence.jsonl"
            result = canonicalize(_paths_file(args.candidate_list), args.rules, args.action_vocabulary, args.state_vocabulary, args.output_dir, args.canonical_graph, args.case_mapping, args.unmapped, evidence, args.report)
        elif command == "review-canonicalization": result = review_canonicalization(args.canonical_graph, args.mapping, args.evidence, args.output, args.report)
        elif command == "verify-dynamic-simulator": result = verify_dynamic_simulator(args.output_root, args.metrics, args.seed, args.workers)
        elif command == "collect-dynamic-dataset": result = collect_dataset(args.split, args.family_count, args.rollouts_per_family, args.scenarios, args.family_seed, args.rollout_seed_base, args.output_root, args.manifest, args.family_split, args.workers)
        elif command == "validate-dynamic-dataset": result = validate_dynamic_dataset(args.root, args.manifest, args.family_split, args.output, args.event_counts, args.report, args.expected_rollouts, args.expected_families)
        elif command == "fit-vision-thresholds":
            manifest = args.dataset_manifest or _dataset_manifest(args.dataset, args.grid_results.parent / "fit_dataset_manifest.reconstructed.csv")
            result = fit_thresholds(args.dataset, manifest, args.family_split, args.fit_split, args.grid, args.output, args.grid_results, args.feature_cache or args.grid_results.parent / "feature_cache")
        elif command == "infer-observable-predicates": result = infer_dataset(args.dataset, args.thresholds, args.output_root, args.manifest, args.camera, args.family_split, args.split, args.selected_graph_lock)
        elif command == "evaluate-predicates":
            prediction_manifest = args.prediction_manifest or _prediction_manifest(args.predictions, args.output.parent / "prediction_manifest.reconstructed.csv")
            dataset_manifest = args.dataset_manifest or _dataset_manifest(args.oracle_diagnostic, args.output.parent / "dataset_manifest.reconstructed.csv")
            result = evaluate_predicates(prediction_manifest, dataset_manifest, args.output, args.per_predicate, args.report)
        elif command == "compile-coarse-graphs": result = compile_graphs(args.canonical_graph, args.thresholds, args.predicate_schema, args.output_g0, args.output_g1, args.binding_table, args.unbound)
        elif command == "execute-graphs":
            dataset_manifest = args.dataset_manifest or _dataset_manifest(args.dataset, args.metrics.parent / "dataset_manifest.reconstructed.csv")
            prediction_manifest = args.prediction_manifest or _prediction_manifest(args.predicate_root, args.metrics.parent / "prediction_manifest.reconstructed.csv")
            result = execute_graphs(args.graphs, dataset_manifest, prediction_manifest, args.family_split, args.split, args.output_root, args.metrics, args.errors, args.per_family)
        elif command == "propose-refinements": result = propose_refinements(args.base_graph, args.errors, set(args.allowed_edits.split(",")), args.max_edits, args.minimum_families, args.output, args.edit_log, args.report)
        elif command == "select-refined-graph": result = select_refined_graph(args.metrics, args.per_family, args.edit_log, {"G0_coarse_direct": args.g0, "G1_predicate_bound": args.g1, "G2_evidence_refined": args.g2, "G3_active_second_view": args.g3}, args.predicate_thresholds, args.camera_config, args.family_generation_lock, args.active_view_query_max, args.output, args.lock, args.report)
        elif command == "generate-fresh-families": result = generate_fresh(args.generation_lock, args.output_root, args.manifest, args.family_lock, args.workers)
        elif command == "evaluate-fresh-confirmation":
            dataset_manifest = args.dataset_manifest or _dataset_manifest(args.dataset, args.output.parent / "dataset_manifest.reconstructed.csv")
            prediction_manifest = args.prediction_manifest or _prediction_manifest(args.predicate_root, args.output.parent / "prediction_manifest.reconstructed.csv")
            execution_root = args.execution_root or args.output.parent.parent / "execution"
            result = evaluate_fresh(args.graphs, args.selection_lock, args.family_lock, dataset_manifest, prediction_manifest, execution_root, args.output, args.paired, args.per_family, args.per_scenario, args.report, args.bootstrap, args.bootstrap_seed)
        elif command == "decide-final": result = decide_final(args.l1v_decision, args.source_lock, args.predicate_metrics, args.selection_lock, args.confirmation, args.paired_effects, args.per_scenario, args.output, args.report, args.final_root or args.output.parent)
        elif command == "package-round": result = package_round(args.round_dir, args.output, args.max_file_mb)
        elif command == "package-complete": result = package_complete(args.root, args.final_root, args.round_zip_dir, args.output, args.max_file_mb)
        else: raise AssertionError(command)
        return _emit(result)
    except Exception as exc:
        print(json.dumps({"status": "FAIL", "error_type": type(exc).__name__, "error": str(exc)[:600]}, ensure_ascii=False), file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
