"""Command-line entry point for the simulator-scoped U2 protocol."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import zipfile
from pathlib import Path
from typing import Any

from .changepoints import evaluate_selected_baselines, run_baselines
from .dataset import (collect_dataset, dataset_gate, verify_observation_action_alignment, verify_state_restore, write_csv, write_json)
from .evaluate import evaluate_weak
from .jobs import (build_inference_jobs, build_jobs, evaluate_models, evaluate_offline_teachers, launch_inference_jobs, launch_training_jobs,
                   run_inference_job, run_training_job, select_checkpoints, select_gold_clips)
from .infer_boundary import infer
from .segment_representation import (build_segments, choose_source, cluster_segments, encode_segments,
                                     evaluate_representation, write_u3_summaries)
from .budgeted_query import (build_budget_jobs, build_queues, evaluate_budget, freeze_protocol,
                             reveal_oracle_queues)
from .value_reference import (aggregate_reward_segments, build_value_jobs, collect_continuations,
                              evaluate_reward_impact, infer_potential, launch_value_jobs,
                              select_value_checkpoints, train_value_job)
from .package import finalize_u2, package_complete
from .weak_labels import aggregate_posteriors, extract_candidates, save_rules


def _handoff() -> Path:
    path = Path(os.environ.get("U2_HANDOFF", "artifacts/pathgraph_sarm/upgrade_v2/results/u1_data_bridge/u2_handoff_v2.json"))
    if not path.is_file(): raise SystemExit(f"U2 authority file missing: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if value.get("u2_eligible") != "SIMULATOR_SCOPED_ONLY" or value.get("physical_generalization_eligible") is not False:
        raise SystemExit("U2 authority file does not permit simulator-scoped execution")
    return path


def cmd_verify(args: argparse.Namespace) -> int:
    _handoff(); summary, details = verify_state_restore(args.families, args.anchors_per_family, args.continuation_steps, args.seed, args.tolerance)
    write_json(args.output, summary); write_csv(args.details, details, list(details[0])); print(json.dumps(summary)); return 0 if summary["status"] == "U2_STATE_RESTORE_PASS" else 2


def cmd_collect(args: argparse.Namespace) -> int:
    _handoff(); ratios = tuple(float(x) for x in args.split_ratios.split(","))
    rows, splits = collect_dataset(args.mode, args.root_families, args.rollouts_per_family, args.seed, args.output_root, ratios)
    if args.manifest: write_csv(args.manifest, rows, list(rows[0]))
    if args.split_table: write_csv(args.split_table, splits, list(splits[0]))
    # A shared copy gives all U2 commands a stable schema location.
    shared = args.output_root.parent / "configs"; shared.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.output_root / "configs" / "event_schema.json", shared / "event_schema.json")
    shutil.copy2(args.output_root / "configs" / "observable_schema.json", shared / "observable_schema.json")
    print(json.dumps({"mode": args.mode, "episodes": len(rows), "root_families": args.root_families, "authority": str(_handoff().resolve())})); return 0


def cmd_validate(args: argparse.Namespace) -> int:
    _handoff(); summary, events, scenarios, report = dataset_gate(args.dataset)
    write_json(args.output, summary); write_csv(args.event_counts, events, list(events[0])); write_csv(args.scenario_counts, scenarios, list(scenarios[0]))
    args.report.parent.mkdir(parents=True, exist_ok=True); args.report.write_text(report, encoding="utf-8"); print(json.dumps(summary)); return 0 if summary["status"] == "U2_EVENTFUL_DATASET_READY" else 2


def cmd_alignment(args: argparse.Namespace) -> int:
    _handoff(); summary = verify_observation_action_alignment(args.dataset)
    write_json(args.output, summary); print(json.dumps(summary));
    return 0 if summary["status"] == "U2_OBSERVATION_ACTION_ALIGNMENT_PASS" else 2


def cmd_rules(args: argparse.Namespace) -> int: _handoff(); save_rules(args.output); return 0
def cmd_extract(args: argparse.Namespace) -> int: _handoff(); extract_candidates(args.dataset, args.rules, args.output_root, args.manifest); return 0
def cmd_aggregate(args: argparse.Namespace) -> int:
    _handoff(); aggregate_posteriors(args.candidate_root, args.dataset, args.modes.split(","), args.calibration_family_fraction, args.seed, args.output_root, args.weight_table, args.calibration_families, args.rules); return 0
def cmd_eval_weak(args: argparse.Namespace) -> int:
    _handoff(); rows = evaluate_weak(args.dataset, args.posterior_root, args.output, args.per_event, args.report, args.boundary_tolerance_steps)
    print(json.dumps(rows)); return 0
def cmd_baselines(args: argparse.Namespace) -> int:
    _handoff(); run_baselines(args.dataset, args.weak_posteriors, args.methods.split(","), args.selection_split, args.output_root, args.selection_table, args.all_configs); return 0
def cmd_eval_baselines(args: argparse.Namespace) -> int:
    _handoff(); rows = evaluate_selected_baselines(args.dataset, args.prediction_root, args.selection, args.split, args.output, args.per_event, args.report, args.boundary_tolerance_steps); print(json.dumps(rows)); return 0
def cmd_select_clips(args: argparse.Namespace) -> int:
    _handoff(); select_gold_clips(args.dataset, args.split, args.budget_clips, args.seed, args.output); return 0
def cmd_build_jobs(args: argparse.Namespace) -> int:
    _handoff(); rows = build_jobs(args.mode, args.dataset, args.weak_posteriors, args.variants.split(","), [int(x) for x in args.seeds.split(",")], args.oracle_clips, args.steps, args.output_root, args.job_table, args.commands_dir); print(json.dumps({"jobs": len(rows)})); return 0
def cmd_run_job(args: argparse.Namespace) -> int:
    _handoff(); print(json.dumps(run_training_job(args.job_table, args.job_id))); return 0
def cmd_launch(args: argparse.Namespace) -> int:
    _handoff(); launch_training_jobs(args.job_table, args.gpu_ids.split(","), args.status_output); return 0
def cmd_select_checkpoints(args: argparse.Namespace) -> int:
    _handoff(); rows = select_checkpoints(args.job_root, args.job_table, args.output, args.lock, args.checkpoint_manifest); print(json.dumps({"selected": len(rows)})); return 0
def cmd_build_infer(args: argparse.Namespace) -> int:
    _handoff(); rows = build_inference_jobs(args.selection, args.dataset, args.splits.split(","), args.output_root, args.job_table); print(json.dumps({"jobs": len(rows)})); return 0
def cmd_run_infer(args: argparse.Namespace) -> int:
    _handoff(); run_inference_job(args.job_table, args.inference_id); return 0
def cmd_launch_infer(args: argparse.Namespace) -> int:
    _handoff(); launch_inference_jobs(args.job_table, args.gpu_ids.split(","), args.status_output); return 0
def cmd_eval_models(args: argparse.Namespace) -> int:
    _handoff(); rows = evaluate_models(args.dataset, args.prediction_root, args.selection, args.output, args.per_event, args.per_seed, args.report); print(json.dumps(rows)); return 0
def cmd_eval_teachers(args: argparse.Namespace) -> int:
    _handoff();rows=evaluate_offline_teachers(args.dataset,args.prediction_root,args.formal_root,args.output,args.per_event,args.report);print(json.dumps(rows));return 0
def cmd_infer_split(args: argparse.Namespace) -> int:
    _handoff(); infer(args.checkpoint, args.dataset, args.weak_posteriors, args.split, args.output); return 0
def cmd_choose_source(args: argparse.Namespace) -> int:
    _handoff(); print(json.dumps(choose_source(args.model_metrics, args.baseline_metrics, args.output))); return 0
def cmd_segments(args: argparse.Namespace) -> int:
    _handoff(); rows = build_segments(args.dataset, args.boundary_source, args.prediction_root, args.baseline_root, args.minimum_length, args.maximum_length, args.output_root, args.manifest); print(json.dumps({"segments":len(rows)})); return 0
def cmd_encode(args: argparse.Namespace) -> int:
    _handoff(); data=encode_segments(args.segments,args.output,args.schema); print(json.dumps({"segments":len(data)})); return 0
def cmd_cluster(args: argparse.Namespace) -> int:
    _handoff(); _, rows=cluster_segments(args.embeddings,args.methods.split(','),[int(x) for x in args.clusters.split(',')],[int(x) for x in args.seeds.split(',')],args.selection_split,args.output_root,args.selection,args.grid_results); print(json.dumps(rows)); return 0
def cmd_eval_segments(args: argparse.Namespace) -> int:
    _handoff(); metric=evaluate_representation(args.segments,args.embeddings,args.dataset,args.output,args.history_ablation,args.transition_table,args.report); write_u3_summaries(args.segments,args.embeddings,args.transition_table,args.summary,args.prototypes,args.support); print(json.dumps(metric)); return 0
def cmd_freeze_budget(args: argparse.Namespace) -> int:
    _handoff(); print(json.dumps(freeze_protocol(args.output))); return 0
def cmd_queues(args: argparse.Namespace) -> int:
    _handoff(); rows=build_queues(args.dataset,args.weak_posteriors,args.boundary_predictions,[int(x) for x in args.budgets.split(',')],args.strategies.split(','),[int(x) for x in args.seeds.split(',')],args.output_root,args.manifest); print(json.dumps({"queues":len(rows)})); return 0
def cmd_reveal(args: argparse.Namespace) -> int:
    _handoff(); rows=reveal_oracle_queues(args.queue_root,args.dataset,args.output_root,args.provenance); print(json.dumps({"revealed_queues":len(rows)})); return 0
def cmd_budget_jobs(args: argparse.Namespace) -> int:
    _handoff(); rows=build_budget_jobs(args.dataset,args.weak_posteriors,args.revealed_queues,[int(x) for x in args.budgets.split(',')],args.strategies.split(','),[int(x) for x in args.seeds.split(',')],args.steps,args.output_root,args.job_table); print(json.dumps({"jobs":len(rows)})); return 0
def cmd_eval_budget(args: argparse.Namespace) -> int:
    _handoff(); rows,supported=evaluate_budget(args.job_root,args.dataset,args.selection_split,args.test_split,args.output,args.per_event,args.report); print(json.dumps({"rows":len(rows),"active_query_supported":supported})); return 0
def cmd_collect_value(args: argparse.Namespace) -> int:
    _handoff(); rows=collect_continuations(args.dataset,args.anchors_per_family,args.continuations_per_anchor,args.horizon,args.seed,args.output,args.target_table,args.summary);print(json.dumps({"anchors":len(rows)}));return 0
def cmd_build_value(args: argparse.Namespace) -> int:
    _handoff(); rows=build_value_jobs(args.dataset,args.targets,args.variants.split(','),[int(x) for x in args.seeds.split(',')],args.steps,args.output_root,args.job_table);print(json.dumps({"jobs":len(rows)}));return 0
def cmd_run_value(args: argparse.Namespace) -> int:
    _handoff();print(json.dumps(train_value_job(args.job_table,args.job_id)));return 0
def cmd_launch_value(args: argparse.Namespace) -> int:
    _handoff();launch_value_jobs(args.job_table,args.gpu_ids.split(','),args.status_output);return 0
def cmd_select_value(args: argparse.Namespace) -> int:
    _handoff();rows=select_value_checkpoints(args.job_table,args.output);print(json.dumps({"selected":len(rows)}));return 0
def cmd_infer_value(args: argparse.Namespace) -> int:
    _handoff();print(json.dumps(infer_potential(args.selection,args.dataset,[float(x) for x in args.alpha_grid.split(',')],args.output_root,args.lock)));return 0
def cmd_aggregate_reward(args: argparse.Namespace) -> int:
    _handoff();rows=aggregate_reward_segments(args.dataset,args.potential_root,args.causal_boundaries,args.rule_boundaries,args.budget_boundaries,args.output_root,args.manifest,args.budget_source,args.causal_source);print(json.dumps({"segments":len(rows)}));return 0
def cmd_eval_reward(args: argparse.Namespace) -> int:
    _handoff();rows=evaluate_reward_impact(args.segment_root,args.targets,args.output,args.per_event,args.report,args.bootstrap,args.bootstrap_seed);print(json.dumps(rows));return 0
def cmd_finalize(args: argparse.Namespace) -> int:
    _handoff();print(json.dumps(finalize_u2(args.repo_root,args.u2_root,args.final_root)));return 0
def cmd_package_complete(args: argparse.Namespace) -> int:
    _handoff();print(json.dumps(package_complete(args.repo_root,args.u2_root,args.output,args.max_file_mb)));return 0


def cmd_package(args: argparse.Namespace) -> int:
    """Build the user's single lightweight package, excluding raw/checkpoint assets."""
    _handoff(); output = args.output; output.parent.mkdir(parents=True, exist_ok=True)
    excluded = {".npz", ".pt", ".pth", ".parquet", ".jsonl", ".log"}
    files = [p for p in args.round_dir.rglob("*") if p.is_file() and p.suffix not in excluded and p.stat().st_size <= args.max_file_mb * 1024 * 1024]
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in files: archive.write(path, path.relative_to(args.round_dir.parent))
        archive.writestr("PACKAGE_POLICY.md", "Single-package policy applied: raw NPZ/JSONL, checkpoints, embeddings, and logs are excluded; manifests and lightweight summaries remain.\n")
    digest = hashlib.sha256(output.read_bytes()).hexdigest(); output.with_suffix(output.suffix + ".sha256").write_text(f"{digest}  {output.name}\n", encoding="utf-8")
    print(json.dumps({"zip": str(output.resolve()), "sha256": digest, "files": len(files)})); return 0


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upgrade_v2.u2.cli"); sub = p.add_subparsers(dest="command", required=True)
    q = sub.add_parser("verify-state-restore"); q.add_argument("--families", type=int, required=True); q.add_argument("--anchors-per-family", type=int, required=True); q.add_argument("--continuation-steps", type=int, required=True); q.add_argument("--seed", type=int, required=True); q.add_argument("--tolerance", type=float, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--details", type=Path, required=True); q.set_defaults(func=cmd_verify)
    q = sub.add_parser("collect-eventful-dataset"); q.add_argument("--mode", required=True); q.add_argument("--root-families", type=int, required=True); q.add_argument("--rollouts-per-family", type=int, required=True); q.add_argument("--scenarios", required=True); q.add_argument("--split-ratios", default="0.70,0.15,0.15"); q.add_argument("--split-unit", default="root_family_id"); q.add_argument("--seed", type=int, required=True); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--manifest", type=Path); q.add_argument("--split-table", type=Path); q.set_defaults(func=cmd_collect)
    q = sub.add_parser("validate-eventful-dataset"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--event-schema", type=Path); q.add_argument("--output", type=Path, required=True); q.add_argument("--event-counts", type=Path, required=True); q.add_argument("--scenario-counts", type=Path, required=True); q.add_argument("--report", type=Path, required=True); q.set_defaults(func=cmd_validate)
    q = sub.add_parser("audit-observation-action-alignment"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.set_defaults(func=cmd_alignment)
    q = sub.add_parser("write-weak-rules"); q.add_argument("--output", type=Path, required=True); q.set_defaults(func=cmd_rules)
    q = sub.add_parser("extract-event-candidates"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--rules", type=Path, required=True); q.add_argument("--workers", type=int, default=1); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--manifest", type=Path, required=True); q.set_defaults(func=cmd_extract)
    q = sub.add_parser("aggregate-weak-events"); q.add_argument("--candidate-root", type=Path, required=True); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--modes", required=True); q.add_argument("--calibration-family-fraction", type=float, required=True); q.add_argument("--seed", type=int, required=True); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--weight-table", type=Path, required=True); q.add_argument("--calibration-families", type=Path, required=True); q.add_argument("--rules", type=Path, required=True); q.set_defaults(func=cmd_aggregate)
    q = sub.add_parser("evaluate-weak-events"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--posterior-root", type=Path, required=True); q.add_argument("--boundary-tolerance-steps", type=int, default=2); q.add_argument("--output", type=Path, required=True); q.add_argument("--per-event", type=Path, required=True); q.add_argument("--report", type=Path, required=True); q.add_argument("--figures", type=Path); q.set_defaults(func=cmd_eval_weak)
    q = sub.add_parser("run-segmentation-baselines"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--weak-posteriors", type=Path, required=True); q.add_argument("--methods", required=True); q.add_argument("--selection-split", default="val"); q.add_argument("--test-used-for-selection", default="false"); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--selection-table", type=Path, required=True); q.add_argument("--all-configs", type=Path, required=True); q.set_defaults(func=cmd_baselines)
    q = sub.add_parser("evaluate-segmentation-baselines"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--prediction-root", type=Path, required=True); q.add_argument("--selection", type=Path, required=True); q.add_argument("--split", default="test"); q.add_argument("--boundary-tolerance-steps", type=int, default=2); q.add_argument("--output", type=Path, required=True); q.add_argument("--per-event", type=Path, required=True); q.add_argument("--report", type=Path, required=True); q.add_argument("--figures", type=Path); q.set_defaults(func=cmd_eval_baselines)
    q = sub.add_parser("select-fixed-gold-clips"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--weak-posteriors", type=Path); q.add_argument("--split", default="train"); q.add_argument("--budget-clips", type=int, required=True); q.add_argument("--selection", default="random_stratified"); q.add_argument("--event-types"); q.add_argument("--seed", type=int, required=True); q.add_argument("--output", type=Path, required=True); q.set_defaults(func=cmd_select_clips)
    q = sub.add_parser("build-boundary-jobs"); q.add_argument("--mode", required=True); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--weak-posteriors", type=Path, required=True); q.add_argument("--variants", required=True); q.add_argument("--seeds", required=True); q.add_argument("--oracle-clips", type=Path, required=True); q.add_argument("--steps", type=int, required=True); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--commands-dir", type=Path); q.set_defaults(func=cmd_build_jobs)
    q = sub.add_parser("run-boundary-job"); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--job-id", required=True); q.set_defaults(func=cmd_run_job)
    q = sub.add_parser("launch-jobs"); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--gpu-ids", default="4"); q.add_argument("--status-output", type=Path, required=True); q.set_defaults(func=cmd_launch)
    q = sub.add_parser("select-boundary-checkpoints"); q.add_argument("--job-root", type=Path, required=True); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--metric", default="boundary_f1_tol2"); q.add_argument("--tiebreaker"); q.add_argument("--split", default="val"); q.add_argument("--output", type=Path, required=True); q.add_argument("--lock", type=Path, required=True); q.add_argument("--checkpoint-manifest", type=Path, required=True); q.set_defaults(func=cmd_select_checkpoints)
    q = sub.add_parser("build-boundary-inference-jobs"); q.add_argument("--selection", type=Path, required=True); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--splits", required=True); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--job-table", type=Path, required=True); q.set_defaults(func=cmd_build_infer)
    q = sub.add_parser("run-boundary-inference-job"); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--inference-id", required=True); q.set_defaults(func=cmd_run_infer)
    q = sub.add_parser("launch-inference-jobs"); q.add_argument("--job-table", type=Path, required=True); q.add_argument("--gpu-ids", default="4"); q.add_argument("--status-output", type=Path, required=True); q.set_defaults(func=cmd_launch_infer)
    q = sub.add_parser("evaluate-boundary-models"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--prediction-root", type=Path, required=True); q.add_argument("--selection", type=Path, required=True); q.add_argument("--baseline-metrics", type=Path); q.add_argument("--tolerance-steps", type=int, default=2); q.add_argument("--output", type=Path, required=True); q.add_argument("--per-event", type=Path, required=True); q.add_argument("--per-seed", type=Path, required=True); q.add_argument("--report", type=Path, required=True); q.add_argument("--figures", type=Path); q.set_defaults(func=cmd_eval_models)
    q = sub.add_parser("evaluate-offline-teachers");q.add_argument("--dataset",type=Path,required=True);q.add_argument("--prediction-root",type=Path,required=True);q.add_argument("--formal-root",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--per-event",type=Path,required=True);q.add_argument("--report",type=Path,required=True);q.set_defaults(func=cmd_eval_teachers)
    q = sub.add_parser("infer-boundary-split"); q.add_argument("--checkpoint", type=Path, required=True); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--weak-posteriors", type=Path, required=True); q.add_argument("--split", required=True); q.add_argument("--output", type=Path, required=True); q.set_defaults(func=cmd_infer_split)
    q = sub.add_parser("choose-u2-boundary-source"); q.add_argument("--model-metrics", type=Path, required=True); q.add_argument("--baseline-metrics", type=Path, required=True); q.add_argument("--selection-split", default="val"); q.add_argument("--output", type=Path, required=True); q.set_defaults(func=cmd_choose_source)
    q = sub.add_parser("build-segments"); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--boundary-source", type=Path, required=True); q.add_argument("--prediction-root", type=Path, required=True); q.add_argument("--baseline-root", type=Path, required=True); q.add_argument("--minimum-length", type=int, required=True); q.add_argument("--maximum-length", type=int, required=True); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--manifest", type=Path, required=True); q.set_defaults(func=cmd_segments)
    q = sub.add_parser("encode-segments"); q.add_argument("--segments", type=Path, required=True); q.add_argument("--boundary-source", type=Path); q.add_argument("--model-selection", type=Path); q.add_argument("--output", type=Path, required=True); q.add_argument("--schema", type=Path, required=True); q.set_defaults(func=cmd_encode)
    q = sub.add_parser("cluster-segments"); q.add_argument("--embeddings", type=Path, required=True); q.add_argument("--methods", required=True); q.add_argument("--clusters", required=True); q.add_argument("--seeds", required=True); q.add_argument("--selection-split", default="val"); q.add_argument("--output-root", type=Path, required=True); q.add_argument("--selection", type=Path, required=True); q.add_argument("--grid-results", type=Path, required=True); q.set_defaults(func=cmd_cluster)
    q = sub.add_parser("evaluate-segment-representation"); q.add_argument("--segments", type=Path, required=True); q.add_argument("--embeddings", type=Path, required=True); q.add_argument("--cluster-root", type=Path); q.add_argument("--dataset", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--history-ablation", type=Path, required=True); q.add_argument("--transition-table", type=Path, required=True); q.add_argument("--report", type=Path, required=True); q.add_argument("--figures", type=Path); q.add_argument("--summary", type=Path, required=True); q.add_argument("--prototypes", type=Path, required=True); q.add_argument("--support", type=Path, required=True); q.set_defaults(func=cmd_eval_segments)
    q = sub.add_parser("freeze-budget-protocol"); q.add_argument("--output",type=Path,required=True); q.set_defaults(func=cmd_freeze_budget)
    q = sub.add_parser("build-query-queues"); q.add_argument("--dataset",type=Path,required=True); q.add_argument("--weak-posteriors",type=Path,required=True); q.add_argument("--boundary-predictions",type=Path,required=True); q.add_argument("--segment-summary",type=Path); q.add_argument("--budgets",required=True); q.add_argument("--strategies",required=True); q.add_argument("--seeds",required=True); q.add_argument("--output-root",type=Path,required=True); q.add_argument("--manifest",type=Path,required=True); q.set_defaults(func=cmd_queues)
    q = sub.add_parser("reveal-oracle-clips"); q.add_argument("--queue-root",type=Path,required=True); q.add_argument("--dataset",type=Path,required=True); q.add_argument("--output-root",type=Path,required=True); q.add_argument("--provenance",default="simulator_gold_oracle_budget"); q.set_defaults(func=cmd_reveal)
    q = sub.add_parser("build-budgeted-training-jobs"); q.add_argument("--base-variant",type=Path); q.add_argument("--dataset",type=Path,required=True); q.add_argument("--weak-posteriors",type=Path,required=True); q.add_argument("--revealed-queues",type=Path,required=True); q.add_argument("--budgets",required=True); q.add_argument("--strategies",required=True); q.add_argument("--seeds",required=True); q.add_argument("--steps",type=int,required=True); q.add_argument("--output-root",type=Path,required=True); q.add_argument("--job-table",type=Path,required=True); q.set_defaults(func=cmd_budget_jobs)
    q = sub.add_parser("evaluate-budgeted-correction"); q.add_argument("--job-root",type=Path,required=True); q.add_argument("--dataset",type=Path,required=True); q.add_argument("--selection-split",default="val"); q.add_argument("--test-split",default="test"); q.add_argument("--output",type=Path,required=True); q.add_argument("--per-event",type=Path,required=True); q.add_argument("--report",type=Path,required=True); q.add_argument("--figures",type=Path); q.set_defaults(func=cmd_eval_budget)
    q = sub.add_parser("collect-boundary-value-continuations"); q.add_argument("--dataset",type=Path,required=True);q.add_argument("--anchors-per-family",type=int,required=True);q.add_argument("--continuations-per-anchor",type=int,required=True);q.add_argument("--policy",default="fixed_recovery_controller");q.add_argument("--horizon",type=int,required=True);q.add_argument("--seed",type=int,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--target-table",type=Path,required=True);q.add_argument("--summary",type=Path,required=True);q.set_defaults(func=cmd_collect_value)
    q = sub.add_parser("build-value-reference-jobs");q.add_argument("--dataset",type=Path,required=True);q.add_argument("--targets",type=Path,required=True);q.add_argument("--variants",required=True);q.add_argument("--seeds",required=True);q.add_argument("--history-steps",type=int,default=32);q.add_argument("--steps",type=int,required=True);q.add_argument("--selection-split",default="val");q.add_argument("--output-root",type=Path,required=True);q.add_argument("--job-table",type=Path,required=True);q.set_defaults(func=cmd_build_value)
    q = sub.add_parser("run-value-reference-job");q.add_argument("--job-table",type=Path,required=True);q.add_argument("--job-id",required=True);q.set_defaults(func=cmd_run_value)
    q = sub.add_parser("launch-value-jobs");q.add_argument("--job-table",type=Path,required=True);q.add_argument("--gpu-ids",default="4");q.add_argument("--status-output",type=Path,required=True);q.set_defaults(func=cmd_launch_value)
    q = sub.add_parser("select-value-checkpoints");q.add_argument("--job-table",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.set_defaults(func=cmd_select_value)
    q = sub.add_parser("infer-value-potential");q.add_argument("--selection",type=Path,required=True);q.add_argument("--dataset",type=Path,required=True);q.add_argument("--alpha-grid",required=True);q.add_argument("--selection-split",default="val");q.add_argument("--output-root",type=Path,required=True);q.add_argument("--lock",type=Path,required=True);q.set_defaults(func=cmd_infer_value)
    q = sub.add_parser("aggregate-reward-by-boundary");q.add_argument("--dataset",type=Path,required=True);q.add_argument("--potential-root",type=Path,required=True);q.add_argument("--gold-boundaries",type=Path);q.add_argument("--causal-boundaries",type=Path,required=True);q.add_argument("--rule-boundaries",type=Path,required=True);q.add_argument("--budget-boundaries",type=Path,required=True);q.add_argument("--output-root",type=Path,required=True);q.add_argument("--manifest",type=Path,required=True);q.add_argument("--budget-source",required=True);q.add_argument("--causal-source",required=True);q.set_defaults(func=cmd_aggregate_reward)
    q = sub.add_parser("evaluate-boundary-reward-impact");q.add_argument("--segment-root",type=Path,required=True);q.add_argument("--targets",type=Path,required=True);q.add_argument("--statistics-unit",default="root_family_id");q.add_argument("--output",type=Path,required=True);q.add_argument("--per-event",type=Path,required=True);q.add_argument("--bootstrap",type=int,default=5000);q.add_argument("--bootstrap-seed",type=int,required=True);q.add_argument("--report",type=Path,required=True);q.add_argument("--figures",type=Path);q.set_defaults(func=cmd_eval_reward)
    q = sub.add_parser("finalize-u2");q.add_argument("--repo-root",type=Path,required=True);q.add_argument("--u2-root",type=Path,required=True);q.add_argument("--final-root",type=Path,required=True);q.set_defaults(func=cmd_finalize)
    q = sub.add_parser("package-u2-complete");q.add_argument("--repo-root",type=Path,required=True);q.add_argument("--u2-root",type=Path,required=True);q.add_argument("--output",type=Path,required=True);q.add_argument("--max-file-mb",type=int,default=20);q.set_defaults(func=cmd_package_complete)
    q = sub.add_parser("package-round"); q.add_argument("--round-dir", type=Path, required=True); q.add_argument("--output", type=Path, required=True); q.add_argument("--max-file-mb", type=int, default=20); q.set_defaults(func=cmd_package)
    return p


def main() -> int:
    args = parser().parse_args(); return int(args.func(args))
if __name__ == "__main__": raise SystemExit(main())
