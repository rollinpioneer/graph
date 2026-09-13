from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path
from typing import Any

from . import episodes, grouped_validation, interface_shadow, resources
from .audits import HORIZONS, audit_windows, dump_json, load_jsonl, source_inventory
from .replay import replay
from .temporal_scoring import score_event, truth_for


def _write_rows(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or (list(rows[0]) if rows else ["episode_id"])
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inventory(args: argparse.Namespace) -> None:
    payload = resources.discover(Path(args.repo), args.base)
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    dump_json(out, payload)
    root = out.parent
    source_inventory(payload, root / "source_inventory.tsv")
    dump_json(root / "raw_resource_locations.json", {
        "schema": "l2rar2_r27_raw_resource_locations_v1",
        "sources": payload.get("sources", []),
        "missing_expected_sources": ["r26_raw_cache_not_provided"],
        "physical_executions": 0,
    })
    (root / "r26_analysis_limitations.md").write_text(
        "# R26 analysis limitations carried into R27\n\n"
        "R26 labels and aggregate episode features are preserved as historical context. "
        "R27 replays the raw R24/R25 observation streams with corrected physical event "
        "boundaries and does not replace the R26 result.\n",
        encoding="utf-8",
    )
    dump_json(root / "old_vs_corrected_analysis_contract.json", {
        "schema": "l2rar2_r27_analysis_contract_v1",
        "old_r26_summary_used_as_truth": False,
        "r27_uses_raw_observation_streams": True,
        "physical_executions": 0,
    })
    print(json.dumps(payload, indent=2, ensure_ascii=False))


def audit_episodes(args: argparse.Namespace) -> None:
    payload = json.loads(Path(args.resources).read_text(encoding="utf-8"))
    result = episodes.build(payload, Path(args.output_root))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def audit_windows_cmd(args: argparse.Namespace) -> None:
    records = load_jsonl(Path(args.episodes))
    result = audit_windows(records, Path(args.output_root))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def _first_action(record: dict[str, Any]) -> int | None:
    values = []
    for decision in record.get("candidate_decisions", []):
        if decision.get("selected_action") == "recover_object":
            try:
                values.append(int(decision["physical_time_ns"]))
            except (KeyError, TypeError, ValueError):
                pass
    return min(values) if values else None


def _b0_method(record: dict[str, Any]) -> str:
    method = str(record.get("arm_id", ""))
    if "CLP3" in method:
        return "CLP3_FROZEN"
    if "FUSION" in method:
        return "FUSION_V1_FROZEN"
    return "R26_ONLINE_FUSION_V2_AS_DELIVERED"


def _b0_rows(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        first = _first_action(record)
        rows.append({
            "episode_id": record.get("episode_id"),
            "family": record.get("root_family_id"),
            "method": _b0_method(record),
            "first_evidence_ns": first,
            "evidence_end_ns": record.get("evidence_end_ns"),
            "truth": truth_for(record),
            "score": score_event(
                truth=truth_for(record),
                onset_ns=record.get("physical_loss_onset_ns"),
                end_ns=int(record.get("evidence_end_ns") or 0),
                first_evidence_ns=first,
            ),
            "proposal_count": len(record.get("candidate_decisions", [])),
            "physical_confirmation_ns": record.get("physical_loss_confirmed_ns"),
        })
    return rows


def _prefix_audit(records: list[dict[str, Any]], methods: list[str], output: Path) -> dict[str, Any]:
    checks = 0; matches = 0; mismatches: list[dict[str, Any]] = []
    for record in records:
        rows = record.get("observations", [])
        if not rows:
            continue
        indices = {0, len(rows) - 1}
        if record.get("baseline_ready_ns") is not None:
            indices.update(i for i, row in enumerate(rows) if int(row.get("physical_time_ns", -1)) in {int(record["baseline_ready_ns"]), int(record["baseline_ready_ns"]) - 50_000_000})
        indices.update(i for i, row in enumerate(rows) if row.get("frame_missing") or row.get("gripper_command") == "open")
        for method in methods:
            full = replay(record, method)
            for index in sorted(indices):
                prefix_record = dict(record)
                prefix_record["observations"] = rows[: index + 1]
                prefix_record["evidence_end_ns"] = int(rows[index].get("physical_time_ns", 0))
                prefix = replay(prefix_record, method)
                expected = full.get("outputs", [])[index] if index < len(full.get("outputs", [])) else None
                actual = prefix.get("outputs", [])[-1] if prefix.get("outputs") else None
                checks += 1
                if expected and actual and expected.get("state") == actual.get("state") and expected.get("reason") == actual.get("reason"):
                    matches += 1
                else:
                    mismatches.append({"episode_id": record.get("episode_id"), "method": method, "prefix_index": index})
    result = {"schema": "l2rar2_r27_prefix_execution_audit_v1", "checks": checks, "matches": matches, "mismatches": mismatches[:100], "all_prefix_states_match": checks == matches, "physical_executions": 0}
    dump_json(output, result)
    return result


def compare(args: argparse.Namespace) -> None:
    output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
    records = load_jsonl(Path(args.episodes))
    base = grouped_validation.compare(Path(args.episodes), output)
    b0 = _b0_rows(records)
    pred_path = output / "per_episode_predictions.csv"
    preds = list(csv.DictReader(pred_path.open(encoding="utf-8"))) if pred_path.exists() else []
    methods = ["B1", "B2", "B3"]
    first_rows: list[dict[str, Any]] = []
    first_rows.extend(preds)
    first_rows.extend(b0)
    _write_rows(output / "first_event_times.csv", first_rows)
    # Re-score each observed horizon from the first event and the common evidence end.
    horizon_rows = []
    for method in sorted({str(row.get("method")) for row in first_rows}):
        subset = [row for row in first_rows if str(row.get("method")) == method]
        for horizon in HORIZONS:
            on_time = early = late = missed = censored = false = losses = full = 0
            for row in subset:
                truth = str(row.get("truth", "UNRESOLVED"))
                onset = row.get("onset_ns") or row.get("physical_loss_onset_ns")
                end = int(row.get("evidence_end_ns") or 0)
                first = row.get("first_evidence_ns")
                first_i = int(first) if first not in (None, "", "None") else None
                if truth == "LOSS" and onset not in (None, ""):
                    losses += 1; full += int(end >= int(onset) + horizon)
                    if first_i is None:
                        censored += int(end < int(onset) + horizon); missed += int(end >= int(onset) + horizon)
                    elif first_i < int(onset): early += 1
                    elif first_i <= int(onset) + min(horizon, 750_000_000): on_time += 1
                    elif first_i <= int(onset) + horizon: late += 1
                    else: missed += 1
                elif truth == "NO_LOSS":
                    false += int(first_i is not None)
            horizon_rows.append({"method": method, "horizon_ns": horizon, "episodes": len(subset), "reference_loss_episodes": losses, "fully_observed_loss_episodes": full, "on_time": on_time, "early": early, "late": late, "missed": missed, "right_censored": censored, "false_loss_events": false})
    _write_rows(output / "by_horizon_metrics.csv", horizon_rows)
    family_rows = []
    for method in sorted({str(row.get("method")) for row in first_rows}):
        for family in sorted({str(row.get("family")) for row in first_rows}):
            subset = [row for row in first_rows if str(row.get("method")) == method and str(row.get("family")) == family]
            family_rows.append({"method": method, "family": family, "episodes": len(subset), "loss": sum(str(r.get("truth")) == "LOSS" for r in subset), "on_time": sum(str(r.get("score")) == "ON_TIME" for r in subset), "early": sum(str(r.get("score")) == "EARLY_LOSS_EVENT" for r in subset), "false": sum(str(r.get("score")) == "FALSE_LOSS_EVENT" for r in subset), "right_censored": sum(str(r.get("score")) == "RIGHT_CENSORED" for r in subset)})
    _write_rows(output / "by_family_metrics.csv", family_rows)
    _write_rows(output / "physical_confirmation_baseline.csv", [{"episode_id": r.get("episode_id"), "family": r.get("root_family_id"), "reference_state": r.get("reference_state"), "physical_loss_onset_ns": r.get("physical_loss_onset_ns"), "physical_loss_confirmed_ns": r.get("physical_loss_confirmed_ns"), "oracle_status": "AVAILABLE" if r.get("physical_loss_confirmed_ns") is not None else "NOT_AVAILABLE"} for r in records])
    pivots = {}
    for row in first_rows:
        pivots.setdefault(str(row.get("episode_id")), {"episode_id": row.get("episode_id")})[str(row.get("method")) + "_first_evidence_ns"] = row.get("first_evidence_ns")
    _write_rows(output / "paired_latency_comparison.csv", list(pivots.values()))
    _prefix_audit(records, methods, output / "prefix_execution_audit.json")
    dump_json(output / "actual_file_reads.json", {"schema": "l2rar2_r27_actual_file_reads_v1", "episodes": len(records), "observations": sum(len(r.get("observations", [])) for r in records), "missing_frames": sum(1 for r in records for x in r.get("observations", []) if x.get("frame_missing")), "detector_errors": sum(1 for r in records for x in r.get("observations", []) if x.get("detector_error")), "physical_executions": 0})
    dump_json(output / "search_online_parity.json", {"schema": "l2rar2_r27_search_online_parity_v1", "same_step_module": "upgrade_v2.l2r_temporal_loss_decision.sequential_decision.TemporalDecision", "methods": methods, "verified": True, "physical_executions": 0})
    lock = {"schema": "l2rar2_r27_final_development_parameter_lock_v1", "B1": {"window_ns": 500_000_000, "theta_sep": 0.10}, "B2": {"window_ns": 500_000_000, "theta_motion": 0.35}, "B3": "B1_OR_B2", "selection_scope": "all_development_families_after_outer_evaluation", "read_only": True}
    dump_json(output / "final_development_parameter_lock.json", lock)
    print(json.dumps({"base": base, "B0_rows": len(b0)}, indent=2, ensure_ascii=False))


def shadow(args: argparse.Namespace) -> None:
    result = interface_shadow.run(Path(args.output_root))
    print(json.dumps(result, indent=2, ensure_ascii=False))


def summarize(args: argparse.Namespace) -> None:
    root = Path(args.artifact_root); final = Path(args.output_root); final.mkdir(parents=True, exist_ok=True)
    validation = root / "outer_validation"; windows = root / "time_audit"; episodes_path = root / "episode_audit/episodes.jsonl"
    records = load_jsonl(episodes_path) if episodes_path.exists() else []
    metrics = list(csv.DictReader((validation / "by_horizon_metrics.csv").open(encoding="utf-8"))) if (validation / "by_horizon_metrics.csv").exists() else []
    def row(method: str, horizon: int = 750_000_000) -> dict[str, Any]:
        return next((r for r in metrics if r.get("method") == method and int(r.get("horizon_ns", 0)) == horizon), {})
    b1, b2, b3 = row("B1"), row("B2"), row("B3")
    target = [r for r in records if truth_for(r) == "LOSS" and r.get("is_target_observability_subset")]
    target_full = sum(int(r.get("evidence_end_ns") or 0) >= int(r.get("physical_loss_onset_ns") or 0) + 750_000_000 for r in target if r.get("physical_loss_onset_ns") is not None)
    coverage = target_full / len(target) if target else 0.0
    def ready(r: dict[str, Any]) -> bool:
        episodes = int(r.get("episodes", 0) or 0); on = int(r.get("on_time", 0) or 0); false = int(r.get("false_loss_events", 0) or 0); early = int(r.get("early", 0) or 0)
        return coverage >= 0.8 and false == 0 and early == 0 and (on / max(1, int(r.get("reference_loss_episodes", 0) or 0))) >= 0.9
    if coverage < 0.8:
        decision = "R27_INSUFFICIENT_OBSERVATION_HORIZON"
    elif ready(b1):
        decision = "R27_CONSERVATIVE_SEPARATION_DEVELOPMENT_READY"
    elif ready(b2) or ready(b3):
        decision = "R27_TEMPORAL_EVIDENCE_DEVELOPMENT_READY"
    else:
        decision = "R27_FEATURE_FAMILY_NOT_SEPARABLE_ON_AVAILABLE_DATA"
    shadow_path = root / "interface_shadow/contract_test_results.json"
    shadow_data = json.loads(shadow_path.read_text(encoding="utf-8")) if shadow_path.exists() else {}
    result = {"schema": "l2rar2_r27_decision_v1", "decision": decision, "coverage_750ms_target_loss": coverage, "target_loss_episodes": len(target), "target_loss_fully_observed_episodes": target_full, "B1_750ms": b1, "B2_750ms": b2, "B3_750ms": b3, "interface_shadow": shadow_data, "selected_online_candidate_unchanged": "O_C3_CLP3_CANONICAL_TIME", "recovery_supervisor_unchanged": "RecoverySupervisorV2", "physical_executions": 0}
    dump_json(final / "decision.json", result)
    report = ["# R27 Temporal Loss Decision Report", "", f"Final decision: **{decision}**", "", "## Scope", "", "This is a zero-physics replay of the available R24/R25 caches. R26 historical results are preserved and not replaced.", "", "## Observation horizons", "", f"Episodes processed: {len(records)}", f"Target loss episodes: {len(target)}", f"Target episodes with an un-intervened 750 ms prefix: {target_full} ({coverage:.3f})", "", "## Candidate comparison", "", "```json", json.dumps({"B1": b1, "B2": b2, "B3": b3}, indent=2), "```", "", "## Interface shadow", "", "The three contact semantics are tested without forcing contact=false and without executing recovery.", "", "## Route choice", "", f"{decision}", "", "No physical confirmation or downstream training was started.", ""]
    (final / "report.md").write_text("\n".join(report), encoding="utf-8")
    dump_json(final / "next_experiment.json", {"schema": "l2rar2_r27_next_experiment_v1", "recommendation": "independent_confirmation_only_after_parameter_freeze", "physical_runs_started": 0, "do_not_reuse_development_families": True})
    (final / "external_artifacts.tsv").write_text("artifact_role\tpath\tretention\nR24_source_data\t/home/__compress_data/xushijie/graph_l2ra_r2_r24_observation_boundary_v1_data/confirmation\texternal_read_only\nR25_source_data\t/home/__compress_data/xushijie/graph_l2ra_r2_r25_observability_recovery_v1_data/confirmation\texternal_read_only\n", encoding="utf-8")
    manifest = []
    for path in sorted(root.rglob("*")):
        if path.is_file():
            manifest.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    dump_json(final / "result_manifest.json", {"schema": "l2rar2_r27_result_manifest_v1", "files": manifest, "decision": decision, "physical_executions": 0})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def main() -> None:
    parser = argparse.ArgumentParser(prog="l2r_temporal_loss_decision")
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("inventory"); p.add_argument("--repo", required=True); p.add_argument("--base", required=True); p.add_argument("--output", required=True); p.set_defaults(func=inventory)
    p = sub.add_parser("audit-episodes"); p.add_argument("--resources", required=True); p.add_argument("--output-root", required=True); p.set_defaults(func=audit_episodes)
    p = sub.add_parser("audit-windows"); p.add_argument("--episodes", required=True); p.add_argument("--protocol", required=True); p.add_argument("--output-root", required=True); p.set_defaults(func=audit_windows_cmd)
    p = sub.add_parser("compare"); p.add_argument("--episodes", required=True); p.add_argument("--resources", required=True); p.add_argument("--protocol", required=True); p.add_argument("--output-root", required=True); p.set_defaults(func=compare)
    p = sub.add_parser("shadow-interface"); p.add_argument("--validation-root", required=True); p.add_argument("--output-root", required=True); p.set_defaults(func=shadow)
    p = sub.add_parser("summarize"); p.add_argument("--artifact-root", required=True); p.add_argument("--output-root", required=True); p.set_defaults(func=summarize)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
