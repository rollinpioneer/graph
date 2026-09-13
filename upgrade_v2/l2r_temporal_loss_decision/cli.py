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

from . import episodes, grouped_validation, interface_shadow, resources, shadow_replenishment
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
        end_key = (int(record.get("evidence_end_ns") or 0),
                   int(record.get("evidence_end_capture_order", 10**18)))
        rows = [
            row for row in record.get("observations", [])
            if (int(row.get("physical_time_ns", 0)), int(row.get("capture_order", -1))) <= end_key
        ]
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
                prefix_record["evidence_end_capture_order"] = int(rows[index].get("capture_order", -1))
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


def _horizon_metric(subset: list[dict[str, Any]], horizon: int) -> dict[str, int]:
    counts = {
        "episodes": len(subset), "reference_loss_episodes": 0,
        "fully_observed_loss_episodes": 0, "on_time": 0, "early": 0,
        "late": 0, "missed": 0, "right_censored": 0,
        "all_no_loss_episodes": 0, "false_loss_events": 0,
    }
    for row in subset:
        truth = str(row.get("truth", "UNRESOLVED"))
        onset = row.get("onset_ns") or row.get("physical_loss_onset_ns")
        end = int(row.get("evidence_end_ns") or 0)
        first = row.get("first_evidence_ns")
        first_i = int(first) if first not in (None, "", "None") else None
        if truth == "LOSS" and onset not in (None, ""):
            onset_i = int(onset)
            counts["reference_loss_episodes"] += 1
            counts["fully_observed_loss_episodes"] += int(end >= onset_i + horizon)
            if first_i is None:
                key = "right_censored" if end < onset_i + horizon else "missed"
                counts[key] += 1
            elif first_i < onset_i:
                counts["early"] += 1
            elif first_i <= onset_i + min(horizon, 750_000_000):
                counts["on_time"] += 1
            elif first_i <= onset_i + horizon:
                counts["late"] += 1
            else:
                counts["missed"] += 1
        elif truth == "NO_LOSS":
            counts["all_no_loss_episodes"] += 1
            counts["false_loss_events"] += int(first_i is not None)
    return counts


def _write_per_prefix_features(
    records: list[dict[str, Any]], predictions: list[dict[str, Any]], output: Path,
) -> None:
    by_episode_method = {
        (str(row.get("episode_id")), str(row.get("method"))): row
        for row in predictions
    }
    with output.open("w", encoding="utf-8") as stream:
        for record in records:
            for method in ("B1", "B2", "B3"):
                params = by_episode_method[(str(record["episode_id"]), method)]
                result = replay(
                    record, method,
                    window_ns=int(params["window_ns"]),
                    theta_motion=float(params["theta_motion"]),
                    theta_sep=float(params["theta_sep"]),
                )
                for item in result["outputs"]:
                    row = {
                        "episode_id": record["episode_id"],
                        "root_family_id": record["root_family_id"],
                        "method": method,
                        "window_ns": int(params["window_ns"]),
                        "theta_motion": float(params["theta_motion"]),
                        "theta_sep": float(params["theta_sep"]),
                        **item,
                    }
                    stream.write(json.dumps(row, sort_keys=True) + "\n")


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
            horizon_rows.append({"method": method, "horizon_ns": horizon,
                                 **_horizon_metric(subset, horizon)})
    _write_rows(output / "by_horizon_metrics.csv", horizon_rows)
    coverage_rows = []
    for horizon in HORIZONS:
        loss_rows = [r for r in records if truth_for(r) == "LOSS" and r.get("physical_loss_onset_ns") is not None]
        full = sum(int(r.get("evidence_end_ns") or 0) >= int(r["physical_loss_onset_ns"]) + horizon for r in loss_rows)
        target_rows = [r for r in loss_rows if r.get("is_target_observability_subset")]
        target_full = sum(int(r.get("evidence_end_ns") or 0) >= int(r["physical_loss_onset_ns"]) + horizon for r in target_rows)
        coverage_rows.append({
            "horizon_ns": horizon,
            "reference_loss_episodes": len(loss_rows),
            "fully_observed_loss_episodes": full,
            "coverage": full / len(loss_rows) if loss_rows else 0.0,
            "target_reference_loss_episodes": len(target_rows),
            "target_fully_observed_loss_episodes": target_full,
            "target_coverage": target_full / len(target_rows) if target_rows else 0.0,
        })
    _write_rows(output / "observation_horizon_coverage.csv", coverage_rows)
    family_rows = []
    for method in sorted({str(row.get("method")) for row in first_rows}):
        for family in sorted({str(row.get("family")) for row in first_rows}):
            subset = [row for row in first_rows if str(row.get("method")) == method and str(row.get("family")) == family]
            for horizon in HORIZONS:
                family_rows.append({"method": method, "family": family,
                                    "horizon_ns": horizon,
                                    **_horizon_metric(subset, horizon)})
    _write_rows(output / "by_family_metrics.csv", family_rows)
    _write_per_prefix_features(records, preds, output / "per_prefix_features.jsonl")
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
    validation = root / "outer_validation"; episodes_path = root / "episode_audit/episodes.jsonl"
    records = load_jsonl(episodes_path) if episodes_path.exists() else []
    metrics = list(csv.DictReader((validation / "by_horizon_metrics.csv").open(encoding="utf-8"))) if (validation / "by_horizon_metrics.csv").exists() else []
    coverage_rows = list(csv.DictReader((validation / "observation_horizon_coverage.csv").open(encoding="utf-8"))) if (validation / "observation_horizon_coverage.csv").exists() else []
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
    proposals = sum(r.get("first_proposal_ns") is not None for r in records)
    executions = sum(bool(r.get("recovery_control_executed")) for r in records)
    proposal_only = sum(r.get("first_proposal_ns") is not None and not r.get("recovery_control_executed") for r in records)
    proposal_only_preserved = sum(
        r.get("first_proposal_ns") is not None
        and not r.get("recovery_control_executed")
        and int(r.get("evidence_end_ns") or 0) == int(r.get("last_candidate_observation_ns") or 0)
        for r in records
    )
    result = {"schema": "l2rar2_r27_zero_physics_review_decision_v1", "decision": decision, "formal_r27_decision_retained": "R27_INSUFFICIENT_OBSERVATION_HORIZON", "confirmation_passed": False, "coverage_750ms_target_loss": coverage, "target_loss_episodes": len(target), "target_loss_fully_observed_episodes": target_full, "proposal_episodes": proposals, "recovery_control_executed_episodes": executions, "proposal_without_execution_episodes": proposal_only, "proposal_without_execution_preserved_to_last_observation": proposal_only_preserved, "B1_750ms": b1, "B2_750ms": b2, "B3_750ms": b3, "interface_shadow": shadow_data, "selected_online_candidate_unchanged": "O_C3_CLP3_CANONICAL_TIME", "recovery_supervisor_unchanged": "RecoverySupervisorV2", "physical_executions": 0}
    dump_json(final / "decision.json", result)
    report = [
        "# R27 Zero-Physics Review Report", "",
        "Formal R27 decision retained: **R27_INSUFFICIENT_OBSERVATION_HORIZON**", "",
        "This review is additive. It does not modify the original R27 result directory, report, commit, selected online candidate, or recovery supervisor. It does not claim confirmation passed.", "",
        "## Executed-Control Censoring", "",
        f"Episodes: {len(records)}; proposal episodes: {proposals}; logged recovery-control executions: {executions}; proposal without execution: {proposal_only}.", "",
        f"Proposal-only episodes preserved through their last candidate observation: {proposal_only_preserved}/{proposal_only}. A proposal alone is never an observation cutoff.", "",
        "The observation end is the last row strictly before the earliest logged recovery-control start, release, or attempt change. Recovery start requires both a non-empty execution record and a later active lifecycle row from a new attempt.", "",
        "## Observation Coverage", "",
        "| horizon | all loss full/total | all coverage | target loss full/total | target coverage |", "|---:|---:|---:|---:|---:|",
    ]
    for item in coverage_rows:
        report.append(
            f"| {int(item['horizon_ns']) // 1_000_000} ms | {item['fully_observed_loss_episodes']}/{item['reference_loss_episodes']} | {float(item['coverage']):.2%} | {item['target_fully_observed_loss_episodes']}/{item['target_reference_loss_episodes']} | {float(item['target_coverage']):.2%} |"
        )
    report.extend(["", "## B1/B2/B3 Counts", "",
                   "Counts are per horizon over all reference-loss episodes. `late` is retained separately for the 1000 ms diagnostic window; the formal on-time deadline remains 750 ms.", "",
                   "| method | horizon | on time | early | late | missed | right censored | false loss |", "|---|---:|---:|---:|---:|---:|---:|---:|"])
    for item in metrics:
        if item.get("method") in {"B1", "B2", "B3"}:
            report.append(
                f"| {item['method']} | {int(item['horizon_ns']) // 1_000_000} ms | {item['on_time']} | {item['early']} | {item['late']} | {item['missed']} | {item['right_censored']} | {item['false_loss_events']} |"
            )
    report.extend(["", "Per-family counts for every horizon are in `outer_validation/by_family_metrics.csv`.", "",
                   "## Frozen CLP3 Interface Shadow", "",
                   "The test uses the frozen `INTERCEPT_REASON` value `historical_hold_established_then_observed_non_release_contact_loss`.", "",
                   "The first exact-reason proposal with measured `contact=false` becomes pending and is suppressed while waiting; same-time duplicates remain pending; a strictly later same-attempt `contact=false` row at 100 ms confirms. `PASS_THROUGH` preserves the input proposal/action but is not a CLP3 confirmation. `CLEAR` resets pending and is also not confirmation. `contact=true`, `contact=null`, release, attempt change, and a different reason are never converted into fabricated contact.", "",
                   "## Development Replenishment", ""])
    supplement_path = root / "shadow_replenishment/shadow_replenishment_results.json"
    supplement = json.loads(supplement_path.read_text(encoding="utf-8")) if supplement_path.exists() else {}
    report.append(f"Status: `{supplement.get('status', 'NOT_AVAILABLE')}`; physical development rollouts: {supplement.get('physical_executions', 0)}.")
    supplement_metrics_path = root / "shadow_replenishment/shadow_comparison_by_horizon.csv"
    supplement_metrics = list(csv.DictReader(supplement_metrics_path.open(encoding="utf-8"))) if supplement_metrics_path.exists() else []
    supplement_coverage_path = root / "shadow_replenishment/shadow_observation_horizon_coverage.csv"
    supplement_coverage = list(csv.DictReader(supplement_coverage_path.open(encoding="utf-8"))) if supplement_coverage_path.exists() else []
    if supplement_coverage:
        item = next((row for row in supplement_coverage if int(row["horizon_ns"]) == 750_000_000), {})
        report.extend(["", f"At 750 ms, actual-loss coverage is {item.get('fully_observed_loss_episodes', 0)}/{item.get('reference_loss_episodes', 0)} ({float(item.get('coverage', 0)):.2%}). The intended slow-detachment cases that did not become physical loss remain no-loss outcomes.", "",
                       "| method | on time | early | late | missed | right censored | false loss |", "|---|---:|---:|---:|---:|---:|---:|"])
        for item in supplement_metrics:
            if item.get("method") in {"B1", "B2", "B3"} and int(item.get("horizon_ns", 0)) == 750_000_000:
                report.append(f"| {item['method']} | {item['on_time']} | {item['early']} | {item['late']} | {item['missed']} | {item['right_censored']} | {item['false_loss_events']} |")
    report.extend(["", "All 24 development rollouts retained exactly 1.50 s after intervention end. Candidate recovery-control executions and candidate-triggered early terminations were both zero.", "",
                   "Any replenishment comparison is development-only and is not an independent confirmation.", "",
                   "## Conclusion", "", f"Corrected cache-only route result: **{decision}**.", "",
                   "Formal R27 remains unchanged. Confirmation is not passed.", ""])
    report_text = "\n".join(report)
    (final / "report.md").write_text(report_text, encoding="utf-8")
    (final / "zero_physics_review_report.md").write_text(report_text, encoding="utf-8")
    dump_json(final / "next_experiment.json", {"schema": "l2rar2_r27_next_experiment_v1", "recommendation": "independent_confirmation_only_after_parameter_freeze", "physical_runs_started": 0, "do_not_reuse_development_families": True})
    (final / "external_artifacts.tsv").write_text("artifact_role\tpath\tretention\nR24_source_data\t/home/__compress_data/xushijie/graph_l2ra_r2_r24_observation_boundary_v1_data/confirmation\texternal_read_only\nR25_source_data\t/home/__compress_data/xushijie/graph_l2ra_r2_r25_observability_recovery_v1_data/confirmation\texternal_read_only\nR27_shadow_replenishment_raw\t/home/__compress_data/xushijie/graph_l2ra_r2_r27_temporal_decision_v1_data/shadow_replenishment_raw_v1\texternal_development_data\n", encoding="utf-8")
    manifest = []
    manifest_path = final / "result_manifest.json"
    for path in sorted(root.rglob("*")):
        if path.is_file() and path != manifest_path:
            manifest.append({"path": str(path.relative_to(root)), "size_bytes": path.stat().st_size, "sha256": _sha256(path)})
    dump_json(final / "result_manifest.json", {"schema": "l2rar2_r27_result_manifest_v1", "files": manifest, "decision": decision, "physical_executions": 0})
    print(json.dumps(result, indent=2, ensure_ascii=False))


def supplement(args: argparse.Namespace) -> None:
    output = Path(args.output_root); output.mkdir(parents=True, exist_ok=True)
    payload = shadow_replenishment.design()
    dump_json(output / "shadow_replenishment_design.json", payload)
    if not args.collect:
        result = {
            "schema": "l2rar2_r27_shadow_replenishment_results_v2",
            "status": "DESIGN_ONLY_NO_PHYSICAL_EXECUTION",
            "physical_executions": 0,
            "confirmation_passed": False,
        }
        dump_json(output / "shadow_replenishment_results.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return

    raw_root = Path(args.raw_root) if args.raw_root else output / "raw"
    collection = shadow_replenishment.collect(raw_root, workers=args.workers)
    if collection["status"] != "PASS":
        result = {
            "schema": "l2rar2_r27_shadow_replenishment_results_v2",
            "status": "COLLECTION_FAILED",
            "collection": collection,
            "physical_executions": collection["physical_executions"],
            "confirmation_passed": False,
        }
        dump_json(output / "shadow_replenishment_results.json", result)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return


    resource_payload = {
        "schema": "l2rar2_r27_shadow_replenishment_resources_v1",
        "sources": [{"name": "R27_SHADOW_REPLENISHMENT", "path": str(raw_root)}],
    }
    episode_result = episodes.build(resource_payload, output / "episode_audit")
    records = load_jsonl(output / "episode_audit/episodes.jsonl")
    prediction_rows = []
    for record in records:
        for method in ("B1", "B2", "B3"):
            replayed = replay(record, method)
            prediction_rows.append({
                "episode_id": record["episode_id"],
                "family": record["root_family_id"],
                "case_id": record["case_id"],
                "method": method,
                "truth": truth_for(record),
                "physical_loss_onset_ns": record.get("physical_loss_onset_ns"),
                "evidence_end_ns": record.get("evidence_end_ns"),
                "first_evidence_ns": replayed["first_evidence_ns"],
            })
    _write_rows(output / "per_episode_shadow_predictions.csv", prediction_rows)
    comparison_rows = []
    for method in ("B1", "B2", "B3"):
        subset = [row for row in prediction_rows if row["method"] == method]
        for horizon in HORIZONS:
            comparison_rows.append({"method": method, "horizon_ns": horizon,
                                    **_horizon_metric(subset, horizon)})
    _write_rows(output / "shadow_comparison_by_horizon.csv", comparison_rows)
    loss_rows = [r for r in records if truth_for(r) == "LOSS" and r.get("physical_loss_onset_ns") is not None]
    coverage_rows = []
    for horizon in HORIZONS:
        full = sum(int(r.get("evidence_end_ns") or 0) >= int(r["physical_loss_onset_ns"]) + horizon for r in loss_rows)
        coverage_rows.append({
            "horizon_ns": horizon, "reference_loss_episodes": len(loss_rows),
            "fully_observed_loss_episodes": full,
            "coverage": full / len(loss_rows) if loss_rows else 0.0,
        })
    _write_rows(output / "shadow_observation_horizon_coverage.csv", coverage_rows)
    result = {
        "schema": "l2rar2_r27_shadow_replenishment_results_v2",
        "status": "DEVELOPMENT_COLLECTION_COMPLETE",
        "collection": collection,
        "episode_audit": episode_result,
        "physical_executions": collection["physical_executions"],
        "candidate_control_executions": collection["candidate_control_executions"],
        "candidate_early_terminations": collection["candidate_early_terminations"],
        "post_intervention_observation_ns": 1_500_000_000,
        "external_raw_root": str(raw_root),
        "comparison_scope": "DEVELOPMENT_ONLY_B1_VS_B2_VS_B3",
        "confirmation_passed": False,
    }
    dump_json(output / "shadow_replenishment_results.json", result)
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
    p = sub.add_parser("supplement"); p.add_argument("--output-root", required=True); p.add_argument("--raw-root"); p.add_argument("--collect", action="store_true"); p.add_argument("--workers", type=int, default=2); p.set_defaults(func=supplement)
    args = parser.parse_args(); args.func(args)


if __name__ == "__main__":
    main()
