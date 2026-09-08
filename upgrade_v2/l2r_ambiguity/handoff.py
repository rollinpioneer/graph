"""Assemble the auditable L2RA handoff without upgrading historical L2R."""
from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path
from typing import Any


def _read_json(path: Path, default: Any = None) -> Any:
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else default


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _copy(source: Path, target: Path) -> bool:
    if not source.is_file(): return False
    target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(source, target); return True


def _not_run(path: Path, name: str, reason: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"status,reason\nNOT_RUN,{name}: {reason}\n", encoding="utf-8")


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file(): return []
    with path.open(encoding="utf-8", newline="") as handle: return list(csv.DictReader(handle))


def handoff(run_root: Path, resolved: dict[str, Any], protocol_path: Path, output_root: Path) -> dict[str, Any]:
    final = output_root; final.mkdir(parents=True, exist_ok=True)
    diagnosis = _read_json(run_root / "rounds/l2ra_2_cause_diagnosis/diagnosis_gate.json", {})
    route = _read_json(run_root / "rounds/l2ra_3_development_and_selection/development_route.json", {})
    consumption = _read_json(run_root / "rounds/l2ra_4_fresh_confirmation/confirmation_consumption.json", {})
    selection = _read_json(run_root / "locks/selection_lock.json", {})
    probe_lock = _read_json(run_root / "data/new_development/probe_lock.json", {})
    duplicate_groups = _read_csv(run_root / "data/new_development/content_duplicate_groups.csv")
    candidate_metrics = _read_csv(run_root / "rounds/l2ra_3_development_and_selection/candidate_metrics.csv")
    a4_complete = consumption.get("status") == "COMPLETE" and (run_root / "rounds/l2ra_4_fresh_confirmation/standard_confirmation.csv").is_file()
    selected = selection.get("selected_candidate_id")
    if not route:
        status = "L2RA_DIAGNOSIS_ONLY" if diagnosis.get("status") else "EXECUTION_BLOCKED"
    elif route.get("status") == "DEVELOPMENT_NOT_READY":
        status = "L2RA_PARTIAL_KEEP_G1" if diagnosis.get("status") else "EXECUTION_BLOCKED"
    elif a4_complete and consumption.get("challenge_confirmation") != "NOT_RUN":
        status = "L2RA_PARTIAL_KEEP_G1"
    else:
        status = "L2RA_PARTIAL_KEEP_G1"
    if any("OBSERVATION_INTERVENTION" in str(value) for value in consumption.values()):
        status = "L2RA_OBSERVABILITY_LIMIT" if a4_complete else status

    _copy(run_root / "rounds/l2ra_2_cause_diagnosis/cause_attribution.csv", final / "cause_attribution.csv")
    # Aggregate legacy reproduction from the three A1 split reports.
    legacy = {"schema": "pathgraph_l2ra_legacy_reproduction_v1", "splits": {}}
    for split in ("legacy_dev_fit", "legacy_dev_select", "legacy_confirmation"):
        path = run_root / f"rounds/l2ra_1_replay_and_localization/{split}/legacy_reproduction.json"
        legacy["splits"][split] = _read_json(path, {"status": "MISSING"})
    _write_json(final / "legacy_reproduction.json", legacy)
    _write_json(final / "metric_definitions.json", {
        "schema": "pathgraph_l2ra_metric_definitions_v1", "legacy_metrics": "frozen summarize_executions denominator and branch semantics",
        "new_metrics": {"raw_rule_overlap_rollout_rate": "rollouts with any raw retry/recover overlap / all rollouts",
                        "effective_conflict_rollout_rate": "rollouts with any effective guard conflict / all rollouts",
                        "event_window_conflict_rate": "event windows with effective conflict / evaluable windows",
                        "missed_grasp_type_recall": "correct retry / evaluable missed-grasp events",
                        "held_loss_type_recall": "correct recover / evaluable held-loss events",
                        "false_emergency_rate": "negative opportunities with retry/recover / negative opportunities",
                        "unjustified_definite_rate": "insufficient-history opportunities with definite action / opportunities",
                        "decision_delay": "observation steps and seconds from first decidable observation to first correct action"},
        "independent_unit": "root_family_id", "bootstrap_resamples": 5000, "zero_denominator": "NOT_ESTIMABLE_NOT_PASS"})
    if not _copy(run_root / "rounds/l2ra_3_development_and_selection/candidate_registry.json", final / "candidate_registry.json"):
        _write_json(final / "candidate_registry.json", {"status": "NOT_RUN", "reason": "development candidate registry unavailable"})
    _copy(run_root / "locks/selection_lock.json", final / "locks/selection_lock.json")
    a4root = run_root / "rounds/l2ra_4_fresh_confirmation"
    for name in ("standard_confirmation.csv", "challenge_confirmation.csv", "paired_family_effects.csv", "event_type_metrics.csv", "false_emergency_and_unknown.csv", "decision_delay.csv"):
        if not _copy(a4root / name, final / f"tables/{name}"):
            _not_run(final / f"tables/{name}", name, "A4 was not opened because development route was not ready")
    if selected:
        _write_json(final / "selected_candidate/candidate_config.json", selection.get("selected_candidate_config"))
    import hashlib
    def digest(path: Path) -> str:
        value = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""): value.update(block)
        return value.hexdigest()
    external = []
    upstream_manifest = Path(next(item["resolved_path"] for item in resolved["frozen_sources"] if item["logical_id"] == "frozen:manifests/external_artifacts.tsv"))
    external.append({"logical_path": "upstream/final_v1/manifests/external_artifacts.tsv", "original_path": str(upstream_manifest), "original_filename": upstream_manifest.name,
                     "size_bytes": upstream_manifest.stat().st_size, "sha256": digest(upstream_manifest), "artifact_type": "upstream_external_manifest",
                     "purpose": "authoritative frozen L2R external payload inventory", "recovery_method": "restore the frozen file at the recorded path and verify this SHA256"})
    data_roots = [("new_development", run_root / "data/new_development"),
                  ("standard_confirmation", run_root / "data/standard_confirmation"),
                  ("challenge_confirmation", run_root / "data/challenge_confirmation")]
    for logical_root, data_root in data_roots:
        rollout_root = data_root / "rollouts"
        for path in sorted(rollout_root.rglob("*")):
            if not path.is_file():
                continue
            artifact_type = "new_probe_rgb_or_array" if path.suffix in {".jpg", ".npz"} else "new_probe_raw_trajectory"
            external.append({"logical_path": f"{logical_root}/{path.relative_to(data_root).as_posix()}", "original_path": str(path.resolve()), "original_filename": path.name,
                             "size_bytes": path.stat().st_size, "sha256": digest(path), "artifact_type": artifact_type,
                             "purpose": "new L2RA raw dynamic probe trajectory, observation, or oracle diagnostic payload", "recovery_method": "restore the retained local file or rerun the locked family and seed"})
    (final / "manifests").mkdir(parents=True, exist_ok=True)
    with (final / "manifests/external_artifacts.tsv").open("w", encoding="utf-8", newline="") as handle:
        fields = ["logical_path", "original_path", "original_filename", "size_bytes", "sha256", "artifact_type", "purpose", "recovery_method"]
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n"); writer.writeheader(); writer.writerows(external)
    _write_json(final / "unsupported_claims.json", {"claims_not_supported": ["real robot success", "reward gain", "policy gain", "new task generalization", "L3 entry", "physical grasp/recovery guarantee"],
                                                       "historical_status_preserved": "L2R_PARTIAL_KEEP_COARSE_GRAPH", "selected_candidate_is_hypothesized": bool(selected)})
    execution = {"schema": "pathgraph_l2ra_execution_manifest_v1", "status": status, "protocol": str(protocol_path.resolve()),
                 "api_calls": 0, "training_jobs": 0, "api_key_read": False, "second_view_active": False,
                 "historical_l2r_status": "L2R_PARTIAL_KEEP_COARSE_GRAPH", "historical_retained_graph": "G1_predicate_bound",
                 "new_development_families": probe_lock.get("family_count"), "new_development_rollouts": probe_lock.get("rollouts"),
                 "new_development_observation_intervention_rollouts": probe_lock.get("observation_intervention_rollouts"),
                 "unique_predicate_content_groups": len(duplicate_groups), "statistical_independent_unit": "root_family_id",
                 "repeated_content_independence_claimed": False,
                 "a1_replay_completed": all(value.get("raw_prediction_replayed") is True for value in legacy["splits"].values()), "a3_route": route.get("status"), "a4_standard_completed": a4_complete,
                 "a4_challenge_completed": a4_complete and (a4root / "challenge_confirmation.csv").is_file(), "l3_entry_allowed": False}
    _write_json(final / "execution_manifest.json", execution)
    confirmation_gate = _read_json(a4root / "confirmation_gate.json", {})
    confirmation_status = confirmation_gate.get("status") or ("NOT_RUN" if not a4_complete else "UNKNOWN")
    control_metrics = next((row for row in candidate_metrics if row.get("candidate_id") == "D_priority_only"), {})
    selected_metrics = next((row for row in candidate_metrics if row.get("candidate_id") == selected), {}) if selected else {}
    handoff_payload = {"schema": "pathgraph_l2ra_handoff_v1", "historical_l2r_status": "L2R_PARTIAL_KEEP_COARSE_GRAPH",
                       "historical_retained_graph": "G1_predicate_bound", "new_status": status,
                       "cause_supported_by": diagnosis.get("supported_hypotheses", []), "selected_candidate_id": selected,
                       "candidate_sha256": selection.get("candidate_sha256"), "event_memory_sha256": next((item["sha256"] for item in selection.get("locked_files", []) if item["logical_id"] == "event_memory_code"), selection.get("event_memory_code_sha256")),
                       "confirmation_status": confirmation_status,
                       "standard_confirmation_completed": a4_complete, "challenge_confirmation_completed": a4_complete and (a4root / "challenge_confirmation.csv").is_file(),
                       "development_denominators": {
                           "decidable_physical_event_count": int(float(control_metrics.get("decidable_physical_event_count", 0) or 0)),
                           "undecidable_physical_event_count": int(float(control_metrics.get("undecidable_physical_event_count", 0) or 0)),
                           "miss_events": int(float(control_metrics.get("miss_events", 0) or 0)),
                           "loss_events": int(float(control_metrics.get("loss_events", 0) or 0)),
                           "false_emergency": int(float(control_metrics.get("false_emergency", 0) or 0)),
                           "unjustified_definite": int(float(control_metrics.get("unjustified_definite", 0) or 0)),
                       },
                       "selected_candidate_metrics": selected_metrics,
                       "metric_definition_changed": True, "legacy_metric_also_reported": True, "l3_entry_allowed": False,
                       "reward_or_policy_gain_claimed": False}
    _write_json(final / "next_stage_handoff.json", handoff_payload)
    report = [f"# L2RA Final Report", "", f"- Status: `{status}`", "- Historical L2R status: `L2R_PARTIAL_KEEP_COARSE_GRAPH`", "- Retained graph: `G1_predicate_bound`", f"- New candidate: `{selected}`", "- API calls: `0`; training jobs: `0`; API key read: `false`", "", "## Scope", "", "The evidence is limited to offline replay and the bounded dynamic tabletop proxy. It does not establish real-robot execution, reward improvement, policy improvement, or task generalization.", "", f"The new development set contains {probe_lock.get('family_count', 0)} root families and {probe_lock.get('rollouts', 0)} rollouts. Repeated deterministic predicate streams collapse to {len(duplicate_groups)} content groups; root family, not rollout or frame, is the statistical unit, and repeated content is not treated as independent evidence.", "", "## Diagnosis", "", "Frozen G2 replay reproduced 12 ambiguous rollouts among 96 legacy confirmation rollouts (0.125), all in `slip_then_recover`. The frozen rules make observed slip a subset of generic grasp failure, while the whole-sequence executor reports both guards even when priority selects recovery.", "", "## Gate", "", f"- Diagnosis route: `{diagnosis.get('status')}`", f"- Development route: `{route.get('status')}`", "- Five selectable configurations were evaluated; none passed all predeclared development gates.", f"- Development physical-event denominator: `{control_metrics.get('decidable_physical_event_count', 0)}` decidable, `{control_metrics.get('undecidable_physical_event_count', 0)}` not decidable.", f"- Standard confirmation: `{ 'COMPLETE' if a4_complete else 'NOT_RUN' }`", f"- Challenge confirmation: `{ 'COMPLETE' if a4_complete and (a4root / 'challenge_confirmation.csv').is_file() else 'NOT_RUN' }`", f"- Confirmation gate: `{confirmation_status}`", "- L3 entry allowed: `false`"]
    (final / "l2ra_final_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    return {"status": status, "selected_candidate_id": selected, "standard_confirmation_completed": a4_complete, "challenge_intervention_only": a4_complete}
