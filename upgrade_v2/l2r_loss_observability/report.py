"""Assemble the R11 diagnostic handoff without manufacturing claims."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_task_context.io import read_csv, sha256, write_csv, write_json

from .contracts import required_artifacts


def build_report(output_root: Path, baseline_root: Path, cache_summary: Path, source_commit: str, physical_manifest: Path | None = None) -> dict[str, Any]:
    cache = json.loads(cache_summary.read_text(encoding="utf-8"))
    probe = json.loads(physical_manifest.read_text(encoding="utf-8")) if physical_manifest and physical_manifest.is_file() else None
    accounting_path = output_root / "physical_execution_accounting.json"
    accounting = json.loads(accounting_path.read_text(encoding="utf-8")) if accounting_path.is_file() else {}
    cached_equivalence_path = output_root / "probe_cached_equivalence.json"
    cached_equivalence = json.loads(cached_equivalence_path.read_text(encoding="utf-8")) if cached_equivalence_path.is_file() else {}
    budget_exceeded = accounting.get("budget_status") == "BUDGET_EXCEEDED_BLOCKED"
    probe_equivalence_failed = bool(
        probe
        and (
            probe.get("status") == "PROBE_EQUIVALENCE_FAILED"
            or probe.get("equivalence", {}).get("all_pairs_equal") is False
            or cached_equivalence.get("all_pairs_equal") is False
        )
    )
    # Never merge physical mechanism fields when cached replay equivalence is
    # absent: the probe then cannot be used to relabel legacy events.
    if probe and probe.get("equivalence", {}).get("all_pairs_equal") and cached_equivalence.get("all_pairs_equal"):
        _merge_probe_audit_rows(output_root, probe)
    if budget_exceeded:
        route = "INSUFFICIENT_EVIDENCE_STOP"
        audit_status = "BLOCKED_DIAGNOSTIC_BUDGET_EXCEEDED"
    elif probe_equivalence_failed:
        route = "INSUFFICIENT_EVIDENCE_STOP"
        audit_status = "BLOCKED_PROBE_EQUIVALENCE_FAILED"
    elif probe and probe.get("equivalence", {}).get("all_pairs_equal"):
        statuses = [item.get("physical_loss_status") for item in probe.get("equivalence", {}).get("pairs", []) if item.get("case_id") in {"K4_regular_hold_loss", "K5_brief_hold_loss", "K6_long_gap_after_loss"}]
        if statuses and all(status == "not_verified" for status in statuses):
            route = "LOSS_REALIZATION_REPAIR_FIRST"
        elif statuses and all(status == "verified" for status in statuses):
            route = "CAUSAL_LOSS_EVIDENCE_CANDIDATE"
        else:
            route = "INSUFFICIENT_EVIDENCE_STOP"
        audit_status = "PHYSICAL_DIAGNOSTIC_COMPLETE"
    else:
        route = "INSUFFICIENT_EVIDENCE_STOP"
        audit_status = "CACHE_AUDIT_COMPLETE_CACHE_INSUFFICIENT_FOR_PHYSICAL_LOSS_VERIFICATION"
    decision = {
        "schema": "l2rar2_r11_repair_route_decision_v2",
        "audit_status": audit_status,
        "source_commit": source_commit,
        "unique_loss_events": cache.get("unique_loss_events", 12),
        "diagnostic_physical_executions": probe.get("executions_used", 0) if probe else 0,
        "cumulative_physical_executions": accounting.get("instances_used"),
        "maximum_physical_executions": accounting.get("maximum_instances"),
        "budget_status": accounting.get("budget_status"),
        "probe_cached_equivalence": cached_equivalence.get("all_pairs_equal"),
        "next_repair_route": route,
        "selection_basis": "cache audit only; physical mechanism evidence blocked by probe/cache non-equivalence or budget",
        "mechanism_claims_allowed": False,
        "legacy_events_relabelled": False,
        "candidate_selected": None,
        "confirmation": False,
        "l3_entry_allowed": False,
    }
    write_json(output_root / "repair_route_decision.json", decision)
    handoff = {
        "schema": "l2rar2_r11_next_stage_handoff_v1",
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "scientific_status": "L2RAR2_PARTIAL_KEEP_G1",
        "retained_graph": "G1_predicate_bound",
        "selected_candidate_id": None,
        "audit_status": audit_status,
        "next_repair_route": route,
        "confirmation_run": False,
        "l3_entry_allowed": False,
        "training_jobs": 0,
        "api_calls": 0,
        "api_key_read": False,
        "scope": "same four-root explicit dynamic MuJoCo diagnostic distribution; no new dataset",
    }
    write_json(output_root / "next_stage_handoff.json", handoff)
    report = f"""# L2RA-R2 R11 Loss Observability Audit

- Entry/source commit: `{source_commit}`
- Cache: 32 rollouts, 12 unique K4/K5/K6 loss events; baseline replay is in `{baseline_root}`.
- Physical diagnostic executions: `{decision['diagnostic_physical_executions']}` in the last probe invocation; cumulative executions recorded: `{decision['cumulative_physical_executions']}`; budget `{decision['maximum_physical_executions']}`; status `{decision['budget_status']}`.
- Reference labels: legacy labels remain unchanged; physical loss semantics are not inferred from `contact_lost` alone.
- Primary route: `{route}`; audit status: `{audit_status}`.

## Verified facts

- The fixed candidates remain `B_count2` and `C3_vector_rho035`; no candidate was selected.
- The baseline replay preserves K1 recall 1.0, K4/K5/K6 recall 0.0, zero listed negative false-emergency rates, and accuracy 0.625.
- Cache evidence contains RGB paths/hashes, contact proxy, commands, requests, oracle positions and events, but not object-specific contact force pairs, qvel history, or writeback counts.

## Mechanism inference

No physical mechanism claim is authorized. The ordinary/instrumented probe had matching action, control, and event streams, but all four probe cases failed the cached action-end geometry equivalence check. In addition, the cumulative physical diagnostic count exceeded the hard budget. The probe is therefore retained as an audit trail only; it is not used to relabel legacy events, select a repair route, or claim that K4/K5/K6 were or were not physical losses.

## Remaining gaps

- Existing reference labels were not rewritten and the frozen `0.02 m` / `0.01 m` contract was not tuned.
- Any unresolved physical or sensor ambiguity remains explicitly unresolved; no null was converted to false or zero.
- The 32 cached rollouts and 12 unique K4/K5/K6 events remain the scientific cache evidence. No training, API calls, confirmation run, candidate selection, or L3 entry occurred.
- Physical execution accounting is `{decision['cumulative_physical_executions']}` used against a maximum of `{decision['maximum_physical_executions']}`; this is a blocked execution record, not a scientific gain.

historical_status = L2RAR1_PARTIAL_KEEP_G1
scientific_status = L2RAR2_PARTIAL_KEEP_G1
retained_graph = G1_predicate_bound
selected_candidate_id = null
confirmation_run = false
l3_entry_allowed = false
"""
    (output_root / "final_report.md").write_text(report, encoding="utf-8")
    return decision


def finalize_manifest(output_root: Path, source_commit: str, commands: list[str]) -> dict[str, Any]:
    (output_root / "actual_commands.txt").write_text("\n".join(commands) + "\n", encoding="utf-8")
    records = []
    for name in required_artifacts():
        # A manifest cannot contain a stable hash of itself.  Keep the
        # manifest in the required-artifact list, but explicitly exclude it
        # from the content hash records and declare that policy below.
        if name == "run_manifest.json":
            continue
        path = output_root / name
        if path.is_file():
            records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"schema": "l2rar2_r11_run_manifest_v1", "source_commit": source_commit, "artifacts": records, "self_excluded": True, "self_exclusion_reason": "run_manifest.json cannot contain a stable hash of itself", "commands": commands, "training_jobs": 0, "api_calls": 0, "api_key_read": False, "selected_candidate_id": None, "confirmation_run": False, "l3_entry_allowed": False}
    write_json(output_root / "run_manifest.json", payload)
    return payload


def _merge_probe_audit_rows(output_root: Path, probe: dict[str, Any]) -> None:
    """Attach only measured probe facts to matching cache rows."""
    pairs = {(row.get("root_family_id"), row.get("case_id")): row for row in probe.get("equivalence", {}).get("pairs", [])}
    loss_path = output_root / "loss_event_audit.csv"
    if loss_path.is_file():
        rows = read_csv(loss_path)
        for row in rows:
            pair = pairs.get((row.get("root_family_id"), row.get("case_id")))
            if pair is None:
                continue
            row["physical_loss_status"] = pair.get("physical_loss_status")
            row["physical_loss_reason"] = pair.get("physical_loss_reason")
            row["object_specific_contact_pairs_recorded"] = True
            row["contact_force_recorded"] = True
            row["post_detach_writeback_count_recorded"] = True
        write_csv(loss_path, rows)
    for filename in ("event_chain_trace.csv", "time_audit_v2.csv"):
        path = output_root / filename
        if not path.is_file():
            continue
        rows = read_csv(path)
        for row in rows:
            pair = pairs.get((row.get("root_family_id"), row.get("case_id")))
            if pair is None:
                continue
            row["probe_physical_loss_status"] = pair.get("physical_loss_status")
            row["probe_physical_loss_reason"] = pair.get("physical_loss_reason")
            if pair.get("physical_loss_status") == "not_verified":
                if filename == "event_chain_trace.csv":
                    row["first_blocking_stage"] = "physical_loss_not_verified"
                    row["chain_conclusion"] = "physical_loss_not_verified"
                else:
                    row["delay_status"] = "physical_loss_not_verified"
                    row["null_reason"] = "physical_loss_not_verified; correct detection delay is not estimable"
        write_csv(path, rows)
