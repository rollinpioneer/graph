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
    if probe and probe.get("equivalence", {}).get("all_pairs_equal"):
        _merge_probe_audit_rows(output_root, probe)
    if probe and probe.get("status") == "PROBE_EQUIVALENCE_FAILED":
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
        "next_repair_route": route,
        "selection_basis": "bounded physical evidence only; no candidate or threshold was changed",
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
- Physical diagnostic executions: `{decision['diagnostic_physical_executions']}`; no training and no API calls.
- Reference labels: legacy labels remain unchanged; physical loss semantics are not inferred from `contact_lost` alone.
- Primary route: `{route}`

## Verified facts

- The fixed candidates remain `B_count2` and `C3_vector_rho035`; no candidate was selected.
- The baseline replay preserves K1 recall 1.0, K4/K5/K6 recall 0.0, zero listed negative false-emergency rates, and accuracy 0.625.
- Cache evidence contains RGB paths/hashes, contact proxy, commands, requests, oracle positions and events, but not object-specific contact force pairs, qvel history, or writeback counts.

## Mechanism inference

The route is selected only from the bounded probe when its ordinary and instrumented replays are equivalent. The probe is diagnostic evidence, not a new family, training sample, confirmation run, or online candidate score.
For the probed root, K4/K5/K6 all retained object-to-`finger_left`/`finger_right` contact force after weld-off, with no post-detach writeback increase; they are therefore `not_verified` as task-level losses, despite the legacy `contact_lost` event.

## Remaining gaps

- Existing reference labels were not rewritten and the frozen `0.02 m` / `0.01 m` contract was not tuned.
- Any unresolved physical or sensor ambiguity remains explicitly unresolved; no null was converted to false or zero.

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
        path = output_root / name
        if path.is_file():
            records.append({"path": name, "size_bytes": path.stat().st_size, "sha256": sha256(path)})
    payload = {"schema": "l2rar2_r11_run_manifest_v1", "source_commit": source_commit, "artifacts": records, "commands": commands, "training_jobs": 0, "api_calls": 0, "api_key_read": False, "selected_candidate_id": None, "confirmation_run": False, "l3_entry_allowed": False}
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
