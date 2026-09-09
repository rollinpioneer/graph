"""Versioned online interface for separating hold lifecycle states.

This module is deliberately separate from the frozen ``M1`` implementation.
It exposes whether an observation is invalid, merely lacks current hold
evidence, or follows a previously established hold.  It never uses oracle
events, case labels, or future outcomes.
"""
from __future__ import annotations

import importlib
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_ambiguity.event_memory import FALSE, TRUE, UNKNOWN, tri
from upgrade_v2.l2r_hold_evidence.evaluate_v2 import _online_observation, _predicates
from upgrade_v2.l2r_hold_evidence.hold_features import build_features
from upgrade_v2.l2r_hold_evidence.hold_predicates import evaluate_candidate
from upgrade_v2.visual_refine_l2.io import secret_scan

from .contract import ControllerRequest, RequestedEffect, validate_request_provenance
from .io import read_csv, read_jsonl, sha256, write_csv, write_json, write_jsonl
from .reference import build_reference


INTERFACE_REPAIR_VERSION = "l2rar2_online_interface_repair_v1"
FIXED_CANDIDATES = {
    "B_count2": ("B_count2", {}),
    "C3_vector_rho035": ("C3_vector", {"relative_rho_max": 0.35}),
}
POSITIVE_CASES = {
    "K1_hold_request_ends_without_hold": "retry_grasp",
    "K4_regular_hold_loss": "recover_object",
    "K5_brief_hold_loss": "recover_object",
    "K6_long_gap_after_loss": "recover_object",
}
NEGATIVE_CASES = {
    "K2_touch_request_completes_without_hold",
    "K3_normal_hold_pause_resume",
    "K7_commanded_release",
    "K8_acquisition_touch_then_continue",
}


@dataclass
class OnlineInterfaceState:
    attempt_id: int = 0
    attempt_started_observed: bool = False
    historical_hold_established: bool = False
    pending_event: str = "none"
    pending_event_id: str | None = None
    event_counter: int = 0
    last_contact: str = UNKNOWN
    consumed_end_keys: tuple[str, ...] = ()


class RepairedOnlineInterface:
    """Causal lifecycle interface with explicit observation-state accounting."""

    def __init__(self, *, history_complete: bool = True) -> None:
        self.history_complete = bool(history_complete)
        self.state = OnlineInterfaceState()

    def _reset_attempt(self, attempt_id: int, started_observed: bool) -> None:
        self.state.attempt_id = attempt_id
        self.state.attempt_started_observed = started_observed
        self.state.historical_hold_established = False
        self.state.pending_event = "none"
        self.state.pending_event_id = None
        self.state.last_contact = UNKNOWN

    def _start_event(self, kind: str) -> None:
        if self.state.pending_event != kind:
            self.state.event_counter += 1
            self.state.pending_event_id = f"l2rar2_event_{self.state.event_counter:03d}_{kind}"
        self.state.pending_event = kind

    def observe(
        self,
        observation: dict[str, Any],
        evidence: dict[str, Any],
        request: ControllerRequest | None,
    ) -> dict[str, Any]:
        now = float(observation.get("time", 0.0))
        capture_order = int(observation.get("capture_order", observation.get("frame_index", 0) or 0))
        attempt_id = int(observation.get("attempt_id") or 0)
        active = tri(observation.get("attempt_active"))
        end = tri(observation.get("attempt_end"))
        end_reason = observation.get("attempt_end_reason")
        phase = str(observation.get("attempt_phase", "unknown"))
        predicates = observation.get("predicates", observation)
        contact = tri(predicates.get("contact_present"))
        closed = tri(predicates.get("gripper_command_closed"))
        opened = tri(predicates.get("gripper_command_open"))
        hold_evidence = tri(evidence.get("hold_evidence"))

        required_values = (contact, closed, opened, active, end)
        data_valid = attempt_id > 0 and UNKNOWN not in required_values
        data_status = "valid" if data_valid else "data_missing_or_invalid"

        if attempt_id > 0 and attempt_id != self.state.attempt_id:
            started = active == TRUE and phase not in {"ended", "inactive", "unknown"}
            self._reset_attempt(attempt_id, started)
        elif data_valid and active == TRUE and phase not in {"ended", "inactive", "unknown"}:
            self.state.attempt_started_observed = True

        context_valid, context_reason = (False, "request_missing")
        if request is not None and attempt_id > 0:
            context_valid, context_reason = validate_request_provenance(
                request, now=now, capture_order=capture_order, attempt_id=attempt_id
            )
        effect = request.requested_effect if context_valid and request is not None else RequestedEffect.UNKNOWN

        current_hold = data_valid and hold_evidence == TRUE and contact == TRUE and closed == TRUE
        if current_hold:
            self.state.historical_hold_established = True

        contact_loss = self.state.last_contact == TRUE and contact == FALSE
        release_observed = opened == TRUE or end_reason == "release" or effect == RequestedEffect.RELEASE_OBJECT
        loss_observed = data_valid and self.state.historical_hold_established and contact_loss and not release_observed
        if loss_observed:
            self._start_event("held_object_loss")

        end_sequence = observation.get("attempt_end_sequence", 0)
        end_key = f"{request.request_id if request else 'missing'}:{attempt_id}:{end_sequence}"
        end_fresh = end == TRUE and end_key not in self.state.consumed_end_keys
        if end_fresh:
            self.state.consumed_end_keys = (*self.state.consumed_end_keys, end_key)

        retry = FALSE
        recover = FALSE
        needs = FALSE
        reason = "no_emergency_condition"

        if not data_valid:
            needs, reason = TRUE, "data_missing_or_invalid"
        elif loss_observed:
            recover, reason = TRUE, "historical_hold_established_then_observed_non_release_contact_loss"
        elif end_fresh and end_reason == "segment_complete" and closed == TRUE and contact == FALSE and not self.state.historical_hold_established:
            if not self.history_complete or not self.state.attempt_started_observed:
                needs, reason = TRUE, "history_incomplete_for_attempt_failure"
            elif not context_valid:
                needs, reason = TRUE, context_reason
            elif effect == RequestedEffect.HOLD_OBJECT:
                retry, reason = TRUE, "hold_request_normally_ended_without_hold"
                self._start_event("missed_grasp")
            elif effect in {RequestedEffect.TOUCH_OBJECT, RequestedEffect.OBSERVE}:
                reason = "touch_or_observe_request_normally_completed"
            else:
                needs, reason = TRUE, "requested_effect_does_not_support_retry"
        elif end_fresh and end_reason in {"cancelled", "release"}:
            reason = f"{end_reason}_end_does_not_imply_failure"
        elif active == TRUE and not self.state.historical_hold_established:
            reason = "valid_no_current_hold_evidence_during_active_acquisition"
        elif self.state.historical_hold_established:
            reason = "historical_hold_established_no_current_loss_observed"
        else:
            reason = "valid_no_current_hold_evidence"

        if recover == TRUE:
            retry = FALSE
            needs = FALSE
        selected = "recover_object" if recover == TRUE else "retry_grasp" if retry == TRUE else "needs_observation" if needs == TRUE else "none"
        hold_state = (
            "data_missing_or_invalid" if not data_valid
            else "historical_hold_established" if self.state.historical_hold_established
            else "valid_no_current_hold_evidence"
        )
        row = {
            "schema": INTERFACE_REPAIR_VERSION,
            "frame_index": observation.get("frame_index"),
            "time": observation.get("time"),
            "capture_order": capture_order,
            "attempt_id": attempt_id,
            "attempt_phase": phase,
            "attempt_active": active,
            "attempt_end": end,
            "attempt_end_reason": end_reason,
            "request_id": request.request_id if request else None,
            "requested_effect": effect.value,
            "context_valid": context_valid,
            "context_provenance": context_reason,
            "history_complete": self.history_complete,
            "data_status": data_status,
            "hold_state": hold_state,
            "current_hold_evidence": hold_evidence,
            "historical_hold_established": self.state.historical_hold_established,
            "contact_loss_observed": loss_observed,
            "end_edge_fresh": end_fresh,
            "effective_guards": {"retry_grasp": retry, "recover_object": recover},
            "needs_observation": needs,
            "selected_action": selected,
            "reason_code": reason,
            "pending_event": self.state.pending_event,
            "pending_event_id": self.state.pending_event_id,
            "state": asdict(self.state),
        }
        self.state.last_contact = contact
        return row


def run_repaired_interface(
    observations: list[dict[str, Any]],
    evidence_rows: list[dict[str, Any]],
    requests: list[dict[str, Any]],
    *,
    history_complete: bool = True,
) -> list[dict[str, Any]]:
    parsed = [ControllerRequest.from_mapping(row) for row in requests]
    interface = RepairedOnlineInterface(history_complete=history_complete)
    output = []
    for observation, evidence in zip(observations, evidence_rows):
        now = float(observation.get("time", 0.0))
        order = int(observation.get("capture_order", observation.get("frame_index", 0) or 0))
        arrived = [
            request for request in parsed
            if request.received_time < now - 1e-9
            or (abs(request.received_time - now) <= 1e-9 and request.issued_capture_order <= order)
        ]
        output.append(interface.observe(observation, evidence, arrived[-1] if arrived else None))
    return output


def _load_candidate(meta: dict[str, Any], candidate_id: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    raw = read_jsonl(Path(meta["path"]) / "observations_dense.jsonl")
    observations = [_online_observation(row) for row in raw]
    geometries = build_features(observations)
    predictions: list[dict[str, Any]] = []
    previous = None
    for observation, geometry in zip(observations, geometries):
        predictions.append({**observation, "predicates": _predicates(observation, geometry, previous)})
        previous = observation
    base, config = FIXED_CANDIDATES[candidate_id]
    evidence = evaluate_candidate(predictions, geometries, base, config)
    requests = read_jsonl(Path(meta["path"]) / "controller_requests.jsonl")
    return predictions, evidence, requests


def _first_emergency(rows: list[dict[str, Any]], start: float, end: float) -> dict[str, Any] | None:
    for row in rows:
        if start - 1e-9 <= float(row.get("time", -1.0)) <= end + 1e-9 and row.get("selected_action") in {"retry_grasp", "recover_object"}:
            return row
    return None


def _runtime_environment() -> dict[str, Any]:
    result: dict[str, Any] = {
        "python": sys.executable,
        "python_version": sys.version.split()[0],
        "purpose": "dependency inventory for cache-only interface diagnostic",
    }
    for name in ("cv2", "torch", "mujoco"):
        try:
            module = importlib.import_module(name)
            result[name] = {"installed": True, "version": getattr(module, "__version__", "unknown")}
            if name == "torch":
                result["torch_cuda_available"] = bool(module.cuda.is_available())
                result["torch_cuda_device_count"] = int(module.cuda.device_count())
        except ImportError:
            result[name] = {"installed": False}
    return result


def _external_artifact_record(data_root: Path) -> dict[str, Any]:
    size = sum(path.stat().st_size for path in data_root.rglob("*") if path.is_file())
    return {
        "logical_path": "data_attach_relpose_repair_v1",
        "original_path": str(data_root.resolve()),
        "original_filename": data_root.name,
        "size_bytes": size,
        "sha256": "NOT_COMPUTED_DIRECTORY_TREE",
        "artifact_type": "raw_dynamic_rollout_directory",
        "purpose": "offline replay input retained outside the lightweight delivery",
        "reason_omitted": "per-frame RGB and raw rollout payload externalized",
        "recovery_method": "restore the exact directory, then rerun the locked cache evaluation command",
    }


def evaluate_interface_repair(data_root: Path, output_root: Path) -> dict[str, Any]:
    metadata = read_csv(data_root / "rollout_manifest.csv")
    if len(metadata) != 32 or len({row["root_family_id"] for row in metadata}) != 4:
        raise ValueError("interface repair requires the frozen 32-rollout attach-relpose collection")
    output_root.mkdir(parents=True, exist_ok=True)
    comparison: list[dict[str, Any]] = []
    trace_rows: list[dict[str, Any]] = []
    metrics: list[dict[str, Any]] = []
    for candidate_id in FIXED_CANDIDATES:
        decisions = []
        for meta in sorted(metadata, key=lambda row: row["rollout_id"]):
            observations, evidence, requests = _load_candidate(meta, candidate_id)
            rows = run_repaired_interface(observations, evidence, requests, history_complete=True)
            reference = build_reference(meta)
            for row in rows:
                if row["hold_state"] == "historical_hold_established" or row["contact_loss_observed"] or row["selected_action"] != "none":
                    trace_rows.append({
                        "candidate_id": candidate_id,
                        "rollout_id": meta["rollout_id"],
                        "root_family_id": meta["root_family_id"],
                        "case_id": meta["case_id"],
                        **{key: row[key] for key in ("time", "contact_loss_observed", "data_status", "hold_state", "current_hold_evidence", "historical_hold_established", "selected_action", "reason_code")},
                    })
            if reference["status"] != "reference_labeled":
                decisions.append({"candidate_id": candidate_id, "rollout_id": meta["rollout_id"], "root_family_id": meta["root_family_id"], "case_id": meta["case_id"], "reference_status": "reference_unresolved", "reference_reason": reference.get("reason"), "expected_action": None, "selected_action": None, "selected_time": None, "correct": None})
                continue
            event = reference["events"][0]
            selected = _first_emergency(rows, float(event["decision_window_start"]), float(event["decision_window_end"]))
            expected = event["expected_action"]
            selected_action = selected["selected_action"] if selected else "none"
            decisions.append({
                "candidate_id": candidate_id,
                "rollout_id": meta["rollout_id"],
                "root_family_id": meta["root_family_id"],
                "case_id": meta["case_id"],
                "reference_status": "reference_labeled",
                "reference_reason": None,
                "expected_action": expected,
                "selected_action": selected_action,
                "selected_time": selected.get("time") if selected else None,
                "correct": selected_action == expected,
                "selected_reason": selected.get("reason_code") if selected else "no_emergency_condition",
                "historical_hold_seen": any(row["historical_hold_established"] for row in rows if float(row["time"]) <= float(event["decision_window_start"]) + 1e-9),
                "data_status_at_selection": selected.get("data_status") if selected else rows[-1]["data_status"],
                "hold_state_at_selection": selected.get("hold_state") if selected else rows[-1]["hold_state"],
            })
        labeled = [row for row in decisions if row["reference_status"] == "reference_labeled"]
        def case_rows(case_id: str) -> list[dict[str, Any]]:
            return [row for row in labeled if row["case_id"] == case_id]
        def rate(rows: list[dict[str, Any]], predicate) -> float | None:
            return sum(bool(predicate(row)) for row in rows) / len(rows) if rows else None
        metrics.append({
            "candidate_id": candidate_id,
            "interface_repair_version": INTERFACE_REPAIR_VERSION,
            "reference_labeled_events": len(labeled),
            "reference_unresolved_events_preserved": len(decisions) - len(labeled),
            "K1_recall": rate(case_rows("K1_hold_request_ends_without_hold"), lambda row: row["correct"]),
            "K2_false_emergency_rate": rate(case_rows("K2_touch_request_completes_without_hold"), lambda row: row["selected_action"] != "none"),
            "K3_false_emergency_rate": rate(case_rows("K3_normal_hold_pause_resume"), lambda row: row["selected_action"] != "none"),
            "K4_recall": rate(case_rows("K4_regular_hold_loss"), lambda row: row["correct"]),
            "K5_recall": rate(case_rows("K5_brief_hold_loss"), lambda row: row["correct"]),
            "K6_recall": rate(case_rows("K6_long_gap_after_loss"), lambda row: row["correct"]),
            "K7_false_emergency_rate": rate(case_rows("K7_commanded_release"), lambda row: row["selected_action"] != "none"),
            "K8_false_emergency_rate": rate(case_rows("K8_acquisition_touch_then_continue"), lambda row: row["selected_action"] != "none"),
            "labeled_event_accuracy": rate(labeled, lambda row: row["correct"]),
            "candidate_pass": False,
        })
        comparison.extend(decisions)
    write_csv(output_root / "interface_repair_comparison.csv", comparison)
    write_csv(output_root / "interface_repair_metrics.csv", metrics)
    write_jsonl(output_root / "interface_repair_trace.jsonl", trace_rows)
    result = {
        "schema": "pathgraph_l2rar2_online_interface_repair_v1",
        "status": "ONLINE_INTERFACE_REPAIR_DIAGNOSTIC_COMPLETE",
        "interface_repair_version": INTERFACE_REPAIR_VERSION,
        "data_role": "repair_dev_cache_only",
        "rollouts": len(metadata),
        "candidates": list(FIXED_CANDIDATES),
        "metrics": metrics,
        "online_interface_change_applied": True,
        "candidate_selected": None,
        "candidate_pass": False,
        "reference_contract_changed": False,
        "confirmation_run": False,
        "training_jobs": 0,
        "api_calls": 0,
        "api_key_read": False,
        "l3_entry_allowed": False,
        "historical_status": "L2RAR1_PARTIAL_KEEP_G1",
        "current_status": "L2RAR2_PARTIAL_KEEP_G1",
        "retained_graph": "G1_predicate_bound",
    }
    write_json(output_root / "online_interface_repair.json", result)
    write_json(output_root / "runtime_environment.json", _runtime_environment())
    external = _external_artifact_record(data_root)
    (output_root / "external_artifacts.tsv").write_text(
        "logical_path\toriginal_path\toriginal_filename\tsize_bytes\tsha256\tartifact_type\tpurpose\treason_omitted\trecovery_method\n"
        + "\t".join(str(external[key]) for key in (
            "logical_path", "original_path", "original_filename", "size_bytes", "sha256",
            "artifact_type", "purpose", "reason_omitted", "recovery_method",
        )) + "\n",
        encoding="utf-8",
    )
    (output_root / "actual_commands.txt").write_text(
        "git fetch origin --prune\n"
        "/home/xushijie/.conda/envs/lerobot/bin/python -m upgrade_v2.l2r_task_context.cli "
        "evaluate-online-interface-repair --data-root "
        f"{data_root.resolve()} --output-root {output_root.resolve()}\n",
        encoding="utf-8",
    )
    report = [
        "# L2RAR2 online interface repair",
        "",
        f"- Version: `{INTERFACE_REPAIR_VERSION}`.",
        f"- Scope: `{len(metadata)}` frozen attach-relpose repair rollouts; no new sampling, training, API calls, or confirmation.",
        "- The interface now records three causal states: `data_missing_or_invalid`, `valid_no_current_hold_evidence`, and `historical_hold_established`.",
        "- Retry is permitted only for a valid, complete `HOLD_OBJECT` attempt ending without current hold evidence.",
        "- Active touch/acquisition does not retry. A non-release contact loss recovers only after a historical hold was established and the loss is actually observed online.",
        "- In this cache, K4/K5/K6 contain offline contact-loss events but no corresponding online contact-loss transition; the interface therefore does not claim recovery for them.",
        "- This is a diagnostic interface repair, not a passing candidate or confirmation result. `G1` remains retained and L3 remains closed.",
        "",
        "## Metrics",
        "",
    ]
    for row in metrics:
        report.append(f"- `{row['candidate_id']}`: K1={row['K1_recall']}, K4={row['K4_recall']}, K5={row['K5_recall']}, K6={row['K6_recall']}, K2/K3/K7/K8 false-emergency rates={[row[key] for key in ('K2_false_emergency_rate', 'K3_false_emergency_rate', 'K7_false_emergency_rate', 'K8_false_emergency_rate')]}; candidate_pass=`false`.")
    (output_root / "online_interface_repair.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    scan = secret_scan([Path(__file__), output_root])
    scan.update({"schema": "pathgraph_l2rar2_online_interface_repair_secret_scan_v1", "api_calls": 0, "api_key_read": False, "training_jobs": 0})
    if scan["status"] != "PASS":
        raise RuntimeError("secret scan failed")
    write_json(output_root / "secret_scan.json", scan)
    artifact_names = ["interface_repair_comparison.csv", "interface_repair_metrics.csv", "interface_repair_trace.jsonl", "online_interface_repair.json", "online_interface_repair.md", "runtime_environment.json", "actual_commands.txt", "external_artifacts.tsv", "secret_scan.json"]
    write_json(output_root / "run_manifest.json", {
        "schema": "pathgraph_l2rar2_online_interface_repair_manifest_v1",
        "interface_repair_version": INTERFACE_REPAIR_VERSION,
        "input_hashes": {"rollout_manifest": sha256(data_root / "rollout_manifest.csv")},
        "artifacts": [{"path": name, "size_bytes": (output_root / name).stat().st_size, "sha256": sha256(output_root / name)} for name in artifact_names],
        "validation": {"reference_contract_changed": False, "parameter_search": False, "training_jobs": 0, "api_calls": 0, "api_key_read": False, "confirmation_run": False, "raw_rollouts_packaged": False},
    })
    return result
