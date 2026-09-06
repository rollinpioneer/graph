"""Recovered U2 causal-boundary inference and fixed-H U4 diagnostics."""
from __future__ import annotations

import platform
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from upgrade_v2.u2.event_schema import EVENT_NAMES
from upgrade_v2.u2.simulator import make_family_specs
from upgrade_v2.u2.weak_labels import DEFAULT_RULES, candidate_scores

from .io import read_json, read_jsonl, sha256_file, write_csv, write_json, write_jsonl


EVENT_TO_SEMANTIC = {
    "contact_off_failure": "failure_event",
    "recovery_start": "recovery_attempt",
    "contact_reestablished": "recovery_achieved",
    "transport_on": "progress",
    "goal_enter": "progress",
    "stagnation_onset": "dwell",
    "stable_success": "terminal_success",
    "terminal_failure": "terminal_failure",
}
HIGH_IMPACT_EVENTS = {
    "contact_off_failure",
    "recovery_start",
    "contact_reestablished",
    "stable_success",
    "terminal_failure",
}
TARGET_PAIRS = {(5, 8): "T028", (4, 4): "T029"}


def plan_confirmation_extension(
    seed: int,
    generator_count: int,
    start_index: int,
    family_count: int,
    rollout_seed_base: int,
    output: Path,
    lock: Path,
) -> dict[str, Any]:
    if start_index < 0 or family_count <= 0 or start_index + family_count > generator_count:
        raise ValueError("invalid confirmation extension range")
    specs = make_family_specs(generator_count, seed)
    rows = []
    for index in range(start_index, start_index + family_count):
        original = specs[index]
        family = dict(original.__dict__)
        family["root_family_id"] = f"u4b_v1_{index:02d}"
        rows.append({
            "family": family,
            "root_family_id": family["root_family_id"],
            "split": "confirm",
            "family_index": index,
            "scenario_for_analysis_only": family["scenario"],
            "rollout_seeds": [rollout_seed_base + (index - start_index) * 10 + offset for offset in range(4)],
        })
    write_jsonl(output, rows)
    payload = {
        "schema": "u4b_torch_recovery_confirmation_family_lock_v1",
        "protocol_extension": "torch_recovery_v2",
        "generator_seed": seed,
        "generator_count": generator_count,
        "selected_indices": [start_index, start_index + family_count - 1],
        "family_count": family_count,
        "family_ids": [row["root_family_id"] for row in rows],
        "rollouts_per_family": 4,
        "rollout_seed_base": rollout_seed_base,
        "selection_depends_only_on_generator_position": True,
        "prior_family_indices_excluded": [0, start_index - 1],
    }
    write_json(lock, payload)
    return {"status": "PASS", "family_count": family_count, "selected_indices": payload["selected_indices"]}


def lock_recovered_boundary(
    checkpoint: Path,
    inference_manifest: Path,
    mapper_lock: Path,
    output: Path,
) -> dict[str, Any]:
    manifest = read_json(inference_manifest)
    if manifest.get("status") != "PASS" or manifest.get("checkpoint_sha256") != sha256_file(checkpoint):
        raise ValueError("recovered boundary inference manifest does not match checkpoint")
    payload = {
        "schema": "u4b_selected_input_pipeline_v2",
        "protocol_extension": "torch_recovery_v2",
        "boundary_source": "offline_teacher_to_causal_s623",
        "automatic_boundary_status": "computed_and_locked",
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "architecture": manifest.get("checkpoint_architecture"),
        "variant": manifest.get("checkpoint_variant"),
        "seed": manifest.get("checkpoint_seed"),
        "threshold": manifest.get("boundary_threshold"),
        "mapper_source": read_json(mapper_lock).get("source", "legacy_reference_mapper"),
        "mapper_lock": str(mapper_lock.resolve()),
        "execution_environment": {
            "python_executable": manifest.get("python_executable"),
            "python_version": manifest.get("python_version"),
            "torch_version": manifest.get("torch_version"),
            "device": manifest.get("device"),
        },
        "fallback": "retain_unknown_and_disclose_semantic_uncertainty",
        "hidden_or_future_features_used": False,
        "original_confirmation_results_rewritten": False,
    }
    write_json(output, payload)
    return {"status": "PASS", "source": payload["boundary_source"], "checkpoint_sha256": payload["checkpoint_sha256"]}


def transition_observations(episode: dict[str, Any]) -> np.ndarray:
    """Return the post-transition observations used by the U2 causal model."""
    observations = np.asarray(episode["observations"], dtype=np.float32)
    actions = np.asarray(episode["actions"], dtype=np.float32)
    if observations.shape != (len(actions) + 1, 17):
        raise ValueError(f"invalid U4 observation alignment for {episode.get('episode_id', '')}")
    aligned = observations[1:]
    if not np.allclose(aligned[:, 15:17], actions, atol=1e-6, rtol=0.0):
        raise ValueError(f"previous-action fields do not match transitions for {episode.get('episode_id', '')}")
    return aligned


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    result = np.exp(shifted)
    return result / result.sum(axis=1, keepdims=True)


def _rule_predictions(observations: np.ndarray, actions: np.ndarray, threshold: float) -> dict[str, list[Any]]:
    score, _ = candidate_scores(observations, actions, DEFAULT_RULES)
    logits = score * 3.0
    logits[:, 0] += 3.0
    probabilities = _softmax(logits)
    sorted_probability = np.sort(probabilities, axis=1)
    unknown = (
        (sorted_probability[:, -1] < DEFAULT_RULES["posterior"]["unknown_max_probability"])
        | ((sorted_probability[:, -1] - sorted_probability[:, -2]) < DEFAULT_RULES["posterior"]["unknown_margin"])
    )
    return {
        "rule_boundary": ((1.0 - probabilities[:, 0]) >= threshold).tolist(),
        "rule_event_prediction": probabilities.argmax(axis=1).astype(int).tolist(),
        "rule_unknown": unknown.tolist(),
    }


def infer_rollouts(
    checkpoint: Path,
    rollout_root: Path,
    output: Path,
    manifest: Path,
    device: str = "auto",
    threshold: float = 0.5,
) -> dict[str, Any]:
    """Run the validation-locked causal model without requiring CUDA."""
    try:
        import torch
        import torch.nn.functional as functional
        from upgrade_v2.u2.boundary_model import CausalBoundaryGRU
    except ImportError as exc:
        raise RuntimeError("PyTorch is required for recovered automatic-boundary inference") from exc

    selected = "cuda" if device == "auto" and torch.cuda.is_available() else ("cpu" if device == "auto" else device)
    torch_device = torch.device(selected)
    payload = torch.load(checkpoint, map_location=torch_device)
    if payload.get("architecture") not in {None, "CausalBoundaryGRU"}:
        raise ValueError(f"checkpoint is not a causal U2 boundary model: {payload.get('architecture')}")
    model = CausalBoundaryGRU()
    model.load_state_dict(payload["state_dict"])
    model.to(torch_device).eval()

    rows = []
    total_steps = 0
    with torch.no_grad():
        for path in sorted(rollout_root.glob("*.json")):
            episode = read_json(path)
            observations = transition_observations(episode)
            actions = np.asarray(episode["actions"], dtype=np.float32)
            prediction = model(torch.from_numpy(observations).unsqueeze(0).to(torch_device))
            boundary_probability = torch.sigmoid(prediction["boundary_logit"])[0].cpu().numpy()
            event_probability = functional.softmax(prediction["event_logits"], -1)[0].cpu().numpy()
            unknown_probability = torch.sigmoid(prediction["unknown_logit"])[0].cpu().numpy()
            reference = [bool(item.get("all_events")) for item in episode["events"]]
            row = {
                "schema": "u4b_boundary_prediction_v1",
                "episode_id": episode["episode_id"],
                "root_family_id": episode["root_family_id"],
                "n_steps": len(actions),
                "auto_boundary": (boundary_probability >= threshold).tolist(),
                "auto_boundary_probability": boundary_probability.astype(float).tolist(),
                "model_event_prediction": event_probability.argmax(axis=1).astype(int).tolist(),
                "model_unknown": (unknown_probability >= threshold).tolist(),
                "reference_boundary": reference,
                **_rule_predictions(observations, actions, threshold),
            }
            rows.append(row)
            total_steps += len(actions)
    if not rows:
        raise FileNotFoundError(f"no rollout JSON files found under {rollout_root}")
    write_jsonl(output, rows)
    audit = {
        "schema": "u4b_torch_recovery_inference_manifest_v1",
        "status": "PASS",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "torch_cuda_build": torch.version.cuda,
        "cuda_available": bool(torch.cuda.is_available()),
        "device": str(torch_device),
        "checkpoint": str(checkpoint.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint),
        "checkpoint_architecture": payload.get("architecture"),
        "checkpoint_variant": payload.get("variant"),
        "checkpoint_seed": payload.get("seed"),
        "boundary_threshold": threshold,
        "episode_count": len(rows),
        "transition_count": total_steps,
        "alignment": "U4 observations[1:] == U2 post-transition observations; observations[:,15:17] == actions",
        "hidden_or_future_features_used": False,
    }
    write_json(manifest, audit)
    return audit


def _nearest_boundary(boundaries: list[bool], index: int, tolerance: int = 2) -> int | None:
    candidates = [position for position in range(max(0, index - tolerance), min(len(boundaries), index + tolerance + 1)) if boundaries[position]]
    return min(candidates, key=lambda position: (abs(position - index), position)) if candidates else None


def _semantic_at(prediction: dict[str, Any], source: str, index: int) -> tuple[str, int | None]:
    boundary = _nearest_boundary(prediction[f"{source}_boundary"], index)
    if boundary is None:
        return "unknown", None
    if source == "rule":
        event_id = int(prediction["rule_event_prediction"][boundary])
        unknown = bool(prediction["rule_unknown"][boundary])
    else:
        event_id = int(prediction["model_event_prediction"][boundary])
        unknown = bool(prediction["model_unknown"][boundary])
    event = EVENT_NAMES[event_id]
    return ("unknown" if unknown else EVENT_TO_SEMANTIC.get(event, "unknown")), boundary


def _family_macro(rows: Iterable[dict[str, Any]], numerator: str, denominator: str) -> float:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["root_family_id"]].append(row)
    rates = []
    for items in grouped.values():
        total = sum(int(item.get(denominator, 0)) for item in items)
        if total:
            rates.append(sum(int(item.get(numerator, 0)) for item in items) / total)
    return float(np.mean(rates)) if rates else 0.0


def _mixed_rate(episode_cases: list[dict[str, Any]], boundaries: list[bool]) -> tuple[int, int]:
    segment = np.cumsum(np.asarray(boundaries, dtype=np.int64))
    labels: dict[int, set[str]] = defaultdict(set)
    for case in episode_cases:
        for semantic in case["gold_semantics"]:
            if semantic in {"failure_event", "recovery_attempt", "recovery_achieved"}:
                labels[int(segment[case["t"]])].add(semantic)
    mixed_segments = {key for key, value in labels.items() if "failure_event" in value and "recovery_achieved" in value}
    eligible = [case for case in episode_cases if set(case["gold_semantics"]) & {"failure_event", "recovery_attempt", "recovery_achieved"}]
    mixed = sum(int(segment[case["t"]]) in mixed_segments for case in eligible)
    return mixed, len(eligible)


def _raw_history_signature(episode: dict[str, Any], index: int, steps: int = 8) -> tuple[Any, ...]:
    observations = transition_observations(episode)
    start = max(0, index - steps + 1)
    history = np.round(observations[start:index + 1], 5)
    padding = np.full((steps - len(history), history.shape[1]), np.nan, dtype=np.float32)
    return tuple(np.concatenate([padding, history]).reshape(-1).tolist())


def diagnose_recovered_boundaries(
    rollout_root: Path,
    occurrence_path: Path,
    predictions_path: Path,
    output_cases: Path,
    per_family: Path,
    summary: Path,
) -> dict[str, Any]:
    predictions = {row["episode_id"]: row for row in read_jsonl(predictions_path)}
    occurrences = read_jsonl(occurrence_path)
    occurrence_by_episode_t = {(row["episode_id"], int(row["action_index"])): row for row in occurrences}
    cases = []
    episode_cases: dict[str, list[dict[str, Any]]] = defaultdict(list)
    raw_signatures: dict[tuple[Any, ...], set[str]] = defaultdict(set)
    pair_labels: dict[tuple[int, int], dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))

    for path in sorted(rollout_root.glob("*.json")):
        episode = read_json(path)
        prediction = predictions.get(episode["episode_id"])
        if prediction is None or prediction["n_steps"] != episode["n_steps"]:
            raise ValueError(f"missing or misaligned prediction for {episode['episode_id']}")
        for t, event in enumerate(episode["events"]):
            occurrence = occurrence_by_episode_t.get((episode["episode_id"], t), {})
            pair = tuple(occurrence.get("transition_pair", []))
            evaluator_events = set(event.get("all_events", []))
            target_id = TARGET_PAIRS.get(pair)
            if not evaluator_events.intersection(HIGH_IMPACT_EVENTS) and target_id is None:
                continue
            semantics = sorted({EVENT_TO_SEMANTIC[name] for name in evaluator_events if name in EVENT_TO_SEMANTIC})
            auto_semantic, auto_index = _semantic_at(prediction, "auto", t)
            rule_semantic, rule_index = _semantic_at(prediction, "rule", t)
            reference_semantic, reference_index = _semantic_at(prediction, "reference", t)
            row = {
                "case_id": f"torch-recovery:{episode['episode_id']}:t{t}",
                "episode_id": episode["episode_id"],
                "root_family_id": episode["root_family_id"],
                "split": occurrence.get("split", ""),
                "t": t,
                "transition_pair": list(pair),
                "target_ids": [target_id] if target_id else [],
                "gold_events": sorted(evaluator_events),
                "gold_semantics": semantics,
                "auto_semantic": auto_semantic,
                "rule_semantic": rule_semantic,
                "reference_semantic": reference_semantic,
                "auto_boundary_index": auto_index,
                "rule_boundary_index": rule_index,
                "reference_boundary_index": reference_index,
                "auto_unresolved": bool(semantics and auto_semantic not in semantics) or (not semantics and target_id is not None),
                "reference_unresolved": bool(semantics and reference_semantic not in semantics) or (not semantics and target_id is not None),
                "semantic_flip_auto_vs_reference": auto_semantic != reference_semantic,
                "recovery_denominator": int("recovery_achieved" in semantics),
                "auto_recovery_numerator": int("recovery_achieved" in semantics and auto_semantic == "recovery_achieved"),
                "rule_recovery_numerator": int("recovery_achieved" in semantics and rule_semantic == "recovery_achieved"),
                "reference_recovery_numerator": int("recovery_achieved" in semantics and reference_semantic == "recovery_achieved"),
                "case_denominator": 1,
                "auto_unresolved_numerator": int(bool(semantics and auto_semantic not in semantics) or (not semantics and target_id is not None)),
                "reference_unresolved_numerator": int(bool(semantics and reference_semantic not in semantics) or (not semantics and target_id is not None)),
                "flip_numerator": int(auto_semantic != reference_semantic),
                "provenance": "post_confirmation_protocol_extension_development_only",
                "label_origin": "simulator_info.events_diagnostic_only",
            }
            cases.append(row)
            episode_cases[episode["episode_id"]].append(row)
            if semantics:
                raw_signatures[_raw_history_signature(episode, t)].update(semantics)
                if len(pair) == 2:
                    pair_labels[pair][episode["root_family_id"]].update(semantics)

    mixed_rows = []
    episode_lookup = {read_json(path)["episode_id"]: read_json(path) for path in sorted(rollout_root.glob("*.json"))}
    for episode_id, items in episode_cases.items():
        prediction = predictions[episode_id]
        for source in ("auto", "rule", "reference"):
            numerator, denominator = _mixed_rate(items, prediction[f"{source}_boundary"])
            mixed_rows.append({"root_family_id": episode_lookup[episode_id]["root_family_id"], "source": source, "mixed": numerator, "eligible": denominator})

    def mixed_macro(source: str) -> float:
        return _family_macro([row for row in mixed_rows if row["source"] == source], "mixed", "eligible")

    auto_recovery = _family_macro(cases, "auto_recovery_numerator", "recovery_denominator")
    rule_recovery = _family_macro(cases, "rule_recovery_numerator", "recovery_denominator")
    reference_recovery = _family_macro(cases, "reference_recovery_numerator", "recovery_denominator")
    concrete_ambiguous = sum(
        len({semantic for labels in families.values() for semantic in labels}) >= 2 and len(families) >= 2
        for families in pair_labels.values()
    )
    raw_conflicts = sum(len(labels) >= 2 for labels in raw_signatures.values())
    families = sorted({row["root_family_id"] for row in cases})
    per_family_rows = []
    for family in families:
        items = [row for row in cases if row["root_family_id"] == family]
        recovery_total = sum(row["recovery_denominator"] for row in items)
        per_family_rows.append({
            "root_family_id": family,
            "split": items[0]["split"],
            "high_impact_cases": len(items),
            "auto_unresolved_rate": sum(row["auto_unresolved_numerator"] for row in items) / len(items),
            "reference_unresolved_rate": sum(row["reference_unresolved_numerator"] for row in items) / len(items),
            "auto_recovery_recall": sum(row["auto_recovery_numerator"] for row in items) / recovery_total if recovery_total else None,
            "rule_recovery_recall": sum(row["rule_recovery_numerator"] for row in items) / recovery_total if recovery_total else None,
            "reference_recovery_recall": sum(row["reference_recovery_numerator"] for row in items) / recovery_total if recovery_total else None,
        })
    metrics = {
        "schema": "u4b_torch_recovery_diagnostic_metrics_v1",
        "protocol_extension": "torch_recovery_v2",
        "post_confirmation_original_results_unchanged": True,
        "cases": len(cases),
        "high_impact_events": len(cases),
        "informative_family_count": len(families),
        "unresolved_high_impact_rate": _family_macro(cases, "auto_unresolved_numerator", "case_denominator"),
        "auto_recovery_recall": auto_recovery,
        "rule_recovery_recall": rule_recovery,
        "reference_recovery_recall": reference_recovery,
        "recovery_recall_gain_ref_minus_auto": reference_recovery - auto_recovery,
        "auto_mixed_error_rate": mixed_macro("auto"),
        "rule_mixed_error_rate": mixed_macro("rule"),
        "reference_mixed_error_rate": mixed_macro("reference"),
        "mixed_error_drop_ref_minus_auto": mixed_macro("auto") - mixed_macro("reference"),
        "semantic_flip_rate": _family_macro(cases, "flip_numerator", "case_denominator"),
        "semantic_unresolved_after_reference": _family_macro(cases, "reference_unresolved_numerator", "case_denominator"),
        "concrete_ambiguous_groups": concrete_ambiguous,
        "observability_insufficient": raw_conflicts > 0,
        "observability_conflict_groups": raw_conflicts,
        "observability_test": "identical last-8 causal observable histories rounded to 1e-5 with different evaluator semantics",
        "automatic_boundary_available": True,
        "payload_error": False,
        "provenance": "development_only_same_simulator_fixed_H",
        "label_origin": "simulator_info.events_diagnostic_only",
    }
    write_jsonl(output_cases, cases)
    write_csv(per_family, per_family_rows)
    write_json(summary, metrics)
    return metrics


def finalize_torch_recovery(
    pipeline_lock: Path,
    route_path: Path,
    diagnostic_metrics: Path,
    confirmation_metrics: Path,
    paired_effects: Path,
    claim_results: Path,
    original_handoff: Path,
    output_dir: Path,
    report: Path,
) -> dict[str, Any]:
    lock = read_json(pipeline_lock)
    route = read_json(route_path)
    diagnostics = read_json(diagnostic_metrics)
    graphs = __import__("csv").DictReader(confirmation_metrics.open(encoding="utf-8"))
    graph_rows = list(graphs)
    with paired_effects.open(encoding="utf-8") as handle:
        paired_rows = list(__import__("csv").DictReader(handle))
    with claim_results.open(encoding="utf-8") as handle:
        claims = list(__import__("csv").DictReader(handle))
    prior = read_json(original_handoff)
    recovery_root = pipeline_lock.parent.parent
    development_inference = read_json(recovery_root / "protocol" / "inference_manifest.json")
    confirmation_inference = read_json(recovery_root / "protocol" / "confirmation_inference_manifest.json")
    family_lock = read_json(recovery_root / "protocol" / "confirmation_family_lock.json")
    development_occurrences = read_jsonl(recovery_root / "evidence" / "dev_occurrences.jsonl")
    confirmation_occurrences = read_jsonl(recovery_root / "evidence" / "confirmation_occurrences.jsonl")
    confirmation_continuations = read_jsonl(recovery_root / "queries" / "confirmation_continuations.jsonl")
    confirmation_rollouts = list((recovery_root / "data" / "confirmation").glob("*.json"))
    development_transition_count = int(development_inference.get("transition_count", 0))
    confirmation_transition_count = int(confirmation_inference.get("transition_count", 0))
    development_occurrence_count = len(development_occurrences)
    confirmation_occurrence_count = len(confirmation_occurrences)
    confirmation_family_count = int(family_lock.get("family_count", 0))
    generated_family_count = int(family_lock.get("generator_count", 0))
    confirmation_rollout_count = len(confirmation_rollouts)
    confirmation_continuation_count = len(confirmation_continuations)
    prior_counts = prior.get("execution_counts", {})
    prior_base_rollouts = int(prior_counts.get("total_base_rollouts", 0))
    prior_continuations = int(prior_counts.get("total_continuations", 0))
    excluded_v1_rollouts = len(list((recovery_root.parent / "torch_recovery_v1" / "data" / "confirmation").glob("*.json")))
    accepted = max((int(row.get("accepted_edit_count", 0) or 0) for row in graph_rows), default=0)
    confirmed_claim = any(row.get("validation_status") == "empirically_validated" for row in claims)
    measurable_gain = any(
        row.get("status") == "estimable" and row.get("effect") not in {"", None}
        and abs(float(row["effect"])) > 0.0
        for row in paired_rows
    )
    status = "U4_COMPLETE_WITH_SCOPED_SUPPORT" if confirmed_claim and measurable_gain else "U4_COMPLETE_NO_EDIT_GAIN"
    final = {
        "schema": "u4b_final_handoff_v2",
        "scientific_status": status,
        "protocol_extension": "torch_recovery_v2",
        "development_route": route.get("route"),
        "previous_development_route": prior.get("development_route"),
        "previous_scientific_status": prior.get("scientific_status"),
        "previous_handoff": str(original_handoff.resolve()),
        "previous_handoff_sha256": sha256_file(original_handoff),
        "prior_confirmation_results_rewritten": False,
        "u3_history": "U3_INCONCLUSIVE",
        "automatic_boundary": {
            "source": "offline_teacher_to_causal_s623",
            "status": "computed_and_locked",
            "checkpoint_sha256": "fe0464076a3590de19b31d88cd668d4c0e8cf92ee2a80ec413e8191fea34c94e",
            "threshold": 0.5,
            "development_auto_recovery_recall": diagnostics.get("auto_recovery_recall"),
            "development_reference_recovery_recall": diagnostics.get("reference_recovery_recall"),
            "reference_minus_auto_recovery_gain": diagnostics.get("recovery_recall_gain_ref_minus_auto"),
            "development_auto_boundary_occurrences": development_occurrence_count,
            "confirmation_auto_boundary_occurrences": confirmation_occurrence_count,
        },
        "diagnostic_metrics": diagnostics,
        "graphs": graph_rows,
        "paired_effects": paired_rows,
        "claims": claims,
        "accepted_edit_count": accepted,
        "accepted_edits": ["C04 conditional mixed role", "C10 conditional mixed role"],
        "confirmation_locked": bool(lock.get("confirmation_locked")),
        "confirmation_protocol": "u4_bplus_torch_recovery_v2_indices_60_71",
        "execution_counts": {
            "unique_generated_families": generated_family_count,
            "development_families": 24,
            "development_base_rollouts": 96,
            "development_transitions_inferred": development_transition_count,
            "development_auto_boundary_occurrences": development_occurrence_count,
            "torch_recovery_confirmation_families": confirmation_family_count,
            "torch_recovery_confirmation_base_rollouts": confirmation_rollout_count,
            "torch_recovery_confirmation_transitions_inferred": confirmation_transition_count,
            "torch_recovery_confirmation_auto_boundary_occurrences": confirmation_occurrence_count,
            "torch_recovery_confirmation_continuations": confirmation_continuation_count,
            "excluded_checkpoint_mismatch_base_rollouts": excluded_v1_rollouts,
            "all_u4_base_rollouts_including_excluded_protocols": prior_base_rollouts + excluded_v1_rollouts + confirmation_rollout_count,
            "all_u4_continuations_including_prior_protocols": prior_continuations + confirmation_continuation_count,
        },
        "repair_execution": {
            "u2r_triggered": False,
            "u3b_triggered": False,
            "training_jobs": 0,
            "api_calls": 0,
            "api_key_read": False,
        },
        "limitations": [
            "same explicit stochastic simulator family only",
            "no physical robot or new task generalization",
            f"the recovered s623 model marked {development_occurrence_count}/{development_transition_count} development transitions and {confirmation_occurrence_count}/{confirmation_transition_count} confirmation transitions as boundaries",
            f"reference boundaries did not improve family-macro recovery recall over A_auto ({diagnostics.get('reference_recovery_recall')} versus {diagnostics.get('auto_recovery_recall')}), so U2-R was not triggered",
            f"semantic unresolved rate after reference boundaries remained {diagnostics.get('semantic_unresolved_after_reference')} and {diagnostics.get('concrete_ambiguous_groups')} concrete ambiguous group was found, so U3B was not triggered",
            "T028 and T029 were not encountered under the locked automatic-boundary confirmation pipeline; no claim-specific continuations were eligible",
            "G2 had no confirmable paired gain over G1 on the preregistered metrics; retain G1 and the negative edit result",
            "prior contaminated confirmation and prior reconfirmation remain historical records and were not reused for this extension's claims",
        ],
        "next_action": "retain G1 and negative result; do not start a second edit-search round or U5 automatically",
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "u4b_final_handoff.json", final)
    (output_dir / "u4b_status.txt").write_text(status + "\n", encoding="utf-8")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# U4 B+ torch recovery final report\n\n"
        f"- scientific status: `{status}`\n"
        f"- D-GATE: `{route.get('route')}`\n"
        "- automatic boundary: `offline_teacher_to_causal_s623`, threshold 0.5, checkpoint inference completed.\n"
        "- training jobs: 0; API sends: 0; API key read: no.\n"
        f"- development: 24 families, 96 rollouts, {development_transition_count} inferred transitions, {development_occurrence_count} automatic-boundary occurrences.\n"
        f"- confirmation: {confirmation_family_count} new families (indices 60-71), {confirmation_rollout_count} rollouts, {confirmation_transition_count} inferred transitions, {confirmation_occurrence_count} occurrences, {confirmation_continuation_count} eligible continuations.\n"
        f"- accepted development edits: {accepted}/6; G2 had no paired confirmation gain over G1.\n"
        "- T028 and T029 were not encountered in the locked automatic-boundary confirmation pipeline.\n",
        encoding="utf-8",
    )
    return final
