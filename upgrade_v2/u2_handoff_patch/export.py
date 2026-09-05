"""Train-only U3 input export and pending candidate-request generation."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .primitives import read_csv_rows, sha256_file, write_csv_rows, write_json


def _repo_root(path: Path) -> Path:
    path = path.resolve()
    for candidate in (path, *path.parents):
        if (candidate / ".git").exists():
            return candidate
    return path


def _episode_npz(row: dict[str, str], repo_root: Path) -> dict[str, np.ndarray]:
    raw = Path(row["npz_path"])
    candidates = [raw, repo_root / raw]
    if not raw.is_absolute():
        candidates.append(repo_root / "artifacts" / "pathgraph_sarm" / "upgrade_v2" / "u2_stochastic_boundary" / "data_v1" / "formal" / "episodes" / f"{row['episode_id']}.npz")
    for path in candidates:
        if path.is_file():
            with np.load(path) as payload:
                return {key: np.asarray(payload[key]) for key in payload.files}
    raise FileNotFoundError(f"missing episode cache: {row['episode_id']} ({candidates})")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _cluster_id(row: dict[str, Any]) -> int:
    try:
        return int(row.get("cluster_id", 0))
    except (TypeError, ValueError):
        return 0


def _safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _candidate_schema(predicate_vocabulary: list[str]) -> dict[str, Any]:
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "$id": "u3_candidate_graph_v1",
        "title": "U3 simulator-scoped hypothesized candidate graph",
        "type": "object",
        "required": ["candidate_id", "scope", "nodes", "edges"],
        "additionalProperties": False,
        "properties": {
            "candidate_id": {"type": "string", "minLength": 1},
            "scope": {"const": "stochastic_simulator_only"},
            "nodes": {"type": "array", "items": {"$ref": "#/$defs/node"}},
            "edges": {"type": "array", "items": {"$ref": "#/$defs/edge"}},
        },
        "$defs": {
            "node": {
                "type": "object",
                "additionalProperties": False,
                "required": ["id", "description", "observable_predicates", "unknown_conditions", "evidence_segment_ids", "status"],
                "properties": {
                    "id": {"type": "string", "minLength": 1},
                    "description": {"type": "string", "minLength": 1},
                    "observable_predicates": {"type": "array", "items": {"enum": predicate_vocabulary}, "uniqueItems": True},
                    "unknown_conditions": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                    "evidence_segment_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                    "status": {"const": "hypothesized"},
                },
            },
            "edge": {
                "type": "object",
                "additionalProperties": False,
                "required": ["src", "dst", "preconditions", "effects", "hypothesized_type", "evidence_segment_ids", "status", "cost_measurement_needed"],
                "properties": {
                    "src": {"type": "string", "minLength": 1},
                    "dst": {"type": "string", "minLength": 1},
                    "preconditions": {"type": "array", "items": {"enum": predicate_vocabulary}, "uniqueItems": True},
                    "effects": {"type": "array", "items": {"enum": predicate_vocabulary}, "uniqueItems": True},
                    "hypothesized_type": {"enum": ["forward", "failure", "recovery", "alternative", "unknown"]},
                    "evidence_segment_ids": {"type": "array", "items": {"type": "string"}, "uniqueItems": True},
                    "status": {"const": "hypothesized"},
                    "cost_measurement_needed": {"type": "string", "minLength": 1},
                },
            },
        },
        "schema_version": "u3_candidate_graph_v1",
        "scope": "stochastic_simulator_only",
        "candidate_status": "hypothesized_only",
        "predicate_policy": {
            "mode": "finite_declarative_vocabulary",
            "allow_arbitrary_code": False,
            "numeric_cost_is_ground_truth": False,
            "allowed_predicates": predicate_vocabulary,
        },
    }


def _transition_evidence(transitions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts: dict[tuple[int, int], dict[str, Any]] = {}
    for row in transitions:
        key = (int(row["from_cluster_id"]), int(row["to_cluster_id"]))
        item = counts.setdefault(key, {"from_cluster_id": key[0], "to_cluster_id": key[1], "n_transitions": 0, "root_family_ids": set()})
        item["n_transitions"] += 1
        item["root_family_ids"].add(str(row["root_family_id"]))
    result = []
    for item in counts.values():
        result.append({
            "from_cluster_id": item["from_cluster_id"],
            "to_cluster_id": item["to_cluster_id"],
            "n_transitions": item["n_transitions"],
            "support_root_families": len(item["root_family_ids"]),
        })
    return sorted(result, key=lambda row: (-row["n_transitions"], row["from_cluster_id"], row["to_cluster_id"]))[:32]


def _condition_prompt(
    condition: str,
    task_contract: dict[str, Any],
    train_summary: list[dict[str, Any]],
    prototypes: dict[str, Any],
    transitions: list[dict[str, Any]],
    fallback: list[dict[str, Any]],
) -> str:
    base = (
        "You are proposing a semantic graph for the explicit-state stochastic simulator only. "
        "Return exactly one JSON object conforming to schema u3_candidate_graph_v1. "
        "Every node and edge must remain hypothesized; do not claim validation. "
        "Use only the supplied finite observable predicate vocabulary, preserve unknown conditions, "
        "cite only supplied train segment IDs, and treat numeric costs as measurements required in U4 rather than ground truth. "
    )
    if condition == "instruction_only":
        return base + "Input condition: instruction, roles, sensors, and actions only; no trajectory examples. Task contract=" + json.dumps(task_contract, ensure_ascii=False, sort_keys=True)
    compact_prototypes = [{key: value for key, value in prototype.items() if key != "root_family_ids"} for _, prototype in sorted(prototypes.items(), key=lambda item: (-int(item[1]["n_segments"]), int(item[0])))[:24]]
    compact_segments = [{
        "segment_id": row.get("segment_id"),
        "cluster_id": row.get("cluster_id"),
        "duration": row.get("duration"),
        "observable_contact_history": row.get("observable_contact_history"),
        "observable_contact_loss_count": row.get("observable_contact_loss_count"),
        "unknown_fraction": row.get("unknown_fraction", row.get("uncertainty")),
    } for row in train_summary[:48]]
    auto_evidence = {
        "task_contract": task_contract,
        "cluster_prototypes": compact_prototypes,
        "observed_cluster_transitions": _transition_evidence(transitions),
        "train_segment_examples": compact_segments,
    }
    if condition == "instruction_plus_auto_train_segments":
        return base + "Input condition: train-only automatic segment summaries with unknown retained. Evidence=" + json.dumps(auto_evidence, ensure_ascii=False, sort_keys=True)
    fallback_evidence = [{
        "clip_id": row.get("clip_id"),
        "episode_id": row.get("episode_id"),
        "root_family_id": row.get("root_family_id"),
        "start_t": row.get("start_t"),
        "end_t": row.get("end_t"),
        "event_id": row.get("event_id"),
        "label_source": row.get("label_source"),
    } for row in fallback]
    auto_evidence["budgeted_train_fallback"] = fallback_evidence
    return base + "Input condition: the same train-only automatic summaries plus explicitly budgeted simulator clip confirmations. Evidence=" + json.dumps(auto_evidence, ensure_ascii=False, sort_keys=True)


def export_u3_train(u2_root: Path, output: Path, split: str = "train", include_unknown: bool = True, fallback_max_clips: int = 30) -> dict[str, Any]:
    """Export a split-pure U3 bundle and generate pending LLM requests."""

    repo_root = _repo_root(u2_root)
    manifest_path = u2_root / "data_v1" / "formal" / "episode_manifest.csv"
    manifest = read_csv_rows(manifest_path)
    episode_by_id = {row["episode_id"]: row for row in manifest}
    train_rows = [row for row in manifest if row["split"] == split]
    forbidden_rows = [row for row in manifest if row["split"] != split]
    train_episode_ids = {row["episode_id"] for row in train_rows}
    train_family_ids = {row["root_family_id"] for row in train_rows}
    forbidden_episode_ids = {row["episode_id"] for row in forbidden_rows}
    forbidden_family_ids = {row["root_family_id"] for row in forbidden_rows}
    if train_episode_ids & forbidden_episode_ids or train_family_ids & forbidden_family_ids:
        raise RuntimeError("episode/root-family split isolation is violated in the authoritative manifest")
    output.mkdir(parents=True, exist_ok=True)

    segments_path = u2_root / "segment_representation_v1" / "segments" / "segments.jsonl"
    segment_summary_path = u2_root / "segment_representation_v1" / "segment_event_summary.jsonl"
    observable_schema_path = u2_root / "data_v1" / "formal" / "configs" / "observable_schema.json"
    event_schema_path = u2_root / "data_v1" / "formal" / "configs" / "event_schema.json"
    if not segments_path.is_file():
        raise FileNotFoundError(f"missing segment summary source: {segments_path}")
    all_segments = _load_jsonl(segments_path)
    # The compact historical summary contains cluster IDs and event posterior
    # fields but omits the episode keys.  Join it to the full segment records
    # by segment_id; no parquet reader is needed for this export.
    compact_by_id = {row.get("segment_id"): row for row in _load_jsonl(segment_summary_path)} if segment_summary_path.is_file() else {}
    for row in all_segments:
        compact = compact_by_id.get(row.get("segment_id"), {})
        for key in ("cluster_id", "event_posterior", "uncertainty", "observable_predicate_summary", "duration_statistics", "following_observed_cluster_id"):
            if key in compact and key not in row:
                row[key] = compact[key]
    train_segments = [row for row in all_segments if row.get("split") == split and row.get("episode_id") in train_episode_ids]
    leaked = [row for row in train_segments if row.get("split") != split or row.get("episode_id") not in train_episode_ids or row.get("root_family_id") not in train_family_ids]
    if leaked:
        raise RuntimeError("train segment export contains a non-train row")

    train_summary_path = output / "segment_event_summary_train.jsonl"
    with train_summary_path.open("w", encoding="utf-8") as handle:
        for row in train_segments:
            # Preserve the original posterior/unknown fields, adding an
            # explicit source and split assertion for downstream prompts.
            value = dict(row)
            value["split"] = split
            value["source_type"] = "auto_boundary_train_only"
            value["status"] = "hypothesized"
            value["unknown_retained"] = bool(value.get("unknown_fraction", value.get("uncertainty", 0.0)) > 0.0)
            if not include_unknown:
                raise ValueError("U3 train export must retain unknown; --include-unknown false is unsupported")
            handle.write(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")

    # Cluster prototypes are recomputed solely from train segments.  The
    # parquet is used for cluster IDs/durations and the JSONL for evidence.
    by_cluster: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in train_segments:
        by_cluster[int(row.get("cluster_id", 0))].append(row)
    prototypes: dict[str, Any] = {}
    for cluster, values in sorted(by_cluster.items()):
        durations = [_safe_float(row.get("duration_statistics", row.get("duration", 0))) for row in values]
        uncertainties = [_safe_float(row.get("uncertainty", row.get("unknown_fraction", 0))) for row in values]
        posterior = [np.asarray(row.get("event_posterior", []), dtype=float) for row in values if row.get("event_posterior")]
        posterior_mean = np.mean(np.stack(posterior), axis=0).tolist() if posterior else []
        families = sorted({str(row["root_family_id"]) for row in values})
        prototypes[str(cluster)] = {
            "cluster_id": cluster,
            "n_segments": len(values),
            "support_root_families": len(families),
            "root_family_ids": families,
            "mean_duration": float(np.nanmean(durations)) if durations else None,
            "mean_unknown_fraction": float(np.nanmean(uncertainties)) if uncertainties else None,
            "mean_event_posterior": posterior_mean,
            "source_split": split,
            "source_type": "train_only_recomputed",
        }
    write_json(output / "cluster_prototypes_train.json", prototypes)

    # Build adjacent transitions in start-time order within each episode.
    by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in train_segments:
        by_episode[str(row["episode_id"])].append(row)
    transitions: list[dict[str, Any]] = []
    for episode_id, values in sorted(by_episode.items()):
        ordered = sorted(values, key=lambda item: (int(item.get("start_t", 0)), int(item.get("end_t", 0)), item["segment_id"]))
        for index, (left, right) in enumerate(zip(ordered, ordered[1:])):
            transitions.append({
                "episode_id": episode_id,
                "root_family_id": left["root_family_id"],
                "split": split,
                "transition_index": index,
                "from_segment_id": left["segment_id"],
                "to_segment_id": right["segment_id"],
                "from_cluster_id": int(left.get("cluster_id", 0)),
                "to_cluster_id": int(right.get("cluster_id", 0)),
                "from_end_t": int(left.get("end_t", 0)),
                "to_start_t": int(right.get("start_t", 0)),
                "source_type": "train_only_observed_transition",
            })
    write_csv_rows(output / "observed_segment_transitions_train.csv", transitions)

    calibration_path = u2_root / "rounds" / "u2_1_event_candidates_and_weak_labels" / "tables" / "weak_calibration_families.csv"
    calibration = read_csv_rows(calibration_path) if calibration_path.is_file() else []
    calibration_families = sorted({row["root_family_id"] for row in calibration if row.get("split") == "train"})
    calibration_frames = 0
    for row in train_rows:
        if row["root_family_id"] in calibration_families:
            calibration_frames += int(row.get("n_steps", 0))
    oracle_path = u2_root / "boundary_models_v1" / "configs" / "oracle30_clip_ids.csv"
    oracle_all = read_csv_rows(oracle_path) if oracle_path.is_file() else []
    oracle = [row for row in oracle_all if row.get("split") == split and row.get("episode_id") in train_episode_ids and row.get("root_family_id") in train_family_ids][:fallback_max_clips]
    if len(oracle) != min(fallback_max_clips, len(oracle_all)):
        raise RuntimeError("fallback budget contains a non-train or manifest-unknown clip")
    fallback_frames = sum(max(0, int(row.get("end_t", 0)) - int(row.get("start_t", 0)) + 1) for row in oracle)
    ledger = []
    for row in oracle:
        ledger.append({"budget_kind": "additional_fallback", "clip_id": row.get("clip_id", ""), "episode_id": row.get("episode_id", ""), "root_family_id": row.get("root_family_id", ""), "split": row.get("split", ""), "start_t": row.get("start_t", ""), "end_t": row.get("end_t", ""), "frame_count": max(0, int(row.get("end_t", 0)) - int(row.get("start_t", 0)) + 1), "source": row.get("label_source", "simulator_gold_oracle_budget")})
    write_csv_rows(output / "fallback_budget_ledger.csv", ledger)

    observability_rows = []
    for row in train_segments:
        observability_rows.append({
            "segment_id": row.get("segment_id", ""),
            "episode_id": row.get("episode_id", ""),
            "root_family_id": row.get("root_family_id", ""),
            "split": split,
            "cluster_id": row.get("cluster_id", ""),
            "observable_ever_contact": row.get("observable_predicate_summary", {}).get("ever_contact", False),
            "observable_contact_loss_count": row.get("observable_predicate_summary", {}).get("contact_loss_count", 0),
            "unknown_fraction": row.get("uncertainty", row.get("unknown_fraction", "")),
            "evidence_status": "train_observed_hypothesis",
        })
    write_csv_rows(output / "candidate_observability_table.csv", observability_rows)
    write_csv_rows(output / "unsupported_predicates.csv", [
        {"predicate": "exact_numeric_cost", "status": "unsupported", "reason": "must be measured in U4; LLM numeric values are not ground truth"},
        {"predicate": "future_segment_label", "status": "forbidden", "reason": "not observable at candidate-generation time"},
        {"predicate": "arbitrary_generated_code", "status": "forbidden", "reason": "schema accepts finite declarative predicates only"},
        {"predicate": "test_gold_state", "status": "forbidden", "reason": "split isolation"},
    ])

    excluded = [{"episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": row["split"], "reason": "excluded_from_U3_prompt_inputs"} for row in forbidden_rows]
    write_csv_rows(output / "excluded_split_manifest.csv", excluded)

    fallback_policy = {
        "scope": "same stochastic simulator family only",
        "automatic_boundary_supported": False,
        "default_source": "validation_locked_causal_or_rule_source",
        "unknown_action": "keep_unknown_and_request_budgeted_confirmation",
        "fallback_source": "explicitly_disclosed_simulator_gold_train_clip",
        "fallback_budget_unit": "unique_clip_and_unique_frame",
        "test_gold_in_prompts": False,
        "confirmed_fragment_is_not_validated_graph": True,
        "shared_calibration_supervision": {
            "family_count": len(calibration_families),
            "frame_count": calibration_frames,
            "family_ids": calibration_families,
            "source": "weak_calibration_families.csv",
        },
        "additional_fallback": {
            "clip_count": len(oracle),
            "frame_count": fallback_frames,
            "unique_clip_count": len({row.get("clip_id") for row in oracle}),
            "unique_frame_count": sum(max(0, int(row.get("end_t", 0)) - int(row.get("start_t", 0)) + 1) for row in oracle),
            "source": "oracle30_clip_ids.csv",
        },
    }
    write_json(output / "fallback_policy.json", fallback_policy)

    # Candidate generation is deliberately pending: no model/API is called.
    candidates_dir = output / "candidates"
    (candidates_dir / "raw_responses").mkdir(parents=True, exist_ok=True)
    (candidates_dir / "parsed_graphs").mkdir(parents=True, exist_ok=True)
    observable_schema = json.loads(observable_schema_path.read_text(encoding="utf-8"))
    event_schema = json.loads(event_schema_path.read_text(encoding="utf-8"))
    predicate_vocabulary = [f"observable:{name}" for name in observable_schema["features"]]
    predicate_vocabulary += [
        "segment:observable_contact_history",
        "segment:observable_contact_loss_count",
        "segment:unknown_fraction",
    ]
    task_contract = {
        "scope": "stochastic_simulator_only",
        "simulator_roles": "A two-dimensional agent approaches, contacts, and transports an object to a goal while an obstacle can cause collision or detour behavior.",
        "action_fields": ["action_x", "action_y"],
        "observable_features": observable_schema["features"],
        "forbidden_prompt_fields": observable_schema["forbidden"],
        "event_timing": event_schema["boundary_timing"],
        "event_names_are_not_prompt_labels": True,
        "finite_predicate_vocabulary": predicate_vocabulary,
        "candidate_status": "hypothesized_only",
        "external_llm_execution": "MODEL_EXECUTION_PENDING",
    }
    write_json(output / "task_contract.json", task_contract)
    schema = _candidate_schema(predicate_vocabulary)
    write_json(candidates_dir / "schema.json", schema)
    conditions = ["instruction_only", "instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"]
    requests_path = candidates_dir / "requests.jsonl"
    request_rows: list[dict[str, Any]] = []
    for condition in conditions:
        for replicate in range(3):
            request_rows.append({
                "request_id": f"{condition}_r{replicate + 1:02d}",
                "condition": condition,
                "replicate": replicate + 1,
                "status": "MODEL_EXECUTION_PENDING",
                "model_version": "unresolved_no_local_or_authorized_model",
                "temperature": 0.0,
                "max_output_tokens": 2500,
                "schema_path": str((candidates_dir / "schema.json").relative_to(repo_root)),
                "prompt": _condition_prompt(condition, task_contract, train_segments, prototypes, transitions, oracle),
                "input_split": split,
                "test_gold_in_prompt": False,
            })
    with requests_path.open("w", encoding="utf-8") as handle:
        for row in request_rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")
    for name in ("raw_responses", "parsed_graphs"):
        (candidates_dir / name / "README.md").write_text("MODEL_EXECUTION_PENDING: no external LLM was called; no response or parsed graph is present.\n", encoding="utf-8")

    embedding_path = u2_root / "segment_representation_v1" / "embeddings" / "segment_embeddings.parquet"
    files_for_manifest = [manifest_path, segments_path, segment_summary_path, embedding_path, calibration_path, oracle_path, observable_schema_path, event_schema_path, train_summary_path, output / "cluster_prototypes_train.json", output / "observed_segment_transitions_train.csv", output / "fallback_policy.json", output / "task_contract.json", candidates_dir / "schema.json", candidates_dir / "requests.jsonl"]
    file_hashes = []
    for path in files_for_manifest:
        if path.is_file():
            file_hashes.append({"path": str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path), "size_bytes": path.stat().st_size, "sha256": sha256_file(path)})
    prompt_manifest = {
        "schema": "u3_prompt_input_manifest_v1",
        "status": "TRAIN_ONLY_VERIFIED_MODEL_EXECUTION_PENDING",
        "source_commit": _git_commit(repo_root),
        "input_split": split,
        "train_episode_count": len(train_rows),
        "train_root_family_count": len({row["root_family_id"] for row in train_rows}),
        "train_segment_count": len(train_segments),
        "train_transition_count": len(transitions),
        "excluded_split_counts": dict(Counter(row["split"] for row in forbidden_rows)),
        "excluded_episode_count": len(forbidden_rows),
        "root_family_split_overlap_count": len(train_family_ids & forbidden_family_ids),
        "train_segment_non_train_id_count": len(leaked),
        "unknown_retained": include_unknown,
        "shared_calibration_family_count": len(calibration_families),
        "shared_calibration_frame_count": calibration_frames,
        "additional_fallback_clip_count": len(oracle),
        "additional_fallback_frame_count": fallback_frames,
        "candidate_request_count": len(request_rows),
        "llm_candidate_count": 0,
        "llm_status": "MODEL_EXECUTION_PENDING",
        "test_gold_in_prompts": False,
        "missing_input_items": [str(path.relative_to(repo_root)) for path in files_for_manifest if not path.is_file()],
        "files": file_hashes,
    }
    write_json(output / "prompt_input_manifest.json", prompt_manifest)
    return prompt_manifest


def _git_commit(repo_root: Path) -> str:
    import subprocess

    try:
        return subprocess.check_output(["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
