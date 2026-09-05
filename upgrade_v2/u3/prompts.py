"""Provider-neutral U3 prompt construction.

The builder deliberately keeps the evidence selection deterministic.  It does
not decide a graph and never injects val/test examples: it only serializes the
already split-pure U2 summaries for a remote model to propose hypotheses.
"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

from .common import read_json, read_jsonl, sha256_file, write_csv, write_jsonl


SYSTEM_PROMPT = """You propose one semantic task graph for the same explicit-state stochastic simulator family. Return one non-empty JSON object conforming to the supplied schema. All nodes and edges are hypotheses. Do not mark anything observed or validated. Use only allowed observable predicates. Preserve unknown conditions. Do not invent numerical costs or rewards. Do not execute or emit code."""

COMPACT_RECOVERY_INSTRUCTION = """\n\nFORMAT RECOVERY (the evidence and task contract are unchanged): return the smallest schema-compliant graph possible: exactly 3 nodes (one start, one intermediate, one success_terminal) and exactly 2 forward edges. Keep every description, unknown condition, and unresolved question concise. Cite at most one supplied evidence item per non-empty evidence array. Return JSON only."""


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ordered_evidence(cluster: dict[str, Any], transition: dict[str, Any], seed: int | None) -> tuple[dict[str, Any], dict[str, Any]]:
    clusters = list(cluster.get("clusters", []))
    transitions = list(transition.get("transitions", []))
    if seed is None:
        clusters.sort(key=lambda row: (-int(row.get("support_root_families", 0)), int(row.get("cluster_id", 0))))
        transitions.sort(key=lambda row: (-int(row.get("support_root_families", 0)), -int(row.get("observation_count", 0)), int(row.get("from_cluster_id", 0)), int(row.get("to_cluster_id", 0))))
    else:
        rng = random.Random(seed)
        rng.shuffle(clusters)
        rng.shuffle(transitions)
    return ({**cluster, "clusters": clusters}, {**transition, "transitions": transitions})


def _request_payload(
    *,
    condition: str,
    task_contract: dict[str, Any],
    vocabulary: dict[str, Any],
    schema: dict[str, Any],
    cluster_evidence: dict[str, Any],
    transition_evidence: dict[str, Any],
    fallback_confirmations: list[dict[str, Any]],
    ledger: dict[str, Any],
) -> dict[str, Any]:
    evidence: dict[str, Any] = {"kind": "instruction_only", "cluster_evidence": [], "transition_evidence": [], "fallback_confirmations": []}
    if condition in {"instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"}:
        evidence = {
            "kind": "train_only_automatic_segments",
            "cluster_evidence": cluster_evidence,
            "transition_evidence": transition_evidence,
            "fallback_confirmations": [],
        }
    if condition == "instruction_plus_budgeted_train_fallback":
        evidence["kind"] = "train_only_automatic_segments_plus_budgeted_confirmations"
        evidence["fallback_confirmations"] = fallback_confirmations
    return {
        "task_contract": task_contract,
        "predicate_vocabulary": vocabulary,
        "condition_name": condition,
        "condition_evidence": evidence,
        "boundary_fallback_disclosure": task_contract["boundary_fallback"],
        "supervision_cost_ledger": ledger[condition],
        "output_schema_summary": {
            "schema_version": "u3_candidate_graph_v1",
            "scope": "stochastic_simulator_only",
            "node_status": "hypothesized",
            "edge_status": "hypothesized",
            "schema": schema,
        },
    }


def build_requests(
    *,
    task_contract_path: Path,
    vocabulary_path: Path,
    schema_path: Path,
    cluster_path: Path,
    transition_path: Path,
    fallback_path: Path,
    ledger_path: Path,
    conditions: list[str],
    replicates: int,
    ordering_seeds: list[int],
    max_prompt_chars: int,
    output: Path,
    prompt_dir: Path,
    manifest: Path,
) -> dict[str, Any]:
    expected = ["instruction_only", "instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"]
    if conditions != expected:
        raise ValueError(f"conditions must exactly be {expected}")
    if replicates != 3 or len(ordering_seeds) != 2:
        raise ValueError("U3 requires exactly three replicates and two ordering seeds")
    task_contract = read_json(task_contract_path)
    vocabulary = read_json(vocabulary_path)
    schema = read_json(schema_path)
    cluster = read_json(cluster_path)
    transition = read_json(transition_path)
    fallback = read_jsonl(fallback_path)
    ledger = read_json(ledger_path)
    if len(fallback) != 30:
        raise ValueError("expected exactly 30 fallback confirmations")
    if any(item.get("input_split") not in {None, "train"} for item in (cluster, transition)):
        raise ValueError("compact evidence must be train-only")
    if any(item.get("confirmation_source") != "simulator_gold_train_clip" for item in fallback):
        raise ValueError("fallback confirmations are not train simulator-gold clips")

    requests: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    schema_sha = sha256_file(schema_path)
    contract_sha = sha256_file(task_contract_path)
    for condition in conditions:
        for replicate in range(1, 4):
            order_seed = None if replicate == 1 else ordering_seeds[replicate - 2]
            ordered_cluster, ordered_transition = _ordered_evidence(cluster, transition, order_seed)
            payload = _request_payload(
                condition=condition,
                task_contract=task_contract,
                vocabulary=vocabulary,
                schema=schema,
                cluster_evidence=ordered_cluster,
                transition_evidence=ordered_transition,
                fallback_confirmations=fallback,
                ledger=ledger,
            )
            # Instruction-only is intentionally identical across replicates.
            if condition == "instruction_only":
                payload = _request_payload(
                    condition=condition,
                    task_contract=task_contract,
                    vocabulary=vocabulary,
                    schema=schema,
                    cluster_evidence={"clusters": []},
                    transition_evidence={"transitions": []},
                    fallback_confirmations=[],
                    ledger=ledger,
                )
                order_seed = None
            user_text = _canonical(payload)
            if len(SYSTEM_PROMPT) + len(user_text) > max_prompt_chars:
                raise ValueError(f"prompt exceeds max chars for {condition} r{replicate:02d}")
            request_id = f"{condition}_r{replicate:02d}"
            request_dir = prompt_dir / request_id
            request_dir.mkdir(parents=True, exist_ok=True)
            system_path = request_dir / "system_prompt.txt"
            user_path = request_dir / "user_prompt.json"
            system_path.write_text(SYSTEM_PROMPT + "\n", encoding="utf-8")
            user_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
            record = {
                "request_id": request_id,
                "condition": condition,
                "replicate": replicate,
                "ordering_seed": "support_desc" if order_seed is None else order_seed,
                "system_prompt_path": str(system_path),
                "user_prompt_path": str(user_path),
                "prompt_sha256": sha256_file(system_path) + ":" + sha256_file(user_path),
                "schema_sha256": schema_sha,
                "task_contract_sha256": contract_sha,
                "input_modality": "text_only",
                "image_count": 0,
                "video_count": 0,
                "test_gold_in_prompt": False,
                "status": "REQUEST_READY",
            }
            requests.append(record)
            rows.append({**record, "prompt_chars": len(SYSTEM_PROMPT) + len(user_text)})
    write_jsonl(output, requests)
    write_csv(manifest, rows)
    return {
        "status": "U3_PROMPT_PACKAGE_REPAIRED",
        "request_count": len(requests),
        "condition_count": len(conditions),
        "max_prompt_chars_observed": max(int(row["prompt_chars"]) for row in rows),
        "schema_sha256": schema_sha,
        "task_contract_sha256": contract_sha,
    }


def refine_condition_once(*, requests: Path, condition: str, prompt_dir: Path, output: Path, manifest: Path) -> dict[str, Any]:
    """Create the single allowed compact-format retry without changing evidence."""
    records = [row for row in read_jsonl(requests) if row["condition"] == condition]
    if len(records) != 3:
        raise ValueError("a one-time refinement requires exactly three condition replicates")
    revised: list[dict[str, Any]] = []
    summary: list[dict[str, Any]] = []
    for row in sorted(records, key=lambda item: int(item["replicate"])):
        old_system = Path(row["system_prompt_path"]).read_text(encoding="utf-8").rstrip()
        old_user = Path(row["user_prompt_path"]).read_text(encoding="utf-8")
        if COMPACT_RECOVERY_INSTRUCTION.strip() in old_system:
            raise ValueError("refinement instruction is already present")
        request_dir = prompt_dir / row["request_id"]
        request_dir.mkdir(parents=True, exist_ok=True)
        system_path = request_dir / "system_prompt.txt"
        user_path = request_dir / "user_prompt.json"
        system_path.write_text(old_system + COMPACT_RECOVERY_INSTRUCTION + "\n", encoding="utf-8")
        user_path.write_text(old_user, encoding="utf-8")
        current = {**row, "system_prompt_path": str(system_path), "user_prompt_path": str(user_path), "prompt_sha256": sha256_file(system_path) + ":" + sha256_file(user_path), "prompt_refinement_round": 1, "refinement_reason": "all_three_condition_outputs_truncated_at_length", "status": "REFINEMENT_REQUEST_READY"}
        revised.append(current)
        summary.append({**current, "prompt_chars": len(system_path.read_text(encoding="utf-8")) + len(old_user)})
    write_jsonl(output, revised)
    write_csv(manifest, summary)
    return {"status": "REFINE_U3_PROMPT_ONCE", "condition": condition, "request_count": len(revised), "refinement_round": 1, "evidence_unchanged": True, "model_unchanged": True, "max_prompt_chars_observed": max(int(row["prompt_chars"]) for row in summary)}
