"""Deterministic hard validation for hypothesized U3 candidate graphs."""

from __future__ import annotations

import json
import re
from collections import deque
from pathlib import Path
from typing import Any

from jsonschema import Draft202012Validator

from .common import read_csv, read_json, write_csv, write_jsonl


def candidate_identity(path: Path) -> tuple[str, int]:
    name = path.stem
    match = re.match(r"(.+)_r(\d\d)$", name)
    if not match:
        raise ValueError(f"candidate filename must end in _rNN: {path.name}")
    return match.group(1), int(match.group(2))


def _schema_errors(candidate: dict[str, Any], schema: dict[str, Any]) -> list[str]:
    validator = Draft202012Validator(schema)
    return [f"{'/'.join(str(item) for item in error.absolute_path) or '$'}: {error.message}" for error in validator.iter_errors(candidate)]


def _reachable(nodes: set[str], edges: list[dict[str, Any]], start: str) -> set[str]:
    seen = {start}
    queue: deque[str] = deque([start])
    adjacency: dict[str, list[str]] = {key: [] for key in nodes}
    for edge in edges:
        if edge.get("src") in nodes and edge.get("dst") in nodes:
            adjacency[str(edge["src"])].append(str(edge["dst"]))
    while queue:
        for nxt in adjacency[queue.popleft()]:
            if nxt not in seen:
                seen.add(nxt); queue.append(nxt)
    return seen


def _extract_ids(candidate: dict[str, Any]) -> tuple[set[str], set[int], set[tuple[int, int]]]:
    segments: set[str] = set()
    clusters: set[int] = set()
    pairs: set[tuple[int, int]] = set()
    for node in candidate.get("nodes", []):
        segments.update(str(value) for value in node.get("evidence_segment_ids", []))
        clusters.update(int(value) for value in node.get("source_cluster_ids", []))
    for edge in candidate.get("edges", []):
        segments.update(str(value) for value in edge.get("evidence_segment_ids", []))
        pairs.update((int(value["from_cluster_id"]), int(value["to_cluster_id"])) for value in edge.get("source_transition_pairs", []))
    return segments, clusters, pairs


def _instruction_only_empty(candidate: dict[str, Any]) -> bool:
    for node in candidate.get("nodes", []):
        if node.get("source_cluster_ids") or node.get("evidence_segment_ids"):
            return False
    for edge in candidate.get("edges", []):
        if edge.get("source_transition_pairs") or edge.get("evidence_segment_ids"):
            return False
    return True


def validate_candidates(
    *, qwen_root: Path, deepseek_root: Path, schema_path: Path, vocabulary_path: Path,
    evidence_registry_path: Path, cluster_evidence_path: Path, transition_evidence_path: Path,
    excluded_splits: Path, output: Path, details: Path, report: Path,
) -> dict[str, Any]:
    schema = read_json(schema_path)
    vocabulary = read_json(vocabulary_path)
    registry = read_json(evidence_registry_path)
    cluster_evidence = read_json(cluster_evidence_path)
    transition_evidence = read_json(transition_evidence_path)
    excluded = read_csv(excluded_splits)
    allowed_predicates = {str(row["name"]) for row in vocabulary["allowed_predicates"]}
    allowed_segments = set(map(str, registry["segment_ids"]))
    allowed_clusters = {int(value) for value in registry["cluster_ids"]}
    allowed_pairs = {(int(row["from_cluster_id"]), int(row["to_cluster_id"])) for row in registry["transition_pairs"]}
    excluded_episodes = {str(row["episode_id"]) for row in excluded}
    cluster_ids_from_evidence = {int(row["cluster_id"]) for row in cluster_evidence["clusters"]}
    transition_pairs_from_evidence = {(int(row["from_cluster_id"]), int(row["to_cluster_id"])) for row in transition_evidence["transitions"]}
    rows: list[dict[str, Any]] = []
    details_rows: list[dict[str, Any]] = []
    for provider, root in (("qwen", qwen_root), ("deepseek", deepseek_root)):
        for path in sorted(root.glob("*.json")):
            candidate = read_json(path)
            condition, replicate = candidate_identity(path)
            errors = _schema_errors(candidate, schema)
            nodes = candidate.get("nodes", [])
            edges = candidate.get("edges", [])
            node_ids = [str(row.get("id", "")) for row in nodes]
            edge_ids = [str(row.get("id", "")) for row in edges]
            if len(node_ids) != len(set(node_ids)):
                errors.append("duplicate node id")
            if len(edge_ids) != len(set(edge_ids)):
                errors.append("duplicate edge id")
            node_set = set(node_ids)
            if any(str(edge.get("src")) not in node_set or str(edge.get("dst")) not in node_set for edge in edges):
                errors.append("edge references missing node")
            starts = [str(row.get("id")) for row in nodes if row.get("role") == "start"]
            successes = [str(row.get("id")) for row in nodes if row.get("role") == "success_terminal"]
            if len(starts) != 1:
                errors.append("requires exactly one start node")
            if not successes:
                errors.append("requires a success_terminal node")
            if len(starts) == 1 and successes and not any(item in _reachable(node_set, edges, starts[0]) for item in successes):
                errors.append("no start-to-success reachability")
            predicates: set[str] = set()
            for node in nodes:
                predicates.update(map(str, node.get("observable_predicates", [])))
            for edge in edges:
                predicates.update(map(str, edge.get("preconditions", [])))
                predicates.update(map(str, edge.get("effects", [])))
            forbidden = sorted(predicates - allowed_predicates)
            if forbidden:
                errors.append("forbidden predicate(s): " + ", ".join(forbidden))
            segments, clusters, pairs = _extract_ids(candidate)
            missing_segments = sorted(segments - allowed_segments)
            missing_clusters = sorted(clusters - allowed_clusters)
            missing_pairs = sorted(pairs - allowed_pairs)
            if missing_segments:
                errors.append("nonexistent evidence segment id(s): " + ", ".join(missing_segments[:6]))
            if missing_clusters:
                errors.append("nonexistent cluster id(s): " + ", ".join(map(str, missing_clusters[:6])))
            if missing_pairs:
                errors.append("nonexistent transition pair(s): " + ", ".join(map(str, missing_pairs[:6])))
            if clusters - cluster_ids_from_evidence:
                errors.append("cluster reference omitted from compact evidence")
            if pairs - transition_pairs_from_evidence:
                errors.append("transition reference omitted from compact evidence")
            leaked_episode_evidence = sorted(segment for segment in segments if any(episode in segment for episode in excluded_episodes))
            if leaked_episode_evidence:
                errors.append("val/test evidence id leak")
            if condition == "instruction_only" and not _instruction_only_empty(candidate):
                errors.append("instruction_only must have empty evidence arrays")
            numeric_claim = any(re.search(r"\b(?:cost|reward)\b[^\n]{0,30}\b\d+(?:\.\d+)?", str(value), flags=re.I) for value in candidate.values())
            if numeric_claim:
                errors.append("numeric cost/reward claim detected")
            identifier = f"{provider}:{path.stem}"
            topology_keywords = ("duplicate node id", "duplicate edge id", "edge references missing node", "requires exactly one start node", "requires a success_terminal node", "no start-to-success reachability")
            row = {
                "candidate_id": identifier, "provider": provider, "request_id": path.stem,
                "condition": condition, "replicate": replicate, "schema_valid": not _schema_errors(candidate, schema),
                "topology_valid": not any(any(keyword in error for keyword in topology_keywords) for error in errors),
                "evidence_valid": not (missing_segments or missing_clusters or missing_pairs or leaked_episode_evidence),
                "predicate_valid": not forbidden, "instruction_only_evidence_valid": condition != "instruction_only" or _instruction_only_empty(candidate),
                "node_count": len(nodes), "edge_count": len(edges), "evidence_segment_count": len(segments), "cluster_count": len(clusters), "transition_pair_count": len(pairs),
                "val_test_evidence_leak": bool(leaked_episode_evidence), "hallucinated_evidence_id_count": len(missing_segments),
                "hard_valid": not errors,
                "error_count": len(errors),
            }
            rows.append(row)
            details_rows.append({"candidate_id": identifier, "errors": errors, "missing_segment_ids": missing_segments, "missing_cluster_ids": missing_clusters, "missing_transition_pairs": [list(value) for value in missing_pairs]})
    write_csv(output, rows)
    write_jsonl(details, details_rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    by_provider = {provider: sum(row["hard_valid"] for row in rows if row["provider"] == provider) for provider in ("qwen", "deepseek")}
    report.write_text("# U3 candidate hard validation\n\n" + "\n".join([f"- candidates: `{len(rows)}`", f"- hard-valid Qwen: `{by_provider['qwen']}`", f"- hard-valid DeepSeek: `{by_provider['deepseek']}`", f"- val/test evidence leaks: `{sum(bool(row['val_test_evidence_leak']) for row in rows)}`"]) + "\n", encoding="utf-8")
    return {"status": "PASS" if rows and all(not row["val_test_evidence_leak"] for row in rows) else "FAIL", "candidate_count": len(rows), "hard_valid_qwen": by_provider["qwen"], "hard_valid_deepseek": by_provider["deepseek"]}
