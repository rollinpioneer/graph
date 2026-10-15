"""Within-Qwen stability and cross-provider structural agreement summaries."""

from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

from .common import read_csv, read_json, write_csv
from .validate import candidate_identity


def _node_set(candidate: dict[str, Any]) -> list[tuple[str, frozenset[str]]]:
    return [(str(node["role"]), frozenset(map(str, node.get("observable_predicates", [])))) for node in candidate["nodes"]]


def _jaccard(a: set[Any] | frozenset[Any], b: set[Any] | frozenset[Any]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


def _node_similarity(left: dict[str, Any], right: dict[str, Any]) -> float:
    a = _node_set(left); b = _node_set(right)
    if not a and not b:
        return 1.0
    scores: list[float] = []
    used: set[int] = set()
    for role, predicates in a:
        candidates = [(index, _jaccard(predicates, other_predicates)) for index, (other_role, other_predicates) in enumerate(b) if index not in used and role == other_role]
        if candidates:
            index, score = max(candidates, key=lambda item: item[1]); used.add(index); scores.append(score)
        else:
            scores.append(0.0)
    return sum(scores) / max(len(a), len(b))


def _edge_set(candidate: dict[str, Any]) -> set[tuple[str, frozenset[str], frozenset[str]]]:
    return {(str(edge["hypothesized_type"]), frozenset(map(str, edge.get("preconditions", []))), frozenset(map(str, edge.get("effects", [])))) for edge in candidate["edges"]}


def graph_distance(left: dict[str, Any], right: dict[str, Any]) -> dict[str, float]:
    node = _node_similarity(left, right)
    edge = _jaccard(_edge_set(left), _edge_set(right))
    return {"node_predicate_jaccard": node, "edge_semantic_jaccard": edge, "graph_distance": 1.0 - 0.5 * (node + edge)}


def _load(root: Path, hard_checks: Path | None = None) -> dict[str, tuple[dict[str, Any], str, int]]:
    valid: set[str] | None = None
    if hard_checks:
        valid = {row["candidate_id"] for row in read_csv(hard_checks) if str(row["hard_valid"]).lower() in {"true", "1"}}
    provider = root.parent.name
    result = {}
    for path in root.glob("*.json"):
        cid = f"{provider}:{path.stem}"
        if valid is not None and cid not in valid:
            continue
        condition, replicate = candidate_identity(path)
        result[cid] = (read_json(path), condition, replicate)
    return result


def analyze_stability(*, qwen_root: Path, hard_checks: Path, conditions: list[str], output: Path, pairwise: Path, report: Path) -> dict[str, Any]:
    candidates = _load(qwen_root, hard_checks)
    pairs: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for condition in conditions:
        selected = [(cid, candidate) for cid, candidate in candidates.items() if candidate[1] == condition]
        distances: list[dict[str, float]] = []
        sizes: list[int] = []
        types: dict[str, list[bool]] = {name: [] for name in ("alternative", "failure", "recovery")}
        for _, (candidate, _, _) in selected:
            sizes.append(len(candidate["nodes"]) + len(candidate["edges"]))
            edge_types = {edge["hypothesized_type"] for edge in candidate["edges"]}
            for name in types:
                types[name].append(name in edge_types)
        for (left_id, (left, _, _)), (right_id, (right, _, _)) in itertools.combinations(selected, 2):
            value = graph_distance(left, right)
            distances.append(value)
            pairs.append({"condition": condition, "left_candidate_id": left_id, "right_candidate_id": right_id, **{key: round(item, 8) for key, item in value.items()}})
        mean = lambda values: round(sum(values) / len(values), 8) if values else "not_estimable"
        mean_size = sum(sizes) / len(sizes) if sizes else 0.0
        cv = (math.sqrt(sum((value - mean_size) ** 2 for value in sizes) / len(sizes)) / mean_size) if sizes and mean_size else "not_estimable"
        agreement = lambda values: round(max(sum(values) / len(values), 1 - sum(values) / len(values)), 8) if values else "not_estimable"
        summaries.append({"condition": condition, "hard_valid_replicates": len(selected), "matched_node_predicate_jaccard": mean([row["node_predicate_jaccard"] for row in distances]), "edge_semantic_jaccard": mean([row["edge_semantic_jaccard"] for row in distances]), "pairwise_graph_distance": mean([row["graph_distance"] for row in distances]), "graph_size_coefficient_of_variation": round(cv, 8) if isinstance(cv, float) else cv, "alternative_edge_presence_agreement": agreement(types["alternative"]), "failure_edge_presence_agreement": agreement(types["failure"]), "recovery_edge_presence_agreement": agreement(types["recovery"]), "replicate_stability": (round(1 - sum(row["graph_distance"] for row in distances) / len(distances), 8) if distances else "not_estimable")})
    write_csv(output, summaries)
    write_csv(pairwise, pairs)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Qwen replicate stability\n\n" + "\n".join(f"- {row['condition']}: valid replicates `{row['hard_valid_replicates']}`, stability `{row['replicate_stability']}`" for row in summaries) + "\n", encoding="utf-8")
    return {"status": "PASS", "pairwise_count": len(pairs)}


def compare_providers(*, qwen_root: Path, deepseek_root: Path, replicate: int, output: Path, report: Path) -> dict[str, Any]:
    qwen = _load(qwen_root)
    deepseek = _load(deepseek_root)
    rows: list[dict[str, Any]] = []
    conditions = sorted({condition for _, condition, rep in qwen.values() if rep == replicate})
    for condition in conditions:
        qs = [(cid, candidate) for cid, candidate in qwen.items() if candidate[1] == condition and candidate[2] == replicate]
        ds = [(cid, candidate) for cid, candidate in deepseek.items() if candidate[1] == condition and candidate[2] == replicate]
        if not qs or not ds:
            rows.append({"condition": condition, "status": "not_estimable", "reason": "missing qwen or deepseek replicate-1 candidate"})
            continue
        value = graph_distance(qs[0][1][0], ds[0][1][0])
        rows.append({"condition": condition, "qwen_candidate_id": qs[0][0], "deepseek_candidate_id": ds[0][0], "status": "structural_crosscheck_only", **{key: round(item, 8) for key, item in value.items()}})
    write_csv(output, rows)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Qwen/DeepSeek cross-provider agreement\n\n" + "\n".join(f"- {row['condition']}: `{row['status']}`" for row in rows) + "\n", encoding="utf-8")
    return {"status": "PASS", "condition_count": len(rows)}
