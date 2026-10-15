"""Evidence-grounded preliminary scoring for hard-valid U3 candidates."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import read_csv, read_json, write_csv


def _paths(roots: list[Path]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for root in roots:
        provider = root.parent.name
        for path in root.glob("*.json"):
            result[f"{provider}:{path.stem}"] = path
    return result


def _safe_ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def score_candidates(
    *, hard_checks: Path, candidate_roots: list[Path], cluster_evidence_path: Path,
    transition_evidence_path: Path, ledger_path: Path, output: Path, condition_summary: Path,
    report: Path,
) -> dict[str, Any]:
    checks = {row["candidate_id"]: row for row in read_csv(hard_checks)}
    candidates = _paths(candidate_roots)
    clusters = {int(row["cluster_id"]): row for row in read_json(cluster_evidence_path)["clusters"]}
    transitions = {(int(row["from_cluster_id"]), int(row["to_cluster_id"])): row for row in read_json(transition_evidence_path)["transitions"]}
    ledger = read_json(ledger_path)
    rows: list[dict[str, Any]] = []
    contradiction_rows: list[dict[str, Any]] = []
    for candidate_id, check in sorted(checks.items()):
        base = {**check, "candidate_path": str(candidates.get(candidate_id, ""))}
        if candidate_id not in candidates or str(check["hard_valid"]).lower() not in {"true", "1"}:
            rows.append({**base, "eligible": False, "preliminary_score": "not_applicable", "grounding_score": "not_applicable", "observability_score": "not_applicable", "transition_score": "not_applicable", "unknown_score": "not_applicable", "simplicity_score": "not_applicable"})
            continue
        candidate = read_json(candidates[candidate_id])
        node_refs = [node for node in candidate["nodes"] if node["source_cluster_ids"] or node["evidence_segment_ids"]]
        edge_refs = [edge for edge in candidate["edges"] if edge["source_transition_pairs"] or edge["evidence_segment_ids"]]
        referenced_clusters = [int(value) for node in candidate["nodes"] for value in node["source_cluster_ids"]]
        referenced_pairs = [(int(pair["from_cluster_id"]), int(pair["to_cluster_id"])) for edge in candidate["edges"] for pair in edge["source_transition_pairs"]]
        support_values = [int(clusters[value].get("support_root_families", 0)) for value in referenced_clusters if value in clusters]
        grounding = 0.5 * _safe_ratio(len(node_refs) + len(edge_refs), len(candidate["nodes"]) + len(candidate["edges"]))
        grounding += 0.5 * min(1.0, _safe_ratio(sum(support_values), max(1, len(referenced_clusters)) * 5))
        all_predicate_fields = [node.get("observable_predicates", []) for node in candidate["nodes"]] + [edge.get("preconditions", []) + edge.get("effects", []) for edge in candidate["edges"]]
        observability = _safe_ratio(sum(bool(field) for field in all_predicate_fields), len(all_predicate_fields))
        pair_support = [int(transitions[pair].get("support_root_families", 0)) for pair in referenced_pairs if pair in transitions]
        transition = min(1.0, _safe_ratio(sum(pair_support), max(1, len(referenced_pairs)) * 5)) if referenced_pairs else 0.0
        unknown_items = sum(bool(item.get("unknown_conditions")) for item in candidate["nodes"] + candidate["edges"])
        unknown = min(1.0, 0.65 * _safe_ratio(unknown_items, len(candidate["nodes"]) + len(candidate["edges"])) + 0.35 * bool(candidate.get("unresolved_questions")))
        simplicity = max(0.0, 1.0 - max(0, len(candidate["nodes"]) - 5) * 0.04 - max(0, len(candidate["edges"]) - 5) * 0.025)
        condition = check["condition"]
        if condition == "instruction_only":
            score = 0.55 * observability + 0.25 * unknown + 0.20 * simplicity
            grounding_value: float | str = "not_applicable"; transition_value: float | str = "not_applicable"
        else:
            score = 0.30 * grounding + 0.25 * observability + 0.20 * transition + 0.15 * unknown + 0.10 * simplicity
            grounding_value = round(grounding, 8); transition_value = round(transition, 8)
        rows.append({**base, "eligible": True, "preliminary_score": round(score, 8), "grounding_score": grounding_value, "observability_score": round(observability, 8), "transition_score": transition_value, "unknown_score": round(unknown, 8), "simplicity_score": round(simplicity, 8)})
        for edge in candidate["edges"]:
            edge_type = edge["hypothesized_type"]
            if edge_type in {"failure", "recovery", "alternative"} and (not edge["source_transition_pairs"] or edge["unknown_conditions"]):
                contradiction_rows.append({
                    "candidate_id": candidate_id, "node_or_edge_id": edge["id"],
                    "claim": f"{edge_type} transition {edge['src']} -> {edge['dst']}",
                    "missing_or_conflicting_evidence": "no observed transition pair or unresolved condition retained",
                    "affected_cluster_or_transition": ";".join(f"{item['from_cluster_id']}->{item['to_cluster_id']}" for item in edge["source_transition_pairs"]) or "not_cited",
                    "recommended_U4_query": "test continuation anchors for the claimed branch under matching observables",
                    "priority": "high" if edge_type in {"failure", "recovery"} else "medium",
                })
    write_csv(output, rows)
    contradiction_output = output.parent.parent / "candidate_contradiction_queue.csv"
    write_csv(contradiction_output, contradiction_rows)
    summaries: list[dict[str, Any]] = []
    for condition in ("instruction_only", "instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"):
        eligible = [row for row in rows if row["condition"] == condition and row["eligible"]]
        def avg(field: str) -> str | float:
            vals = [float(row[field]) for row in eligible if isinstance(row[field], (float, int))]
            return round(sum(vals) / len(vals), 8) if vals else "not_estimable"
        cost = ledger[condition]
        summaries.append({"condition": condition, "eligible_count": len(eligible), "candidate_count": sum(row["condition"] == condition for row in rows), "preliminary_score_mean": avg("preliminary_score"), "grounding_score_mean": avg("grounding_score") if condition != "instruction_only" else "not_applicable", "observability_score_mean": avg("observability_score"), "transition_score_mean": avg("transition_score") if condition != "instruction_only" else "not_applicable", "unknown_score_mean": avg("unknown_score"), "simplicity_score_mean": avg("simplicity_score"), **cost})
    write_csv(condition_summary, summaries)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# U3 candidate preliminary scoring\n\n" + "\n".join(f"- {row['condition']}: hard-valid/eligible `{row['eligible_count']}/{row['candidate_count']}`, mean score `{row['preliminary_score_mean']}`" for row in summaries) + "\n", encoding="utf-8")
    return {"status": "PASS", "scored_candidate_count": sum(bool(row["eligible"]) for row in rows), "contradiction_count": len(contradiction_rows)}
