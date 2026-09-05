"""Deterministic candidate selection and conservative U4 handoff generation."""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from .common import read_csv, read_json, write_csv, write_json
from .stability import graph_distance


def _candidate_paths(roots: list[Path]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for root in roots:
        provider = root.parent.name
        for path in root.glob("*.json"):
            output[f"{provider}:{path.stem}"] = path
    return output


def _float(value: str | Any) -> float:
    return float(value)


def select_candidates(
    *, scores: Path, stability: Path, pairwise_distance: Path, cross_provider: Path,
    candidate_roots: list[Path], max_extra_diverse: int, diversity_min_distance: float,
    diversity_max_score_drop: float, output: Path, copy_dir: Path, report: Path,
) -> dict[str, Any]:
    rows = read_csv(scores)
    paths = _candidate_paths(candidate_roots)
    selected: list[dict[str, Any]] = []
    by_condition: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if row["provider"] == "qwen" and str(row["eligible"]).lower() in {"true", "1"}:
            by_condition.setdefault(row["condition"], []).append(row)
    for condition in ("instruction_only", "instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"):
        candidates = sorted(by_condition.get(condition, []), key=lambda row: (-_float(row["preliminary_score"]), row["candidate_id"]))
        if not candidates:
            continue
        row = dict(candidates[0])
        row.update({"selection_reason": "highest_scoring_hard_valid_qwen_for_condition", "selection_rank": len(selected) + 1, "selected": True, "is_diverse_extra": False})
        selected.append(row)
    required_conditions = {"instruction_only", "instruction_plus_auto_train_segments", "instruction_plus_budgeted_train_fallback"}
    selected_conditions = {row["condition"] for row in selected}
    missing_conditions = sorted(required_conditions - selected_conditions)
    if missing_conditions:
        # A candidate pool with a missing input condition is not silently
        # narrowed.  Emit an explicit terminal-status row and do not hand any
        # graph to U4 as if it were a balanced 3-condition comparison.
        terminal_rows = [{"candidate_id": "", "condition": condition, "provider": "qwen", "eligible": False, "selected": False, "selection_reason": "U3_INCONCLUSIVE_no_hard_valid_qwen_candidate_after_allowed_refinement", "selection_rank": "", "is_diverse_extra": False, "selected_candidate_path": ""} for condition in missing_conditions]
        write_csv(output, terminal_rows)
        report.parent.mkdir(parents=True, exist_ok=True)
        report.write_text("# U3 candidate selection\n\n- Decision: `U3_INCONCLUSIVE`\n- Missing hard-valid Qwen conditions after the single allowed refinement: `" + ", ".join(missing_conditions) + "`.\n- No candidate is handed to U4.\n", encoding="utf-8")
        return {"status": "U3_INCONCLUSIVE", "selected_count": 0, "missing_conditions": missing_conditions, "extra_diverse_count": 0}
    selected_ids = {row["candidate_id"] for row in selected}
    extras: list[dict[str, str]] = []
    all_eligible = [row for row in rows if str(row["eligible"]).lower() in {"true", "1"} and row["candidate_id"] not in selected_ids]
    for candidate in sorted(all_eligible, key=lambda row: (-_float(row["preliminary_score"]), row["candidate_id"])):
        candidate_path = paths.get(candidate["candidate_id"])
        if candidate_path is None:
            continue
        candidate_graph = read_json(candidate_path)
        distances = []
        for chosen in selected:
            chosen_path = paths.get(chosen["candidate_id"])
            if chosen_path:
                distances.append(graph_distance(candidate_graph, read_json(chosen_path))["graph_distance"])
        best_same_condition = next((row for row in selected if row["condition"] == candidate["condition"]), None)
        baseline = _float(best_same_condition["preliminary_score"]) if best_same_condition else _float(candidate["preliminary_score"])
        if distances and min(distances) >= diversity_min_distance and _float(candidate["preliminary_score"]) >= baseline - diversity_max_score_drop:
            extras.append(candidate)
            if len(extras) >= max_extra_diverse:
                break
    for candidate in extras:
        row = dict(candidate)
        row.update({"selection_reason": "structurally_diverse_hard_valid_candidate", "selection_rank": len(selected) + 1, "selected": True, "is_diverse_extra": True})
        selected.append(row)
    copy_dir.mkdir(parents=True, exist_ok=True)
    for row in selected:
        source = paths[row["candidate_id"]]
        target = copy_dir / source.name
        shutil.copy2(source, target)
        row["selected_candidate_path"] = str(target)
    write_csv(output, selected)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# U3 candidate selection\n\n" + "\n".join(f"- `{row['candidate_id']}` — {row['selection_reason']}" for row in selected) + "\n", encoding="utf-8")
    return {"status": "PASS" if len(selected) in {3, 4} else "U3_INCONCLUSIVE", "selected_count": len(selected), "extra_diverse_count": len(extras), "missing_conditions": []}


def build_u4_handoff(
    *, selected: Path, candidate_dir: Path, task_contract: Path, predicate_vocabulary: Path,
    supervision_ledger: Path, fallback_policy: Path, contradiction_queue: Path,
    output: Path, report: Path,
) -> dict[str, Any]:
    selected_rows = [row for row in read_csv(selected) if str(row.get("selected", "")).lower() in {"true", "1"} and row.get("candidate_id")]
    selected_graphs: list[dict[str, Any]] = []
    for row in selected_rows:
        candidate_path = candidate_dir / Path(row["selected_candidate_path"]).name
        graph = read_json(candidate_path)
        selected_graphs.append({"candidate_id": row["candidate_id"], "condition": row["condition"], "provider": row["provider"], "preliminary_score": row["preliminary_score"], "selection_reason": row["selection_reason"], "graph": graph})
    policy = read_json(fallback_policy)
    decision = "GO_U4_CANDIDATE_VALIDATION" if len(selected_graphs) in {3, 4} else "U3_INCONCLUSIVE"
    handoff = {
        "schema": "u3_to_u4_candidate_handoff_v1",
        "u3_decision": decision,
        "scope": "same stochastic simulator family only",
        "primary_model": "qwen3.7-plus", "crosscheck_model": "deepseek-v4-flash",
        "selected_candidates": selected_graphs, "all_statuses": "hypothesized",
        "boundary_fallback_required": bool(not policy["automatic_boundary_supported"]),
        "unknown_retained": True, "test_gold_used_for_generation": False,
        "physical_generalization_eligible": False, "original_task_generalization_eligible": False,
        "task_contract_path": str(task_contract), "predicate_vocabulary_path": str(predicate_vocabulary),
        "supervision_cost_ledger": read_json(supervision_ledger),
        "fallback_policy": policy,
        "contradiction_queue_path": str(contradiction_queue),
        "allowed_next": (["U4 train/validation trajectory support checks", "U4 continuation validation", "U4 contradicted-edge removal", "U4 budgeted active confirmation"] if decision == "GO_U4_CANDIDATE_VALIDATION" else ["preserve U2/U3 artifacts", "review failed candidate evidence grounding before a future separately-authorized U3 run"]),
    }
    write_json(output, handoff)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# U3 final candidate handoff\n\n" + "\n".join([f"- Decision: `{decision}`", "- Scope: same stochastic simulator family only", f"- Selected candidates: `{len(selected_graphs)}`", "- All graph elements remain `hypothesized`.", ("- U4 must validate or contradict candidate edges; U3 does not promote them." if decision == "GO_U4_CANDIDATE_VALIDATION" else "- No graph is handed to U4 because one or more conditions has no hard-valid Qwen candidate.")]) + "\n", encoding="utf-8")
    # Final lightweight tables are inferred from sibling validation artifacts.
    final_tables = output.parent / "tables"
    final_tables.mkdir(parents=True, exist_ok=True)
    root = output.parents[1]
    scores = root / "candidate_validation_v1" / "scores" / "candidate_scores.csv"
    condition_summary = root / "candidate_validation_v1" / "scores" / "condition_summary.csv"
    agreement = root / "candidate_validation_v1" / "stability" / "qwen_deepseek_agreement.csv"
    if scores.is_file():
        write_csv(final_tables / "u3_candidate_summary.csv", read_csv(scores))
    if condition_summary.is_file():
        write_csv(final_tables / "u3_condition_comparison.csv", read_csv(condition_summary))
    if agreement.is_file():
        write_csv(final_tables / "u3_provider_agreement.csv", read_csv(agreement))
    ledger = read_json(supervision_ledger)
    write_csv(final_tables / "u3_supervision_cost.csv", [{"condition": key, **value} for key, value in ledger.items()])
    write_csv(final_tables / "u3_selected_candidates.csv", selected_rows)
    return handoff
