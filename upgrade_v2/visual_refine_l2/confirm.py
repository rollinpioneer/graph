"""One-shot fresh-family graph confirmation and predeclared L2R decision gate."""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Any

from .evaluate import paired_family_bootstrap, summarize_executions
from .execute_graph import execute_graphs
from .io import read_csv, read_json, read_jsonl, sha256_file, write_csv, write_json, write_report


def verify_confirmation_lock(selection_lock: Path, family_lock: Path) -> dict[str, Any]:
    selection = read_json(selection_lock); family = read_json(family_lock)
    failures = []
    for path_key, sha_key in (("selected_graph_path", "selected_graph_sha256"), ("predicate_thresholds_path", "predicate_thresholds_sha256"), ("camera_config_path", "camera_config_sha256"), ("family_generation_lock_path", "family_generation_lock_sha256")):
        path = Path(selection[path_key])
        if not path.is_file() or sha256_file(path) != selection[sha_key]:
            failures.append(f"selection lock mismatch: {path_key}")
    if family.get("status") != "LOCKED" or family.get("opened_once") is not True:
        failures.append("fresh family lock is not valid")
    return {"status": "PASS" if not failures else "BLOCKED", "failures": failures, "selected_graph_id": selection.get("selected_graph_id"), "fresh_families": family.get("family_count")}


def evaluate_fresh(graph_paths: list[Path], selection_lock: Path, family_lock: Path, dataset_manifest: Path, prediction_manifest: Path, execution_root: Path, output: Path, paired: Path, per_family: Path, per_scenario: Path, report: Path, bootstrap: int, bootstrap_seed: int) -> dict[str, Any]:
    gate = verify_confirmation_lock(selection_lock, family_lock)
    if gate["status"] != "PASS":
        raise RuntimeError(f"confirmation lock failed: {gate['failures']}")
    executions = execute_graphs(graph_paths, dataset_manifest, prediction_manifest, None, None, execution_root, execution_root / "metrics.csv", None, per_family)
    all_rows = []
    for graph_path in graph_paths:
        graph_id = read_json(graph_path)["graph_id"]
        all_rows.extend(read_jsonl(execution_root / f"{graph_id}.jsonl"))
    metrics_rows = []
    for graph_id in sorted({row["graph_id"] for row in all_rows}):
        summary = summarize_executions([row for row in all_rows if row["graph_id"] == graph_id])
        summary["graph_id"] = graph_id
        metrics_rows.append(summary)
    write_csv(output, metrics_rows)
    selected = read_json(selection_lock)["selected_graph_id"]
    effects = [paired_family_bootstrap(all_rows, selected, baseline, "branch_accuracy", bootstrap, bootstrap_seed + index) for index, baseline in enumerate(("G0_coarse_direct", "G1_predicate_bound"))]
    write_csv(paired, effects)
    metadata = {row["rollout_id"]: row for row in read_csv(dataset_manifest)}
    scenario_rows = []
    for graph_id in sorted({row["graph_id"] for row in all_rows}):
        for scenario in sorted({row["scenario"] for row in metadata.values()}):
            subset = [row for row in all_rows if row["graph_id"] == graph_id and metadata[row["rollout_id"]]["scenario"] == scenario]
            summary = summarize_executions(subset); summary.update(graph_id=graph_id, scenario=scenario)
            scenario_rows.append(summary)
    write_csv(per_scenario, scenario_rows)
    write_report(report, "Fresh Confirmation", [("status", "L2R_FRESH_CONFIRMATION_COMPLETE"), ("families", gate["fresh_families"]), ("selected graph", selected), ("graphs", len(metrics_rows)), ("bootstrap", bootstrap), ("post-confirmation tuning", False)])
    return {"status": "L2R_FRESH_CONFIRMATION_COMPLETE", "selected_graph": selected, "metrics": metrics_rows, "paired_effects": effects}


def decide_status(selection_lock: Path, confirmation: Path, paired_effects: Path, per_scenario: Path) -> tuple[str, list[str]]:
    selection = read_json(selection_lock); selected = selection["selected_graph_id"]
    metrics = {row["graph_id"]: row for row in read_csv(confirmation)}
    row = metrics[selected]
    numeric = {key: float(value) if value not in {"", None} else None for key, value in row.items() if key != "graph_id"}
    effects = {row["comparison"].split("-")[-1]: row for row in read_csv(paired_effects)}
    scenarios = read_csv(per_scenario)
    selected_scenario = {row["scenario"]: float(row["branch_accuracy"]) for row in scenarios if row["graph_id"] == selected}
    g1_scenario = {row["scenario"]: float(row["branch_accuracy"]) for row in scenarios if row["graph_id"] == "G1_predicate_bound"}
    noninferior = sum(selected_scenario[name] + 1e-12 >= g1_scenario[name] for name in selected_scenario)
    reasons = []
    stop = numeric["goal_precision"] is not None and numeric["goal_precision"] < .65 or numeric["false_ready_rate"] is not None and numeric["false_ready_rate"] > .25 or numeric["graph_completion_coverage"] is not None and numeric["graph_completion_coverage"] < .60 or ((numeric["failure_recall"] is None or numeric["failure_recall"] < .40) and (numeric["recovery_recall"] is None or numeric["recovery_recall"] < .40))
    if stop:
        return "STOP_VISUAL_REFINEMENT", ["one or more predeclared stop thresholds were crossed"]
    g0_effect = float(effects["G0_coarse_direct"]["mean_effect"])
    g1_effect = float(effects["G1_predicate_bound"]["mean_effect"])
    go = numeric["branch_accuracy"] >= .80 and numeric["goal_precision"] >= .85 and numeric["false_ready_rate"] <= .10 and numeric["unnecessary_manipulation_rate"] <= .15 and numeric["failure_denominator"] >= 20 and numeric["failure_recall"] >= .65 and numeric["recovery_denominator"] >= 20 and numeric["recovery_recall"] >= .65 and numeric["unknown_rate"] <= .25 and numeric["ambiguous_edge_rate"] <= .10 and numeric["graph_completion_coverage"] >= .80 and g0_effect >= .15 and g1_effect >= .05 and noninferior >= 6
    if go and selected != "G3_active_second_view":
        return "GO_L3_REWARD_GROUNDING_SINGLE_VIEW", [f"all GO thresholds passed; scenario noninferiority={noninferior}/8"]
    if go and selected == "G3_active_second_view" and numeric["second_view_query_rate"] <= .35:
        return "GO_L3_REWARD_GROUNDING_ACTIVE_MULTIVIEW", [f"all active-view GO thresholds passed; query rate={numeric['second_view_query_rate']}"]
    return "L2R_PARTIAL_KEEP_COARSE_GRAPH", [f"GO structural gain gate not met; selected-G0={g0_effect:.6f}, selected-G1={g1_effect:.6f}, noninferior={noninferior}/8"]
