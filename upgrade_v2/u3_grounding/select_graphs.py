"""Select graphs using validation only, before the single test evaluation."""

from pathlib import Path

from .common import read_csv, read_json, sha256_file, write_csv, write_json


def select_graphs(*, val_metrics: Path, require_start_success_reachable: bool, max_contradicted_edge_rate: float, max_unresolved_edge_rate: float, one_per_source: bool, output: Path, lock: Path, report: Path) -> dict:
    rows = read_csv(val_metrics); selected = []
    for row in rows:
        reachable = row.get("start_success_reachable", "").lower() in {"true", "1"}
        if require_start_success_reachable and not reachable: continue
        if float(row.get("contradicted_edge_rate", 1)) > max_contradicted_edge_rate or float(row.get("unresolved_edge_rate", 1)) > max_unresolved_edge_rate: continue
        if row.get("source_candidate_id") == "data_only" or float(row.get("recovery_edge_recall", 0)) > 0 or float(row.get("failure_edge_recall", 0)) > 0: selected.append(row)
    selected.sort(key=lambda r: (-float(r.get("score", 0)), r.get("graph_id", "")))
    chosen = []
    sources = set()
    for row in selected:
        source = row.get("source_candidate_id", "data_only")
        if one_per_source and source in sources: continue
        chosen.append(row); sources.add(source)
    write_csv(output, [{**row, "selected": True, "selection_reason": "validation_locked"} for row in chosen])
    lock_value = {"schema": "u3_selection_lock_v1", "selection_split": "val", "test_evaluation_not_used": True, "selected_count": len(chosen), "selected_graph_ids": [x["graph_id"] for x in chosen], "criteria": {"require_start_success_reachable": require_start_success_reachable, "max_contradicted_edge_rate": max_contradicted_edge_rate, "max_unresolved_edge_rate": max_unresolved_edge_rate}, "sha256": sha256_file(output)}
    write_json(lock, lock_value)
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Validation graph selection\n\n" + "\n".join([f"- selected: `{len(chosen)}`", "- selection split: `val`", "- test used for selection: `false`", f"- selected graph IDs: `{', '.join(x['graph_id'] for x in chosen) or 'none'}`"]) + "\n", encoding="utf-8")
    qwen = [x for x in chosen if x.get("source_candidate_id", "").startswith("qwen:")]
    data_only = any(x.get("source_candidate_id") == "data_only" for x in chosen)
    decision = "GO_U4_CANDIDATE_VALIDATION" if qwen else "GO_U4_DATA_ONLY" if data_only else "STOP_GRAPH_AUTOMATION"
    return {"status": decision, "selected_count": len(chosen), "qwen_selected": len(qwen), "data_only_selected": data_only}
