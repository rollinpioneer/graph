"""Create the explicit evidence-state handoff consumed by U4."""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

from .common import read_csv, read_json, repo_root, write_csv, write_json
from .evaluate_graphs import _data


def _split_support_counts(copy_graphs: Path) -> tuple[dict[str, Counter], dict[str, Counter]]:
    """Count split-local segment and transition support without using gold labels."""
    segments, transitions = _data(repo_root(copy_graphs))
    node_counts: dict[str, Counter] = defaultdict(Counter)
    edge_counts: dict[str, Counter] = defaultdict(Counter)
    for row in segments:
        node_counts[str(row.get("split", ""))][int(row["evaluation_cluster_id"])] += 1
    for row in transitions:
        split = str(row.get("split", ""))
        edge_counts[split][(int(row["from_cluster_id"]), int(row["to_cluster_id"]))] += 1
    return node_counts, edge_counts


def _cluster_id(handle: str | None, fallback: object = None) -> int | None:
    if handle and handle.startswith("C") and handle[1:].isdigit():
        return int(handle[1:])
    try:
        return int(fallback) if fallback is not None and str(fallback) != "" else None
    except (TypeError, ValueError):
        return None


def _edge_pair(edge: dict, grounding: dict) -> tuple[int, int] | None:
    source = grounding.get("from_cluster_handle")
    target = grounding.get("to_cluster_handle")
    source_id = _cluster_id(source, grounding.get("from_cluster_id"))
    target_id = _cluster_id(target, grounding.get("to_cluster_id"))
    if source_id is not None and target_id is not None:
        return source_id, target_id
    raw_pair = edge.get("raw_pair")
    if isinstance(raw_pair, list) and len(raw_pair) == 2:
        return int(raw_pair[0]), int(raw_pair[1])
    return None


def build_u4_handoff(*, decision: Path, selected: Path, val_metrics: Path, test_metrics: Path, edge_details: Path, fallback_policy: Path, output: Path, element_table: Path, query_queue: Path, copy_graphs: Path, report: Path) -> dict:
    lock = read_json(decision); selected_rows = read_csv(selected); val = {x["graph_id"]: x for x in read_csv(val_metrics)}; test = {x["graph_id"]: x for x in read_csv(test_metrics)}
    node_counts, edge_counts = _split_support_counts(copy_graphs)
    elements = []; queries = []
    for row in selected_rows:
        gid = row["graph_id"]; graph_path = Path(row.get("path", ""))
        if not graph_path.is_file():
            graph_path = next((p for p in copy_graphs.parent.glob("*.json") if read_json(p).get("graph_id") == gid), graph_path)
        if not graph_path.is_file(): continue
        graph = read_json(graph_path)
        for node in graph.get("nodes", []):
            grounding = node.get("grounding", {}); status = grounding.get("status", node.get("status", "unresolved"))
            status = status if status in {"observed", "unresolved", "contradicted", "hypothesized"} else "unresolved"
            cluster_id = _cluster_id(grounding.get("cluster_handle"), node.get("raw_cluster_id"))
            elements.append({"graph_id": gid, "element_type": "node", "element_id": node["id"], "semantic_source": row.get("source_candidate_id", "data_only"), "status": status, "train_support_count": int(grounding.get("n_segments") or node.get("n_segments", 0) or 0), "train_root_family_support": grounding.get("support_root_families", node.get("support_root_families", "not_derived")), "val_support_count": node_counts["val"].get(cluster_id, 0) if cluster_id is not None else 0, "test_support_count": node_counts["test"].get(cluster_id, 0) if cluster_id is not None else 0, "contradiction_count": 0, "observable_predicates": ";".join(node.get("observable_predicates", [])), "evidence_handles": grounding.get("cluster_handle") or node.get("id", ""), "unresolved_reason": "no train cluster match" if status == "unresolved" else "", "recommended_u4_query": "existing continuation lookup" if status == "unresolved" else "none"})
        for edge in graph.get("edges", []):
            grounding = edge.get("grounding", {}); status = grounding.get("status", edge.get("status", "unresolved"))
            status = status if status in {"observed", "unresolved", "contradicted", "hypothesized"} else "unresolved"
            query = "none"
            if edge.get("hypothesized_type") in {"failure", "recovery", "alternative"}: query = "new simulator continuation" if status == "observed" else "fallback clip review"
            if row.get("source_candidate_id") == "data_only" and status == "unresolved": query = "train/val transition contradiction check"
            pair = _edge_pair(edge, grounding)
            elements.append({"graph_id": gid, "element_type": "edge", "element_id": edge["id"], "semantic_source": row.get("source_candidate_id", "data_only"), "status": status, "train_support_count": int(grounding.get("transition_count") or edge.get("transition_count", 0) or 0), "train_root_family_support": grounding.get("support_root_families", edge.get("support_root_families", "not_derived")), "val_support_count": edge_counts["val"].get(pair, 0) if pair is not None else 0, "test_support_count": edge_counts["test"].get(pair, 0) if pair is not None else 0, "contradiction_count": 0, "observable_predicates": ";".join(edge.get("effects", [])), "evidence_handles": grounding.get("transition_handle") or edge.get("id", ""), "unresolved_reason": "no observed transition handle" if status == "unresolved" else "", "recommended_u4_query": query})
            if query != "none": queries.append({"priority": "high" if edge.get("hypothesized_type") in {"failure", "recovery"} else "medium", "graph_id": gid, "element_id": edge["id"], "query_type": query, "reason": "branch structure needs independent U4 evidence"})
        target = copy_graphs / f"{gid}.json"; target.parent.mkdir(parents=True, exist_ok=True); target.write_text(graph_path.read_text(encoding="utf-8"), encoding="utf-8")
    write_csv(element_table, elements); write_csv(query_queue, queries)
    policy = read_json(fallback_policy); qwen_count = sum(row.get("source_candidate_id", "").startswith("qwen:") for row in selected_rows)
    decision_name = "GO_U4_CANDIDATE_VALIDATION" if qwen_count else "GO_U4_DATA_ONLY" if any(row.get("source_candidate_id") == "data_only" for row in selected_rows) else "STOP_GRAPH_AUTOMATION"
    handoff = {"schema": "u4_graph_validation_handoff_v1", "decision": decision_name, "source_u3_v1_decision": "U3_INCONCLUSIVE", "selected_graphs": [row["graph_id"] for row in selected_rows], "elements": elements, "query_queue": queries, "boundary_fallback_required": True, "fallback_policy": policy, "scope": "same stochastic simulator family only", "physical_generalization_eligible": False, "original_task_generalization_eligible": False, "all_u3g_statuses": ["hypothesized", "observed", "unresolved", "contradicted"], "empirically_validated_by_u3g": False, "new_external_llm_calls": 0, "test_run_once": True, "support_count_definition": {"node": "number of episode-local segments assigned to the cluster; train uses the frozen train catalog", "edge": "number of episode-local adjacent segment transitions matching the endpoint cluster pair"}, "test_not_used_for_selection": True}
    write_json(output, handoff); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# U3 grounding bridge final report\n\n" + "\n".join([f"- Decision: `{decision_name}`", "- U3 v1: `U3_INCONCLUSIVE`", f"- Selected graphs: `{len(selected_rows)}`", f"- Observed elements: `{sum(x['status'] == 'observed' for x in elements)}`", f"- Unresolved elements: `{sum(x['status'] == 'unresolved' for x in elements)}`", f"- U4 query queue: `{len(queries)}`", "- empirically_validated assigned by U3-G: `false`"]) + "\n", encoding="utf-8")
    return {"status": "U4_HANDOFF_READY", "decision": decision_name, "selected_count": len(selected_rows), "element_count": len(elements), "query_count": len(queries)}
