"""Train/development semantic summaries for nodes and transitions."""
from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any

from .io import read_csv, read_json, read_jsonl, safe_float, write_csv, write_json, write_jsonl


EVENT_TO_SEMANTIC = {
    "contact_off_failure": "failure_event", "recovery_start": "recovery_attempt",
    "contact_reestablished": "recovery_achieved", "transport_on": "progress",
    "stagnation_onset": "dwell", "stable_success": "terminal_success",
}


def _macro(rows: list[dict[str, Any]], predicate) -> tuple[float | None, int]:
    by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if predicate(row):
            by_family[str(row.get("root_family_id", ""))].append(row)
    values = []
    for family, items in by_family.items():
        values.append(sum(1 for x in items if x.get("label")) / len(items))
    return (sum(values) / len(values), len(values)) if values else (None, 0)


def propose_semantics(graph_path, occurrences_path, node_table, edge_table, graph_out, report) -> dict[str, Any]:
    graph = read_json(graph_path)
    legacy = read_jsonl(occurrences_path)
    node_rows = []
    for node in graph.get("nodes", []):
        cid = int(node.get("raw_cluster_id", str(node["id"])[1:]))
        matches = [x for x in legacy if x.get("src_cluster_id") == cid or x.get("dst_cluster_id") == cid]
        families = sorted({x.get("root_family_id") for x in matches if x.get("root_family_id")})
        node_rows.append({"node_id": node["id"], "raw_cluster_id": cid, "legacy_occurrences": len(matches), "eligible_family_count": len(families), "role_source": "legacy_topology_only", "role": node.get("role", "intermediate"), "semantic_status": "unknown", "observable_predicates": ";".join(node.get("observable_predicates", [])), "counterexample_ids": ""})
    edge_rows = []
    for edge in graph.get("edges", []):
        pair = tuple(edge.get("raw_pair", [int(str(edge["src"])[1:]), int(str(edge["dst"])[1:])]))
        matches = [x for x in legacy if tuple(x.get("transition_pair", [])) == pair]
        semantics = Counter()
        for item in matches:
            for label in item.get("legacy_events", []):
                if label in EVENT_TO_SEMANTIC: semantics[EVENT_TO_SEMANTIC[label]] += 1
        families = sorted({x.get("root_family_id") for x in matches if x.get("root_family_id")})
        top = semantics.most_common(1)[0][0] if semantics else "unknown"
        status = "proposed" if len(families) >= 3 and (semantics and semantics.most_common(1)[0][1] / max(1, len(matches)) >= .8) else "mixed/unresolved"
        edge_rows.append({"edge_id": edge["id"], "src": edge["src"], "dst": edge["dst"], "raw_pair": list(pair), "occurrence_count": len(matches), "eligible_family_count": len(families), "proposed_semantic": top, "semantic_status": status, "failure_event_count": semantics["failure_event"], "recovery_attempt_count": semantics["recovery_attempt"], "recovery_achieved_count": semantics["recovery_achieved"], "dwell_count": semantics["dwell"], "unknown_count": max(0, len(matches) - sum(semantics.values())), "provenance": "legacy_train_cache"})
    write_csv(node_table, node_rows)
    write_csv(edge_table, edge_rows)
    out = dict(graph)
    out["schema"] = "u4b_g1_pre_diagnosis_v1"
    out["semantic_evidence"] = {"nodes": node_rows, "edges": edge_rows, "source": "legacy_train_only"}
    write_json(graph_out, out)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("# Semantic proposals\n\n- semantic labels are hypotheses; no empirically_validated status assigned.\n- unresolved and mixed occurrences are retained.\n", encoding="utf-8")
    return {"status": "PASS", "node_count": len(node_rows), "edge_count": len(edge_rows)}


def plan_queries(original_queue, nodes, edges, occurrences, output, max_query_classes=6) -> dict[str, Any]:
    edge_rows = read_csv(edges)
    node_rows = read_csv(nodes)
    selected = [{"query_id": "T028", "priority": 1, "target": "C05->C08", "query_type": "transition_semantics", "reason": "explicitly queued rare edge"}, {"query_id": "T029", "priority": 2, "target": "C04->C04", "query_type": "dwell_or_repeat_boundary", "reason": "explicitly queued self-loop"}]
    classes = ["termination_role_conflict", "failure_recovery_mix", "history_dependent_success"]
    for index, kind in enumerate(classes, 3):
        if len(selected) >= max_query_classes: break
        selected.append({"query_id": f"Q{index:02d}", "priority": index, "target": "C04/C05/C08/C10", "query_type": kind, "reason": "high-impact semantic ambiguity"})
    write_jsonl(output, selected)
    return {"status": "PASS", "query_classes": len(selected), "not_encountered": [x["query_id"] for x in selected if x["query_id"] in {"T028", "T029"} and not any(row.get("edge_id") == x["query_id"] for row in edge_rows)]}


def fit_final_semantics(source_graph, legacy_path, dev_occurrences_path, output) -> dict[str, Any]:
    graph = read_json(source_graph)
    legacy = read_jsonl(legacy_path)
    dev = read_jsonl(dev_occurrences_path) if dev_occurrences_path and dev_occurrences_path.is_file() else []
    by_pair: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in legacy:
        pair = tuple(row.get("transition_pair", []))
        if len(pair) == 2: by_pair[pair].append(row)
    for row in dev:
        pair = tuple(row.get("transition_pair", []))
        if row.get("split") == "dev_fit" and len(pair) == 2:
            by_pair[pair].append(row)
    result = dict(graph); result["schema"] = "u4b_g1_semantic_only_v1"; result["fit_splits"] = ["train", "dev_fit"]
    for edge in result.get("edges", []):
        pair = tuple(edge.get("raw_pair", [])); items = by_pair.get(pair, [])
        counts = Counter(label for x in items for label in (x.get("proposed_semantics") or ["unknown"]))
        edge["semantic_distribution"] = dict(counts)
        edge["semantic_type"] = counts.most_common(1)[0][0] if counts else "unknown"
        edge["semantic_status"] = "mixed" if len([k for k in counts if k != "unknown"]) > 1 else "unknown" if not counts or set(counts) == {"unknown"} else "proposed"
        edge["semantic_provenance"] = "train_dev_fit_only"
    write_json(output, result)
    return {"status": "PASS", "dev_fit_occurrences": len(dev), "edge_count": len(result.get("edges", []))}
