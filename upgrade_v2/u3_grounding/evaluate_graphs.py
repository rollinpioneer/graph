"""Evaluate grounded graphs on one split, with root-family bootstrap summaries."""

from __future__ import annotations

import random
import math
from collections import defaultdict
from pathlib import Path

from .common import read_csv, read_json, read_jsonl, repo_root, safe_float, write_csv


def _jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a | b else 1.0


EVENT_NAMES = ["none", "contact_on", "transport_on", "contact_off_failure", "recovery_start", "contact_reestablished", "detour_start", "goal_enter", "stable_success", "terminal_failure", "stagnation_onset"]


def _observable_predicates(row: dict) -> set[str]:
    raw = row.get("raw_mean", [])
    predicates = {"contact_present" if safe_float(raw[12]) >= 0.5 else "contact_absent"} if len(raw) > 12 else set()
    if len(raw) > 7:
        moving = math.hypot(safe_float(raw[6]), safe_float(raw[7])) >= 0.06
        predicates.add("object_moving" if moving else "object_stationary")
    if len(raw) > 14 and safe_float(raw[14]) >= 0.5:
        predicates.add("object_inside_goal")
    if len(raw) > 13 and safe_float(raw[13]) >= 0.5:
        predicates.add("collision_detected")
    if int(row.get("observable_contact_loss_count", 0) or 0) > 0:
        predicates.add("contact_recently_lost")
    if safe_float(row.get("unknown_fraction", 0.0)) > 0.25:
        predicates.add("progress_unknown")
    return predicates or {"progress_unknown"}


def _events(row: dict) -> set[str]:
    posterior = row.get("event_posterior_mean", [])
    if not posterior: return set()
    order = sorted(range(len(posterior)), key=lambda i: safe_float(posterior[i]), reverse=True)[:3]
    return {EVENT_NAMES[i] for i in order if i < len(EVENT_NAMES)}


def _reference_clusters(repo: Path) -> dict[int, dict]:
    path = repo / "artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph/inputs_v1/segment_event_summary_train.jsonl"
    rows = read_jsonl(path)
    grouped: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[int(row["cluster_id"])].append(row)
    refs = {}
    for cid, items in grouped.items():
        event_len = max((len(x.get("event_posterior_mean", [])) for x in items), default=0)
        raw_len = max((len(x.get("raw_mean", [])) for x in items), default=0)
        event = [sum(safe_float(x.get("event_posterior_mean", [])[i]) for x in items) / len(items) for i in range(event_len)]
        raw = [sum(safe_float(x.get("raw_mean", [])[i]) for x in items) / len(items) for i in range(raw_len)]
        refs[cid] = {"predicates": set().union(*(_observable_predicates(x) for x in items)), "event": event, "raw": raw}
    return refs


def _assign_cluster(row: dict, refs: dict[int, dict]) -> int:
    if "cluster_id" in row:
        return int(row["cluster_id"])
    row_event = row.get("event_posterior_mean", [])
    row_raw = row.get("raw_mean", [])
    predicates = _observable_predicates(row)
    best = None
    for cid, ref in refs.items():
        predicate_score = _jaccard(predicates, ref["predicates"])
        event_score = 0.0
        if row_event and ref["event"]:
            event_score = sum(min(safe_float(row_event[i]), ref["event"][i]) for i in range(min(len(row_event), len(ref["event"]))))
        raw_score = 0.0
        if row_raw and ref["raw"]:
            distance = math.sqrt(sum((safe_float(row_raw[i]) - ref["raw"][i]) ** 2 for i in range(min(len(row_raw), len(ref["raw"])))))
            raw_score = 1.0 / (1.0 + distance)
        score = 0.45 * predicate_score + 0.35 * event_score + 0.20 * raw_score
        key = (score, -cid)
        if best is None or key > best[0]:
            best = (key, cid)
    return best[1] if best else 0


def _data(repo: Path) -> tuple[list[dict], list[dict]]:
    path = repo / "artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/segment_representation_v1/segments/segments.jsonl"
    segments = read_jsonl(path); refs = _reference_clusters(repo); transitions = []
    for row in segments:
        row["evaluation_cluster_id"] = _assign_cluster(row, refs)
    episodes: dict[str, list[dict]] = defaultdict(list)
    for row in segments: episodes[row["episode_id"]].append(row)
    for rows in episodes.values():
        rows.sort(key=lambda r: (int(r["start_t"]), int(r["end_t"])))
        for left, right in zip(rows, rows[1:]):
            if left.get("split") == right.get("split"):
                transitions.append({"split": left.get("split"), "root_family_id": left.get("root_family_id"), "from_cluster_id": int(left["evaluation_cluster_id"]), "to_cluster_id": int(right["evaluation_cluster_id"]), "events_before": _events(left), "events_after": _events(right)})
    return segments, transitions


def _cluster_id(handle: object, fallback: object = None) -> int | None:
    value = str(handle or "")
    if value.startswith("C") and value[1:].isdigit():
        return int(value[1:])
    try:
        return int(fallback) if fallback is not None and str(fallback) != "" else None
    except (TypeError, ValueError):
        return None


def _element_status(element: dict) -> str:
    grounding = element.get("grounding") or {}
    return str(grounding.get("status") or element.get("status") or "unresolved")


def _reachable(graph: dict) -> bool:
    nodes = graph.get("nodes", [])
    starts = {graph["start_cluster_handle"]} if graph.get("start_cluster_handle") else {n["id"] for n in nodes if n.get("role") == "start"}
    successes = {graph["success_cluster_handle"]} if graph.get("success_cluster_handle") else {n["id"] for n in nodes if n.get("role") == "success_terminal"}
    if not starts or not successes:
        return False
    adjacency: dict[str, set[str]] = defaultdict(set)
    for edge in graph.get("edges", []):
        if _element_status(edge) not in {"contradicted", "contradicted_on_train"}:
            if edge.get("src") is not None and edge.get("dst") is not None:
                adjacency[str(edge["src"])].add(str(edge["dst"]))
    seen = {str(x) for x in starts}; frontier = list(seen)
    while frontier:
        current = frontier.pop()
        for target in adjacency.get(current, set()):
            if target not in seen:
                seen.add(target); frontier.append(target)
    return bool(seen & {str(x) for x in successes})


def _predicate_fit(graph: dict, segments: list[dict], refs: dict[int, dict], split: str) -> float:
    available = {int(s["evaluation_cluster_id"]) for s in segments if s.get("split") == split}
    scores = []
    for node in graph.get("nodes", []):
        grounding = node.get("grounding") or {}
        cid = _cluster_id(grounding.get("cluster_handle"), grounding.get("raw_cluster_id", node.get("raw_cluster_id")))
        if cid is None or cid not in available:
            continue
        declared = set(node.get("observable_predicates", []))
        expected = set(refs.get(cid, {}).get("predicates", set()))
        if declared or expected:
            scores.append(_jaccard(declared, expected))
    return sum(scores) / len(scores) if scores else 0.0


def _graph_metrics(graph: dict, segments: list[dict], transitions: list[dict], refs: dict[int, dict], split: str, bootstrap: int, seed: int) -> dict:
    segs = [x for x in segments if x.get("split") == split]; trans = [x for x in transitions if x.get("split") == split]
    edges = graph.get("edges", []); nodes = graph.get("nodes", [])
    graph_pairs = set()
    for edge in edges:
        grounding = edge.get("grounding", {})
        if grounding.get("transition_handle"):
            graph_pairs.add((int(grounding["from_cluster_handle"][1:]), int(grounding["to_cluster_handle"][1:])))
        elif edge.get("raw_pair"):
            graph_pairs.add(tuple(map(int, edge["raw_pair"])))
    # Handles are assigned in raw cluster order, so the numeric suffix is stable.
    explained = [t for t in trans if (t["from_cluster_id"], t["to_cluster_id"]) in graph_pairs]
    total = len(trans); transition_coverage = len(explained) / total if total else 0.0
    predicate_fit = _predicate_fit(graph, segments, refs, split)
    recovery_den = sum("recovery_start" in t["events_after"] or "contact_off_failure" in t["events_before"] for t in trans)
    failure_den = sum(bool({"contact_off_failure", "terminal_failure", "stagnation_onset"} & t["events_after"]) for t in trans)
    recovery_pairs = set()
    failure_pairs = set()
    for edge in edges:
        grounding = edge.get("grounding", {})
        pair = None
        if grounding.get("transition_handle"):
            pair = (int(grounding["from_cluster_handle"][1:]), int(grounding["to_cluster_handle"][1:]))
        elif edge.get("raw_pair"):
            pair = tuple(map(int, edge["raw_pair"]))
        if pair and edge.get("hypothesized_type") == "recovery": recovery_pairs.add(pair)
        if pair and edge.get("hypothesized_type") in {"failure", "alternative"}: failure_pairs.add(pair)
    recovery_recall = sum((t["from_cluster_id"], t["to_cluster_id"]) in recovery_pairs and ("recovery_start" in t["events_after"] or "contact_off_failure" in t["events_before"]) for t in trans) / recovery_den if recovery_den else 0.0
    failure_recall = sum((t["from_cluster_id"], t["to_cluster_id"]) in failure_pairs and bool({"contact_off_failure", "terminal_failure", "stagnation_onset"} & t["events_after"]) for t in trans) / failure_den if failure_den else 0.0
    statuses = [_element_status(edge) for edge in edges]
    unresolved = sum(status == "unresolved" for status in statuses)
    contradicted = sum(status in {"contradicted", "contradicted_on_train"} for status in statuses)
    unknown_honesty = unresolved / len(edges) if edges else 1.0
    reachable = _reachable(graph)
    family_values = []
    families = sorted({t["root_family_id"] for t in trans})
    for family in families:
        ft = [t for t in trans if t["root_family_id"] == family]; family_values.append(sum((t["from_cluster_id"], t["to_cluster_id"]) in graph_pairs for t in ft) / len(ft) if ft else 0.0)
    rng = random.Random(seed); samples = []
    if family_values:
        for _ in range(min(bootstrap, 5000)): samples.append(sum(rng.choice(family_values) for _ in family_values) / len(family_values))
    samples.sort(); low = samples[int(0.025 * (len(samples) - 1))] if samples else 0.0; high = samples[int(0.975 * (len(samples) - 1))] if samples else 0.0
    complexity = len(nodes) + len(edges); complexity_penalty = min(1.0, max(0, complexity - 10) / 40)
    score = 0.25 * transition_coverage + 0.20 * predicate_fit + 0.20 * recovery_recall + 0.15 * failure_recall + 0.10 * float(reachable) + 0.10 * unknown_honesty - 0.10 * complexity_penalty
    return {"graph_id": graph.get("graph_id", "data_only_transition_graph"), "source_candidate_id": graph.get("source_candidate_id", "data_only"), "split": split, "root_family_count": len(families), "transition_count": total, "explained_transition_mass": round(transition_coverage, 8), "predicate_fit": round(predicate_fit, 8), "recovery_edge_recall": round(recovery_recall, 8), "failure_edge_recall": round(failure_recall, 8), "start_success_reachable": reachable, "contradicted_edge_rate": round(contradicted / len(edges), 8) if edges else 0.0, "unresolved_edge_rate": round(unresolved / len(edges), 8) if edges else 0.0, "unknown_honesty": round(unknown_honesty, 8), "graph_complexity": complexity, "complexity_penalty": round(complexity_penalty, 8), "value_consistent_edge_direction": "not_measured", "root_family_bootstrap_resamples": min(bootstrap, 5000), "explained_transition_mass_ci_low": round(low, 8), "explained_transition_mass_ci_high": round(high, 8), "score": round(score, 8), "physical_generalization_eligible": False}


def evaluate_graphs(*, grounded_graph_root: Path | None, data_only_graph: Path, selected: Path | None, dataset: Path, split: str, statistics_unit: str, protocol: Path, output: Path, details: Path, report: Path, bootstrap: int = 5000, seed: int = 20261011) -> dict:
    repo = repo_root(dataset); segments, transitions = _data(repo); refs = _reference_clusters(repo); paths = []
    if selected and selected.is_file():
        for row in read_csv(selected):
            if row.get("selected", "").lower() in {"true", "1"} and row.get("path"): paths.append(Path(row["path"]))
    elif grounded_graph_root:
        paths.extend(sorted(grounded_graph_root.glob("*.json")))
    if data_only_graph.is_file() and not selected: paths.append(data_only_graph)
    rows = []
    for path in paths:
        row = _graph_metrics(read_json(path), segments, transitions, refs, split, bootstrap, seed)
        row["path"] = str(path)
        rows.append(row)
    details_rows = [{"graph_id": row["graph_id"], "split": split, "detail": "metrics computed from episode-local segment transitions", "evaluation_role": "final_evaluation_only" if split == "test" else "validation_for_selection", "test_gold_used_for_selection": False, "test_not_used_for_selection": True} for row in rows]
    write_csv(output, rows); write_csv(details, details_rows); report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# Graph evaluation\n\n" + "\n".join([f"- split: `{split}`", f"- graph_count: `{len(rows)}`", f"- statistics_unit: `{statistics_unit}`", f"- test gold used for threshold selection: `{False if split == 'val' else 'not_applicable'}`"]) + "\n", encoding="utf-8")
    return {"status": "PASS", "split": split, "graph_count": len(rows), "test_gold_used_for_selection": False if split == "val" else "not_applicable"}
