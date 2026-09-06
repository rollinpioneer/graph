"""Deterministically score semantic nodes against train cluster profiles."""

from pathlib import Path

from .common import read_json, safe_float, write_csv, write_json


def _jaccard(left: set[str], right: set[str]) -> float:
    return len(left & right) / len(left | right) if left | right else 1.0


def _role_score(role: str, profile: dict) -> float:
    events = {row["event"]: safe_float(row.get("posterior")) for row in profile.get("top_event_posterior", [])}
    if role == "start": return 1.0 if "contact_on" in events or "none" in events else 0.25
    if role == "success_terminal": return max(events.get("goal_enter", 0.0), events.get("stable_success", 0.0), 0.1 if "object_inside_goal" in profile.get("top_predicates", []) else 0.0)
    if role == "failure_terminal": return max(events.get("terminal_failure", 0.0), events.get("stagnation_onset", 0.0), events.get("contact_off_failure", 0.0), 0.1 if "collision_detected" in profile.get("top_predicates", []) else 0.0)
    return 0.8


def _history_score(predicates: set[str], profile: dict) -> float:
    events = {row["event"] for row in profile.get("top_event_posterior", [])}
    score = 0.5
    if "contact_recently_lost" in predicates: score += 0.4 if "contact_off" in events or "recovery_start" in events else -0.1
    if "contact_reestablished" in predicates: score += 0.4 if "recovery_start" in events else 0.0
    if "stagnation_detected" in predicates: score += 0.3 if "stagnation_onset" in events else 0.0
    return max(0.0, min(1.0, score))


def ground_nodes(*, candidate_dir: Path, cluster_profiles: Path, thresholds: list[float], top_k: int, output_dir: Path, table: Path) -> dict:
    profiles = read_json(cluster_profiles)["profiles"]
    output_dir.mkdir(parents=True, exist_ok=True); rows = []
    for path in sorted(candidate_dir.glob("*.json")):
        cid = f"qwen:{path.stem}"; candidate = read_json(path); result = {"schema": "u3_node_grounding_v1", "candidate_id": cid, "nodes": []}
        for node in candidate["nodes"]:
            predicates = set(node.get("observable_predicates", [])); scored = []
            for profile in profiles:
                predicate_score = _jaccard(predicates, set(profile.get("top_predicates", [])))
                role_score = _role_score(node.get("role", "unknown"), profile)
                history_score = _history_score(predicates, profile)
                family_score = min(1.0, safe_float(profile.get("support_root_families")) / 84.0)
                score = 0.50 * predicate_score + 0.20 * role_score + 0.15 * history_score + 0.15 * family_score
                scored.append({"cluster_handle": profile["cluster_handle"], "raw_cluster_id": profile["raw_cluster_id"], "score": round(score, 8), "predicate_score": round(predicate_score, 8), "role_score": round(role_score, 8), "history_score": round(history_score, 8), "family_support_score": round(family_score, 8)})
            scored.sort(key=lambda row: (-row["score"], row["cluster_handle"])); top = scored[:top_k]
            result["nodes"].append({"node_id": node["id"], "candidates": top})
            for item in top:
                rows.append({"candidate_id": cid, "node_id": node["id"], **item, "thresholds": ";".join(str(x) for x in thresholds), "grounding_status": "grounded_candidate" if item["score"] >= min(thresholds) else "unresolved_no_match"})
        write_json(output_dir / path.name, result)
    write_csv(table, rows)
    return {"status": "PASS", "candidate_count": len(list(candidate_dir.glob("*.json"))), "node_candidate_count": len(rows), "top_k": top_k}
