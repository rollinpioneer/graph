"""Ground semantic edges to observed train transition handles."""

from pathlib import Path

from .common import read_json, safe_float, write_csv, write_json


def _effect_score(edge: dict, src: dict, dst: dict) -> float:
    claimed = set(edge.get("effects", [])) | set(edge.get("preconditions", []))
    target = set(src.get("top_predicates", [])) | set(dst.get("top_predicates", []))
    return len(claimed & target) / len(claimed | target) if claimed | target else 0.0


def ground_edges(*, candidate_dir: Path, node_grounding: Path, transition_catalog: Path, fallback_catalog: Path, top_k: int, output_dir: Path, table: Path) -> dict:
    transitions = read_json(transition_catalog)["transitions"]
    fallback_rows = read_json(fallback_catalog)["fallback"]
    by_pair = {(int(x["from_cluster_id"]), int(x["to_cluster_id"])): x for x in transitions}
    profile_path = node_grounding.parent.parent / "evidence_catalog_v1/profiles/cluster_predicate_profiles.json"
    profiles = {x["cluster_handle"]: x for x in read_json(profile_path)["profiles"]} if profile_path.is_file() else {}
    output_dir.mkdir(parents=True, exist_ok=True); rows = []
    for path in sorted(candidate_dir.glob("*.json")):
        cid = f"qwen:{path.stem}"; candidate = read_json(path); node_result = read_json(node_grounding / path.name)
        node_map = {x["node_id"]: x["candidates"] for x in node_result["nodes"]}
        result = {"schema": "u3_edge_grounding_v1", "candidate_id": cid, "edges": []}
        for edge in candidate["edges"]:
            scored = []
            for src in node_map.get(edge["src"], []):
                for dst in node_map.get(edge["dst"], []):
                    pair = (int(src["raw_cluster_id"]), int(dst["raw_cluster_id"])); trans = by_pair.get(pair)
                    if not trans: continue
                    support = min(1.0, safe_float(trans.get("support_root_families")) / 84.0)
                    node_pair = (safe_float(src["score"]) + safe_float(dst["score"])) / 2
                    effect = _effect_score(edge, profiles.get(src["cluster_handle"], {}), profiles.get(dst["cluster_handle"], {}))
                    type_score = 0.75 if edge.get("hypothesized_type") in {"forward", "failure", "recovery", "alternative"} else 0.25
                    score = 0.40 * node_pair + 0.30 * effect + 0.20 * type_score + 0.10 * support
                    scored.append({"transition_handle": trans["handle"], "from_cluster_handle": trans["from_cluster_handle"], "to_cluster_handle": trans["to_cluster_handle"], "from_cluster_id": pair[0], "to_cluster_id": pair[1], "transition_count": int(trans.get("transition_count", 0)), "score": round(score, 8), "node_pair_score": round(node_pair, 8), "predicate_effect_score": round(effect, 8), "edge_type_score": round(type_score, 8), "family_support_score": round(support, 8), "support_root_families": trans.get("support_root_families", 0)})
            scored.sort(key=lambda row: (-row["score"], row["transition_handle"])); top = scored[:top_k]
            edge_type = edge.get("hypothesized_type", "")
            fallback_events = {
                "failure": {"contact_off_failure", "terminal_failure", "stagnation_onset"},
                "recovery": {"recovery_start", "contact_reestablished"},
                "alternative": {"detour_start"},
                "forward": {"contact_on", "transport_on", "goal_enter", "stable_success"},
            }.get(edge_type, set())
            fallback_handles = [row["handle"] for row in fallback_rows if row.get("confirmed_event") in fallback_events][:top_k]
            provenance = {
                "fallback_handles": fallback_handles,
                "fallback_status": "train_fragment_support" if fallback_handles else "no_matching_train_fallback_fragment",
                "independent_continuation_status": "not_run_u3g_required_u4",
            }
            result["edges"].append({"edge_id": edge["id"], "candidates": top, **provenance})
            for item in top:
                rows.append({"candidate_id": cid, "edge_id": edge["id"], **item, "fallback_handles": ";".join(fallback_handles), "fallback_status": provenance["fallback_status"], "independent_continuation_status": provenance["independent_continuation_status"], "grounding_status": "grounded_candidate"})
        write_json(output_dir / path.name, result)
    write_csv(table, rows)
    return {"status": "PASS", "candidate_count": len(list(candidate_dir.glob("*.json"))), "edge_candidate_count": len(rows), "top_k": top_k}
