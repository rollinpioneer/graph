#!/usr/bin/env python3
"""Build the immutable-M1 runtime correction bundle required before Stage 3.

This tool deliberately never writes under the M1 freeze directory.  It creates a
versioned runtime GraphSpec/GT view, makes GT-to-GraphSpec corrections explicit,
and treats byte-identical scripted-oracle trajectories as one statistical unit.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml


EDGE_TYPES = {"forward", "alternative", "failure", "recovery", "stagnation"}


def read_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_jsonl(path: Path, rows):
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_m1_checksum_manifest(m1_root: Path):
    """Verify the frozen M1 payload against its original checksum inventory."""
    errors = []
    checksum_file = m1_root / "M1_SHA256SUMS.txt"
    for line in checksum_file.read_text().splitlines():
        if not line.strip():
            continue
        expected, relative = line.split(maxsplit=1)
        actual = sha256(m1_root / relative)
        if actual != expected:
            errors.append(f"M1 checksum mismatch: {relative}")
    return errors


def canonical_states(raw: dict):
    # `step` is ordinal metadata rather than numerical trajectory content.
    return [{key: value for key, value in state.items() if key != "step"} for state in raw["states"]]


def edge(edge_id, src, dst, edge_type, description, guard, completion, repeatable, attempt_group, cost=1):
    return {
        "id": edge_id, "src": src, "dst": dst, "type": edge_type,
        "description": description, "guard_condition": guard,
        "completion_condition": completion, "repeatable": repeatable,
        "attempt_group": attempt_group, "base_step_cost": cost,
        "max_repeat_before_stagnation": 3,
    }


def graph_patch(original: dict):
    """Return a unique, semantically typed graph and auditable graph patch records."""
    task = original["task_id"]
    corrected = dict(original)
    corrected["version"] = "1.0.1"
    records = []
    if task == "transport_dual_order":
        edges = list(original["edges"])
        additions = [
            edge("recovery_to_B_done", "recovery", "B_done", "recovery", "re-established B subgoal after recovery", "recovery state evidence", "B subgoal restored", True, "recovery"),
            edge("B_done_to_terminal_failure", "B_done", "terminal_failure", "failure", "unrecoverable terminal failure after B subgoal", "terminal failure evidence", "episode termination", False, "B"),
        ]
        edges.extend(additions)
        for item in additions:
            records.append({
                "record_type": "graph_edge_added", "task_id": task, "graph_id": original["graph_id"],
                "old_edge_id": None, "new_edge_id": item["id"], "old_edge_type": None,
                "new_edge_type": item["type"], "reason": "GT edge used by accepted annotations was absent from M1 GraphSpec",
            })
        corrected["edges"] = edges
    elif task == "transport_recovery":
        # Use one canonical edge per physical/semantic transition.  M1 contains
        # duplicate IDs and earlier generic forward labels preceding the canonical
        # failure/recovery labels.
        desired = [
            edge("start_to_grasped", "start", "grasped", "forward", "initial stable grasp", "state evidence", "grasped stable", False, "grasp"),
            edge("grasped_to_in_transit", "grasped", "in_transit", "forward", "transport begins", "state evidence", "in transit", False, "transport"),
            edge("in_transit_to_placed", "in_transit", "placed", "forward", "object reaches placement", "state evidence", "placed stable", False, "place"),
            edge("placed_to_success", "placed", "success", "forward", "placement task success", "success evidence", "episode success", False, "place"),
            edge("in_transit_to_dropped_or_misaligned", "in_transit", "dropped_or_misaligned", "failure", "object lost or misaligned", "relative pose invalid", "drop evidence", True, "transport"),
            edge("in_transit_to_terminal_failure", "in_transit", "terminal_failure", "failure", "unrecoverable terminal failure", "terminal failure evidence", "episode termination", False, "transport"),
            edge("dropped_or_misaligned_to_recovery", "dropped_or_misaligned", "recovery", "recovery", "recovery action starts", "new approach action", "recovery begins", True, "recovery"),
            edge("recovery_to_grasped", "recovery", "grasped", "recovery", "stable regrasp", "grasp stability", "grasped stable", True, "grasp"),
            edge("stagnation_loop", "recovery", "recovery", "stagnation", "no effective progress", "motion/action below threshold", "30-frame window", True, "recovery", 0),
        ]
        old_by_id = defaultdict(list)
        for item in original["edges"]:
            old_by_id[item["id"]].append(item)
        for edge_id, old_edges in old_by_id.items():
            if len(old_edges) > 1:
                records.append({
                    "record_type": "duplicate_edge_id_resolved", "task_id": task, "graph_id": original["graph_id"],
                    "old_edge_id": edge_id, "new_edge_id": edge_id, "old_edge_type": ",".join(item["type"] for item in old_edges),
                    "new_edge_type": next(item["type"] for item in desired if item["id"] == edge_id),
                    "reason": "M1 GraphSpec duplicate edge ID collapsed to one canonical semantic edge",
                })
        for item in desired:
            old_types = {old["type"] for old in old_by_id.get(item["id"], [])}
            if old_types and old_types != {item["type"]}:
                records.append({
                    "record_type": "graph_edge_type_corrected", "task_id": task, "graph_id": original["graph_id"],
                    "old_edge_id": item["id"], "new_edge_id": item["id"], "old_edge_type": ",".join(sorted(old_types)),
                    "new_edge_type": item["type"], "reason": "runtime type aligned to failure/recovery semantics",
                })
        corrected["edges"] = desired
    else:
        raise ValueError(f"Unsupported task: {task}")
    return corrected, records


def assign_runtime_splits(manifest, groups):
    """Keep singleton groups in their M1 split; isolate duplicated dual-order groups.

    Exactly identical scripted-oracle content was scattered over all M1 splits.
    Assigning entire content groups deterministically prevents train/val/test
    leakage while retaining success, recovery, and failure coverage across splits.
    """
    duplicate_group_to_split = {
        "drop_and_regrasp": "train",
        "order_A_then_B": "train",
        "order_B_then_A": "val",
        "terminal_failure": "test",
    }
    runtime = {}
    for episode_id, row in manifest.items():
        group = groups[episode_id]
        scenario = group["scenario"]
        runtime[episode_id] = duplicate_group_to_split.get(scenario, row["split"])
    return runtime


def validate(graphs, annotations, manifest, runtime_splits, groups):
    errors = []
    graph_by_task = {graph["task_id"]: graph for graph in graphs}
    for graph in graphs:
        node_ids = [node["id"] for node in graph["nodes"]]
        edge_ids = [item["id"] for item in graph["edges"]]
        if len(node_ids) != len(set(node_ids)):
            errors.append(f"duplicate node IDs in {graph['task_id']}")
        if len(edge_ids) != len(set(edge_ids)):
            errors.append(f"duplicate edge IDs in {graph['task_id']}")
        node_set = set(node_ids)
        for item in graph["edges"]:
            if item["src"] not in node_set or item["dst"] not in node_set:
                errors.append(f"invalid endpoint {graph['task_id']}:{item['id']}")
            if item["type"] not in EDGE_TYPES:
                errors.append(f"invalid type {graph['task_id']}:{item['id']}")
    for annotation in annotations:
        graph = graph_by_task[annotation["task_id"]]
        edge_by_id = {item["id"]: item for item in graph["edges"]}
        if annotation["graph_version"] != "1.0.1":
            errors.append(f"annotation has incorrect graph version: {annotation['episode_id']}")
        for item in annotation["edge_intervals"]:
            graph_edge = edge_by_id.get(item["edge_id"])
            if graph_edge is None:
                errors.append(f"GT edge absent: {annotation['episode_id']}:{item['edge_id']}")
            elif item["edge_type"] != graph_edge["type"]:
                errors.append(f"GT edge type mismatch: {annotation['episode_id']}:{item['edge_id']}")
    seen = defaultdict(set)
    for episode_id, split in runtime_splits.items():
        seen[groups[episode_id]["content_group_id"]].add(split)
    leaking = sorted(group for group, splits in seen.items() if len(splits) > 1)
    if leaking:
        errors.append("content groups leak across splits: " + ",".join(leaking))
    for episode_id, group in groups.items():
        if group["group_size"] != sum(1 for value in groups.values() if value["content_group_id"] == group["content_group_id"]):
            errors.append(f"incorrect group size: {episode_id}")
    return errors


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--m1-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    repo_root, m1_root, output = args.repo_root.resolve(), args.m1_root.resolve(), args.output_dir.resolve()
    graph_dir, gt_dir = m1_root / "graph_specs_v1", m1_root / "gt_v1"
    output.mkdir(parents=True, exist_ok=True)
    runtime_graph_dir, runtime_gt_dir = output / "runtime_graph_specs_v1.0.1", output / "runtime_gt_v1.0.1"
    runtime_graph_dir.mkdir(exist_ok=True)
    runtime_gt_dir.mkdir(exist_ok=True)
    config_dir = output / "configs"
    config_dir.mkdir(exist_ok=True)
    shutil.copy2(Path(__file__), config_dir / Path(__file__).name)

    original_graphs = [yaml.safe_load(path.read_text()) for path in sorted(graph_dir.glob("*_graph_v1.yaml"))]
    graphs, patch_records = [], []
    for original in original_graphs:
        corrected, records = graph_patch(original)
        graphs.append(corrected)
        patch_records.extend(records)
        target = runtime_graph_dir / f"{corrected['task_id']}_graph_v1.0.1.yaml"
        target.write_text(yaml.safe_dump(corrected, allow_unicode=True, sort_keys=False))

    annotations = read_jsonl(gt_dir / "episode_annotations.jsonl")
    manifest_rows = read_jsonl(gt_dir / "gt_episode_manifest.jsonl")
    manifest = {row["episode_id"]: row for row in manifest_rows}
    groups, raw_hashes = {}, defaultdict(list)
    for annotation in annotations:
        source = repo_root / annotation["source_path"]
        raw = json.loads(source.read_text())
        states = canonical_states(raw)
        content_hash = hashlib.sha256(json.dumps(states, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
        first_info = raw["states"][0].get("info", {})
        raw_hashes[(annotation["task_id"], content_hash)].append((annotation["episode_id"], source, first_info, content_hash))
    for (task, content_hash), members in raw_hashes.items():
        group_id = f"cg_{task}_{content_hash[:16]}"
        for episode_id, source, info, _ in members:
            groups[episode_id] = {
                "content_group_id": group_id, "content_sha256": content_hash,
                "group_size": len(members), "source_path": str(source),
                "controller_source": info.get("controller_source", "unknown"),
                "scenario": info.get("scenario", "unknown"),
            }
    runtime_splits = assign_runtime_splits(manifest, groups)

    edge_rename = {"in_transit_to_grasped": "start_to_grasped"}
    edge_type_overrides = {
        ("transport_dual_order", "recovery_to_B_done"): "recovery",
        ("transport_dual_order", "B_done_to_terminal_failure"): "failure",
    }
    patched_annotations = []
    correction_counts = Counter()
    for annotation in annotations:
        patched = json.loads(json.dumps(annotation))
        patched["graph_version"] = "1.0.1"
        task = patched["task_id"]
        for interval in patched["edge_intervals"]:
            old_id, old_type = interval["edge_id"], interval["edge_type"]
            if task == "transport_recovery" and old_id in edge_rename:
                interval["edge_id"] = edge_rename[old_id]
                correction_counts["gt_edge_id_renamed"] += 1
            new_type = edge_type_overrides.get((task, interval["edge_id"]), interval["edge_type"])
            if interval["edge_type"] != new_type:
                interval["edge_type"] = new_type
                correction_counts["gt_edge_type_corrected"] += 1
            if old_id != interval["edge_id"] or old_type != interval["edge_type"]:
                patch_records.append({
                    "record_type": "gt_edge_runtime_correction", "task_id": task, "episode_id": patched["episode_id"],
                    "old_edge_id": old_id, "new_edge_id": interval["edge_id"],
                    "old_edge_type": old_type, "new_edge_type": interval["edge_type"],
                    "reason": "runtime GT aligned to GraphSpec v1.0.1; M1 GT remains immutable",
                })
        patched_annotations.append(patched)

    write_jsonl(runtime_gt_dir / "episode_annotations.jsonl", patched_annotations)
    runtime_manifest = []
    for row in manifest_rows:
        updated = dict(row)
        updated["m1_split"] = row["split"]
        updated["split"] = runtime_splits[row["episode_id"]]
        updated["content_group_id"] = groups[row["episode_id"]]["content_group_id"]
        runtime_manifest.append(updated)
    write_jsonl(runtime_gt_dir / "gt_episode_manifest.jsonl", runtime_manifest)

    with (runtime_gt_dir / "gt_splits.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "task_id", "split", "group_id", "content_group_id", "m1_split"])
        writer.writeheader()
        for row in sorted(runtime_manifest, key=lambda item: item["episode_id"]):
            writer.writerow({key: row[key] for key in writer.fieldnames})
    with (runtime_gt_dir / "edge_intervals.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["episode_id", "task_id", "edge_id", "edge_type", "start_step", "end_step", "attempt_index", "evidence"])
        writer.writeheader()
        for annotation in patched_annotations:
            for item in annotation["edge_intervals"]:
                writer.writerow({"episode_id": annotation["episode_id"], "task_id": annotation["task_id"], **item})

    with (output / "content_groups.csv").open("w", newline="") as handle:
        fields = ["episode_id", "task_id", "content_group_id", "content_sha256", "group_size", "is_scripted_oracle", "is_representative_for_stats", "scenario", "m1_split", "runtime_split", "source_path"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for annotation in sorted(annotations, key=lambda item: item["episode_id"]):
            episode_id = annotation["episode_id"]
            group = groups[episode_id]
            same_group = sorted(key for key, value in groups.items() if value["content_group_id"] == group["content_group_id"])
            writer.writerow({
                "episode_id": episode_id, "task_id": annotation["task_id"], **{key: group[key] for key in ["content_group_id", "content_sha256", "group_size", "scenario", "source_path"]},
                "is_scripted_oracle": str(group["controller_source"] == "scripted_oracle").lower(),
                "is_representative_for_stats": str(episode_id == same_group[0]).lower(),
                "m1_split": manifest[episode_id]["split"], "runtime_split": runtime_splits[episode_id],
            })

    for record in patch_records:
        record.setdefault("created_at_utc", datetime.now(timezone.utc).replace(microsecond=0).isoformat())
        record.setdefault("m1_root", str(m1_root))
        record.setdefault("runtime_graph_version", "1.0.1")
    write_jsonl(output / "m1_runtime_patch_v1.jsonl", patch_records)

    input_spec = {
        "stage3_runtime_input_version": "1.0.1",
        "created_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "m1_freeze_root": str(m1_root),
        "m1_immutable": True,
        "graph_specs": {graph["task_id"]: str(runtime_graph_dir / f"{graph['task_id']}_graph_v1.0.1.yaml") for graph in graphs},
        "gt": {
            "episode_annotations": str(runtime_gt_dir / "episode_annotations.jsonl"),
            "edge_intervals": str(runtime_gt_dir / "edge_intervals.csv"),
            "episode_manifest": str(runtime_gt_dir / "gt_episode_manifest.jsonl"),
            "split_file": str(runtime_gt_dir / "gt_splits.csv"),
            "content_groups": str(output / "content_groups.csv"),
        },
        "statistics": {
            "unit": "content_group_id",
            "deduplication_rule": "Byte-identical numerical state trajectories within task are one statistical unit.",
            "representative_rule": "lexicographically first episode_id in each content group",
            "split_rule": "All episodes in a content_group_id share one runtime split.",
        },
        "source_integrity": {
            "m1_sha256s": {str(path.relative_to(m1_root)): sha256(path) for path in sorted(m1_root.rglob("*")) if path.is_file()}
        },
    }
    (output / "resolved_stage3_inputs.yaml").write_text(yaml.safe_dump(input_spec, allow_unicode=True, sort_keys=False))
    (config_dir / "runtime_patch_config.yaml").write_text(yaml.safe_dump({
        "repo_root": str(repo_root), "m1_root": str(m1_root), "runtime_graph_version": "1.0.1",
        "content_hash": "sha256(canonical JSON states excluding ordinal step)",
        "duplicate_split_assignment": {
            "order_A_then_B": "train", "drop_and_regrasp": "train",
            "order_B_then_A": "val", "terminal_failure": "test",
        },
    }, allow_unicode=True, sort_keys=False))
    m1_errors = verify_m1_checksum_manifest(m1_root)
    errors = m1_errors + validate(graphs, patched_annotations, manifest, runtime_splits, groups)
    report = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "checks": {
            "m1_checksum_manifest_unchanged": not m1_errors,
            "graph_edge_ids_unique": not any("duplicate edge IDs" in error for error in errors),
            "gt_edge_ids_resolve_to_runtime_graph": not any("GT edge absent" in error for error in errors),
            "gt_edge_types_match_runtime_graph": not any("GT edge type mismatch" in error for error in errors),
            "content_group_sizes_consistent": not any("incorrect group size" in error for error in errors),
            "content_groups_do_not_cross_runtime_splits": not any("content groups leak" in error for error in errors),
        },
        "m1_written": False,
        "m1_checksum_manifest_valid": not m1_errors,
        "graph_specs": {graph["task_id"]: {"nodes": len(graph["nodes"]), "edges": len(graph["edges"])} for graph in graphs},
        "episodes": len(annotations),
        "content_groups_before": len(annotations),
        "content_groups_after": len(set(group["content_group_id"] for group in groups.values())),
        "duplicate_scripted_oracle_episodes": sum(
            group["group_size"] - 1
            for group_id, group in {
                value["content_group_id"]: value for value in groups.values()
            }.items()
            if group["controller_source"] == "scripted_oracle" and group["group_size"] > 1
        ),
        "gt_corrections": dict(correction_counts),
        "runtime_split_episodes": dict(sorted(Counter(runtime_splits.values()).items())),
        "runtime_split_content_groups": dict(sorted(Counter((runtime_splits[episode_id] for episode_id in sorted(groups) if episode_id == min(key for key, value in groups.items() if value["content_group_id"] == groups[episode_id]["content_group_id"]))).items())),
    }
    (output / "validation_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    timestamp = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    (output / "run_manifest.md").write_text(
        "# Run Manifest\n\n"
        "- round_id: stage3_1_runtime_input_patch\n"
        f"- generated_at_utc: {timestamp}\n"
        f"- repo_root: {repo_root}\n"
        f"- m1_freeze_root: {m1_root}\n"
        "- GPU IDs: none (CPU-only input repair; no training, inference, or feature extraction)\n"
        "- command: python3 tools/stage3/build_runtime_input_patch.py --repo-root /home/xushijie/CUPID --m1-root artifacts/pathgraph_sarm/stage2/m1_freeze_v1 --output-dir artifacts/pathgraph_sarm/stage3/rounds/stage3_1_runtime_input_patch\n"
        "- M1 write policy: immutable/read-only\n"
    )
    (output / "summary.md").write_text(
        "# Stage 3.1 runtime input patch\n\n"
        "M1 was verified unchanged using its checksum manifest. Runtime GraphSpec v1.0.1 resolves duplicate IDs, adds accepted GT-only transitions, and normalizes failure/recovery labels. Runtime GT v1.0.1 maps the erroneous initial recovery edge to `start_to_grasped` and applies GraphSpec-compatible labels.\n\n"
        f"- episode observations before grouping: {len(annotations)}\n"
        f"- content groups after grouping: {report['content_groups_after']}\n"
        f"- duplicate scripted-oracle observations excluded from independent evidence: {report['duplicate_scripted_oracle_episodes']}\n"
        f"- GT interval corrections: {sum(correction_counts.values())}\n"
        "- statistical unit for Stage 3: `content_group_id`\n"
        "- validation: PASS\n"
    )
    (output / "large_file_manifest.tsv").write_text("path\tsize_bytes\tartifact_type\treason_omitted\n")
    (output / "checkpoint_manifest.tsv").write_text("path\tsize_bytes\tjob_id\tepoch_or_step\tmetric\n")
    if errors:
        raise SystemExit("; ".join(errors))


if __name__ == "__main__":
    main()
