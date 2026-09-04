#!/usr/bin/env python3
"""Merge three frozen-checkpoint predictions by sample identity."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np

from tools.stage8.common import dump_json, write_csv


def entropy(probabilities: np.ndarray) -> tuple[list[float], float, float]:
    mean = probabilities.mean(axis=0)
    predictive_entropy = float(-(mean * np.log(np.clip(mean, 1e-12, 1))).sum())
    per_model = -np.sum(probabilities * np.log(np.clip(probabilities, 1e-12, 1)), axis=1)
    return mean.tolist(), predictive_entropy, float(predictive_entropy - per_model.mean())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--prediction-root", type=Path, required=True)
    parser.add_argument("--seeds", required=True)
    parser.add_argument("--suites", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    args = parser.parse_args()
    seeds = args.seeds.split(",")
    manifests = []
    args.output_root.mkdir(parents=True, exist_ok=True)
    for suite in args.suites.split(","):
        seed_maps = []
        for seed in seeds:
            source = args.prediction_root / f"s{seed}__{suite}.jsonl.gz"
            records = {}
            with gzip.open(source, "rt", encoding="utf-8") as handle:
                for line in handle:
                    record = json.loads(line)
                    key = record["sample_id"]
                    records[key] = record
            seed_maps.append(records)
        keys = set(seed_maps[0])
        if any(set(values) != keys for values in seed_maps[1:]):
            raise ValueError(f"seed sample identity mismatch for {suite}")
        output = args.output_root / f"ensemble_{suite}_predictions.jsonl.gz"
        digest = hashlib.sha256()
        with gzip.open(output, "wt", encoding="utf-8") as handle:
            for key in sorted(keys):
                records = [values[key] for values in seed_maps]
                reference = records[0]
                for field in ("gt_node_id", "gt_edge_type", "gt_edge_id", "gt_phi", "gt_remaining_cost"):
                    if len({str(record[field]) for record in records}) != 1:
                        raise ValueError(f"ground truth mismatch in {suite}: {field}")
                node_mean, node_entropy, node_mi = entropy(np.asarray([record["pred_node_probs"] for record in records]))
                edge_type_mean, edge_entropy, edge_mi = entropy(np.asarray([record["pred_edge_type_probs"] for record in records]))
                edge_id_mean, _, _ = entropy(np.asarray([record["pred_edge_id_probs"] for record in records]))
                phi = np.asarray([record["pred_phi"] for record in records], dtype=float)
                cost = np.asarray([record["pred_remaining_cost"] for record in records], dtype=float)
                merged = {
                    **{field: reference[field] for field in ("sample_id", "episode_id", "content_group_id", "task_id", "scenario", "outcome", "path_signature", "split", "t", "gt_node_id", "gt_edge_type", "gt_edge_id", "gt_phi", "gt_remaining_cost")},
                    "node_probs_mean": node_mean, "edge_type_probs_mean": edge_type_mean, "edge_id_probs_mean": edge_id_mean,
                    "node_pred": int(np.argmax(node_mean)), "edge_type_pred": int(np.argmax(edge_type_mean)), "edge_id_pred": int(np.argmax(edge_id_mean)),
                    "node_predictive_entropy": node_entropy, "node_mutual_information": node_mi,
                    "edge_predictive_entropy": edge_entropy, "edge_mutual_information": edge_mi,
                    "phi_mean": float(phi.mean()), "phi_std": float(phi.std(ddof=1)),
                    "remaining_cost_mean": float(cost.mean()), "remaining_cost_std": float(cost.std(ddof=1)),
                    "per_seed_phi": phi.tolist(), "per_seed_remaining_cost": cost.tolist(),
                    "checkpoint_sha256s": [record["checkpoint_sha256"] for record in records],
                }
                line = json.dumps(merged, separators=(",", ":"), allow_nan=False) + "\n"
                handle.write(line)
                digest.update(line.encode("utf-8"))
        manifests.append({"suite": suite, "output_path": str(output.resolve()), "prediction_count": len(keys), "ensemble_members": len(seeds), "uncompressed_jsonl_sha256": digest.hexdigest()})
    write_csv(args.manifest, manifests)
    dump_json(args.metrics, {"decision": "ENSEMBLE_MERGE_COMPLETE", "suite_count": len(manifests), "all_samples_have_three_predictions": all(row["ensemble_members"] == 3 for row in manifests), "suites": manifests})


if __name__ == "__main__":
    main()
