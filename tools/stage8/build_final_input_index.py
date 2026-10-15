#!/usr/bin/env python3
"""Index and hash every frozen Stage 8 input without changing source artifacts."""
from __future__ import annotations

import argparse
from pathlib import Path

from tools.stage8.common import dump_yaml, sha256, write_csv


def entry(name: str, path: Path) -> dict:
    return {
        "name": name,
        "path": str(path.resolve()),
        "exists": path.exists(),
        "size_bytes": path.stat().st_size if path.is_file() else "",
        "sha256": sha256(path) if path.is_file() else "",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--claim-scope", type=Path, required=True)
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--checkpoint-manifest", type=Path, required=True)
    parser.add_argument("--supervision", type=Path, required=True)
    parser.add_argument("--diagnostic", type=Path, required=True)
    parser.add_argument("--reference-predictions", type=Path, required=True)
    parser.add_argument("--reward-v1", type=Path, required=True)
    parser.add_argument("--g4-r1", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--hash-output", type=Path, required=True)
    args = parser.parse_args()

    inputs = [
        entry("claim_scope", args.claim_scope),
        entry("model_bundle", args.model_bundle),
        entry("checkpoint_manifest", args.checkpoint_manifest),
        entry("sample_index", args.supervision / "tables/sample_index.csv.gz"),
        entry("episode_manifest", args.supervision / "tables/episode_manifest.csv"),
        entry("content_group_split", args.supervision / "tables/content_group_split.csv"),
        entry("feature_schema", args.supervision / "configs/feature_schema.json"),
        entry("label_maps", args.supervision / "configs/label_maps.json"),
        entry("cost_target_spec", args.supervision / "configs/cost_target_spec.yaml"),
        entry("diagnostic_episodes", args.diagnostic / "tables/diagnostic_episodes.csv"),
        entry("reference_val_predictions", args.reference_predictions / "tables/ensemble_val_predictions.jsonl.gz"),
        entry("reference_test_predictions", args.reference_predictions / "tables/ensemble_test_predictions.jsonl.gz"),
        entry("reference_diagnostic_predictions", args.reference_predictions / "tables/ensemble_stage3_diagnostic_predictions.jsonl.gz"),
        entry("reward_config", args.reward_v1 / "configs/reward_config_v1.yaml"),
        entry("reward_lock", args.reward_v1 / "configs/reward_selection_lock.json"),
        entry("reward_engine", args.reward_v1 / "code/reward_engine.py"),
        entry("reward_main_table", args.g4_r1 / "tables/reward_main_table.csv"),
        entry("core_ablation_effects", args.g4_r1 / "tables/core_ablation_effects.csv"),
        entry("history_granularity", args.g4_r1 / "tables/history_granularity_summary.csv"),
        entry("uncertainty", args.g4_r1 / "tables/uncertainty_error_detection.csv"),
    ]
    write_csv(args.hash_output, inputs, delimiter="\t")
    dump_yaml(args.output, {
        "mode": "core_reward_only",
        "statistics_unit": "content_group_id",
        "model": {"ensemble_size": 3, "bundle": str(args.model_bundle.resolve()), "checkpoint_manifest": str(args.checkpoint_manifest.resolve())},
        "suites": {
            "val": {"supervision_root": str(args.supervision.resolve()), "selection": "split_original=val"},
            "test": {"supervision_root": str(args.supervision.resolve()), "selection": "split_original=test"},
            "stage3_diagnostic": {"diagnostic_root": str(args.diagnostic.resolve()), "selection": "frozen Stage 3 reporting panel: val plus test recovery episodes"},
        },
        "reward": {"config": str((args.reward_v1 / "configs/reward_config_v1.yaml").resolve()), "lock": str((args.reward_v1 / "configs/reward_selection_lock.json").resolve()), "engine": str((args.reward_v1 / "code/reward_engine.py").resolve())},
        "reference_results": {"reward_main_table": str((args.g4_r1 / "tables/reward_main_table.csv").resolve()), "core_ablation_effects": str((args.g4_r1 / "tables/core_ablation_effects.csv").resolve()), "history_granularity": str((args.g4_r1 / "tables/history_granularity_summary.csv").resolve()), "uncertainty": str((args.g4_r1 / "tables/uncertainty_error_detection.csv").resolve())},
        "unsupported_extensions": {"coverage_scaling": False, "unseen_order": False, "stable_policy": False, "auto_graph_main": False},
        "input_hash_manifest": str(args.hash_output.resolve()),
    })
    if not all(item["exists"] for item in inputs):
        missing = [item["name"] for item in inputs if not item["exists"]]
        raise SystemExit(f"missing frozen inputs: {missing}")


if __name__ == "__main__":
    main()
