#!/usr/bin/env python3
"""Freeze the exact Transport-MH demo IDs only after the strong offline gate."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd


STRONG_DECISION = "PASS_STABILITY_WEIGHTED_CORE_CANDIDATE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def verify_diagnosis_inputs(audit: dict, split_path: Path) -> None:
    if int(audit.get("num_rollouts", -1)) != 100:
        raise ValueError("Diagnosis audit does not describe 100 rollouts")
    if int(audit.get("num_demos", -1)) != 192:
        raise ValueError("Diagnosis audit does not describe 192 training demos")
    reconstruction = audit.get("reconstruction", {})
    if float(reconstruction.get("max_abs_error", np.inf)) > 2e-3:
        raise ValueError("Official score reconstruction exceeded 2e-3")

    audited_inputs = (
        ("matrix_path", "matrix_sha256"),
        ("manifest_path", "manifest_sha256"),
        ("split_json_path", "split_json_sha256"),
        ("official_scores_path", "official_scores_sha256"),
    )
    for path_key, hash_key in audited_inputs:
        path = Path(audit[path_key])
        if not path.is_file():
            raise FileNotFoundError(path)
        actual = sha256(path)
        if actual != audit[hash_key]:
            raise ValueError(f"Diagnosis input hash mismatch: {path}")
    if sha256(split_path) != audit["split_json_sha256"]:
        raise ValueError("Requested split JSON differs from the diagnosed split")


def freeze_ids(
    diagnosis_dir: Path,
    split_path: Path,
    strong_pass_status: Path,
    output_dir: Path,
    threshold: float = 0.90,
    proportion: float = 0.20,
) -> dict:
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if not 0.0 < threshold <= 1.0:
        raise ValueError("threshold must be in (0,1]")
    if not 0.0 < proportion < 1.0:
        raise ValueError("proportion must be in (0,1)")

    decision_path = diagnosis_dir / "stage2b_decision.txt"
    audit_path = diagnosis_dir / "audit.json"
    mapping_path = diagnosis_dir / "full100_demo_boundary_mapping.csv"
    for path in (decision_path, audit_path, mapping_path, split_path, strong_pass_status):
        if not path.is_file():
            raise FileNotFoundError(path)

    status_lines = strong_pass_status.read_text(encoding="utf-8").splitlines()
    if not status_lines or status_lines[0].strip() != "STRONG_PASS":
        raise PermissionError("Strong-pass status is absent")
    decision = decision_path.read_text(encoding="utf-8").strip()
    if decision != STRONG_DECISION:
        raise PermissionError(f"Filtering is not authorized: {decision}")
    expected_status = f"decision={STRONG_DECISION}"
    if expected_status not in {line.strip() for line in status_lines}:
        raise ValueError("Strong-pass status and diagnosis decision disagree")

    audit = read_json(audit_path)
    verify_diagnosis_inputs(audit, split_path)
    split = read_json(split_path)
    train_ids = np.asarray(split["train_demo_indices"])
    if (
        train_ids.shape != (192,)
        or not np.issubdtype(train_ids.dtype, np.integer)
        or len(np.unique(train_ids)) != 192
    ):
        raise ValueError("Invalid frozen training-demo IDs")
    train_ids = train_ids.astype(np.int64, copy=False)

    mapping = pd.read_csv(mapping_path)
    probability_column = "bootstrap_delete_probability_%.2f" % proportion
    required = {
        "matrix_column",
        "original_demo_index",
        "ascending_rank",
        probability_column,
    }
    missing = required - set(mapping.columns)
    if missing:
        raise ValueError(f"Boundary mapping is missing columns: {sorted(missing)}")
    if len(mapping) != 192:
        raise ValueError("Boundary mapping must contain exactly 192 demos")

    mapping_ids = pd.to_numeric(
        mapping["original_demo_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    ranks = pd.to_numeric(
        mapping["ascending_rank"], errors="raise"
    ).to_numpy(dtype=np.int64)
    probabilities = pd.to_numeric(
        mapping[probability_column], errors="raise"
    ).to_numpy(dtype=float)
    if len(np.unique(mapping_ids)) != 192 or set(mapping_ids) != set(train_ids):
        raise ValueError("Boundary mapping IDs differ from the frozen train split")
    if set(ranks) != set(range(1, 193)):
        raise ValueError("Boundary mapping ranks are not a 1..192 permutation")
    if not np.all(np.isfinite(probabilities)) or not np.all(
        (probabilities >= 0.0) & (probabilities <= 1.0)
    ):
        raise ValueError("Invalid bootstrap deletion probabilities")

    selected = mapping.loc[probabilities >= threshold].copy()
    selected = selected.sort_values(
        ["ascending_rank", "original_demo_index"], kind="stable"
    )
    selected_ids = pd.to_numeric(
        selected["original_demo_index"], errors="raise"
    ).to_numpy(dtype=np.int64)
    if len(selected_ids) == 0:
        raise ValueError("Strong pass produced an empty full-100 stable core")
    if len(np.unique(selected_ids)) != len(selected_ids):
        raise ValueError("Selected filter IDs are not unique")
    if not set(selected_ids).issubset(set(train_ids)):
        raise ValueError("Selected filter IDs include a non-training demo")

    manifest = {
        "status": "FROZEN",
        "authorization_decision": decision,
        "selection": {
            "method": "full100_bottom20_bootstrap_delete_probability",
            "probability_threshold": threshold,
            "comparison": ">=",
            "target_proportion": proportion,
            "filter_count": int(len(selected_ids)),
            "ordered_filter_episode_ids": selected_ids.tolist(),
        },
        "split_counts_before_filter": {
            "train": len(train_ids),
            "validation": len(split["val_demo_indices"]),
            "holdout": len(split["holdout_demo_indices"]),
        },
        "input_files": {
            "strong_pass_status": {
                "path": str(strong_pass_status.resolve()),
                "sha256": sha256(strong_pass_status),
            },
            "decision": {
                "path": str(decision_path.resolve()),
                "sha256": sha256(decision_path),
            },
            "diagnosis_audit": {
                "path": str(audit_path.resolve()),
                "sha256": sha256(audit_path),
            },
            "boundary_mapping": {
                "path": str(mapping_path.resolve()),
                "sha256": sha256(mapping_path),
            },
            "split_json": {
                "path": str(split_path.resolve()),
                "sha256": sha256(split_path),
            },
        },
        "diagnosed_input_hashes": {
            key: audit[key]
            for key in (
                "matrix_sha256",
                "manifest_sha256",
                "split_json_sha256",
                "official_scores_sha256",
            )
        },
    }

    output_dir.mkdir(parents=True, exist_ok=False)
    selected.to_csv(output_dir / "selected_filter_demos.csv", index=False)
    manifest_path = output_dir / "filter_ids.json"
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    override = (
        "+task.dataset.dataset_mask_kwargs.filter_episode_ids="
        + json.dumps(selected_ids.tolist(), separators=(",", ":"))
    )
    (output_dir / "hydra_override.txt").write_text(
        override + "\n", encoding="utf-8"
    )
    output_files = (
        output_dir / "filter_ids.json",
        output_dir / "hydra_override.txt",
        output_dir / "selected_filter_demos.csv",
    )
    (output_dir / "output_files_sha256.txt").write_text(
        "".join(f"{sha256(path)}  {path.name}\n" for path in output_files),
        encoding="utf-8",
    )
    return manifest


def self_test() -> None:
    with tempfile.TemporaryDirectory() as temp_dir:
        root = Path(temp_dir)
        diagnosis = root / "diagnosis"
        diagnosis.mkdir()
        inputs = root / "inputs"
        inputs.mkdir()
        for name in ("matrix.npy", "manifest.csv", "scores.csv"):
            (inputs / name).write_bytes((name + "\n").encode("ascii"))
        split = {
            "train_demo_indices": list(range(192)),
            "val_demo_indices": list(range(192, 204)),
            "holdout_demo_indices": list(range(204, 300)),
        }
        split_path = inputs / "split.json"
        split_path.write_text(json.dumps(split), encoding="utf-8")
        audit = {
            "matrix_path": str(inputs / "matrix.npy"),
            "matrix_sha256": sha256(inputs / "matrix.npy"),
            "manifest_path": str(inputs / "manifest.csv"),
            "manifest_sha256": sha256(inputs / "manifest.csv"),
            "split_json_path": str(split_path),
            "split_json_sha256": sha256(split_path),
            "official_scores_path": str(inputs / "scores.csv"),
            "official_scores_sha256": sha256(inputs / "scores.csv"),
            "num_rollouts": 100,
            "num_demos": 192,
            "reconstruction": {"max_abs_error": 1e-6},
        }
        (diagnosis / "audit.json").write_text(json.dumps(audit), encoding="utf-8")
        (diagnosis / "stage2b_decision.txt").write_text(
            STRONG_DECISION + "\n", encoding="utf-8"
        )
        probability = np.zeros(192)
        probability[[7, 3, 11, 5, 2]] = [0.91, 0.99, 0.95, 0.90, 0.98]
        pd.DataFrame({
            "matrix_column": np.arange(192),
            "original_demo_index": np.arange(192),
            "ascending_rank": np.arange(1, 193),
            "bootstrap_delete_probability_0.20": probability,
        }).to_csv(diagnosis / "full100_demo_boundary_mapping.csv", index=False)
        status = root / "strong_pass.status"
        status.write_text(
            "STRONG_PASS\n" + f"decision={STRONG_DECISION}\n",
            encoding="utf-8",
        )
        output = root / "frozen_ids"
        result = freeze_ids(diagnosis, split_path, status, output)
        assert result["selection"]["ordered_filter_episode_ids"] == [2, 3, 5, 7, 11]
        assert result["selection"]["filter_count"] == 5
        assert (output / "output_files_sha256.txt").is_file()

        stop_status = root / "stop.status"
        stop_status.write_text("STOP\n", encoding="utf-8")
        try:
            freeze_ids(diagnosis, split_path, stop_status, root / "forbidden")
        except PermissionError:
            pass
        else:
            raise AssertionError("STOP status must not freeze filter IDs")
    print("SELF-TEST PASS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnosis-dir", type=Path)
    parser.add_argument("--split-json", type=Path)
    parser.add_argument("--strong-pass-status", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--threshold", type=float, default=0.90)
    parser.add_argument("--proportion", type=float, default=0.20)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if not args.self_test:
        required = ("diagnosis_dir", "split_json", "strong_pass_status", "output_dir")
        missing = [name for name in required if getattr(args, name) is None]
        if missing:
            parser.error(f"Missing arguments: {missing}")
    return args


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    manifest = freeze_ids(
        args.diagnosis_dir,
        args.split_json,
        args.strong_pass_status,
        args.output_dir,
        args.threshold,
        args.proportion,
    )
    print(json.dumps(manifest, indent=2, ensure_ascii=False))
    print("FILTER ID FREEZE PASS")


if __name__ == "__main__":
    main()
