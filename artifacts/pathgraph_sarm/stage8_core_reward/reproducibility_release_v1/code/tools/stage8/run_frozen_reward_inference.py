#!/usr/bin/env python3
"""Run one frozen reward checkpoint on one immutable evaluation suite."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch

from tools.stage4.lib.model import load_model
from tools.stage8.common import dump_json, load_yaml, sha256


def suite_rows(supervision: Path, diagnostic: Path, suite: str) -> list[dict[str, str]]:
    with (supervision / "tables" / "episode_manifest.csv").open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if suite in {"val", "test"}:
        selected = [row for row in rows if row["split_original"] == suite]
    else:
        with (diagnostic / "tables" / "diagnostic_episodes.csv").open(newline="", encoding="utf-8") as handle:
            diagnostic_ids = {row["episode_id"] for row in csv.DictReader(handle) if row["task_id"] == "transport_recovery"}
        selected = [row for row in rows if row["episode_id"] in diagnostic_ids and row["split_original"] in {"val", "test"}]
    if not selected:
        raise ValueError(f"empty frozen suite: {suite}")
    return sorted(selected, key=lambda row: (row["task_id"], row["episode_id"]))


def windows(x: np.ndarray, history_steps: int) -> np.ndarray:
    out = np.zeros((len(x), history_steps, x.shape[-1]), dtype=np.float32)
    for index in range(len(x)):
        start = max(0, index - history_steps + 1)
        out[index, -(index - start + 1):] = x[start:index + 1]
    return out


def finite(values: list[float]) -> bool:
    return all(math.isfinite(float(value)) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--checkpoint-sha256", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--suite", choices=["val", "test", "stage3_diagnostic"], required=True)
    parser.add_argument("--supervision-root", type=Path, required=True)
    parser.add_argument("--diagnostic-root", type=Path, required=True)
    parser.add_argument("--inference-protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metrics-output", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=512)
    args = parser.parse_args()

    protocol = load_yaml(args.inference_protocol)
    checkpoint_sha = sha256(args.checkpoint)
    if checkpoint_sha != args.checkpoint_sha256:
        raise ValueError("checkpoint SHA256 does not match frozen manifest")
    device = args.device if args.device.startswith("cuda") and torch.cuda.is_available() else "cpu"
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    model = load_model(args.checkpoint, device)
    model.eval()
    rows = suite_rows(args.supervision_root, args.diagnostic_root, args.suite)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256()
    prediction_count = 0
    all_finite = True
    with gzip.open(args.output, "wt", encoding="utf-8") as handle, torch.inference_mode():
        for row in rows:
            archive = np.load(args.supervision_root / row["file"])
            history = windows(archive["x"], int(protocol["model"]["history_steps"]))
            for start in range(0, len(history), args.batch_size):
                batch = torch.from_numpy(history[start:start + args.batch_size]).to(device)
                result = model(batch)
                node_logits = result["node_logits"].detach().cpu().numpy()
                edge_type_logits = result["edge_type_logits"].detach().cpu().numpy()
                edge_id_logits = result["edge_id_logits"].detach().cpu().numpy()
                node_probs = result["node_probs"].detach().cpu().numpy()
                edge_type_probs = result["edge_type_probs"].detach().cpu().numpy()
                edge_id_probs = result["edge_id_probs"].detach().cpu().numpy()
                phi = result["phi"].detach().cpu().numpy()
                cost = result["remaining_cost"].detach().cpu().numpy()
                for offset in range(len(node_logits)):
                    t = start + offset
                    record = {
                        "sample_id": f"{row['episode_id']}:{t}",
                        "episode_id": row["episode_id"],
                        "content_group_id": row["content_group_id"],
                        "task_id": row["task_id"],
                        "scenario": row["scenario"],
                        "outcome": row["outcome"],
                        "path_signature": row["path_signature"],
                        "split": args.suite,
                        "t": t,
                        "gt_node_id": int(archive["node_y"][t]),
                        "pred_node_logits": node_logits[offset].tolist(),
                        "pred_node_probs": node_probs[offset].tolist(),
                        "pred_node_id": int(node_logits[offset].argmax()),
                        "gt_edge_type": int(archive["edge_type_y"][t]),
                        "pred_edge_type_logits": edge_type_logits[offset].tolist(),
                        "pred_edge_type_probs": edge_type_probs[offset].tolist(),
                        "pred_edge_type": int(edge_type_logits[offset].argmax()),
                        "gt_edge_id": int(archive["edge_id_y"][t]),
                        "pred_edge_id_logits": edge_id_logits[offset].tolist(),
                        "pred_edge_id_probs": edge_id_probs[offset].tolist(),
                        "pred_edge_id": int(edge_id_logits[offset].argmax()),
                        "gt_phi": float(archive["phi_y"][t]),
                        "pred_phi": float(phi[offset]),
                        "gt_remaining_cost": float(archive["cost_y_norm"][t]),
                        "pred_remaining_cost": float(cost[offset]),
                        "model_seed": args.seed,
                        "checkpoint_path": str(args.checkpoint.resolve()),
                        "checkpoint_sha256": checkpoint_sha,
                    }
                    all_finite &= finite([record["pred_phi"], record["pred_remaining_cost"], *record["pred_node_logits"], *record["pred_edge_type_logits"], *record["pred_edge_id_logits"]])
                    line = json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n"
                    handle.write(line)
                    digest.update(line.encode("utf-8"))
                    prediction_count += 1
    result = {
        "status": "PASS" if prediction_count > 0 and all_finite else "FAIL",
        "exit_code": 0,
        "cuda_used": device.startswith("cuda"),
        "device": device,
        "loaded_checkpoint_path": str(args.checkpoint.resolve()),
        "loaded_checkpoint_sha256": checkpoint_sha,
        "prediction_count": prediction_count,
        "prediction_sha256_uncompressed_jsonl": digest.hexdigest(),
        "all_prediction_values_finite": all_finite,
        "model_seed": args.seed,
        "suite": args.suite,
        "input_episode_count": len(rows),
        "provenance": "stage8_real_frozen_checkpoint_inference",
    }
    dump_json(args.metrics_output, result)
    (args.output.parent / "DONE").write_text("PASS\n", encoding="utf-8")
    if result["status"] != "PASS":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
