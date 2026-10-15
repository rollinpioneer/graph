#!/usr/bin/env python3
"""Build a CPU-verified manifest for the immutable Stage 6 reward checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from tools.stage8.common import dump_json, sha256, write_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--load-check", type=Path, required=True)
    args = parser.parse_args()

    bundle = json.loads(args.model_bundle.read_text(encoding="utf-8"))
    rows: list[dict] = []
    reports: list[dict] = []
    for item in bundle.get("checkpoints", []):
        checkpoint = Path(item["path"])
        exists = checkpoint.is_file()
        actual_sha = sha256(checkpoint) if exists else ""
        obj = None
        error = ""
        try:
            obj = torch.load(checkpoint, map_location="cpu")
            load_ok = isinstance(obj, dict) and isinstance(obj.get("model", obj), dict)
        except Exception as exc:  # preserve diagnostic detail in evidence
            load_ok = False
            error = f"{type(exc).__name__}: {exc}"
        model_state = obj.get("model", obj) if isinstance(obj, dict) else {}
        model_config_id = ";".join(f"{key}:{tuple(value.shape)}" for key, value in sorted(model_state.items()))
        row = {
            "model_seed": item.get("seed"),
            "checkpoint_path": str(checkpoint.resolve()),
            "size_bytes": checkpoint.stat().st_size if exists else 0,
            "sha256": actual_sha,
            "bundle_sha256": item.get("sha256", ""),
            "bundle_sha256_match": actual_sha == item.get("sha256", ""),
            "torch_load_ok": load_ok,
            "checkpoint_seed": obj.get("seed") if isinstance(obj, dict) else "",
            "checkpoint_seed_match": bool(isinstance(obj, dict) and obj.get("seed") == item.get("seed")),
            "history_steps": obj.get("history_steps") if isinstance(obj, dict) else "",
            "history_steps_match": bool(isinstance(obj, dict) and obj.get("history_steps") == item.get("history_steps")),
            "state_dict_present": bool(model_state),
            "model_config_id": sha256_text(model_config_id),
            "load_error": error,
        }
        rows.append(row)
        reports.append({key: row[key] for key in row if key != "model_config_id"})
    write_csv(args.output, rows, delimiter="\t")
    valid = len(rows) == 3 and all(
        row["bundle_sha256_match"] and row["torch_load_ok"] and row["checkpoint_seed_match"]
        and row["history_steps_match"] and row["state_dict_present"] for row in rows
    )
    dump_json(args.load_check, {
        "decision": "FINAL_CHECKPOINTS_VERIFIED" if valid else "MISSING_OR_INVALID_FINAL_CHECKPOINT",
        "checkpoint_count": len(rows),
        "sha256_match_count": sum(bool(row["bundle_sha256_match"]) for row in rows),
        "torch_load_ok_count": sum(bool(row["torch_load_ok"]) for row in rows),
        "cpu_only_validation": True,
        "reports": reports,
    })
    if not valid:
        raise SystemExit(2)


def sha256_text(value: str) -> str:
    import hashlib
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


if __name__ == "__main__":
    main()
