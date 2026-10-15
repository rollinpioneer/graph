#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import pickle
from pathlib import Path

import torch


def load_pickled_scalar(payload: dict, key: str) -> int:
    pickles = payload.get("pickles")
    assert isinstance(pickles, dict), "checkpoint is missing the pickles mapping"
    value = pickles.get(key)
    assert isinstance(value, bytes), f"checkpoint pickle {key!r} is missing"
    decoded = pickle.loads(value)
    assert isinstance(decoded, int), f"checkpoint pickle {key!r} is not an integer"
    return decoded


def audit_checkpoint(checkpoint: Path, expected_last_epoch: int) -> tuple[int, int, int]:
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert isinstance(payload, dict), "checkpoint payload is not a mapping"

    checkpoint_epoch = load_pickled_scalar(payload, "epoch")
    checkpoint_global_step = load_pickled_scalar(payload, "global_step")
    assert checkpoint_epoch >= expected_last_epoch, (
        checkpoint_epoch,
        expected_last_epoch,
    )
    assert checkpoint_global_step > 0, checkpoint_global_step

    state_dicts = payload.get("state_dicts")
    assert isinstance(state_dicts, dict), "checkpoint is missing state_dicts"
    required_state_dicts = {"model", "ema_model", "optimizer"}
    assert required_state_dicts.issubset(state_dicts), sorted(state_dicts)

    tensor_count = 0
    for state_name, state_dict in state_dicts.items():
        assert isinstance(state_dict, dict), f"state_dict {state_name!r} is invalid"
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                assert torch.isfinite(value).all(), (
                    f"non-finite checkpoint tensor: {state_name}.{key}"
                )
                tensor_count += 1
    assert tensor_count > 0
    return checkpoint_epoch, checkpoint_global_step, tensor_count


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--expected-last-epoch", type=int, default=1750)
    args = parser.parse_args()

    log_path = args.train_dir / "logs.json.txt"
    checkpoint = args.train_dir / "checkpoints" / "latest.ckpt"
    assert log_path.exists(), log_path
    assert checkpoint.exists(), checkpoint

    last_epoch = -1
    last_epoch_global_step = -1
    numeric_entries = 0
    test_scores: list[tuple[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in item:
            item_epoch = int(item["epoch"])
            if item_epoch > last_epoch:
                last_epoch = item_epoch
                last_epoch_global_step = -1
            if item_epoch == last_epoch and "global_step" in item:
                last_epoch_global_step = max(
                    last_epoch_global_step, int(item["global_step"])
                )
        for key, value in item.items():
            if isinstance(value, (int, float)):
                assert math.isfinite(value), f"non-finite: {key}={value}"
                numeric_entries += 1
            if ("test" in key.lower() and "score" in key.lower()
                    and isinstance(value, (int, float))):
                test_scores.append((key, float(value)))

    print("last epoch:", last_epoch)
    print("numeric entries:", numeric_entries)
    print("last test-score entries:", test_scores[-10:])
    print("latest checkpoint:", checkpoint)
    assert numeric_entries > 0
    assert last_epoch >= args.expected_last_epoch, (last_epoch, args.expected_last_epoch)
    checkpoint_epoch, checkpoint_global_step, tensor_count = audit_checkpoint(
        checkpoint, args.expected_last_epoch
    )
    assert checkpoint_epoch == last_epoch, (checkpoint_epoch, last_epoch)
    assert checkpoint_global_step == last_epoch_global_step, (
        checkpoint_global_step,
        last_epoch_global_step,
    )
    print("checkpoint epoch:", checkpoint_epoch)
    print("checkpoint global step:", checkpoint_global_step)
    print("checkpoint tensors checked:", tensor_count)
    print("TRAINING AUDIT PASS")


if __name__ == "__main__":
    main()
