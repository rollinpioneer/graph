#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


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
    numeric_entries = 0
    test_scores: list[tuple[str, float]] = []
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if "epoch" in item:
            last_epoch = max(last_epoch, int(item["epoch"]))
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
    print("TRAINING AUDIT PASS")


if __name__ == "__main__":
    main()
