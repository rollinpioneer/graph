#!/usr/bin/env python3
"""Emit a short, uniform manifest for a Stage 8 evidence round."""
from __future__ import annotations

import argparse
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--round-dir", type=Path, required=True)
    parser.add_argument("--round-id", required=True)
    parser.add_argument("--purpose", required=True)
    parser.add_argument("--gpu-ids", default="none")
    parser.add_argument("--command", action="append", default=[])
    args = parser.parse_args()
    try:
        commit = subprocess.check_output(["git", "-C", "/home/__compress_data/xushijie/CUPID/repo", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        commit = "unavailable_non_git_workspace_root"
    text = ["# Run Manifest", "", f"- round_id: {args.round_id}", f"- generated_at: {datetime.now(timezone.utc).isoformat()}", f"- purpose: {args.purpose}", f"- gpu_ids: {args.gpu_ids}", f"- code_commit: {commit}"]
    if args.command:
        text.extend(["", "## Executed Commands", *[f"- `{command}`" for command in args.command]])
    args.round_dir.mkdir(parents=True, exist_ok=True)
    (args.round_dir / "run_manifest.md").write_text("\n".join(text) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
