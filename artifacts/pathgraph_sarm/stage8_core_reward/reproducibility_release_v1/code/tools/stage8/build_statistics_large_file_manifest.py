#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
from tools.stage8.common import sha256, write_csv

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--distribution-dir", type=Path, required=True)
    p.add_argument("--observations", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    files = [a.observations, *sorted(a.distribution_dir.glob("*"))]
    rows = [{"path": str(f.resolve()), "size_bytes": f.stat().st_size, "sha256": sha256(f), "artifact_type": "bootstrap_distribution" if f.parent == a.distribution_dir else "group_level_observations", "reason_omitted": "large_recomputable_statistical_artifact", "required_for_full_recompute": True} for f in files if f.is_file()]
    write_csv(a.output, rows, delimiter="\t")
if __name__ == "__main__": main()
