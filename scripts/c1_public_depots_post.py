#!/usr/bin/env python
"""Packaging helper of the public-Depots card (run AFTER the DAG finished; not part of the registered pipeline).

    exact_summary --run-root R     data/exact.json (large: trajectories and successor labels) -> data/exact_summary.json (light)
    light_list    --run-root R     prints the repository-relative paths of the light result files to commit (no .pt, no planner work directories)
"""
import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAX_BYTES = 8_000_000


def exact_summary(rr):
    ex = json.loads((Path(rr) / "data" / "exact.json").read_text())
    keep = ("case_id", "status", "optimal_length", "n_states", "n_goal_states", "max_dist", "n_actions", "n_dyn_atoms", "n_optimal_first_actions", "requires_goal_destruction", "seconds", "light")
    out = {}
    for k, v in ex.items():
        row = {a: v.get(a) for a in keep if a in v}
        row["trajectories"] = len(v.get("trajectories", []))
        row["labelled_states"] = len(v.get("rank_labels", {}))
        out[k] = row
    p = Path(rr) / "data" / "exact_summary.json"
    p.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    return p


def light_list(rr):
    rr = Path(rr)
    take, skip = [], []
    skip_dirs = ("data/fd/", "goose/eval_", "goose/train_typed/training", "goose/train_ipc/training", "driver_logs/")
    for p in sorted(rr.rglob("*")):
        if not p.is_file():
            continue
        rel = p.relative_to(rr).as_posix()
        if p.suffix == ".pt" or p.name.endswith(".tmp") or rel.endswith("resume_last.pt") or any(rel.startswith(s) for s in skip_dirs) or rel == "data/exact.json":
            skip.append((rel, "excluded class"))
        elif p.stat().st_size > MAX_BYTES:
            skip.append((rel, "larger than %d bytes" % MAX_BYTES))
        else:
            take.append(rel)
    return take, skip


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd")
    ap.add_argument("--run-root", required=True)
    a = ap.parse_args()
    if a.cmd == "exact_summary":
        print(exact_summary(a.run_root))
    elif a.cmd == "light_list":
        take, skip = light_list(a.run_root)
        rr = Path(a.run_root).resolve()
        base = rr.relative_to(ROOT)
        Path(rr / "light_files.txt").write_text("\n".join(str(base / t) for t in take) + "\n", encoding="utf-8")
        print(len(take), "files to commit;", sum((rr / t).stat().st_size for t in take) / 1e6, "MB;", len(skip), "excluded")
        for rel, why in skip:
            if why != "excluded class":
                print("EXCLUDED", rel, why)
    return 0


if __name__ == "__main__":
    sys.exit(main())
