#!/usr/bin/env python3
"""CLI for the S4 Family A soft-ordering MVP (CP-DISR-S4-FAMILY-A-SOFT-ORDERING-MVP-1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m  # noqa: E402

COMMANDS = ("freeze-spec", "derive-anchor-bank", "freeze-configs", "prepare-branches", "technical-wave",
            "amend-technical-gate", "run-remaining", "classify", "summarize", "verify",
            "physical-worker", "protected-after")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "configs/final_master/s4_family_a_soft_ordering_mvp.yaml"))
    ap.add_argument("--output", default="")
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--branch-id", default="")
    a = ap.parse_args(argv)
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    try:
        if a.command == "freeze-spec":
            out = Path(a.output) if a.output else m.out_dir_for(a.root, a.config)
            res = m.freeze_spec(a.root, a.config, out)
        else:
            if not a.output:
                raise SystemExit("--output is required")
            out = Path(a.output).resolve()
            if a.command == "derive-anchor-bank":
                res = m.derive_anchor_bank(a.root, a.config, out)
            elif a.command == "freeze-configs":
                res = m.freeze_configs(a.root, a.config, out)
            elif a.command == "prepare-branches":
                res = m.prepare_branches(a.root, a.config, out)
            elif a.command == "technical-wave":
                res = m.technical_wave(a.root, a.config, out, gpus, a.max_workers)
            elif a.command == "amend-technical-gate":
                res = m.amend_technical_gate(a.root, a.config, out)
            elif a.command == "run-remaining":
                res = m.run_remaining(a.root, a.config, out, gpus, a.max_workers)
            elif a.command == "physical-worker":
                res = m.physical_worker(a.root, out, a.branch_id)
            elif a.command == "classify":
                res = m.classify(a.root, a.config, out)
            elif a.command == "summarize":
                res = m.summarize(a.root, a.config, out)
            elif a.command == "verify":
                res = m.verify(a.root, a.config, out)
            elif a.command == "protected-after":
                cfg = m.load_config(a.config)
                inv = m.protected_inventory(a.root, cfg, out, "after")
                same, diff = m.protected_unchanged(out)
                res = {"protected_files": inv["file_count"], "unchanged": same, "diff": diff}
            else:  # pragma: no cover
                raise SystemExit("unhandled command")
    except m.StopRun as exc:
        print(json.dumps({"status": exc.code, "detail": exc.detail}, ensure_ascii=False))
        return 2
    print(json.dumps(m.jsonable(res), ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())