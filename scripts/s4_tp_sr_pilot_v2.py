#!/usr/bin/env python3
"""CLI for the S4 T_P_SR V2 pilot (CP-DISR-S4-TP-SR-PILOT-V2-1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cp_disr.analysis import s4_tp_sr_pilot_v2 as m  # noqa: E402

COMMANDS = ("freeze-spec", "prepare-diagnostics", "run-diagnostics", "classify-diagnostics", "freeze-prototypes", "prepare-prototype-branches",
            "run-prototypes", "classify-prototypes", "summarize", "verify", "physical-worker", "protected-after")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "configs/final_master/s4_tp_sr_pilot_v2.yaml"))
    ap.add_argument("--output", default="")
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--branch-id", default="")
    ap.add_argument("--wave", default="D", choices=("D", "P"))
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
            if a.command == "prepare-diagnostics":
                res = m.prepare_diagnostics(a.root, a.config, out)
            elif a.command == "run-diagnostics":
                res = m.run_wave(a.root, a.config, out, "D", gpus, a.max_workers)
            elif a.command == "physical-worker":
                r = m.physical_worker(a.root, out, a.branch_id, a.wave)
                res = {"branch_id": a.branch_id, "execution_status": r.get("execution_status")}
            elif a.command == "classify-diagnostics":
                from cp_disr.analysis import s4_tp_sr_pilot_v2_diag as d
                res = d.classify_diagnostics(a.root, a.config, out)
            elif a.command == "protected-after":
                cfg = m.load_config(a.config)
                inv = m.protected_inventory(a.root, cfg, out, "after")
                same, diff = m.protected_unchanged(out)
                res = {"protected_files": inv["file_count"], "unchanged": same, "diff": diff}
            else:
                from cp_disr.analysis import s4_tp_sr_pilot_v2_proto as p
                fn = {"freeze-prototypes": p.freeze_prototypes, "prepare-prototype-branches": p.prepare_prototype_branches, "run-prototypes": None,
                      "classify-prototypes": p.classify_prototypes, "summarize": p.summarize, "verify": p.verify}[a.command]
                if a.command == "run-prototypes":
                    res = m.run_wave(a.root, a.config, out, "P", gpus, a.max_workers)
                else:
                    res = fn(a.root, a.config, out)
    except m.StopRun as exc:
        print(json.dumps({"status": exc.code, "detail": exc.detail}, ensure_ascii=False))
        return 2
    print(json.dumps(m.jsonable(res), ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
