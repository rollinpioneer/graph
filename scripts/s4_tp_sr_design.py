#!/usr/bin/env python3
"""CLI for the S4 T_P_SOFT_RELOCATION_V1 design-feasibility card (CP-DISR-S4-TASK-REDESIGN-1)."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from cp_disr.analysis import s4_tp_sr_design as d  # noqa: E402

COMMANDS = ("freeze-spec", "generate-pool", "static-screen", "prepare-physical-probe", "run-physical-probe", "classify-opportunity",
            "freeze-provider-sample", "call-provider", "probe-representation", "summarize", "verify", "physical-worker")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=str(ROOT))
    ap.add_argument("--config", default=str(ROOT / "configs/final_master/s4_tp_sr_design.yaml"))
    ap.add_argument("--output", default="")
    ap.add_argument("--gpus", default="1,3")
    ap.add_argument("--max-workers", type=int, default=2)
    ap.add_argument("--scene-shard", default="")
    ap.add_argument("--branch-id", default="")
    ap.add_argument("--provider-workers", type=int, default=2)
    a = ap.parse_args(argv)
    a.root = str(Path(a.root).resolve())
    a.config = str(Path(a.config).resolve())
    a.output = str(Path(a.output).resolve()) if a.output else ""
    gpus = [int(x) for x in a.gpus.split(",") if x != ""]
    try:
        if a.command == "freeze-spec":
            out = Path(a.output) if a.output else d.out_dir_for(a.root, a.config)
            res = d.freeze_spec(a.root, a.config, out)
        else:
            if not a.output:
                raise SystemExit("--output is required")
            out = Path(a.output)
            if a.command == "generate-pool":
                res = d.generate_pool_phase(a.root, a.config, out)
            elif a.command == "static-screen":
                res = d.static_screen(a.root, a.config, out, gpus, a.max_workers, a.scene_shard or None)
            elif a.command == "prepare-physical-probe":
                res = d.prepare_physical_probe(a.root, a.config, out)
            elif a.command == "run-physical-probe":
                res = d.run_physical_probe(a.root, a.config, out, gpus, a.max_workers)
            elif a.command == "physical-worker":
                res = d.physical_worker(a.root, out, a.branch_id)
                res = {"branch_id": a.branch_id, "execution_status": res.get("execution_status")}
            elif a.command == "classify-opportunity":
                res = d.classify_opportunity(a.root, a.config, out)
            else:
                from cp_disr.analysis import s4_tp_sr_provider as p
                if a.command == "freeze-provider-sample":
                    res = p.freeze_provider_sample(a.root, a.config, out)
                elif a.command == "call-provider":
                    res = p.call_provider(a.root, a.config, out, a.provider_workers)
                elif a.command == "probe-representation":
                    res = p.probe_representation(a.root, a.config, out, gpus)
                elif a.command == "summarize":
                    res = p.summarize(a.root, a.config, out)
                else:
                    res = p.verify(a.root, a.config, out)
    except d.StopRun as exc:
        print(json.dumps({"status": exc.code, "detail": exc.detail}, ensure_ascii=False))
        return 2
    print(json.dumps(d.jsonable(res), ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    sys.exit(main())
