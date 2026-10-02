#!/usr/bin/env python
"""CLI for CP-DISR-TP-EF-DISCOVERY-1 (0 RL / 0 optimizer / 0 provider requests)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from cp_disr.analysis import tp_ef_discovery as m

COMMANDS = ["gate0", "inventory", "register", "run-wave", "freeze-epsilon", "worker", "analyze", "verify", "gpus"]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=COMMANDS)
    ap.add_argument("--root", default=".")
    ap.add_argument("--output")
    ap.add_argument("--branch-id")
    ap.add_argument("--wave", type=int)
    ap.add_argument("--gpus", default="")
    a = ap.parse_args(argv)
    root = Path(a.root).resolve()
    out = Path(a.output).resolve() if a.output else None
    if a.command == "gpus":
        print(json.dumps(m._gpu_pick(8)))
    elif a.command == "gate0":
        print(json.dumps({"status": m.gate0(root, out)["status"]}))
    elif a.command == "inventory":
        m.write_json(out / "protected_before.json", m.protected_inventory(root))
        print("ok")
    elif a.command == "register":
        print(json.dumps(m.register(root, out)))
    elif a.command == "run-wave":
        print(json.dumps(m.run_wave(root, out, a.wave, [int(x) for x in a.gpus.split(",") if x])))
    elif a.command == "freeze-epsilon":
        e = m.freeze_epsilon(out)
        print(json.dumps({k: e[k] for k in ("epsilon_q", "epsilon_t", "repeat_groups_complete")}))
    elif a.command == "worker":
        r = m.worker(root, out, a.branch_id)
        print(json.dumps({"branch_id": r["branch_id"], "valid": r["valid"], "termination": r["termination"]}))
    elif a.command == "analyze":
        v = m.analyze(root, out)
        print(json.dumps({"decision": v["decision"], "gates": v["gates"], "e6": v["e6_classification"]}))
    elif a.command == "verify":
        print(json.dumps({"status": m.verify(root, out)["status"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
