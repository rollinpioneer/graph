#!/usr/bin/env python
"""CLI for the S1-REV1 E4 recovery (see cp_disr.analysis.s1_e4_recovery)."""
import argparse, json, sys
from cp_disr.analysis import s1_e4_recovery as m


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in ("prepare", "summarize"):
        s = sub.add_parser(name); s.add_argument("--root", required=True)
    s = sub.add_parser("worker"); s.add_argument("--root", required=True); s.add_argument("--branch-id", required=True)
    s = sub.add_parser("run-wave"); s.add_argument("--root", required=True); s.add_argument("--wave", type=int, nargs="+", required=True)
    s.add_argument("--gpus", type=int, nargs="+", required=True); s.add_argument("--max-workers", type=int)
    s = sub.add_parser("check"); s.add_argument("--root", required=True); s.add_argument("--wave", type=int, default=1)
    for sp in sub.choices.values():
        sp.add_argument("--round", choices=["r2"], default=None)
    a = p.parse_args()
    m.set_round(a.round)
    if a.cmd == "prepare": out = m.prepare(a.root)
    elif a.cmd == "worker": out = m.worker(a.root, a.branch_id); out = {"branch_id": a.branch_id, "execution_status": out.get("execution_status")}
    elif a.cmd == "run-wave": out = m.run_wave(a.root, a.wave, a.gpus, a.max_workers)
    elif a.cmd == "check": out = m.technical_check(a.root, a.wave)
    else: out = m.summarize(a.root)
    print(json.dumps(out, ensure_ascii=False, indent=2, default=str))


if __name__ == "__main__":
    sys.exit(main())
