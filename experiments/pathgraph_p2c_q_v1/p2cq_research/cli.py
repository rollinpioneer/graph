"""P2C-Q research CLI. No undocumented auto-retry. Formal export uses the shared ledger."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .compatibility import inspect_legacy
from .export_pairs import export_split
from .source_audit import run_all
from .task_contract import ACTION_NAMES


def cmd_help():
    print("p2cq_research commands: export, audit, actions, version")
    print("export --split development|confirmation --out DIR")
    print("audit --exp-root DIR --repo DIR")
    return 0


def cmd_export(ns):
    summary = export_split(ns.split, ns.out)
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


def cmd_audit(ns):
    ast = run_all(ns.exp_root)
    legacy = inspect_legacy(ns.repo)
    print(json.dumps({"source_audit": ast, "legacy": legacy}, indent=2, default=str))
    return 0 if ast["passed"] else 2


def cmd_actions(_):
    print(json.dumps(list(ACTION_NAMES)))
    return 0


def main(argv=None):
    p = argparse.ArgumentParser(prog="p2cq_research", description="P2C-Q finite skill benchmark tools")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("help").set_defaults(func=lambda n: cmd_help())
    sub.add_parser("version").set_defaults(func=lambda n: (print("P2CQ_RESEARCH_V1"), 0)[1] or 0)
    e = sub.add_parser("export")
    e.add_argument("--split", required=True, choices=("development", "confirmation"))
    e.add_argument("--out", required=True)
    e.set_defaults(func=cmd_export)
    a = sub.add_parser("audit")
    a.add_argument("--exp-root", required=True)
    a.add_argument("--repo", required=True)
    a.set_defaults(func=cmd_audit)
    sub.add_parser("actions").set_defaults(func=cmd_actions)
    ns = p.parse_args(argv)
    if ns.cmd == "version":
        print("P2CQ_RESEARCH_V1")
        return 0
    return ns.func(ns)


if __name__ == "__main__":
    raise SystemExit(main())
