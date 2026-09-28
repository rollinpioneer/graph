#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from cp_disr.analysis.s1_revision import (
    adjudicate_history,
    finalize_eligibility,
    freeze_discovery_manifest,
    freeze_history,
    freeze_source_revision,
    load_revision_config,
    offline_rescore_planner,
    probe_production_representation,
    register_physical_branches,
    run_provider_call,
    run_witnesses,
    verify_revision_output,
)


COMMANDS = (
    "freeze-history", "adjudicate-history", "freeze-source-revision",
    "prepare-discovery", "call-provider", "probe-representation",
    "register-witnesses", "run-witnesses", "rescore-planner", "finalize", "verify",
)


def main():
    parser = argparse.ArgumentParser()
    subs = parser.add_subparsers(dest="command", required=True)
    for command in COMMANDS:
        sub = subs.add_parser(command)
        sub.add_argument("--root", required=True)
        sub.add_argument("--config", required=True)
        sub.add_argument("--output", required=True)
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config = Path(args.config)
    if not config.is_absolute():
        config = root / config
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    os.environ["S1_REV1_COMMAND"] = args.command
    os.environ["S1_REV1_SOURCE_COMMIT"] = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    frozen = load_revision_config(config)
    if args.command == "freeze-history":
        result = freeze_history(root, config, output)
    elif args.command == "adjudicate-history":
        result = adjudicate_history(root, config, output)
    elif args.command == "freeze-source-revision":
        result = freeze_source_revision(root, config, output)
    elif args.command == "prepare-discovery":
        result = freeze_discovery_manifest(root, frozen, output)
    elif args.command == "call-provider":
        result = run_provider_call(root, None, output)
    elif args.command == "probe-representation":
        result = probe_production_representation(root, None, output)
    elif args.command == "register-witnesses":
        result = register_physical_branches(root, output)
    elif args.command == "run-witnesses":
        result = run_witnesses(root, output)
    elif args.command == "rescore-planner":
        result = offline_rescore_planner(root, None, output)
    elif args.command == "finalize":
        result = finalize_eligibility(root, output)
    elif args.command == "verify":
        result = verify_revision_output(root, output)
    else:
        raise AssertionError(args.command)
    print(json.dumps(result, sort_keys=True, ensure_ascii=False))


if __name__ == "__main__":
    main()
