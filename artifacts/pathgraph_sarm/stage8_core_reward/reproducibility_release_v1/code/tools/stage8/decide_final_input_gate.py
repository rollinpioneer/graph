#!/usr/bin/env python3
"""Decide the Stage 8 input gate from frozen machine-readable evidence."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from tools.stage8.common import dump_json, load_yaml, read_csv


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--g4", type=Path, required=True)
    parser.add_argument("--claim-scope", type=Path, required=True)
    parser.add_argument("--input-index", type=Path, required=True)
    parser.add_argument("--checkpoint-check", type=Path, required=True)
    parser.add_argument("--evidence-registry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    g4 = json.loads(args.g4.read_text(encoding="utf-8"))
    scope = json.loads(args.claim_scope.read_text(encoding="utf-8"))
    index = load_yaml(args.input_index)
    checkpoints = json.loads(args.checkpoint_check.read_text(encoding="utf-8"))
    registry = read_csv(args.evidence_registry)
    unsupported = index.get("unsupported_extensions", {})
    ok = (
        g4.get("decision") == "GO_STAGE8_CORE_REWARD_ONLY"
        and g4.get("alternative_structural_support") is True
        and g4.get("recovery_structural_support") is True
        and g4.get("scaling_extension_supported") is False
        and g4.get("order_holdout_extension_supported") is False
        and checkpoints.get("sha256_match_count") == 3
        and checkpoints.get("torch_load_ok_count") == 3
        and scope.get("new_training_allowed") is False
        and scope.get("main_reward_retuning_allowed") is False
        and all(value is False for value in unsupported.values())
        and bool(registry)
    )
    decision = "FINAL_SCOPE_AND_INPUTS_LOCKED" if ok else "REPAIR_FINAL_INPUT_PATHS"
    result = {
        "decision": decision,
        "g4_r1": g4.get("decision"),
        "checkpoints_verified": checkpoints.get("sha256_match_count"),
        "checkpoint_loads": checkpoints.get("torch_load_ok_count"),
        "claim_scope_locked": scope.get("locked_before_stage8_reproduction") is True,
        "unsupported_claims_marked_false": all(value is False for value in unsupported.values()),
        "new_training_allowed": scope.get("new_training_allowed"),
        "evidence_rows": len(registry),
    }
    dump_json(args.output, result)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(f"# Final Input Gate\n\nDecision: `{decision}`.\n", encoding="utf-8")
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
