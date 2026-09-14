"""Separate execution status from scientific claims."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Any

from .util import write_csv, write_json, write_text


def build(payload: dict[str, Any]) -> dict[str, Any]:
    decision = {
        "schema": "p1_v3_final_decision_v1",
        "execution_status": payload.get("execution_status", "PASS"),
        "source_parent_commit": "de3e54310270eeb47e00a99745b3ea6987aa8795",
        "runner_commit": payload.get("runner_commit"),
        "result_commit": payload.get("result_commit"),
        "parent_preserved": payload.get("parent_preserved"),
        "engine_mirror_parity": payload.get("engine_mirror_parity"),
        "long_phi_loop": payload.get("long_phi_loop"),
        "residual_debt": payload.get("residual_debt"),
        "dual_order_guard": payload.get("dual_order_guard"),
        "causal_replay": payload.get("causal_replay"),
        "G1": payload.get("G1"),
        "overall_claim": payload.get("overall_claim"),
        "new_physical_runs": 0,
        "new_training_runs": 0,
        "new_llm_calls": 0,
        "confirmation_passed": False,
        "policy_gain_claimed": False,
        "visual_grounding_confirmed": False,
    }
    return decision


def write_final(out: Path, payload: dict[str, Any]) -> None:
    decision = build(payload)
    write_json(out / "decision.json", decision)
    report = []
    report.append("# P1 continuation V3")
    report.append("")
    report.append("## Execution vs science")
    report.append("Packaged first-pass tools and agent modules completed without new physics, training, or LLM calls.")
    report.append("Scientific claims below are scoped. confirmation_passed remains false.")
    report.append("")
    report.append("## Long phi loop")
    report.append(json.dumps(payload.get("long_phi_loop"), ensure_ascii=False, indent=2))
    report.append("")
    report.append("## Residual debt")
    report.append(json.dumps(payload.get("residual_debt"), ensure_ascii=False, indent=2))
    report.append("")
    report.append("## Dual-order guard")
    report.append(json.dumps(payload.get("dual_order_guard"), ensure_ascii=False, indent=2))
    report.append("")
    report.append("## Causal replay")
    report.append(json.dumps(payload.get("causal_replay"), ensure_ascii=False, indent=2))
    report.append("")
    report.append("## G1")
    report.append(json.dumps(payload.get("G1"), ensure_ascii=False, indent=2))
    report.append("")
    report.append("## Claims that are not made")
    report.append("- G1 does not drive the controller.")
    report.append("- 24 symbolic episodes are not 24 physical experiments.")
    report.append("- Residual debt without suffix reward change is not harm, and not a license to reset debt.")
    write_text(out / "report.md", "\n".join(report))
    write_csv(out / "claim_scope.csv", payload["claim_scope"])
    write_json(out / "result_manifest.json", payload.get("result_manifest", {}))
    write_text(out / "external_artifacts.tsv", payload.get("external_tsv", "role\tpath\tsha256\n"))
    write_json(out / "run_manifest.json", payload.get("run_manifest", {}))