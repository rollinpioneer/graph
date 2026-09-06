"""Final U4R1 handoff with explicit scientific exit states."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .io import read_csv, read_json, write_json


def decide(selection: Path, metrics: Path, paired: Path, separability: Path, output: Path, report: Path) -> dict[str, Any]:
    selected = read_json(selection)
    metric_data = read_json(metrics) if metrics.suffix == ".json" else {"graphs": read_csv(metrics)}
    effects = read_csv(paired)
    sep = read_json(separability) if separability.is_file() else {}
    g2_gain = [float(x["effect"]) for x in effects if x.get("metric", "").startswith("typed") and x.get("effect") not in {None, ""}]
    blocked = not selected.get("selected_graph") or not metric_data.get("graphs")
    if blocked:
        exit_state = "EXECUTION_BLOCKED"
    elif any("SEMANTICS_NOT_OBSERVABLE" in str(x) for x in sep.get("results", [])):
        exit_state = "REFINE_U2_OBSERVABLES_ONCE"
    elif g2_gain and g2_gain[0] > 0:
        exit_state = "GO_U5_CONDITIONAL_GRAPH"
    else:
        exit_state = "STOP_AUTO_GRAPH_KEEP_MANUAL"
    payload = {"schema": "u4r1_final_handoff_v1", "exit_state": exit_state, "selected_graph": selected.get("selected_graph"), "d_gate": "CONTINUE_U4", "api_calls": 0, "api_key_read": False, "training_jobs": 0, "device": "cpu", "g2_minus_g1": effects, "limits": ["same explicit stochastic simulator family only", "horizon is censored_unknown and excluded from failure claims", "scenario/phase/gold_mode/future labels are analysis-only and not online features", "conditional semantics are not physical robot or new-task validation"], "next_action": exit_state}
    write_json(output, payload)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(f"# U4R1 final handoff\n\n- exit: `{exit_state}`\n- selected graph: `{selected.get('selected_graph')}`\n- API calls: `0`\n- training jobs: `0`\n- device: `cpu`\n", encoding="utf-8")
    return payload
