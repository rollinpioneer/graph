from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_v2.l2r_logical_observation_clock.guard import LogicalObservationClockGuard
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input
from upgrade_v2.l2r_rgb_temporal_confirmation.evaluation import _run_o

from .io_utils import write_json, write_jsonl


def apply_logical_guard(rows: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guard = LogicalObservationClockGuard()
    return [guard.step(observation, proposal) for observation, proposal in zip(rows, proposals)]


def run_online_rollout(rollout_root: Path) -> dict[str, Any]:
    rows = load_candidate_input(rollout_root / "candidate_input")
    b2 = _run_o(rows, "O_B2")
    c3 = _run_o(rows, "O_C3")
    clp2 = apply_logical_guard(rows, c3)
    output = rollout_root / "candidate_output"; output.mkdir(parents=True, exist_ok=False)
    write_jsonl(output / "O_B2_RAW.jsonl", b2)
    write_jsonl(output / "O_C3_RAW.jsonl", c3)
    write_jsonl(output / "O_C3_CLP2_LOGICAL_CLOCK.jsonl", clp2)
    status = {"schema": "l2rar2_r20_candidate_output_status_v1", "status": "PASS",
              "rows": len(rows), "methods": ["O_B2_RAW", "O_C3_RAW", "O_C3_CLP2_LOGICAL_CLOCK"]}
    write_json(output / "status.json", status)
    return {"rollout_id": rollout_root.name, **status}


def run_online_candidates(confirmation_root: Path) -> dict[str, Any]:
    results = [run_online_rollout(path.parent) for path in sorted(confirmation_root.glob("*/metadata.json"))]
    status = {"schema": "l2rar2_r20_candidate_execution_v1",
              "status": "PASS" if len(results) == 72 else "FAIL", "rollouts": len(results)}
    write_json(confirmation_root / "candidate_execution_status.json", status)
    return status
