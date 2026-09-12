from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import write_json
from upgrade_v2.l2r_logical_observation_clock.guard import INTERCEPT_REASON, LogicalObservationClockGuard

from .protocol import FROZEN_BLOBS


def _observation(time: float, order: int, contact: bool, **extra: Any) -> dict[str, Any]:
    return {"time": time, "capture_order": order, "attempt_id": 1, "contact_present": contact,
            "gripper_command": "closed", "requested_effect": "HOLD_OBJECT", "context_valid": True,
            "attempt_end": False, "attempt_end_reason": None, **extra}


def _proposal(action: str = "none", reason: str = "no_action") -> dict[str, Any]:
    return {"selected_action": action, "reason_code": reason, "hold_state": "historical_hold_established"}


def _run(rows: list[dict[str, Any]], proposals: list[dict[str, Any]]) -> list[dict[str, Any]]:
    guard = LogicalObservationClockGuard()
    return [guard.step(row, proposal) for row, proposal in zip(rows, proposals)]


def run(repo: Path, output: Path) -> dict:
    trigger = _proposal("recover_object", INTERCEPT_REASON)
    neutral = _proposal()
    cases = []
    def check(name: str, rows: list[dict[str, Any]], proposals: list[dict[str, Any]], expected: tuple[str, str]) -> None:
        result = _run(rows, proposals)
        actual = (result[-1]["guard_state"], result[-1]["selected_action"])
        cases.append({"case_id": name, "passed": actual == expected, "expected": list(expected),
                      "actual": list(actual), "records": result})
    check("C01_same_time_false_false_confirms", [_observation(1,10,False),_observation(1,11,False)], [trigger,neutral], ("CONFIRMED","recover_object"))
    check("C02_same_time_false_true_clears", [_observation(1,10,False),_observation(1,11,True)], [trigger,neutral], ("CLEAR","none"))
    check("C03_later_050_false_confirms", [_observation(1,10,False),_observation(1.05,11,False)], [trigger,neutral], ("CONFIRMED","recover_object"))
    check("C04_nominal_100_boundary_confirms", [_observation(1,10,False),_observation(1.10,11,False)], [trigger,neutral], ("CONFIRMED","recover_object"))
    check("C05_later_110_clears", [_observation(1,10,False),_observation(1.11,11,False)], [trigger,neutral], ("CLEAR","none"))
    check("C06_release_open_clears", [_observation(1,10,False),_observation(1.05,11,False,gripper_command="open")], [trigger,neutral], ("CLEAR","none"))
    check("C07_release_request_clears", [_observation(1,10,False),_observation(1.05,11,False,requested_effect="RELEASE_OBJECT")], [trigger,neutral], ("CLEAR","none"))
    check("C08_attempt_end_clears", [_observation(1,10,False),_observation(1.05,11,False,attempt_end=True)], [trigger,neutral], ("CLEAR","none"))
    check("C09_attempt_change_clears", [_observation(1,10,False),_observation(1.05,11,False,attempt_id=2)], [trigger,neutral], ("CLEAR","none"))
    check("C10_equal_order_invalid_clears", [_observation(1,10,False),_observation(1.05,10,False)], [trigger,neutral], ("CLEAR","none"))
    check("C11_backwards_time_invalid_clears", [_observation(1,10,False),_observation(.99,11,False)], [trigger,neutral], ("CLEAR","none"))
    check("C12_nontrigger_never_pending", [_observation(1,10,False),_observation(1,11,False)], [neutral,neutral], ("PASS_THROUGH","none"))
    blob = subprocess.run(["git", "-C", str(repo), "hash-object", "upgrade_v2/l2r_logical_observation_clock/guard.py"],
                          capture_output=True, text=True, check=True).stdout.strip()
    result = {"schema": "l2rar2_r21_logical_clock_conformance_v1", "status": "PASS" if all(row["passed"] for row in cases) and blob == FROZEN_BLOBS["guard"] else "FAIL",
              "cases_passed": sum(row["passed"] for row in cases), "cases_total": len(cases),
              "guard_blob": blob, "expected_guard_blob": FROZEN_BLOBS["guard"], "mujoco_imported": "mujoco" in sys.modules,
              "physical_executions": 0, "cases": cases}
    write_json(output, result)
    return result
