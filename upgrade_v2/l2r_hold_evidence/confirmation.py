from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .inputs import write_json


def confirm(selection_lock: Path, protocol: dict[str, Any], inputs: dict[str, Any], output_root: Path) -> dict[str, Any]:
    lock = json.loads(selection_lock.read_text(encoding="utf-8"))
    output_root.mkdir(parents=True, exist_ok=True)
    if not lock.get("ready_for_confirmation") or not lock.get("selected_candidate_id"):
        result = {"schema": "pathgraph_l2rar1_confirmation_consumption_v1", "status": "NOT_RUN_DEVELOPMENT_FAILED", "reason": "selection_lock.ready_for_confirmation is false or selected_candidate_id is null", "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN", "new_data_consumed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    else:
        result = {"schema": "pathgraph_l2rar1_confirmation_consumption_v1", "status": "NOT_RUN_CONFIRMATION_GENERATOR_REQUIRED", "reason": "Confirmation data generator is intentionally not reused from the historical L2RA cache; no independent R4 family was consumed in this run.", "standard_confirmation_status": "NOT_RUN", "challenge_confirmation_status": "NOT_RUN", "new_data_consumed": False, "api_calls": 0, "training_jobs": 0, "api_key_read": False}
    write_json(output_root / "confirmation_consumption.json", result)
    write_json(output_root / "confirmation_status.json", result)
    return result
