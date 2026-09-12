from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input

from .io_utils import read_csv, write_csv, write_json


def audit_r17_clocks(r17_data_root: Path, output_root: Path) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=False)
    groups, rows = [], []
    rollouts = sorted((r17_data_root / "confirmation").glob("*/metadata.json"))
    for metadata in rollouts:
        rollout = metadata.parent
        online = load_candidate_input(rollout / "candidate_input")
        frames = {int(row["capture_order"]): row for row in read_csv(rollout / "online_raw/frame_manifest.csv")}
        by_time: dict[float, list[dict[str, Any]]] = {}
        for row in online: by_time.setdefault(float(row["time"]), []).append(row)
        for physical_time, same_time in by_time.items():
            if len(same_time) < 2: continue
            groups.append({"rollout_id": rollout.name, "physical_time": physical_time,
                           "capture_orders": [int(row["capture_order"]) for row in same_time],
                           "count": len(same_time)})
            for row in same_time:
                frame = frames[int(row["capture_order"])]
                rows.append({"rollout_id": rollout.name, "case_id": rollout.name.split("__", 1)[1],
                             "physical_time": physical_time, "capture_order": row["capture_order"],
                             "phase": frame.get("phase"), "action": frame.get("action"),
                             "action_end": frame.get("action_end"), "contact_present": row.get("contact_present"),
                             "attempt_id": row.get("attempt_id"), "attempt_phase": row.get("attempt_phase")})
    affected = len({row["rollout_id"] for row in groups})
    case_counts = Counter(row["rollout_id"].split("__", 1)[1].split("_", 1)[0] for row in groups)
    summary = {"schema": "l2rar2_r19_r17_clock_forensics_v1", "rollouts": len(rollouts),
               "same_physical_time_groups": len(groups), "same_physical_time_rows": len(rows),
               "affected_rollouts": affected, "all_rollouts_affected": affected == len(rollouts),
               "all_groups_strict_capture_order": all(all(b > a for a, b in zip(group["capture_orders"], group["capture_orders"][1:])) for group in groups),
               "group_count_by_case": dict(sorted(case_counts.items()))}
    write_csv(output_root / "same_physical_time_observations.csv", rows)
    write_json(output_root / "clock_group_index.json", {"schema": "l2rar2_r19_clock_group_index_v1", "groups": groups})
    write_json(output_root / "clock_forensics_summary.json", summary)
    return summary

