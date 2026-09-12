from __future__ import annotations

from pathlib import Path

from upgrade_v2.l2r_logical_clock_confirmation.io_utils import read_csv, read_jsonl
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import load_candidate_input

PHYSICS_STEP_NS = 10_000_000


def load_with_canonical_time(rollout_root: Path) -> list[dict]:
    rows = load_candidate_input(rollout_root / "candidate_input")
    contacts = {int(row["capture_order"]): row for row in read_jsonl(rollout_root / "candidate_input/contact_proxy.jsonl")}
    if all(isinstance(contacts[int(row["capture_order"])].get("physical_time_ns"), int) for row in rows):
        return [{**row, "physical_time_ns": contacts[int(row["capture_order"])]["physical_time_ns"]} for row in rows]
    frames = {int(row["capture_order"]): int(row["physics_step_index"])
              for row in read_csv(rollout_root / "online_raw/frame_manifest.csv")}
    logical_path = rollout_root / "online_raw/logical_observation_manifest.csv"
    mapping = ({int(row["candidate_capture_order"]): int(row["source_capture_order"])
                for row in read_csv(logical_path)} if logical_path.is_file() else {})
    enriched = []
    for row in rows:
        order = int(row["capture_order"]); source_order = mapping.get(order, order)
        if source_order not in frames: raise RuntimeError(f"CANONICAL_TIME_SOURCE_MISSING:{rollout_root.name}:{order}")
        enriched.append({**row, "physical_time_ns": frames[source_order] * PHYSICS_STEP_NS})
    return enriched
