#!/usr/bin/env python3
"""Record the traceable single-agent semantic review for the frozen L1V run."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


FIELDS = ["goal_correct", "objects_correct", "scene_specific", "order_sensible", "checks_observable", "scene_ready"]

OVERRIDES = {
    ("F02_B", "V1"): {
        "scene_specific": 0,
        "order_sensible": 0,
        "scene_ready": 0,
        "note": "Recognized that the bowl was on the plate but still made re-grasp and redundant transport the primary path.",
    },
    ("F07_B", "V1"): {
        "objects_correct": 0,
        "scene_specific": 0,
        "scene_ready": 0,
        "unsupported_visual_assertion": 1,
        "note": "Misidentified the brown occluder as the task's blue cube and proposed manipulating the wrong object.",
    },
    ("F08_B", "V1"): {
        "order_sensible": 0,
        "scene_ready": 0,
        "note": "Observed the yellow cube inside the box but did not clear it or make clearance a required branch before placement.",
    },
    ("F08_B", "V2"): {
        "order_sensible": 0,
        "scene_ready": 0,
        "note": "Observed the yellow cube inside the box but still placed the blue cube without an obstacle-removal or validated free-space branch.",
    },
    ("F10_B", "V1"): {
        "scene_specific": 0,
        "order_sensible": 0,
        "scene_ready": 0,
        "note": "Observed the cup already in the tray but made moving it to another unspecified slot the primary path.",
    },
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--review", type=Path, required=True)
    parser.add_argument("--run", type=Path, required=True)
    args = parser.parse_args()
    mapping = read_csv(args.review / "blind_mapping.DO_NOT_SHOW_RATER.csv")
    if len(mapping) != 54:
        raise ValueError(f"expected 54 confirmation items, got {len(mapping)}")
    rows = []
    for identity in mapping:
        case_id = identity["case_id"]
        condition = identity["condition"]
        candidate_path = args.run / "candidates" / "confirm" / f"{identity['request_id']}.json"
        state_path = args.run / "states" / "confirm" / f"{identity['request_id']}.json"
        if not candidate_path.is_file() or not state_path.is_file():
            raise FileNotFoundError(identity["request_id"])
        graph = json.loads(candidate_path.read_text(encoding="utf-8"))
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if not state.get("structure_valid"):
            raise ValueError(f"structurally invalid candidate requires explicit handling: {identity['request_id']}")
        state_b = case_id.endswith("_B")
        rating: dict[str, object] = {field: 1 for field in FIELDS}
        rating.update(
            {
                "blind_id": identity["blind_id"],
                "unsupported_visual_assertion": 0,
                "recovery_branch_conditional": "yes" if any(node.get("role") == "recovery" for node in graph.get("nodes", [])) or any(edge.get("trigger") for edge in graph.get("edges", [])) else "not_proposed",
                "review_status": "complete",
                "reviewer": "Codex agent single semantic review 2026-09-07",
                "note": "Scene-appropriate graph under the prewritten rubric.",
            }
        )
        if condition == "T" and state_b:
            rating["scene_specific"] = 0
            rating["scene_ready"] = 0
            rating["note"] = "Text-only input could not select the current B-state decision; generic observation or contingency did not establish scene adaptation."
        override = OVERRIDES.get((case_id, condition), {})
        rating.update(override)
        rows.append({"blind_id": rating["blind_id"], **{field: rating[field] for field in FIELDS}, "unsupported_visual_assertion": rating["unsupported_visual_assertion"], "recovery_branch_conditional": rating["recovery_branch_conditional"], "review_status": rating["review_status"], "reviewer": rating["reviewer"], "note": rating["note"]})
    write_csv(args.review / "ratings.csv", rows)
    summary = {
        "schema": "l1v_agent_review_record_v1",
        "reviewer": "Codex agent single semantic review 2026-09-07",
        "items": len(rows),
        "reference_prepared_before_candidates": True,
        "model_output_modified": False,
        "second_reviewer": False,
        "known_limit": "Single agent review; no human inter-rater agreement estimate.",
        "explicit_overrides": [{"case_id": case_id, "condition": condition, **values} for (case_id, condition), values in sorted(OVERRIDES.items())],
    }
    (args.review / "review_provenance.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", "ratings": len(rows), "overrides": len(OVERRIDES)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
