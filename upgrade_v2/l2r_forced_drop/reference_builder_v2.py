from __future__ import annotations

import csv
import json
import os
from pathlib import Path


def _case_family(row):
    return str(row.get("case_id", "")).split("_", 1)[0]


def build_reference_v2(development_root: Path, output_root: Path):
    output_root.mkdir(parents=True, exist_ok=False)
    rows = []
    for path in sorted(development_root.rglob("reference/physical_reference.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        value["reference_path"] = os.path.relpath(path, output_root)
        value["online_trace_path"] = os.path.relpath(
            path.parent.parent / "online" / "physics_trace.jsonl", output_root)
        rows.append(value)
    counts = {f"F{i}": 0 for i in range(1, 9)}
    for row in rows:
        counts[_case_family(row)] = counts.get(_case_family(row), 0) + 1
    trace_complete = sum(bool(r.get("trace_complete")) for r in rows)
    numeric_pass = sum(bool(r.get("numeric_health_pass")) for r in rows)
    prehold_cases = {f"F{i}": sum(bool(r.get("pre_hold_verified")) for r in rows
                                    if _case_family(r) == f"F{i}") for i in range(1, 9)}
    forced_cases = {f"F{i}": sum(bool(r.get("physical_loss_confirmed")) for r in rows
                                   if _case_family(r) == f"F{i}") for i in (5, 6, 8)}
    no_loss_f3 = sum(not bool(r.get("physical_loss_confirmed")) for r in rows
                     if _case_family(r) == "F3")
    release_f7 = sum(bool(r.get("commanded_release")) and
                     r.get("reference_action") == "none" for r in rows
                     if _case_family(r) == "F7")
    resolvable = {f"F{i}": sum(bool(r.get("resolvable")) for r in rows
                                if _case_family(r) == f"F{i}") for i in (1, 2, 4)}
    gate = {
        "schema": "l2rar2_r16_generator_gate_v2",
        "generator_gate": True,
        "rollouts": len(rows),
        "case_counts": counts,
        "trace_complete": trace_complete == 32 and all(counts[f"F{i}"] == 4 for i in range(1, 9)),
        "trace_complete_count": trace_complete,
        "numeric_health": numeric_pass == 32,
        "numeric_health_count": numeric_pass,
        "prehold_24_of_24": sum(prehold_cases[f"F{i}"] for i in range(3, 9)) == 24,
        "prehold_counts_F3_F8": {f"F{i}": prehold_cases[f"F{i}"] for i in range(3, 9)},
        "forced_loss_12_of_12": sum(forced_cases.values()) == 12,
        "forced_loss_counts_F5_F6_F8": forced_cases,
        "F3_no_loss_4_of_4": no_loss_f3 == 4,
        "F3_no_loss_count": no_loss_f3,
        "F7_commanded_release_4_of_4": release_f7 == 4,
        "F7_commanded_release_count": release_f7,
        "COMMANDED_RELEASE": release_f7,
        "resolvable_F1_F2_F4": all(resolvable[f"F{i}"] == 4 for i in (1, 2, 4)),
        "resolvable_counts_F1_F2_F4": resolvable,
    }
    gate["candidate_evaluation_allowed"] = bool(
        gate["trace_complete"] and gate["numeric_health"] and gate["prehold_24_of_24"]
        and gate["forced_loss_12_of_12"] and gate["F3_no_loss_4_of_4"]
        and gate["F7_commanded_release_4_of_4"] and gate["resolvable_F1_F2_F4"])
    fields = ("family_id", "case_id", "trace_complete", "numeric_health_pass",
              "pre_hold_verified", "physical_loss_confirmed", "commanded_release")
    with (output_root / "physical_reference_events.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row.get(key, False) for key in fields} for row in rows)
    (output_root / "generator_gate.json").write_text(json.dumps(gate, indent=2) + "\n", encoding="utf-8")
    (output_root / "physical_reference_index.json").write_text(
        json.dumps({"schema": "l2rar2_r16_physical_reference_index_v2", "rows": rows}, indent=2) + "\n",
        encoding="utf-8")
    return gate
