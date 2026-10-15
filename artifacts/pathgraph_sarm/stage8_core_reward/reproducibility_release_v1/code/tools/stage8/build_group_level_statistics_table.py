#!/usr/bin/env python3
"""Build auditable, group-level—not frame-level—Stage 8 observations."""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import math
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq


def finite(value: float) -> bool:
    return math.isfinite(float(value))


def add(rows: list[dict], *, group: str, task: str, provenance: str, metric: str,
        value: float, method: str = "", variant: str = "", outcome: str = "",
        scenario: str = "", source: str = "", numerator: float | None = None,
        denominator: float | None = None) -> None:
    if not finite(value):
        raise ValueError(f"non-finite {metric} for {group}")
    rows.append({
        "content_group_id": group, "task_id": task, "provenance": provenance,
        "path_order": "", "scenario_type": scenario, "method": method,
        "variant": variant, "metric_name": metric, "metric_value": float(value),
        "model_seed": "ensemble", "outcome": outcome, "source": source,
        "event_numerator": None if numerator is None else float(numerator),
        "event_denominator": None if denominator is None else float(denominator),
    })


def read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ensemble-test", type=Path, required=True)
    parser.add_argument("--trace-returns", type=Path, required=True)
    parser.add_argument("--main-table", type=Path, required=True)
    parser.add_argument("--ablation-table", type=Path, required=True)
    parser.add_argument("--reward-transitions", type=Path, required=True)
    parser.add_argument("--ablation-transitions", type=Path, required=True)
    parser.add_argument("--group-key", default="content_group_id")
    parser.add_argument("--stratify", default="task_id,provenance")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    if args.group_key != "content_group_id":
        raise SystemExit("Stage 8 statistics unit is fixed to content_group_id")

    observations: list[dict] = []
    # Secondary, checkpoint-derived model observables are averaged within a
    # group. They never turn individual frames into independent observations.
    test_records = []
    with gzip.open(args.ensemble_test, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                test_records.append(json.loads(line))
    split_by_group: dict[str, set[str]] = defaultdict(set)
    raw_by_group: dict[str, list[dict]] = defaultdict(list)
    for record in test_records:
        split_by_group[record["content_group_id"]].add(record["split"])
        raw_by_group[record["content_group_id"]].append(record)
    crossing = {group: values for group, values in split_by_group.items() if len(values) != 1}
    if crossing:
        raise SystemExit(f"content groups cross splits: {crossing}")
    for group, records in raw_by_group.items():
        first = records[0]
        add(observations, group=group, task=first["task_id"], provenance="real_test_checkpoint_ensemble",
            metric="node_accuracy", value=sum(int(r["node_pred"] == r["gt_node_id"]) for r in records) / len(records),
            method="pathgraph_reward_v1_locked", outcome=first["outcome"], scenario=first["scenario"], source="ensemble_test_predictions")
        add(observations, group=group, task=first["task_id"], provenance="real_test_checkpoint_ensemble",
            metric="phi_mae", value=sum(abs(float(r["phi_mean"]) - float(r["gt_phi"])) for r in records) / len(records),
            method="pathgraph_reward_v1_locked", outcome=first["outcome"], scenario=first["scenario"], source="ensemble_test_predictions")

    # Episode returns are already one observation per content group.
    for row in read_csv(args.trace_returns):
        group = row.get("content_group_id", "")
        if not group:
            continue
        controlled = row.get("provenance") == "frozen_controlled_symbolic_trace"
        add(observations, group=group, task=row.get("task_id") or ("controlled_symbolic" if controlled else "unknown"),
            provenance="frozen_controlled_symbolic_trace" if controlled else "real_test_checkpoint_ensemble",
            metric="return", value=float(row["return"]), method=row.get("method", ""), outcome=row.get("outcome", ""),
            scenario=row.get("scenario", ""), source="reproduced_trace_returns")

    reward_transitions = read_csv(args.reward_transitions)
    event_groups: dict[tuple[str, str, str], list[dict]] = defaultdict(list)
    for row in reward_transitions:
        group = row.get("content_group_id", "") or f"controlled_trace:{row.get('trace_id', 'unknown')}"
        controlled = row.get("provenance") == "frozen_controlled_symbolic_trace"
        provenance = "frozen_controlled_symbolic_trace" if controlled else "real_test_checkpoint_ensemble"
        task = row.get("task_id") or ("controlled_symbolic" if controlled else "unknown")
        event_groups[(row.get("method", ""), group, provenance)].append({**row, "_task": task})
    for (method, group, provenance), rows in event_groups.items():
        task = rows[0]["_task"]
        def edge(row: dict) -> int:
            return int(row.get("edge_type_gt") or row.get("edge_type") or -1)
        failures = [row for row in rows if edge(row) == 4]
        recoveries = [row for row in rows if edge(row) == 3]
        if failures:
            numerator = sum(float(row["reward"]) < 0 for row in failures)
            add(observations, group=group, task=task, provenance=provenance, metric="failure_negative_rate",
                value=numerator / len(failures), method=method, source="reproduced_reward_transitions", numerator=numerator, denominator=len(failures))
        if recoveries:
            numerator = sum(float(row["reward"]) > 0 for row in recoveries)
            add(observations, group=group, task=task, provenance=provenance, metric="recovery_positive_rate",
                value=numerator / len(recoveries), method=method, source="reproduced_reward_transitions", numerator=numerator, denominator=len(recoveries))
        cycle = failures + recoveries
        if failures and recoveries:
            cycle_value = float(sum(float(row["reward"]) for row in cycle) <= 1e-12)
            add(observations, group=group, task=task, provenance=provenance, metric="recovery_cycle_nonpositive_rate",
                value=cycle_value, method=method, source="reproduced_reward_transitions", numerator=cycle_value, denominator=1)
        loops = [row for row in rows if str(row.get("trace_id", "")).startswith("failure_recovery_loop_x")]
        if loops:
            numerator = sum(float(row.get("loop_penalty", 0.0)) > 1e-12 for row in loops)
            add(observations, group=group, task=task, provenance=provenance, metric="positive_loop_rate",
                value=numerator / len(loops), method=method, source="reproduced_reward_transitions", numerator=numerator, denominator=len(loops))

    # Path consistency is a controlled, paired two-order calculation. It is a
    # single immutable pair, so later bootstrap output reports its n=1 support.
    trace_rows = read_csv(args.trace_returns)
    for method in sorted({row.get("method", "") for row in trace_rows}):
        values = {row.get("path_signature"): float(row["return"]) for row in trace_rows if row.get("method") == method and row.get("provenance") == "frozen_controlled_symbolic_trace"}
        if "legal_A_then_B" in values and "legal_B_then_A" in values:
            value = abs(values["legal_A_then_B"] - values["legal_B_then_A"]) / max(1.0, abs(values["legal_A_then_B"]), abs(values["legal_B_then_A"]))
            add(observations, group="controlled_pair:legal_orders", task="controlled_symbolic", provenance="frozen_controlled_symbolic_trace",
                metric="legal_path_normalized_gap", value=value, method=method, source="reproduced_trace_returns")

    # Core ablation observations: controlled alternatives/recovery plus real
    # test within-node density. No Stage-7 aggregate values are copied.
    ablation_metrics = read_csv(args.ablation_table.with_name("reproduced_core_ablation_metrics.csv"))
    for row in ablation_metrics:
        variant = row["variant_id"]
        for order, field in (("legal_A_then_B", "A_first_return"), ("legal_B_then_A", "B_first_return")):
            add(observations, group=f"controlled_trace:{order}", task="controlled_symbolic", provenance="frozen_controlled_symbolic_trace",
                metric="alternate_path_return", value=float(row[field]), variant=variant, source="reproduced_core_ablation_metrics")
    ablation_transitions = read_csv(args.ablation_transitions)
    ablation_grouped: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for row in ablation_transitions:
        group = row.get("content_group_id", "") or f"controlled_trace:{row.get('trace_id', 'unknown')}"
        ablation_grouped[(row["variant_id"], group)].append(row)
    for (variant, group), rows in ablation_grouped.items():
        controlled = group.startswith("controlled_trace:")
        provenance = "frozen_controlled_symbolic_trace" if controlled else "real_test_checkpoint_ensemble"
        task = "controlled_symbolic" if controlled else "transport_recovery"
        def edge(row: dict) -> int:
            return int(row.get("edge_type_gt") or row.get("edge_type") or -1)
        recovery = [row for row in rows if edge(row) == 3]
        if recovery:
            numerator = sum(float(row["reward"]) > 0 for row in recovery)
            add(observations, group=group, task=task, provenance=provenance, metric="recovery_positive_rate",
                value=numerator / len(recovery), variant=variant, source="reproduced_core_ablation_transitions", numerator=numerator, denominator=len(recovery))
            overshoot = sum(float(row["reward"]) > max(0.0, float(row.get("debt_before") or 0.0)) + 1e-12 for row in recovery)
            add(observations, group=group, task=task, provenance=provenance, metric="recovery_overshoot_rate",
                value=overshoot / len(recovery), variant=variant, source="reproduced_core_ablation_transitions", numerator=overshoot, denominator=len(recovery))
        same = [row for row in rows if str(row.get("same_predicted_node", "")).lower() == "true"]
        if same and not controlled:
            numerator = sum(abs(float(row["reward"])) > 1e-12 for row in same)
            add(observations, group=group, task=task, provenance=provenance, metric="within_node_reward_density",
                value=numerator / len(same), variant=variant, source="reproduced_core_ablation_transitions", numerator=numerator, denominator=len(same))

    frame = pd.DataFrame(observations)
    if frame.empty:
        raise SystemExit("no group-level observations")
    primary = {"legal_path_normalized_gap", "failure_negative_rate", "recovery_positive_rate", "recovery_cycle_nonpositive_rate", "success_failure_margin", "alternate_path_return", "recovery_overshoot_rate", "within_node_reward_density"}
    # success_failure_margin is constructed by bootstrap from per-group returns.
    invalid = frame[frame["metric_name"].isin(primary) & ~pd.to_numeric(frame["metric_value"], errors="coerce").map(math.isfinite)]
    if not invalid.empty:
        raise SystemExit("non-finite primary group observations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), args.output)
    summary = frame.groupby(["metric_name", "method", "variant", "provenance"], dropna=False).agg(observation_rows=("metric_value", "size"), content_groups=("content_group_id", "nunique"), min_value=("metric_value", "min"), max_value=("metric_value", "max")).reset_index()
    summary["statistics_unit"] = "content_group_id"
    summary["split_crossing_groups"] = len(crossing)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(args.summary, index=False)


if __name__ == "__main__":
    main()
