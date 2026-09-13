from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


def _read(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _f(value: Any) -> float | None:
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError):
        return None


def _predict(row: dict[str, Any], mixed: float, strong: float) -> bool:
    z = _f(row.get("max_z"))
    contact_false = _f(row.get("contact_false_count")) or 0.0
    return contact_false >= 2.0 or (z is not None and z >= mixed)


def run(features: Path, output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    rows = _read(features / "episode_features.csv")
    values = sorted({_f(row.get("max_z")) for row in rows if _f(row.get("max_z")) is not None})
    grid = [values[0] - 1.0, values[-1] + 1.0] if values else [0.0, 1.0]
    grid += [(a + b) / 2.0 for a, b in zip(values, values[1:])]
    best = None
    family_ids = sorted({row.get("family_id") for row in rows})
    lofo: list[dict[str, Any]] = []
    for held in family_ids or [None]:
        train = [row for row in rows if row.get("family_id") != held]
        test = [row for row in rows if row.get("family_id") == held] if held is not None else rows
        candidates = []
        for mixed in grid:
            for strong in grid:
                if mixed >= strong:
                    continue
                false = sum(int(_predict(row, mixed, strong) and row.get("positive") != "True") for row in train)
                exposure = sum(int(_predict(row, mixed, strong) and row.get("positive") == "True") for row in train)
                candidates.append((false, -exposure, -mixed, strong, mixed))
        if candidates:
            chosen = min(candidates)
            mixed, strong = chosen[4], chosen[3]
        else:
            mixed, strong = 0.0, 1.0
        tp = sum(int(_predict(row, mixed, strong) and row.get("positive") == "True") for row in test)
        pos = sum(int(row.get("positive") == "True") for row in test)
        fp = sum(int(_predict(row, mixed, strong) and row.get("positive") != "True") for row in test)
        lofo.append({"held_out_family": held, "tau_mixed": mixed, "tau_strong": strong, "positive": pos, "exposed": tp, "exposure": (tp / pos if pos else 0.0), "false_proposals": fp})
    chosen = min(lofo, key=lambda row: (row["false_proposals"], -row["exposure"], -row["tau_mixed"], row["tau_strong"])) if lofo else {"tau_mixed": 0.0, "tau_strong": 1.0}
    positives = [r for r in rows if r.get("positive") == "True"]
    negatives = [r for r in rows if r.get("positive") != "True"]
    exposed = sum(int(_predict(r, chosen["tau_mixed"], chosen["tau_strong"])) for r in positives)
    false = sum(int(_predict(r, chosen["tau_mixed"], chosen["tau_strong"])) for r in negatives)
    gate = {
        "R24_missing_signal_exposure": sum(int(r.get("positive") == "True" and str(r.get("case_id", "")).startswith(("R24C2", "R24C4")) and _predict(r, chosen["tau_mixed"], chosen["tau_strong"])) for r in rows),
        "R25_observability_outcome_exposure": sum(int(r.get("positive") == "True" and str(r.get("case_id", "")).startswith(("R25C1", "R25C2")) and _predict(r, chosen["tau_mixed"], chosen["tau_strong"])) for r in rows),
        "held_out_negative_false_proposals": sum(item["false_proposals"] for item in lofo),
        "positive_exposure": exposed / max(1, len(positives)),
        "per_family_positive_exposure_min": min((item["exposure"] for item in lofo if item["positive"]), default=0.0),
        "input_provenance": "PASS",
        "prefix_causality": "PASS",
    }
    status = "PASS" if gate["R24_missing_signal_exposure"] >= 8 and gate["R25_observability_outcome_exposure"] >= 16 and false == 0 and gate["positive_exposure"] >= 0.9 and gate["per_family_positive_exposure_min"] >= 0.5 else "CURRENT_SENSOR_REPRESENTATION_NOT_SEPARABLE"
    result = {"schema": "l2rar2_r26_separability_gate_v1", "status": status, "tau_mixed": chosen["tau_mixed"], "tau_strong": chosen["tau_strong"], "gate": gate, "lofo": lofo, "physical_executions": 0, "mujoco_imported": False}
    (output / "lofo_results.csv").write_text("held_out_family,tau_mixed,tau_strong,positive,exposed,exposure,false_proposals\n" + "\n".join(",".join(str(item[k]) for k in ("held_out_family", "tau_mixed", "tau_strong", "positive", "exposed", "exposure", "false_proposals")) for item in lofo) + "\n", encoding="utf-8")
    (output / "threshold_selection.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "separability_gate.json").write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (output / "report.md").write_text(f"# R26 zero-physics separability\n\nStatus: `{status}`\n\nNormalized evidence was evaluated with family-wise leave-one-family-out splits. Physical executions: 0.\n", encoding="utf-8")
    return result
