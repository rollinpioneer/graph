"""Stable contracts for the PathGraph-SARM U4 B+ pipeline.

The module intentionally keeps the wire format plain JSON/CSV friendly.  The
scientific pipeline can therefore be resumed or audited without importing a
training framework.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
from enum import Enum
from typing import Any


class Route(str, Enum):
    REPAIR_EXECUTION_ONLY = "REPAIR_EXECUTION_ONLY"
    CONTINUE_WITH_FALLBACK = "CONTINUE_WITH_FALLBACK"
    RUN_U2R = "RUN_U2R"
    RUN_U3B = "RUN_U3B"
    CONTINUE_U4 = "CONTINUE_U4"


class FinalStatus(str, Enum):
    SCOPED_SUPPORT = "U4_COMPLETE_WITH_SCOPED_SUPPORT"
    PARTIAL = "U4_COMPLETE_PARTIAL"
    NO_EDIT_GAIN = "U4_COMPLETE_NO_EDIT_GAIN"
    BLOCKED = "U4_BLOCKED_INPUT_OR_EXECUTION"


@dataclass(frozen=True)
class Protocol:
    version: str = "u4_bplus_v1"
    source_commit: str = "4c3520d6ca07b5f39e2b5a7e0ea3cb15a5c2e1f6"
    family_seed: int = 840100
    family_total: int = 36
    dev_fit: int = 12
    dev_route: int = 12
    confirm: int = 12
    rollouts_per_family: int = 4
    continuation_horizon: int = 32
    dev_continuation_limit: int = 144
    confirm_continuation_limit: int = 96
    max_query_classes: int = 6
    max_accepted_edits: int = 6
    bootstrap: int = 5000
    event_tolerance: int = 2
    stable_recontact_steps: int = 2
    candidate_family_rate: float = 0.80
    minimum_candidate_families: int = 3
    api_authorized: bool = False

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data.update({
            "scope": "same_explicit_stochastic_simulator_distribution",
            "history_read_only": True,
            "main_api_calls": 0,
            "main_training_jobs": 0,
            "confirmation_once": True,
            "hidden_features_allowed": False,
            "u2_before_u3": True,
            "u2_repair_limits": {"jobs": 3, "steps": 1200, "clips": 40, "unique_frames": 480},
            "u3b_limits": {"qwen": 2, "deepseek": 1, "format_repairs": 1, "total_sends": 4},
            "edit_limits": {"splits": 2, "adds": 2, "deletes": 2},
        })
        return data


def load_protocol(path) -> Protocol:
    """Load YAML when available, with a deterministic lightweight fallback."""
    import json
    from pathlib import Path

    raw = Path(path).read_text(encoding="utf-8")
    try:
        import yaml  # type: ignore
        value = yaml.safe_load(raw) or {}
    except Exception:
        value = {}
        for line in raw.splitlines():
            if ":" in line and not line.startswith(" "):
                key, val = line.split(":", 1)
                value[key.strip()] = val.strip().strip("'\"")
    flat = {}
    family = value.get("family", {}) if isinstance(value, dict) else {}
    cont = value.get("continuations", {}) if isinstance(value, dict) else {}
    flat.update({
        "version": value.get("version", "u4_bplus_v1"),
        "source_commit": value.get("source_commit", Protocol.source_commit),
        "family_seed": family.get("seed", 840100),
        "family_total": family.get("total", 36),
        "dev_fit": family.get("dev_fit", 12),
        "dev_route": family.get("dev_route", 12),
        "confirm": family.get("confirm", 12),
        "rollouts_per_family": family.get("rollouts_per_family", 4),
        "continuation_horizon": cont.get("horizon", 32),
        "dev_continuation_limit": cont.get("dev_limit", 144),
        "confirm_continuation_limit": cont.get("confirm_limit", 96),
        "max_query_classes": cont.get("max_query_classes", 6),
        "api_authorized": bool(value.get("api_authorized", False)),
    })
    return Protocol(**{k: v for k, v in flat.items() if k in Protocol.__dataclass_fields__})


def metric_record(name: str, numerator: int, denominator: int, *, family_count: int = 0,
                  split: str = "", provenance: str = "", label_origin: str = "") -> dict[str, Any]:
    return {
        "metric": name, "numerator": int(numerator), "denominator": int(denominator),
        "eligible_family_count": int(family_count), "split": split,
        "provenance": provenance, "label_origin": label_origin,
        "value": (float(numerator) / denominator) if denominator else None,
        "status": "estimable" if denominator else "not_estimable",
    }
