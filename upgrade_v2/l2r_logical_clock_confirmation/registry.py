from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .io_utils import read_json


@dataclass(frozen=True)
class ConfirmationCase:
    case_id: str
    level: str | None = None
    phase_offset_ms: int | None = None
    camera_jitter: bool = False
    t3_dropout: bool = False
    t4_same_time_pair: bool = False
    rgb_dropout: bool = False
    commanded_release: bool = False


CASES = (
    ConfirmationCase("T1_transport_stable_clean"),
    ConfirmationCase("T2_transport_camera_jitter", camera_jitter=True),
    ConfirmationCase("T3_transport_single_contact_proxy_dropout", t3_dropout=True),
    ConfirmationCase("T4_transport_same_time_contact_false_then_true", t4_same_time_pair=True),
    ConfirmationCase("T5_transport_weak_outcome", "weak", 20),
    ConfirmationCase("T6_transport_medium_outcome", "medium", 20),
    ConfirmationCase("T7_transport_strong_phase_00ms", "strong", 0),
    ConfirmationCase("T8_transport_strong_phase_10ms", "strong", 10),
    ConfirmationCase("T9_transport_strong_phase_20ms", "strong", 20),
    ConfirmationCase("T10_transport_strong_phase_30ms", "strong", 30),
    ConfirmationCase("T11_transport_strong_phase_40ms_rgb_dropout", "strong", 40, rgb_dropout=True),
    ConfirmationCase("T12_transport_commanded_release", phase_offset_ms=20, commanded_release=True),
)


def load_families(registry_path: Path) -> tuple[tuple[str, int, int], ...]:
    value = read_json(registry_path)
    families = tuple((str(row["family_id"]), int(row["family_seed"]), int(row["rollout_seed_base"]))
                     for row in value["families"])
    expected = tuple((f"L2RAR2_R19_CONF_{index:02d}_{886000 + index}", 886000 + index,
                      88700000 + index * 100) for index in range(6))
    if families != expected or value.get("case_order") != [case.case_id for case in CASES] or value.get("rollouts") != 72:
        raise ValueError("R19_CONFIRMATION_REGISTRY_MISMATCH")
    return families


def unique_keys(registry_path: Path) -> tuple[tuple[str, str, int], ...]:
    return tuple((family, case.case_id, seed_base + index)
                 for family, _, seed_base in load_families(registry_path)
                 for index, case in enumerate(CASES))
