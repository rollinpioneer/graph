from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TransportCase:
    case_id: str
    level: str | None
    phase_offset_ms: int | None
    camera_jitter: bool = False
    contact_proxy_dropout: bool = False
    rgb_dropout: bool = False
    commanded_release: bool = False


CASES = (
    TransportCase("T1_transport_stable_clean", None, None),
    TransportCase("T2_transport_camera_jitter", None, None, camera_jitter=True),
    TransportCase("T3_transport_single_contact_proxy_dropout", None, None, contact_proxy_dropout=True),
    TransportCase("T4_transport_weak_outcome", "weak", None),
    TransportCase("T5_transport_medium_outcome", "medium", None),
    TransportCase("T6_transport_strong_phase_00ms", "strong", 0),
    TransportCase("T7_transport_strong_phase_10ms", "strong", 10),
    TransportCase("T8_transport_strong_phase_20ms", "strong", 20),
    TransportCase("T9_transport_strong_phase_30ms", "strong", 30),
    TransportCase("T10_transport_strong_phase_40ms", "strong", 40),
    TransportCase("T11_transport_strong_phase_20ms_rgb_dropout", "strong", 20, rgb_dropout=True),
    TransportCase("T12_transport_commanded_release", None, None, commanded_release=True),
)

FAMILIES = tuple((f"L2RAR2_R18_CONF_{index:02d}_{884000+index}", 884000 + index,
                  88500000 + index * 100) for index in range(6))


def unique_keys() -> tuple[tuple[str, str, int], ...]:
    return tuple((family, case.case_id, seed_base + index)
                 for family, _, seed_base in FAMILIES for index, case in enumerate(CASES))
