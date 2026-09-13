from __future__ import annotations

import math
import statistics
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

CONTACT_MAX_GAP_NS = 100_000_000
HOLD_REQUIRED_OBSERVATIONS = 2


@dataclass(frozen=True)
class FusionProposal:
    selected_action: str = "none"
    reason_code: str = "NO_NORMALIZED_LOSS_EVIDENCE"
    proposal_source: str | None = None
    signal_observed: bool = False
    physical_time_ns: int | None = None
    capture_order: int | None = None
    z_evidence: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class LossObservabilityFusionV2_NormalizedEvidence:
    """Online proposal source using scale/noise-normalized RGB evidence.

    Parameters are frozen by the R26 zero-physics audit.  ``tau_mixed`` and
    ``tau_strong`` are supplied as locked values; no confirmation data is
    consulted here.  The proposal reason is intentionally the stable contract
    ``historical_hold_established_then_observed_non_release_contact_loss``.
    """

    def __init__(self, tau_mixed: float, tau_strong: float) -> None:
        if not (math.isfinite(tau_mixed) and math.isfinite(tau_strong) and tau_mixed < tau_strong):
            raise ValueError("tau_mixed must be finite and smaller than tau_strong")
        self.tau_mixed = float(tau_mixed)
        self.tau_strong = float(tau_strong)
        self.previous_order: int | None = None
        self.previous_ns: int | None = None
        self.hold_count = 0
        self.historical_hold = False
        self.anchor: tuple[float, float] | None = None
        self.scale = 1.0
        self.prehold_u: list[float] = []
        self.contact_count = 0
        self.last_contact_false_ns: int | None = None
        self.last_rgb_ns: int | None = None
        self.rgb_count = 0
        self.emitted = False

    @staticmethod
    def _relative(row: dict[str, Any]) -> tuple[float, float] | None:
        obj, grip = row.get("object_centroid"), row.get("gripper_centroid")
        if not (isinstance(obj, (list, tuple)) and len(obj) == 2 and isinstance(grip, (list, tuple)) and len(grip) == 2):
            return None
        try:
            value = (float(obj[0]) - float(grip[0]), float(obj[1]) - float(grip[1]))
        except (TypeError, ValueError):
            return None
        return value if all(math.isfinite(x) for x in value) else None

    def _z(self, relative: tuple[float, float] | None) -> float | None:
        if relative is None or self.anchor is None:
            return None
        u = math.dist(relative, self.anchor) / max(self.scale, 1e-6)
        if not self.prehold_u:
            return None
        center = statistics.median(self.prehold_u)
        deviations = [abs(x - center) for x in self.prehold_u]
        mad = statistics.median(deviations) if deviations else 0.0
        noise = max(1.4826 * mad, 1e-4)
        return (u - center) / noise

    def rearm_after_verified_hold(self) -> None:
        self.historical_hold = True
        self.hold_count = HOLD_REQUIRED_OBSERVATIONS
        self.emitted = False
        self.contact_count = 0
        self.last_contact_false_ns = None
        self.last_rgb_ns = None
        self.rgb_count = 0

    def step(self, row: dict[str, Any]) -> dict[str, Any]:
        ns, order = row.get("physical_time_ns"), row.get("capture_order")
        if not (isinstance(ns, int) and not isinstance(ns, bool) and isinstance(order, int)):
            return FusionProposal(reason_code="INVALID_OR_NONMONOTONIC_OBSERVATION").to_dict()
        if self.previous_order is not None and order <= self.previous_order:
            return FusionProposal(reason_code="INVALID_OR_NONMONOTONIC_OBSERVATION", physical_time_ns=ns, capture_order=order).to_dict()
        if self.previous_ns is not None and ns < self.previous_ns:
            return FusionProposal(reason_code="INVALID_OR_NONMONOTONIC_OBSERVATION", physical_time_ns=ns, capture_order=order).to_dict()
        self.previous_order, self.previous_ns = order, ns
        if row.get("context_valid") is not True:
            return FusionProposal(reason_code="INVALID_CONTEXT", physical_time_ns=ns, capture_order=order).to_dict()
        release = row.get("gripper_command") == "open" or row.get("requested_effect") == "RELEASE_OBJECT" or row.get("attempt_end_reason") == "release"
        if release:
            self.contact_count = self.rgb_count = 0
            self.last_contact_false_ns = self.last_rgb_ns = None
            self.emitted = False
            return FusionProposal(reason_code="RELEASE_SUPPRESSES_FALLBACK", physical_time_ns=ns, capture_order=order).to_dict()

        relative = self._relative(row)
        holding = row.get("gripper_command") == "closed" and row.get("requested_effect") == "HOLD_OBJECT" and row.get("contact_present") is True
        if holding:
            self.hold_count += 1
            if not self.historical_hold and relative is not None:
                self.prehold_u.append(0.0)
                self.anchor = tuple(statistics.median([relative[i]]) for i in range(2))
                area = float(row.get("object_area") or 1.0)
                self.scale = math.sqrt(max(area, 1e-6) / math.pi)
            if self.hold_count >= HOLD_REQUIRED_OBSERVATIONS:
                self.historical_hold = True
        elif not self.historical_hold:
            self.hold_count = 0
        if not self.historical_hold or self.emitted:
            return FusionProposal(reason_code="HISTORICAL_HOLD_NOT_ARMED" if not self.historical_hold else "FALLBACK_ALREADY_EMITTED", physical_time_ns=ns, capture_order=order).to_dict()

        contact = row.get("contact_present")
        if contact is False:
            if self.last_contact_false_ns is not None and 0 < ns - self.last_contact_false_ns <= CONTACT_MAX_GAP_NS:
                self.contact_count += 1
            elif self.last_contact_false_ns != ns:
                self.contact_count = 1
            self.last_contact_false_ns = ns
        elif contact is True:
            self.contact_count = 0
            self.last_contact_false_ns = None

        z = self._z(relative)
        rgb = bool(z is not None and z >= self.tau_mixed and float(row.get("gripper_confidence") or 0.0) >= 0.5)
        if rgb:
            if self.last_rgb_ns is not None and 0 < ns - self.last_rgb_ns <= CONTACT_MAX_GAP_NS:
                self.rgb_count += 1
            elif self.last_rgb_ns != ns:
                self.rgb_count = 1
            self.last_rgb_ns = ns
        else:
            self.rgb_count = 0
            self.last_rgb_ns = None

        source = None
        if self.contact_count >= 2 and z is not None and z >= self.tau_mixed:
            source = "MIXED_SINGLE_CONTACT_FALSE_PLUS_NORMALIZED_RGB"
        elif self.contact_count >= 2:
            source = "SUSTAINED_CONTACT_ABSENCE"
        elif self.rgb_count >= 2 and z is not None and z >= self.tau_strong:
            source = "STRONG_NORMALIZED_RGB_DETACH"
        if source is None:
            return FusionProposal(physical_time_ns=ns, capture_order=order, z_evidence=z).to_dict()
        self.emitted = True
        return FusionProposal("recover_object", "historical_hold_established_then_observed_non_release_contact_loss", source, True, ns, order, z).to_dict()


def run_fusion(rows: list[dict[str, Any]], tau_mixed: float, tau_strong: float) -> list[dict[str, Any]]:
    guard = LossObservabilityFusionV2_NormalizedEvidence(tau_mixed, tau_strong)
    return [guard.step(row) for row in rows]
