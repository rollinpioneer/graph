from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np


STATES = (
    "PREHOLD_UNVERIFIED", "WELD_SUPPORTED_HOLD", "WELD_OFF_CONTACT_SUPPORTED",
    "INCIPIENT_OR_UNRESOLVED", "PHYSICAL_LOSS_CONFIRMED", "COMMANDED_RELEASE",
    "NUMERICAL_INVALID", "TRACE_INCOMPLETE",
)


@dataclass(frozen=True)
class CaptureEnvelope:
    minimum: np.ndarray
    maximum: np.ndarray

    def contains(self, point: Any) -> bool:
        value = np.asarray(point, dtype=float)
        return bool(np.all(value >= self.minimum) and np.all(value <= self.maximum))


def box_corners(center: Any, half_extents: Any) -> np.ndarray:
    center = np.asarray(center, dtype=float)
    half = np.asarray(half_extents, dtype=float)
    if center.shape != (3,) or half.shape != (3,):
        raise ValueError("center and half_extents must be 3-vectors")
    signs = np.asarray([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=float)
    return center + signs * half


def finger_corner_envelope(fingers: Iterable[tuple[Any, Any]]) -> CaptureEnvelope:
    corners = np.concatenate([box_corners(center, half) for center, half in fingers], axis=0)
    return CaptureEnvelope(corners.min(axis=0), corners.max(axis=0))


def inflate_for_object(envelope: CaptureEnvelope, object_half_extents: Any, clearance_m: float = 0.005) -> CaptureEnvelope:
    half = np.asarray(object_half_extents, dtype=float)
    if half.shape != (3,) or clearance_m < 0:
        raise ValueError("object_half_extents must be a non-negative 3-vector")
    margin = half + clearance_m
    return CaptureEnvelope(envelope.minimum - margin, envelope.maximum + margin)


def support_force_ratio(normal_forces_n: Iterable[float], mass_kg: float, gravity_mps2: float = 9.81) -> float:
    if mass_kg <= 0 or gravity_mps2 <= 0:
        raise ValueError("mass and gravity must be positive")
    support = sum(max(float(value), 0.0) for value in normal_forces_n)
    return support / (mass_kg * gravity_mps2)


def summarize_contacts(contacts: Iterable[dict[str, Any]], allowed_pairs: set[tuple[str, str]]) -> dict[str, Any]:
    forces: list[float] = []
    pairs: list[tuple[str, str]] = []
    for contact in contacts:
        pair = (str(contact.get("geom1")), str(contact.get("geom2")))
        reverse = (pair[1], pair[0])
        if pair not in allowed_pairs and reverse not in allowed_pairs:
            continue
        if int(contact.get("exclude", 0)) != 0 or int(contact.get("efc_address", -1)) < 0:
            continue
        if contact.get("normal_force") is None:
            continue
        pairs.append(pair)
        forces.append(float(contact["normal_force"]))
    return {"pairs": pairs, "normal_forces_n": forces, "support_force_n": sum(max(v, 0.0) for v in forces)}


def evaluate_loss_trace(rows: list[dict[str, Any]], *, dt_s: float = 0.01, mass_kg: float = 0.18,
                        pre_hold_verified: bool | None = None,
                        force_start_time: float | None = None) -> dict[str, Any]:
    if not rows:
        return {"state": "TRACE_INCOMPLETE", "physical_loss_confirmed": False}
    for row in rows:
        for key in ("object_xyz", "gripper_xyz"):
            if key in row and not np.isfinite(np.asarray(row[key], dtype=float)).all():
                return {"state": "NUMERICAL_INVALID", "physical_loss_confirmed": False}
        health = row.get("numeric_health") or {}
        speed = row.get("object_speed_mps", health.get("object_speed_mps", 0.0))
        angular = row.get("object_angular_speed_rads", health.get("object_angular_speed_rads", 0.0))
        z = row.get("object_z_m", health.get("object_z_m", 1.0))
        if float(speed) >= 100.0 or float(angular) >= 500.0 or float(z) <= -5.0:
            return {"state": "NUMERICAL_INVALID", "physical_loss_confirmed": False}
    verified = bool(rows[0].get("pre_hold_verified", False)) if pre_hold_verified is None else bool(pre_hold_verified)
    if not verified:
        return {"state": "PREHOLD_UNVERIFIED", "physical_loss_confirmed": False}
    if any(bool(row.get("commanded_release", False)) for row in rows):
        return {"state": "COMMANDED_RELEASE", "physical_loss_confirmed": False}
    outside_needed = max(1, int(np.ceil(0.10 / dt_s)))
    support_needed = max(1, int(np.ceil(0.05 / dt_s)))
    outside_run = support_run = 0
    outside_start = support_start = None
    loss_start = loss_confirmed = None
    for index, row in enumerate(rows):
        outside = bool(row.get("outside_capture", False))
        absent = float(row.get("support_force_ratio_mg", 1.0)) < 0.05
        outside_run = outside_run + 1 if outside else 0
        support_run = support_run + 1 if absent else 0
        if outside_run == outside_needed:
            outside_start = index - outside_needed + 1
        if support_run == support_needed:
            support_start = index - support_needed + 1
        if outside_run >= outside_needed and support_run >= support_needed:
            candidates = [x for x in (outside_start, support_start) if x is not None]
            loss_start = min(candidates)
            loss_confirmed = index
            break
    if loss_confirmed is not None:
        onset_abs = float(rows[loss_start].get("time", loss_start * dt_s))
        confirmed_abs = float(rows[loss_confirmed].get("time", loss_confirmed * dt_s))
        force_delay = None if force_start_time is None else confirmed_abs - float(force_start_time)
        return {"state": "PHYSICAL_LOSS_CONFIRMED", "physical_loss_confirmed": True,
                "loss_onset_index": loss_start, "loss_confirmed_index": loss_confirmed,
                "loss_onset_time_abs": onset_abs, "loss_confirmed_time_abs": confirmed_abs,
                "loss_onset_time_s": onset_abs, "loss_confirmed_time_s": confirmed_abs,
                "loss_confirmed_delay_from_force_s": force_delay}
    if any(bool(row.get("outside_capture", False)) for row in rows) or any(float(row.get("support_force_ratio_mg", 1.0)) < 0.05 for row in rows):
        return {"state": "INCIPIENT_OR_UNRESOLVED", "physical_loss_confirmed": False}
    return {"state": "WELD_OFF_CONTACT_SUPPORTED", "physical_loss_confirmed": False}


def evaluate_no_force_control(rows: list[dict[str, Any]], *, window_s: float = 0.50,
                              support_threshold: float = 0.05) -> dict[str, Any]:
    """Require capture containment and real finger support throughout the tail window."""
    if not rows:
        return {"passed": False, "reason": "TRACE_INCOMPLETE"}
    end = float(rows[-1].get("time", 0.0)); start = end - window_s
    tail = [r for r in rows if float(r.get("time", 0.0)) >= start]
    inside = all(bool(r.get("inside_capture", False)) for r in tail)
    support = all(float(r.get("support_force_ratio_mg", 0.0)) >= support_threshold for r in tail)
    return {"passed": bool(tail and inside and support), "sample_count": len(tail),
            "inside_capture": inside, "finger_support": support,
            "window_start_time_abs": start, "window_end_time_abs": end}
