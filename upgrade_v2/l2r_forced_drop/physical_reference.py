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


def evaluate_loss_trace(rows: list[dict[str, Any]], *, dt_s: float = 0.01, mass_kg: float = 0.18) -> dict[str, Any]:
    if not rows:
        return {"state": "TRACE_INCOMPLETE", "physical_loss_confirmed": False}
    for row in rows:
        for key in ("object_xyz", "gripper_xyz"):
            if key in row and not np.isfinite(np.asarray(row[key], dtype=float)).all():
                return {"state": "NUMERICAL_INVALID", "physical_loss_confirmed": False}
        if float(row.get("object_speed_mps", 0.0)) >= 5.0 or float(row.get("object_angular_speed_rads", 0.0)) >= 50.0 or float(row.get("object_z_m", 1.0)) <= 0.25:
            return {"state": "NUMERICAL_INVALID", "physical_loss_confirmed": False}
    if not all(bool(row.get("pre_hold_verified", False)) for row in rows[:1]):
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
        return {"state": "PHYSICAL_LOSS_CONFIRMED", "physical_loss_confirmed": True, "loss_onset_index": loss_start, "loss_confirmed_index": loss_confirmed, "loss_onset_time_s": loss_start * dt_s, "loss_confirmed_time_s": loss_confirmed * dt_s}
    if any(bool(row.get("outside_capture", False)) for row in rows) or any(float(row.get("support_force_ratio_mg", 1.0)) < 0.05 for row in rows):
        return {"state": "INCIPIENT_OR_UNRESOLVED", "physical_loss_confirmed": False}
    return {"state": "WELD_OFF_CONTACT_SUPPORTED", "physical_loss_confirmed": False}
