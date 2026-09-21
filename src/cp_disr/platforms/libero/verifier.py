"""Fact verifier: TRUE/FALSE/UNKNOWN from RGB-D perception only."""
from __future__ import annotations

from cp_disr.facts import FactRecord, Truth
from .perception import THRESHOLDS
from .d0_env import CONTAINER_INNER, BUFFER_HALF
import time


VERIFIER_VERSION = "cp-disr-d0-verifier-v1"


def _t():
    return time.time()


class FactVerifier:
    def __init__(self, env):
        self.env = env
        self.prev = {}

    def verify(self, measurement, execution=None):
        m = measurement.measurements
        blobs = m.get("blobs", {})
        eef = m.get("eef_pos")
        grip = m.get("gripper_qpos") or [0.04, -0.04]
        grip_open = abs(float(grip[0])) > THRESHOLDS["gripper_open"]
        now = _t()
        recs = []

        def blob(name):
            b = blobs.get(name)
            if not b:
                return None
            return b

        def xyz(name):
            b = blob(name)
            return None if b is None else b["xyz"]

        def add(fid, value, reason, evidence):
            last = self.prev.get(fid, Truth.UNKNOWN)
            recs.append(FactRecord(
                fact_id=fid,
                value=value,
                capture_time=now,
                available_time=now,
                evidence_ids=tuple(evidence),
                last_confirmed_value=value if value != Truth.UNKNOWN else last,
                last_confirmed_time=now if value != Truth.UNKNOWN else None,
                reason=f"{VERIFIER_VERSION}:{reason}",
            ))
            self.prev[fid] = recs[-1].last_confirmed_value

        # GripperEmpty
        near = []
        for name in ("target", "second_object", "lid"):
            p = xyz(name)
            if p is not None and eef is not None:
                dxy = ((p[0] - eef[0]) ** 2 + (p[1] - eef[1]) ** 2) ** 0.5
                if dxy < THRESHOLDS["held_xy"] and abs(p[2] - eef[2]) < THRESHOLDS["held_z"]:
                    near.append(name)
        if grip_open and not near:
            add("p:GripperEmpty", Truth.TRUE, "gripper_open_and_no_nearby_blob", ["gripper", "rgb"])
        elif (not grip_open) and near:
            add("p:GripperEmpty", Truth.FALSE, "gripper_closed_with_nearby_blob", ["gripper", "rgb"])
        else:
            add("p:GripperEmpty", Truth.UNKNOWN, "conflicting_or_missing_gripper_evidence", ["gripper", "rgb"])

        container = xyz("container")
        buffer = xyz("buffer")
        table_z = float(self.env.public_layout()["table_top_z"])

        for obj in ("target", "second_object"):
            p = xyz(obj)
            held_id = f"p:Held:{obj}"
            on_id = f"p:OnTable:{obj}"
            inside_id = f"p:Inside:{obj}:container"
            buf_id = f"p:AtBuffer:{obj}:buffer"
            if p is None:
                add(held_id, Truth.UNKNOWN, "object_blob_missing", ["rgb"])
                add(on_id, Truth.UNKNOWN, "object_blob_missing", ["rgb"])
                add(inside_id, Truth.UNKNOWN, "object_blob_missing", ["rgb"])
                add(buf_id, Truth.UNKNOWN, "object_blob_missing", ["rgb"])
                continue
            close = eef is not None and ((p[0]-eef[0])**2+(p[1]-eef[1])**2)**0.5 < THRESHOLDS["held_xy"] and abs(p[2]-eef[2]) < THRESHOLDS["held_z"]
            if (not grip_open) and close:
                add(held_id, Truth.TRUE, "closed_gripper_near_object_blob", ["gripper", "rgb", "depth"])
            elif grip_open and not close:
                add(held_id, Truth.FALSE, "open_gripper_or_object_away", ["gripper", "rgb"])
            else:
                add(held_id, Truth.UNKNOWN, "held_evidence_conflict", ["gripper", "rgb"])

            on_table = abs(p[2] - (table_z + 0.025)) < THRESHOLDS["table_z_tol"]
            if on_table and not ((not grip_open) and close):
                add(on_id, Truth.TRUE, "blob_near_table_plane", ["depth"])
            elif (not grip_open) and close:
                add(on_id, Truth.FALSE, "object_tracked_at_gripper", ["depth"])
            else:
                add(on_id, Truth.UNKNOWN, "table_contact_uncertain", ["depth"])

            if container is None:
                add(inside_id, Truth.UNKNOWN, "container_blob_missing", ["rgb"])
            else:
                inside = abs(p[0]-container[0]) <= (CONTAINER_INNER[0] + THRESHOLDS["inside_margin"]) and abs(p[1]-container[1]) <= (CONTAINER_INNER[1] + THRESHOLDS["inside_margin"]) and p[2] < table_z + 0.12
                add(inside_id, Truth.TRUE if inside else Truth.FALSE, "xy_inside_container_estimate", ["rgb", "depth"])

            if buffer is None:
                add(buf_id, Truth.UNKNOWN, "buffer_blob_missing", ["rgb"])
            else:
                atb = abs(p[0]-buffer[0]) <= (BUFFER_HALF[0] + THRESHOLDS["buffer_margin"]) and abs(p[1]-buffer[1]) <= (BUFFER_HALF[1] + THRESHOLDS["buffer_margin"])
                add(buf_id, Truth.TRUE if atb else Truth.FALSE, "xy_inside_buffer_estimate", ["rgb", "depth"])

        lid = xyz("lid")
        if lid is None or container is None:
            add("p:Open:container", Truth.UNKNOWN, "lid_or_container_blob_missing", ["rgb"])
        else:
            dist = ((lid[0]-container[0])**2 + (lid[1]-container[1])**2) ** 0.5
            add("p:Open:container", Truth.TRUE if dist > THRESHOLDS["open_offset"] else Truth.FALSE, "lid_offset_from_container", ["rgb"])
        return tuple(recs)
