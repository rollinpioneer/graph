from __future__ import annotations

import hashlib
import json
from typing import Any

import numpy as np

from .protocol import canonical_hash


def array_hash(value: Any) -> dict[str, Any]:
    array = np.asarray(value)
    little = np.ascontiguousarray(array).astype(array.dtype.newbyteorder("<"), copy=False)
    raw = little.tobytes(order="C")
    return {
        "shape": list(array.shape),
        "dtype": array.dtype.str,
        "byte_length": len(raw),
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def _array_fields(obj: Any, fields: tuple[str, ...]) -> dict[str, Any]:
    return {field: array_hash(getattr(obj, field)) for field in fields if hasattr(obj, field)}


def model_fingerprint(model: Any, mujoco: Any, generated_xml: str) -> dict[str, Any]:
    fields = {
        "nq": int(model.nq), "nv": int(model.nv), "na": int(model.na), "nbody": int(model.nbody),
        "ngeom": int(model.ngeom), "neq": int(model.neq),
        "timestep": float(model.opt.timestep), "gravity": array_hash(model.opt.gravity),
        "integrator": int(model.opt.integrator),
        "bodies": _array_fields(model, ("body_pos", "body_quat", "body_mass")),
        "geoms": _array_fields(model, ("geom_type", "geom_size", "geom_pos", "geom_quat", "geom_friction")),
        "joints": _array_fields(model, ("jnt_type", "jnt_qposadr", "jnt_dofadr")),
        "equality": _array_fields(model, ("eq_type", "eq_obj1id", "eq_obj2id", "eq_data")),
        "cameras": _array_fields(model, ("cam_pos", "cam_quat", "cam_mat0")),
        "names_sha256": hashlib.sha256(bytes(model.names)).hexdigest(),
        "generated_model_xml_sha256": hashlib.sha256(generated_xml.encode("utf-8")).hexdigest(),
    }
    fields["model_fingerprint_sha256"] = canonical_hash(fields)
    del mujoco
    return fields


def state_hash(state: dict[str, Any]) -> str:
    return canonical_hash(state)


def checkpoint_state(sim: Any, *, sequence: int, sampling_point: str, action: str | None) -> dict[str, Any]:
    rng_state = sim.rng.bit_generator.state
    arrays = {
        "qpos": array_hash(sim.data.qpos),
        "qvel": array_hash(sim.data.qvel),
        "qacc": array_hash(sim.data.qacc),
        "qacc_warmstart": array_hash(sim.data.qacc_warmstart),
        "mocap_pos": array_hash(sim.data.mocap_pos),
        "mocap_quat": array_hash(sim.data.mocap_quat),
        "eq_active": array_hash(sim.data.eq_active),
        "model_eq_data": array_hash(sim.model.eq_data),
        "xfrc_applied": array_hash(sim.data.xfrc_applied),
    }
    visible = {
        "sequence": sequence,
        "sampling_point": sampling_point,
        "action": action,
        "action_index": int(sim.action_index),
        "control_index": len(getattr(sim, "_active_control_sequence", [])),
        "time": float(sim.data.time),
        "time_hex": float(sim.data.time).hex(),
        "arrays": arrays,
        "object_xyz": [float(x) for x in sim.object_xyz],
        "gripper_xyz": [float(x) for x in sim.data.mocap_pos[0]],
        "object_qpos": [float(x) for x in sim.data.qpos[sim.object_qpos:sim.object_qpos + 7]],
        "object_qvel": [float(x) for x in sim.data.qvel[sim.object_dof:sim.object_dof + 6]],
        "rng_state": rng_state,
        "events": list(sim.events),
        "attempt_lifecycle": sim.attempt_lifecycle.snapshot(),
        "simulator_flags": {key: bool(getattr(sim, key)) for key in ("gripper_closed", "attached", "failed_once", "recovered", "contact_lost")},
    }
    visible["state_sha256"] = state_hash(visible)
    return visible
