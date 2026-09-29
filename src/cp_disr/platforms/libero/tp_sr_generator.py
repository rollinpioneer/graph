"""Deterministic pool generator for T_P_SOFT_RELOCATION_V1 (S4 design; no environment, no reset, no provider).

Scene coordinates are drawn from the workspace envelope of the *existing valid resets* (configs/splits) and shrunk by
object/container/buffer/lid footprints; nothing here is an unverified absolute coordinate. Geometry features are
generator/QA fields only: they never enter policy input, provider payloads, relation admission or reward.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

GENERATOR_VERSION = "tp-sr-generator-v1"
TASK_FAMILY_ID = "T_P_SOFT_RELOCATION_V1"
TASK_ID = "T_P_SR"
POOL_SIZE = 64
N_BINS = 4

OBJECT_HALF = 0.020            # d0_env.OBJECT_HALF
CONTAINER_OUTER_HALF = 0.042   # CONTAINER_INNER 0.030 + CONTAINER_WALL 0.012
BUFFER_HALF = 0.080            # d0_env.BUFFER_HALF (x/y)
LID_HALF = 0.036               # d0_env.LID_HALF (x/y)
LID_OFFSET_X = 0.26            # d0_env._apply_case_poses: open lid is parked at container_x + 0.26
TABLE_SKIN_HALF = 0.36         # d0_env table skin half extent (visible table)
FOOTPRINT_MARGIN = 0.012       # extra clearance kept around every static footprint
MIN_AXIS_SEPARATION = 0.048    # axis-aligned cubes of side 0.04 do not touch when max(|dx|,|dy|) >= this
FINGER_CLEARANCE_PROXY = 0.060 # centre distance below which a gripper pad plausibly meets the neighbour (proxy only)

# Fixed agentview camera as recorded by camera_calibration() in production captures (experiments/stage_0c_inputs/*/camera_config.json).
CAMERA = {"pos": [0.5, 0.0, 1.35], "fovy": 45.0, "width": 128, "height": 128,
          "mat": [[0.0, -0.706147844353306, 0.7080644193257978], [1.0, 0.0, 0.0], [0.0, 0.7080644193257978, 0.706147844353306]]}
TABLE_TOP_Z = 0.825            # d0_env: table_offset z 0.8 + 0.025
CUBE_CENTER_Z = TABLE_TOP_Z + OBJECT_HALF + 0.001

# Occlusion bins (design axis): centre-distance bands between target and interferer, in metres.
BIN_BANDS = {3: (0.050, 0.075), 2: (0.075, 0.105), 1: (0.105, 0.150), 0: (0.150, 0.220)}
CONTAINER_X_RANGE = (0.02, 0.07)   # keeps the parked open lid (x + 0.26) on the table skin
CONTAINER_Y_RANGE = (0.08, 0.16)
BUFFER_X_RANGE = (-0.22, -0.14)
BUFFER_Y_RANGE = (0.08, 0.16)


def _sha(x) -> str:
    return hashlib.sha256(x if isinstance(x, bytes) else json.dumps(x, sort_keys=True).encode()).hexdigest()


def seed_for(base_sha: str, index: int) -> int:
    return int(hashlib.sha256(f"{base_sha}:{TASK_FAMILY_ID}:{index}".encode()).hexdigest()[:8], 16) % 2147483648


def production_bounds(root) -> dict:
    """Object-placement envelope from existing valid resets (T_C splits; the wider non-overlap split included)."""
    files = sorted(Path(root, "configs/splits").glob("T_C_*.json"))
    xs, ys, srcs = [], [], {}
    for f in files:
        doc = json.loads(f.read_text(encoding="utf-8"))
        n = 0
        for sp in ("train", "dev"):
            for r in doc.get(sp, []):
                if isinstance(r, dict) and "target_xy" in r and "second_xy" in r:
                    for k in ("target_xy", "second_xy"):
                        xs.append(r[k][0]); ys.append(r[k][1]); n += 1
        srcs[f.name] = {"sha256": _sha(f.read_bytes()), "positions": n}
    if not xs:
        raise RuntimeError("no production reset rows found for workspace bounds")
    return {"x": [min(xs), max(xs)], "y": [min(ys), max(ys)], "sources": srcs}


def _rect_overlap(c1, h1, c2, h2) -> bool:
    return abs(c1[0] - c2[0]) < h1 + h2 and abs(c1[1] - c2[1]) < h1 + h2


def _static_footprints(container_xy, buffer_xy):
    lid_xy = (container_xy[0] + LID_OFFSET_X, container_xy[1])
    return [("container", container_xy, CONTAINER_OUTER_HALF + FOOTPRINT_MARGIN + OBJECT_HALF),
            ("buffer", buffer_xy, BUFFER_HALF + FOOTPRINT_MARGIN + OBJECT_HALF),
            ("lid", lid_xy, LID_HALF + FOOTPRINT_MARGIN + OBJECT_HALF)]


def _cube_ok(xy, statics, bounds):
    if not (bounds["x"][0] <= xy[0] <= bounds["x"][1] and bounds["y"][0] <= xy[1] <= bounds["y"][1]):
        return False
    if max(abs(xy[0]), abs(xy[1])) > TABLE_SKIN_HALF - OBJECT_HALF - 0.01:
        return False
    return not any(_rect_overlap(xy, OBJECT_HALF, c, h - OBJECT_HALF) for _, c, h in statics)


def project(point, cam=CAMERA):
    """Pinhole projection matching perception.backproject_mask; returns (col, row, depth)."""
    R = np.asarray(cam["mat"], dtype=float)
    t = np.asarray(cam["pos"], dtype=float)
    p = R.T @ (np.asarray(point, dtype=float) - t)
    z = -p[2]
    f = 0.5 * cam["height"] / np.tan(np.deg2rad(cam["fovy"]) / 2.0)
    return (cam["width"] - 1) / 2.0 + f * p[0] / z, (cam["height"] - 1) / 2.0 - f * p[1] / z, z


def cube_bbox(xy, cam=CAMERA):
    pts = [(xy[0] + sx * OBJECT_HALF, xy[1] + sy * OBJECT_HALF, TABLE_TOP_Z + dz)
           for sx in (-1, 1) for sy in (-1, 1) for dz in (0.0, 2 * OBJECT_HALF)]
    proj = np.array([project(p, cam) for p in pts])
    return proj[:, 0].min(), proj[:, 1].min(), proj[:, 0].max(), proj[:, 1].max(), float(np.mean(proj[:, 2]))


def camera_projected_overlap(target_xy, interferer_xy, cam=CAMERA) -> float:
    """Fraction of the target's projected bounding box covered by the interferer's, only if the interferer is nearer to the camera."""
    tx0, ty0, tx1, ty1, tz = cube_bbox(target_xy, cam)
    ix0, iy0, ix1, iy1, iz = cube_bbox(interferer_xy, cam)
    if iz >= tz:
        return 0.0
    w = max(0.0, min(tx1, ix1) - max(tx0, ix0)); h = max(0.0, min(ty1, iy1) - max(ty0, iy0))
    area = max((tx1 - tx0) * (ty1 - ty0), 1e-9)
    return float(w * h / area)


def geometry_features(cfg: dict) -> dict:
    t, i, c, b = (np.asarray(cfg[k], dtype=float) for k in ("target_xy", "second_xy", "container_xy", "buffer_xy"))
    d = float(np.linalg.norm(t - i))
    edge = min(TABLE_SKIN_HALF - abs(t[0]), TABLE_SKIN_HALF - abs(t[1])) - OBJECT_HALF
    return {"target_interferer_xy_distance": d,
            "camera_projected_overlap": camera_projected_overlap(cfg["target_xy"], cfg["second_xy"]),
            "approach_clearance_proxy": d - FINGER_CLEARANCE_PROXY,
            "target_container_distance": float(np.linalg.norm(t - c)),
            "interferer_buffer_distance": float(np.linalg.norm(i - b)),
            "target_grasp_margin_proxy": float(min(edge, np.linalg.norm(t - c) - CONTAINER_OUTER_HALF - OBJECT_HALF))}


def generate_config(index: int, base_sha: str, bounds: dict) -> dict:
    seed = seed_for(base_sha, index)
    rng = np.random.default_rng(seed)
    occ_bin = index % N_BINS
    axis = "y" if (index // N_BINS) % 2 == 0 else "x"
    lo, hi = BIN_BANDS[occ_bin]
    for attempt in range(2000):
        container = (float(rng.uniform(*CONTAINER_X_RANGE)), float(rng.uniform(*CONTAINER_Y_RANGE)))
        buffer = (float(rng.uniform(*BUFFER_X_RANGE)), float(rng.uniform(*BUFFER_Y_RANGE)))
        statics = _static_footprints(container, buffer)
        target = (float(rng.uniform(*bounds["x"])), float(rng.uniform(*bounds["y"])))
        d = float(rng.uniform(lo, hi))
        base_angle = 0.5 * np.pi if axis == "y" else 0.0
        theta = base_angle + (0.0 if rng.random() < 0.5 else np.pi) + float(rng.uniform(-0.35, 0.35))
        inter = (target[0] + d * float(np.cos(theta)), target[1] + d * float(np.sin(theta)))
        if max(abs(inter[0] - target[0]), abs(inter[1] - target[1])) < MIN_AXIS_SEPARATION:
            continue
        if _cube_ok(target, statics, bounds) and _cube_ok(inter, statics, bounds):
            cfg = {"case_id": f"{TASK_ID}_pool_{index:02d}", "pool_index": index, "seed": seed, "split": "dev", "task_id": TASK_ID,
                   "target_xy": [round(target[0], 6), round(target[1], 6)], "second_xy": [round(inter[0], 6), round(inter[1], 6)],
                   "container_xy": [round(container[0], 6), round(container[1], 6)], "buffer_xy": [round(buffer[0], 6), round(buffer[1], 6)],
                   "lid_closed": False, "second_role": "interferer", "occlusion_bin": occ_bin, "axis": axis, "attempts": attempt + 1}
            cfg["features"] = geometry_features(cfg)
            return cfg
    raise RuntimeError(f"generator could not place pool index {index}")


def generate_pool(base_sha: str, bounds: dict, n: int = POOL_SIZE) -> list[dict]:
    pool = [generate_config(i, base_sha, bounds) for i in range(n)]
    keys = {json.dumps({k: c[k] for k in ("target_xy", "second_xy", "container_xy", "buffer_xy")}, sort_keys=True) for c in pool}
    if len(keys) != len(pool):
        raise RuntimeError("duplicate configurations in pool")
    return pool


def split_row(cfg: dict) -> dict:
    """The exact row consumed by tp_sr_runtime (CaseSpec fields)."""
    return {k: cfg[k] for k in ("case_id", "split", "seed", "target_xy", "second_xy", "container_xy", "buffer_xy", "lid_closed", "second_role")}
