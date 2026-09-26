"""Freeze the deterministic tc_nonoverlap_v1 pool and static qualification.

This generator is independent of policy, controller outcomes, and VLM responses.
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path

from cp_disr.platforms.libero.d0_env import OBJECT_HALF, LID_HALF, BUFFER_HALF

ROOT = Path(__file__).resolve().parents[1]
NS = "cp_disr_v1.2_tc_nonoverlap_v1"
VERSION = "tc_nonoverlap_v1"
G_MIN = 0.010
CORRIDOR_T = (0.25, 0.75)
CORRIDOR_RADIUS = 0.090
CONTAINER = (0.18, 0.12)
BUFFER = (-0.24, 0.24)


def digest(value: object) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def unit(seed: int, label: str) -> float:
    h = hashlib.sha256(f"{NS}\0{label}\0{seed}".encode()).digest()
    return int.from_bytes(h[:8], "big") / float(2**64)


def make_row(split: str, index: int, pool_index: int) -> dict:
    seed = int.from_bytes(hashlib.sha256(f"{NS}\0{split}\0{pool_index}".encode()).digest()[:4], "big")
    tx = -0.245 + 0.175 * unit(seed, "target-x")
    ty = -0.205 + 0.285 * unit(seed, "target-y")
    cx, cy = CONTAINER
    vx, vy = cx - tx, cy - ty
    length = math.hypot(vx, vy)
    px, py = -vy / length, vx / length
    side = -1.0 if int(unit(seed, "side") * 2) == 0 else 1.0
    offset = side * (0.060 + 0.020 * unit(seed, "corridor"))
    # Keep the interferer in the interior corridor while leaving a generous
    # clearance from the container rim and walls.
    t = 0.30 + 0.10 * unit(seed, "projection")
    ix, iy = tx + t * vx + offset * px, ty + t * vy + offset * py
    return {
        "case_id": f"T_C_v12_{split}_{index:03d}", "task_id": "T_C", "split": split,
        "pool_index": pool_index, "seed": seed,
        "target_xy": [round(tx, 9), round(ty, 9)], "second_xy": [round(ix, 9), round(iy, 9)],
        "container_xy": list(CONTAINER), "buffer_xy": list(BUFFER), "lid_closed": True,
        "second_role": "interferer", "task_data_version": VERSION, "generator_namespace": NS,
    }


def qualify(row: dict) -> dict:
    tx, ty = row["target_xy"]; ix, iy = row["second_xy"]
    cx, cy = row["container_xy"]; bx, by = row["buffer_xy"]
    hx, hy = map(float, OBJECT_HALF[:2])
    dx, dy = ix - tx, iy - ty
    gx, gy = abs(dx) - 2 * hx, abs(dy) - 2 * hy
    vx, vy = cx - tx, cy - ty
    denom = vx * vx + vy * vy
    proj = ((ix - tx) * vx + (iy - ty) * vy) / denom
    perp = abs((ix - tx) * vy - (iy - ty) * vx) / math.sqrt(denom)
    def outside_aabb(x: float, y: float, ox: float, oy: float, ahx: float, ahy: float) -> float:
        return max(abs(x - ox) - (hx + ahx), abs(y - oy) - (hy + ahy))
    container_clear = min(outside_aabb(tx, ty, cx, cy, 0.054, 0.054),
                          outside_aabb(ix, iy, cx, cy, 0.054, 0.054))
    buffer_clear = min(outside_aabb(tx, ty, bx, by, float(BUFFER_HALF[0]), float(BUFFER_HALF[1])),
                       outside_aabb(ix, iy, bx, by, float(BUFFER_HALF[0]), float(BUFFER_HALF[1])))
    workspace_clear = min(0.36 - abs(tx) - hx, 0.36 - abs(ty) - hy,
                          0.36 - abs(ix) - hx, 0.36 - abs(iy) - hy)
    # Conservative controller envelope derived from the production table and
    # exercised OSC_POSE workspace: all movable starts stay in [-0.30, 0.30]^2.
    controller_clear = min(0.30 - abs(tx), 0.30 - abs(ty), 0.30 - abs(ix), 0.30 - abs(iy))
    static_clear = min(container_clear, buffer_clear, workspace_clear, controller_clear)
    passed = (max(gx, gy) >= G_MIN and CORRIDOR_T[0] <= proj <= CORRIDOR_T[1]
              and perp <= CORRIDOR_RADIUS and container_clear >= G_MIN
              and buffer_clear >= G_MIN and workspace_clear >= G_MIN
              and controller_clear >= 0.0)
    return {**row, "dx": dx, "dy": dy, "abs_dx": abs(dx), "abs_dy": abs(dy),
            "gap_x": gx, "gap_y": gy, "min_axis_gap": max(gx, gy),
            "aabb_overlap": bool(gx < 0 and gy < 0), "corridor_projection": proj,
            "corridor_perpendicular": perp, "container_clearance": container_clear,
            "buffer_clearance": buffer_clear, "workspace_clearance": workspace_clear,
            "controller_envelope_clearance": controller_clear,
            "static_clearance_min": static_clear,
            "classification": "PASS" if passed else "FAIL_STATIC_GEOMETRY",
            "pass_static_geometry": passed}


def main() -> None:
    out = ROOT / "runs" / "tc_nonoverlap_remediation" / "pool_generation"
    out.mkdir(parents=True, exist_ok=True)
    pools = {"train_pool": [make_row("train", i, i) for i in range(128)],
             "dev_pool": [make_row("dev", i, 128 + i) for i in range(20)],
             "test_pool": [make_row("test", i, 148 + i) for i in range(50)]}
    qualified = {k: [qualify(r) for r in rows] for k, rows in pools.items()}
    ranked = sorted(qualified["train_pool"], key=lambda r: digest({"ns": NS, "id": r["case_id"]}))
    selected = {r["case_id"] for r in ranked[:64]}
    active_train = [r for r in qualified["train_pool"] if r["case_id"] in selected]
    enabled = {"task_id": "T_C", "version": VERSION, "generator_namespace": NS,
               "train": active_train, "dev": qualified["dev_pool"], "test": [],
               "train_pool_count": 128, "dev_count": 20, "test_pool_count": 50,
               "test_reserved_only": True, "selection_rule": "sha256_rank_then_source_order",
               "geometry_contract": {"g_min_m": G_MIN, "corridor_projection": CORRIDOR_T,
                                     "corridor_radius_m": CORRIDOR_RADIUS,
                                     "object_half_m": list(map(float, OBJECT_HALF)),
                                     "lid_half_m": list(map(float, LID_HALF)),
                                     "buffer_half_m": list(map(float, BUFFER_HALF)),
                                     "container_xy": list(CONTAINER), "buffer_xy": list(BUFFER)}}
    for name, data in pools.items():
        (out / f"{name}.json").write_text(json.dumps(data, indent=2) + "\n")
    (out / "enabled_split.json").write_text(json.dumps(enabled, indent=2) + "\n")
    all_rows = qualified["train_pool"] + qualified["dev_pool"] + qualified["test_pool"]
    with (out / "geometry_static_audit.csv").open("w", newline="") as f:
        fields = ["case_id", "split", "pool_index", "target_xy", "second_xy", "gap_x", "gap_y", "min_axis_gap", "aabb_overlap", "corridor_projection", "corridor_perpendicular", "container_clearance", "buffer_clearance", "workspace_clearance", "controller_envelope_clearance", "static_clearance_min", "classification"]
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in all_rows: w.writerow({k: r.get(k) for k in fields})
    manifest = {"task_data_version": VERSION, "generator_namespace": NS,
                "object_half": list(map(float, OBJECT_HALF)), "lid_half": list(map(float, LID_HALF)),
                "buffer_half": list(map(float, BUFFER_HALF)), "g_min_m": G_MIN,
                "corridor_projection": CORRIDOR_T, "corridor_radius_m": CORRIDOR_RADIUS,
                "pool_counts": {k: len(v) for k, v in pools.items()},
                "static_pass": {k: sum(bool(r["pass_static_geometry"]) for r in v) for k, v in qualified.items()},
                "static_failures": [r["case_id"] for r in all_rows if not r["pass_static_geometry"]],
                "pool_sha256": {k: digest(v) for k, v in pools.items()},
                "enabled_split_sha256": digest(enabled), "policy_independent": True}
    (out / "scene_generator_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
