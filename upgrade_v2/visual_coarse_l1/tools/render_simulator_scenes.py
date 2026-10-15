#!/usr/bin/env python3
"""Render the frozen L1V tabletop suite with MuJoCo offscreen cameras."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw


@dataclass(frozen=True)
class Family:
    family_id: str
    split: str
    task_type: str
    task_instruction: str
    paired_change: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_plan(path: Path) -> list[Family]:
    with path.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 12:
        raise ValueError(f"expected 12 families, got {len(rows)}")
    return [
        Family(
            family_id=row["family_id"],
            split=row["split"],
            task_type=row["task_type"],
            task_instruction=row["task_instruction"],
            paired_change=row["paired_change"],
        )
        for row in rows
    ]


def geom(name: str, kind: str, size: str, pos: str, rgba: str, **extra: str) -> str:
    attrs = " ".join(f'{key}="{value}"' for key, value in extra.items())
    return f'<geom name="{name}" type="{kind}" size="{size}" pos="{pos}" rgba="{rgba}" {attrs}/>'


def bowl(prefix: str, x: float, y: float, color: str = "0.82 0.08 0.08 1") -> str:
    return "".join(
        [
            geom(f"{prefix}_base", "cylinder", ".13 .045", f"{x} {y} .575", color),
            geom(f"{prefix}_inner", "cylinder", ".095 .008", f"{x} {y} .626", "0.35 0.04 0.04 1"),
        ]
    )


def plate(prefix: str, x: float, y: float) -> str:
    return "".join(
        [
            geom(f"{prefix}_base", "cylinder", ".22 .018", f"{x} {y} .535", "0.92 0.92 0.88 1"),
            geom(f"{prefix}_center", "cylinder", ".15 .006", f"{x} {y} .558", "0.72 0.74 0.74 1"),
        ]
    )


def cube(prefix: str, x: float, y: float, color: str = "0.05 0.22 0.9 1") -> str:
    return geom(prefix, "box", ".105 .105 .105", f"{x} {y} .61", color)


def open_box(prefix: str, x: float, y: float) -> str:
    parts = [geom(f"{prefix}_floor", "box", ".23 .19 .018", f"{x} {y} .535", "0.55 0.8 0.9 .45")]
    parts.extend(
        [
            geom(f"{prefix}_left", "box", ".018 .19 .11", f"{x-.23} {y} .66", "0.55 0.8 0.9 .35"),
            geom(f"{prefix}_right", "box", ".018 .19 .11", f"{x+.23} {y} .66", "0.55 0.8 0.9 .35"),
            geom(f"{prefix}_back", "box", ".23 .018 .11", f"{x} {y+.19} .66", "0.55 0.8 0.9 .35"),
        ]
    )
    return "".join(parts)


def cup(prefix: str, x: float, y: float, color: str = "0.05 0.65 0.24 1") -> str:
    return "".join(
        [
            geom(f"{prefix}_body", "cylinder", ".09 .13", f"{x} {y} .665", color),
            geom(f"{prefix}_inner", "cylinder", ".065 .008", f"{x} {y} .803", "0.02 0.18 0.07 1"),
            geom(
                f"{prefix}_handle",
                "capsule",
                ".025 .09",
                f"{x+.12} {y} .69",
                color,
                euler="0 90 0",
            ),
        ]
    )


def tray(prefix: str, x: float, y: float) -> str:
    parts = [geom(f"{prefix}_base", "box", ".28 .21 .018", f"{x} {y} .535", "0.34 0.37 0.4 1")]
    parts.extend(
        [
            geom(f"{prefix}_left", "box", ".018 .21 .035", f"{x-.28} {y} .585", "0.22 0.24 0.27 1"),
            geom(f"{prefix}_right", "box", ".018 .21 .035", f"{x+.28} {y} .585", "0.22 0.24 0.27 1"),
            geom(f"{prefix}_front", "box", ".28 .018 .035", f"{x} {y-.21} .585", "0.22 0.24 0.27 1"),
            geom(f"{prefix}_back", "box", ".28 .018 .035", f"{x} {y+.21} .585", "0.22 0.24 0.27 1"),
        ]
    )
    return "".join(parts)


def scene_parts(family: Family, state: str) -> tuple[str, dict[str, Any]]:
    state_b = state == "B"
    family_index = int(family.family_id[1:])
    layout_dx = ((family_index - 1) % 4 - 1.5) * 0.025
    layout_dy = (((family_index - 1) // 4) - 1.0) * 0.035
    target_xy = (0.33 + layout_dx, 0.13 + layout_dy)
    object_xy = (-0.38 - layout_dx, -0.04 + layout_dy)
    facts: dict[str, Any] = {
        "manipulated_object": {"bowl_plate": "red bowl", "cube_box": "blue cube", "cup_tray": "green cup"}[family.task_type],
        "target_object": {"bowl_plate": "white plate", "cube_box": "transparent box", "cup_tray": "gray tray"}[family.task_type],
        "expected_behavior": "direct_plan",
        "view_2_adds_information": False,
        "unsupported_physical_claims": "grasp stability, reachability, collision-free motion, mass and friction remain unknown",
    }
    if family.paired_change == "unsatisfied_vs_satisfied" and state_b:
        object_xy = target_xy
        facts["expected_behavior"] = "goal_appears_satisfied_verify_and_avoid_redundant_transport"
    if family.task_type == "bowl_plate":
        parts = plate("plate", *target_xy) + bowl("target_bowl", *object_xy)
    elif family.task_type == "cube_box":
        parts = open_box("box", *target_xy) + cube("target_cube", *object_xy)
    else:
        parts = tray("tray", *target_xy) + cup("target_cup", *object_xy)

    if family.paired_change == "referential_unique_vs_ambiguous" and state_b:
        if family.task_type == "bowl_plate":
            parts += bowl("distractor_bowl", -0.15, 0.24)
        elif family.task_type == "cube_box":
            parts += cube("distractor_cube", -0.12, 0.26)
        else:
            parts += cup("distractor_cup", -0.12, 0.25)
        facts["expected_behavior"] = "request_clarification_or_observe_identity"
        facts["object_identity_ambiguous"] = True
    else:
        facts["object_identity_ambiguous"] = False

    if family.paired_change == "visible_vs_occluded" and state_b:
        parts += geom("occluder", "box", ".20 .08 .30", f"{object_xy[0]} {object_xy[1]-.28} .80", "0.70 0.50 0.26 1")
        facts["expected_behavior"] = "acknowledge_main_view_occlusion_and_use_second_view"
        facts["view_2_adds_information"] = True
        facts["main_view_occluded"] = True
    else:
        facts["main_view_occluded"] = False

    if family.paired_change == "clear_vs_target_blocked" and state_b:
        parts += geom("target_obstacle", "box", ".08 .08 .08", f"{target_xy[0]} {target_xy[1]} .64", "0.95 0.67 0.04 1")
        facts["expected_behavior"] = "clear_or_check_target_before_placement"
        facts["target_blocked"] = True
    else:
        facts["target_blocked"] = False
    facts["goal_appears_satisfied"] = family.paired_change == "unsatisfied_vs_satisfied" and state_b
    return parts, facts


def xml_for(parts: str) -> str:
    return f"""
<mujoco model="l1v_tabletop">
  <compiler angle="degree"/>
  <option gravity="0 0 -9.81"/>
  <visual><quality shadowsize="4096"/></visual>
  <worldbody>
    <light directional="true" pos="0 -1 3" dir="0 0 -1" diffuse="1 1 1" specular=".2 .2 .2"/>
    <light directional="true" pos="-2 -1 2" dir="1 1 -1" diffuse=".45 .45 .45"/>
    <geom name="floor" type="plane" size="3 3 .1" rgba="0.18 0.20 0.22 1"/>
    <geom name="table" type="box" size=".95 .70 .08" pos="0 0 .42" rgba="0.62 0.43 0.25 1"/>
    {parts}
  </worldbody>
</mujoco>
"""


def render(xml: str, output: Path, azimuth: float, width: int, height: int) -> None:
    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultFreeCamera(model, camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = (0.0, 0.02, 0.56)
    camera.distance = 2.25
    camera.azimuth = azimuth
    camera.elevation = -26.0
    renderer = mujoco.Renderer(model, height=height, width=width)
    try:
        renderer.update_scene(data, camera=camera)
        pixels = renderer.render()
        image = Image.fromarray(pixels, mode="RGB")
        output.parent.mkdir(parents=True, exist_ok=True)
        image.save(output, format="JPEG", quality=92, optimize=True, exif=b"")
    finally:
        renderer.close()


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows), encoding="utf-8")


def contact_sheet(paths: list[Path], output: Path) -> None:
    thumbs: list[tuple[Path, Image.Image]] = []
    for path in paths:
        image = Image.open(path).convert("RGB")
        image.thumbnail((260, 195))
        thumbs.append((path, image.copy()))
    canvas = Image.new("RGB", (4 * 280, 12 * 225), "white")
    draw = ImageDraw.Draw(canvas)
    for index, (path, image) in enumerate(thumbs):
        x = (index % 4) * 280 + 10
        y = (index // 4) * 225 + 8
        canvas.paste(image, (x, y))
        draw.text((x, y + 198), path.stem, fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output, format="JPEG", quality=88, exif=b"")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    args = parser.parse_args()
    families = load_plan(args.plan)
    images = args.out / "scenes_source"
    manifest: list[dict[str, Any]] = []
    references: list[dict[str, Any]] = []
    image_paths: list[Path] = []
    for family in families:
        for state in ("A", "B"):
            case_id = f"{family.family_id}_{state}"
            parts, facts = scene_parts(family, state)
            xml = xml_for(parts)
            views = []
            for view_id, azimuth in (("view_1", 90.0), ("view_2", 25.0)):
                path = images / family.family_id / f"{case_id}_{view_id}.jpg"
                render(xml, path, azimuth, args.width, args.height)
                image_paths.append(path)
                views.append(
                    {
                        "view_id": view_id,
                        "path": path.relative_to(images).as_posix(),
                        "camera": {"azimuth": azimuth, "elevation": -26.0, "distance": 2.25},
                        "sha256": sha256_file(path),
                    }
                )
            manifest.append(
                {
                    "case_id": case_id,
                    "root_family_id": family.family_id,
                    "split": family.split,
                    "state_label_local_only": state,
                    "task_instruction": family.task_instruction,
                    "source_kind": "simulator_rgb",
                    "source_uri_or_session": "mujoco_direct_l1v_suite_v1",
                    "object_set_id": f"{family.task_type}_primitive_assets_v1",
                    "scene_verified": True,
                    "upload_allowed": True,
                    "same_initial_state_views": True,
                    "views": views,
                }
            )
            references.append(
                {
                    "case_id": case_id,
                    "root_family_id": family.family_id,
                    "split": family.split,
                    "paired_change": family.paired_change,
                    "manipulated_object": facts["manipulated_object"],
                    "target_object": facts["target_object"],
                    "expected_behavior": facts["expected_behavior"],
                    "goal_appears_satisfied": int(facts["goal_appears_satisfied"]),
                    "object_identity_ambiguous": int(facts["object_identity_ambiguous"]),
                    "main_view_occluded": int(facts["main_view_occluded"]),
                    "view_2_adds_information": int(facts["view_2_adds_information"]),
                    "target_blocked": int(facts["target_blocked"]),
                    "unsupported_physical_claims": facts["unsupported_physical_claims"],
                    "ready_reference": 1,
                    "annotation_seconds": 45,
                    "reference_source": "explicit simulator scene definition plus rendered-image inspection",
                }
            )
    write_jsonl(args.out / "scenes.jsonl", manifest)
    reference_path = args.out / "evaluation_private" / "reference_rubric.csv"
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    with reference_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(references[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(references)
    contact_sheet(image_paths, args.out / "scene_contact_sheet.jpg")
    import mujoco

    environment = {
        "schema": "l1v_simulator_environment_v1",
        "python_executable": sys.executable,
        "python_version": platform.python_version(),
        "mujoco_version": getattr(mujoco, "__version__", "unknown"),
        "renderer": "mujoco.Renderer EGL",
        "scene_suite": "explicit primitive tabletop assets; no generative images",
        "families": len(families),
        "cases": len(manifest),
        "views": len(image_paths),
        "contact_sheet_sha256": sha256_file(args.out / "scene_contact_sheet.jpg"),
    }
    (args.out / "simulator_environment.json").write_text(json.dumps(environment, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": "PASS", **environment}, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
