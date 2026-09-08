"""HSV and contour based perception for the L2R primitive benchmark."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np


def _components(mask: np.ndarray, minimum_area: float = 20.0) -> list[dict[str, Any]]:
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rows = []
    for contour in contours:
        area = float(cv2.contourArea(contour))
        if area < minimum_area:
            continue
        moments = cv2.moments(contour)
        if not moments["m00"]:
            continue
        x, y, w, h = cv2.boundingRect(contour)
        rows.append({"area": area, "centroid": (moments["m10"] / moments["m00"], moments["m01"] / moments["m00"]), "bbox": (x, y, w, h)})
    return sorted(rows, key=lambda row: row["area"], reverse=True)


def _union(*masks: np.ndarray) -> np.ndarray:
    result = masks[0].copy()
    for mask in masks[1:]:
        result = cv2.bitwise_or(result, mask)
    return result


def detect_frame(path: Path) -> dict[str, Any]:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"cannot read image: {path}")
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    red = _union(cv2.inRange(hsv, (0, 90, 45), (12, 255, 255)), cv2.inRange(hsv, (168, 80, 40), (180, 255, 255)))
    white = cv2.inRange(hsv, (0, 0, 145), (180, 65, 255))
    cyan = cv2.inRange(hsv, (78, 70, 45), (105, 255, 255))
    yellow = cv2.inRange(hsv, (15, 100, 70), (42, 255, 255))
    kernel = np.ones((3, 3), np.uint8)
    masks = {name: cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) for name, mask in {"object": red, "target": white, "gripper": cyan, "obstacle": yellow}.items()}
    components = {name: _components(mask) for name, mask in masks.items()}
    height, width = image.shape[:2]
    object_parts = components["object"]
    target = components["target"][0] if components["target"] else None
    target_pixels = cv2.findNonZero(masks["target"])
    if target_pixels is not None:
        x, y, w, h = cv2.boundingRect(target_pixels)
        target = {"area": float(cv2.countNonZero(masks["target"])), "centroid": (x + (w - 1) / 2, y + (h - 1) / 2), "bbox": (x, y, w, h)}
    obj = object_parts[0] if object_parts else None
    gripper = components["gripper"][0] if components["gripper"] else None
    obstacle = components["obstacle"][0] if components["obstacle"] else None
    target_radius = max(target["bbox"][2:]) / 2 if target else None
    center_ratio = None
    if obj and target and target_radius:
        # Front-camera perspective shifts the visible object's vertical centroid with height.
        # Horizontal image displacement remains the calibrated tabletop centering signal.
        center_ratio = abs(float(obj["centroid"][0]) - float(target["centroid"][0])) / target_radius
    overlap_ratio = 0.0
    if obj and target:
        intersection = cv2.countNonZero(cv2.bitwise_and(masks["object"], masks["target"]))
        overlap_ratio = intersection / max(1, cv2.countNonZero(masks["object"]))
    occupied = False
    if obstacle and target and target_radius:
        occupied = float(np.linalg.norm(np.asarray(obstacle["centroid"]) - np.asarray(target["centroid"]))) <= target_radius
    return {
        "width": width, "height": height,
        "object_centroid": obj["centroid"] if obj else None, "object_area": obj["area"] if obj else 0.0,
        "object_component_count": len(object_parts), "target_centroid": target["centroid"] if target else None,
        "target_area": target["area"] if target else 0.0, "target_radius_px": target_radius,
        "gripper_centroid": gripper["centroid"] if gripper else None, "gripper_area": gripper["area"] if gripper else 0.0,
        "obstacle_centroid": obstacle["centroid"] if obstacle else None, "obstacle_area": obstacle["area"] if obstacle else 0.0,
        "object_target_center_ratio": center_ratio, "object_target_overlap_ratio": overlap_ratio, "target_occupied_visual": occupied,
        "object_confidence": min(1.0, (obj["area"] if obj else 0.0) / 60.0),
        "target_confidence": min(1.0, (target["area"] if target else 0.0) / 180.0),
        "gripper_confidence": min(1.0, (gripper["area"] if gripper else 0.0) / 150.0),
    }
