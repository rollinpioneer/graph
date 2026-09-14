from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from upgrade_v2.l2r_temporal_loss_decision.episodes import as_box
from upgrade_v2.l2r_temporal_loss_decision.visual_separation import (
    BBOX_FORMAT,
    as_xyxy,
    box_gap,
    detect_frame_boxes,
)


class BoundingBoxContractTests(unittest.TestCase):
    def test_real_positive_gap(self) -> None:
        self.assertEqual(box_gap([0, 0, 10, 10], [20, 0, 30, 10], 1.0), 10.0)

    def test_overlap_has_zero_gap(self) -> None:
        self.assertEqual(box_gap([0, 0, 10, 10], [5, 5, 15, 15], 1.0), 0.0)

    def test_missing_bbox_is_unavailable(self) -> None:
        self.assertIsNone(box_gap(None, [0, 0, 10, 10], 1.0))
        self.assertIsNone(as_xyxy([0, 0, 10]))
        self.assertIsNone(as_box([10, 10, 5, 5]))

    def test_detector_and_episode_adapter_are_xyxy_round_trip(self) -> None:
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[10:30, 8:28] = (0, 0, 255)      # red object
        image[10:30, 58:88] = (255, 255, 0)   # cyan gripper in BGR
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.png"
            self.assertTrue(cv2.imwrite(str(path), image))
            detected = detect_frame_boxes(path)
        self.assertEqual(detected["bbox_format"], BBOX_FORMAT)
        self.assertEqual(detected["object_bbox"], [8.0, 10.0, 28.0, 30.0])
        self.assertEqual(detected["gripper_bbox"], [58.0, 10.0, 88.0, 30.0])
        self.assertEqual(as_box(detected["object_bbox"]), detected["object_bbox"])
        self.assertEqual(as_box(detected["gripper_bbox"]), detected["gripper_bbox"])


if __name__ == "__main__":
    unittest.main()
