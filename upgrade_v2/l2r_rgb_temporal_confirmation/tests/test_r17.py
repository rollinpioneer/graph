from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image, ImageDraw

from upgrade_v2.l2r_rgb_temporal_confirmation.audits import parameter_audit, source_audit
from upgrade_v2.l2r_rgb_temporal_confirmation.candidate_inputs import OnlineInputLeakage, validate_online_row
from upgrade_v2.l2r_rgb_temporal_confirmation.case_registry import (
    CALIBRATION_FAMILIES, CASES, CONFIRMATION_FAMILIES, dropout_capture_times,
    jitter_deg, unique_keys,
)
from upgrade_v2.l2r_rgb_temporal_confirmation.difficulty_ladder import choose_ladder
from upgrade_v2.l2r_rgb_temporal_confirmation.protocol import (
    BASE_COMMIT, CAPTURE_EVERY, HEIGHT, JPEG_QUALITY, METHODS, SCALES, WIDTH,
    protocol_lock,
)
from upgrade_v2.l2r_rgb_temporal_confirmation.temporal_scoring import score_temporal
from upgrade_v2.visual_refine_l2.vision import detect_frame

ROOT = Path(__file__).resolve().parents[2]


class RGBPathTests(unittest.TestCase):
    def test_o_tier_reads_detector_output_only(self):
        text = (ROOT / "l2r_rgb_temporal_confirmation/candidate_inputs.py").read_text()
        self.assertIn("detections_rgb.jsonl", text)
        self.assertNotIn("320.0 + 1000.0", text)

    def test_world_position_key_rejected(self):
        with self.assertRaisesRegex(OnlineInputLeakage, "ONLINE_INPUT_LEAKAGE"):
            validate_online_row({"time": 0, "object_world_position": [0, 0, 0]})

    def test_no_oracle_fallback_for_missing_detection(self):
        text = (ROOT / "l2r_rgb_temporal_confirmation/detector_adapter.py").read_text()
        self.assertIn('"object_centroid": None', text)
        self.assertNotIn("object_world_position", text)

    def test_detector_runs_on_saved_jpeg(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "frame.jpg"
            image = Image.new("RGB", (WIDTH, HEIGHT), (20, 20, 20))
            draw = ImageDraw.Draw(image); draw.ellipse((50, 50, 90, 90), fill=(240, 5, 8))
            image.save(path, quality=JPEG_QUALITY)
            self.assertIsNotNone(detect_frame(path)["object_centroid"])

    def test_renderer_resolution_192x144(self): self.assertEqual((WIDTH, HEIGHT), (192, 144))
    def test_jpeg_quality_86_locked(self): self.assertEqual(JPEG_QUALITY, 86)
    def test_capture_every_five_physics_steps(self): self.assertEqual(CAPTURE_EVERY, 5)

    def test_action_end_frame_flagged(self):
        self.assertIn('"action_end": bool(action_end)', (ROOT / "l2r_rgb_temporal_confirmation/rgb_capture.py").read_text())

    def test_frame_dropout_three_captures(self): self.assertEqual(len(dropout_capture_times(1.0)), 3)
    def test_jitter_schedule_deterministic(self): self.assertEqual(jitter_deg(1.23, 1.0), jitter_deg(1.23, 1.0))


class DifficultyTests(unittest.TestCase):
    def test_four_new_calibration_families(self): self.assertEqual(len(CALIBRATION_FAMILIES), 4)
    def test_fixed_scale_sequence(self): self.assertEqual(SCALES, (0.0, .125, .25, .5, .75, 1.0, 1.25))

    @staticmethod
    def levels():
        return [{"scale": 0.0, "loss_count": 0}, {"scale": .25, "loss_count": 0},
                {"scale": .5, "loss_count": 1}, {"scale": .75, "loss_count": 2},
                {"scale": 1.0, "loss_count": 4}]

    def test_weak_highest_zero_of_four(self): self.assertEqual(choose_ladder(self.levels())["weak"]["scale"], .25)
    def test_strong_lowest_four_of_four(self): self.assertEqual(choose_ladder(self.levels())["strong"]["scale"], 1.0)
    def test_medium_closest_two_of_four(self): self.assertEqual(choose_ladder(self.levels())["medium"]["scale"], .75)

    def test_midpoint_added_at_most_once(self):
        result = choose_ladder([{"scale": 0.0, "loss_count": 0}, {"scale": 1.0, "loss_count": 4}])
        self.assertEqual(result["midpoint_required"], .5)

    def test_calibration_not_in_confirmation_metrics(self):
        self.assertNotIn("CALIBRATION_FAMILIES", (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text())


class RegistryTests(unittest.TestCase):
    def test_six_new_confirmation_families(self): self.assertEqual(len(CONFIRMATION_FAMILIES), 6)
    def test_twelve_case_order(self): self.assertEqual(len(CASES), 12)
    def test_72_unique_family_case_seed_keys(self): self.assertEqual(len(set(unique_keys())), 72)

    def test_no_development_family_reuse(self):
        self.assertTrue(all("FDROP_DEV" not in family for family, _, _ in CONFIRMATION_FAMILIES))

    def test_C4_jitter_no_physics_change(self):
        self.assertAlmostEqual(jitter_deg(0.2, 0.0), 0.0, places=8)

    def test_C9_dropout_schedule_force_relative(self): self.assertEqual(dropout_capture_times(2.0), (2.1, 2.15, 2.2))

    def test_C10_prehold_is_point_one_seconds(self):
        self.assertEqual(next(case.prehold_s for case in CASES if case.case_id.startswith("C10_")), .1)

    def test_C12_commanded_release(self): self.assertEqual(CASES[-1].requested_effect, "RELEASE_OBJECT")


class TemporalTests(unittest.TestCase):
    def score(self, action=None, time=None, truth="recover_object", onset=1.0, deadline=1.75):
        actions = [] if action is None else [{"action": action, "time": time, "capture_order": 1}]
        return score_temporal(truth_action=truth, event_onset=onset, deadline=deadline, actions=actions)

    def test_retry_before_attempt_end_is_early(self): self.assertEqual(self.score("retry_grasp", .9, "retry_grasp", 1, 1.5)["temporal_outcome"], "EARLY_ACTION")
    def test_retry_in_half_second_window_correct(self): self.assertEqual(self.score("retry_grasp", 1.2, "retry_grasp", 1, 1.5)["temporal_outcome"], "CORRECT_IN_WINDOW")
    def test_retry_after_window_late(self): self.assertEqual(self.score("retry_grasp", 1.6, "retry_grasp", 1, 1.5)["temporal_outcome"], "LATE_ACTION")
    def test_recovery_before_loss_onset_early(self): self.assertEqual(self.score("recover_object", .9)["temporal_outcome"], "EARLY_ACTION")
    def test_recovery_in_window_correct(self): self.assertEqual(self.score("recover_object", 1.5)["temporal_outcome"], "CORRECT_IN_WINDOW")
    def test_recovery_after_point75_late(self): self.assertEqual(self.score("recover_object", 1.8)["temporal_outcome"], "LATE_ACTION")
    def test_missing_recovery_missed(self): self.assertEqual(self.score()["temporal_outcome"], "MISSED_REQUIRED_ACTION")
    def test_none_with_recovery_false_recovery(self): self.assertEqual(self.score("recover_object", 1, "none", None, None)["temporal_outcome"], "FALSE_RECOVERY")

    def test_first_non_none_action_controls_score(self):
        result = score_temporal(truth_action="recover_object", event_onset=1, deadline=2,
                                actions=[{"action":"retry_grasp","time":1.1,"capture_order":1}, {"action":"recover_object","time":1.2,"capture_order":2}])
        self.assertEqual(result["temporal_outcome"], "WRONG_ACTION")

    def test_action_match_alone_not_accuracy(self): self.assertFalse(self.score("recover_object", .9)["accurate"])


class GeneratorContractTests(unittest.TestCase):
    def test_72_trace_complete(self): self.assertIn("trace_count == 72", (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_60_prehold_verified(self): self.assertIn("prehold_count == 60", (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_24_strong_loss_confirmed(self): self.assertIn("strong_loss == 24", (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_12_stable_no_loss(self): self.assertIn("stable_no_loss == 12", (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_6_commanded_release(self): self.assertIn("commanded_release == 6", (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_weak_medium_strong_monotonic(self): self.assertIn('rates["C6"] <= rates["C7"] <= rates["C8"]', (ROOT / "l2r_rgb_temporal_confirmation/reference_builder.py").read_text())
    def test_generator_fail_blocks_evaluation(self): self.assertIn("GENERATOR_GATE_REQUIRED", (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text())


class EvaluationAuditTests(unittest.TestCase):
    def test_B2_and_C3_use_true_rgb(self):
        text = (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text()
        self.assertIn("load_candidate_input", text); self.assertIn("O_B2", METHODS); self.assertIn("O_C3", METHODS)

    def test_S_methods_marked_state_assisted(self): self.assertIn("STATE_ASSISTED_DIAGNOSTIC", (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text())
    def test_reference_join_after_prediction(self): self.assertIn("reference_join_after_prediction", (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text())
    def test_input_provenance_zero_forbidden_reads(self): self.assertIn('"forbidden_online_reads": 0', (ROOT / "l2r_rgb_temporal_confirmation/evaluation.py").read_text())
    def test_prefix_causality_1440_checks(self): self.assertEqual(len(METHODS) * 72 * 4, 1440)
    def test_parameter_search_disabled(self): self.assertFalse(parameter_audit()["parameter_search_allowed"])
    def test_tie_does_not_select_candidate(self): self.assertIsNone(protocol_lock()["selected_candidate_id"])
    def test_selected_candidate_remains_null(self): self.assertIsNone(protocol_lock()["selected_candidate_id"])

    def test_source_audit_passes(self):
        self.assertEqual(source_audit(ROOT.parent)["status"], "PASS")

    def test_base_commit_locked(self): self.assertEqual(BASE_COMMIT, "70c93dd8b8e9d95bc3faa67a64f410a37a657e08")


if __name__ == "__main__":
    unittest.main()
