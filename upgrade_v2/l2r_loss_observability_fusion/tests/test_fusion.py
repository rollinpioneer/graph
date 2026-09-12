from __future__ import annotations

import unittest

from upgrade_v2.l2r_loss_observability_fusion.fusion import LossObservabilityFusionV1


def row(order, ns, contact=True, obj=(10.0, 20.0), grip=(10.0, 10.0), **extra):
    return {"capture_order": order, "physical_time_ns": ns, "contact_present": contact,
            "object_centroid": obj, "gripper_centroid": grip,
            "object_confidence": 1.0 if obj is not None else 0.0, "gripper_confidence": 1.0,
            "gripper_command": "closed", "requested_effect": "HOLD_OBJECT",
            "context_valid": True, **extra}


class FusionTests(unittest.TestCase):
    def armed(self):
        fusion = LossObservabilityFusionV1()
        fusion.step(row(0, 0)); fusion.step(row(1, 50_000_000))
        return fusion

    def test_sustained_contact_absence(self):
        fusion = self.armed()
        self.assertEqual(fusion.step(row(2, 100_000_000, False))["selected_action"], "none")
        proposal = fusion.step(row(3, 150_000_000, False))
        self.assertEqual((proposal["selected_action"], proposal["proposal_source"]),
                         ("recover_object", "SUSTAINED_CONTACT_ABSENCE"))

    def test_same_time_does_not_persist(self):
        fusion = self.armed()
        fusion.step(row(2, 100_000_000, False))
        self.assertEqual(fusion.step(row(3, 100_000_000, False))["selected_action"], "none")

    def test_rgb_detach(self):
        fusion = self.armed()
        fusion.step(row(2, 100_000_000, True, obj=(18.0, 20.0)))
        proposal = fusion.step(row(3, 150_000_000, True, obj=(19.0, 20.0)))
        self.assertEqual(proposal["proposal_source"], "REAL_RGB_DETACH")

    def test_object_missing_rgb_detach(self):
        fusion = self.armed()
        fusion.step(row(2, 100_000_000, True, obj=None))
        self.assertEqual(fusion.step(row(3, 150_000_000, True, obj=None))["selected_action"], "recover_object")

    def test_single_contact_fault_does_not_fire(self):
        fusion = self.armed()
        fusion.step(row(2, 100_000_000, False)); fusion.step(row(3, 150_000_000, True))
        self.assertFalse(fusion.emitted)

    def test_release_suppresses_and_rearm_is_explicit(self):
        fusion = self.armed()
        release = row(2, 100_000_000, False, gripper_command="open", attempt_end_reason="release")
        self.assertEqual(fusion.step(release)["reason_code"], "RELEASE_SUPPRESSES_FALLBACK")
        fusion.rearm_after_verified_hold()
        fusion.step(row(3, 150_000_000, False))
        self.assertEqual(fusion.step(row(4, 200_000_000, False))["selected_action"], "recover_object")


if __name__ == "__main__": unittest.main()
