import unittest

from upgrade_v2.l2r_normalized_observability_fusion import LossObservabilityFusionV2_NormalizedEvidence


def row(order, ns, contact=True, command="closed", rel=(0.0, 0.0)):
    return {"capture_order": order, "physical_time_ns": ns, "context_valid": True, "contact_present": contact, "gripper_command": command, "requested_effect": "HOLD_OBJECT", "attempt_active": True, "object_centroid": [rel[0], rel[1]], "gripper_centroid": [0.0, 0.0], "object_area": 100.0, "gripper_confidence": 1.0}


class FusionV2Tests(unittest.TestCase):
    def test_historical_hold_then_contact_absence(self):
        guard = LossObservabilityFusionV2_NormalizedEvidence(1.0, 2.0)
        stream = [row(0, 0), row(1, 50_000_000), row(2, 100_000_000, False), row(3, 150_000_000, False)]
        outputs = [guard.step(item) for item in stream]
        self.assertEqual(outputs[-1]["selected_action"], "recover_object")
        self.assertEqual(outputs[-1]["reason_code"], "historical_hold_established_then_observed_non_release_contact_loss")

    def test_release_clears_evidence(self):
        guard = LossObservabilityFusionV2_NormalizedEvidence(1.0, 2.0)
        guard.step(row(0, 0)); guard.step(row(1, 50_000_000)); guard.step(row(2, 100_000_000, False))
        result = guard.step(row(3, 150_000_000, command="open"))
        self.assertEqual(result["selected_action"], "none")
        self.assertEqual(result["reason_code"], "RELEASE_SUPPRESSES_FALLBACK")

    def test_same_time_capture_order_is_valid(self):
        guard = LossObservabilityFusionV2_NormalizedEvidence(1.0, 2.0)
        guard.step(row(0, 0)); guard.step(row(1, 50_000_000)); guard.step(row(2, 100_000_000, False)); result = guard.step(row(3, 100_000_000, False))
        self.assertEqual(result["physical_time_ns"], 100_000_000)


if __name__ == "__main__":
    unittest.main()
