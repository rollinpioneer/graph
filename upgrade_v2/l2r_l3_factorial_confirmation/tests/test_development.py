from __future__ import annotations

import unittest

from upgrade_v2.l2r_l3_factorial_confirmation.protocol import protocol_lock
from upgrade_v2.l2r_l3_factorial_confirmation.registry import ARMS, CASES, FAMILIES, registry
from upgrade_v2.l2r_l3_factorial_confirmation.static_lock import FROZEN_PATHS


class DevelopmentContractTests(unittest.TestCase):
    def test_factorial(self):
        self.assertEqual(len(ARMS), 4)
        self.assertEqual({(arm.fusion_enabled, arm.supervisor_v2_enabled) for arm in ARMS},
                         {(False, False), (True, False), (False, True), (True, True)})

    def test_new_physical_registry(self):
        self.assertEqual(len(FAMILIES), 4)
        self.assertEqual(len(CASES), 7)
        self.assertEqual(registry()["physical_rollouts"], 112)

    def test_protocol_freezes_clp3(self):
        protocol = protocol_lock()
        self.assertFalse(protocol["clp3_modified"])
        self.assertFalse(protocol["signal_not_observed_is_candidate_error"])
        self.assertFalse(protocol["confirmation_parameter_search"])

    def test_r22_r23_r24_frozen_paths(self):
        self.assertIn("clp3_guard", FROZEN_PATHS)
        self.assertIn("r23_controller", FROZEN_PATHS)
        self.assertIn("r24_runner", FROZEN_PATHS)
        self.assertIn("r24_decision", FROZEN_PATHS)


if __name__ == "__main__": unittest.main()
