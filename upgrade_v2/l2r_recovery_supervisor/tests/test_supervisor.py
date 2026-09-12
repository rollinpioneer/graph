from __future__ import annotations

import unittest

from upgrade_v2.l2r_recovery_supervisor import RecoverySupervisorV2


class SupervisorTests(unittest.TestCase):
    def supervise(self, executions, goals, losses):
        return RecoverySupervisorV2().run(
            initial_action="recover_object",
            execute_recovery=lambda action, cycle: executions[cycle - 1],
            verify_goal=lambda cycle: goals[cycle - 1],
            next_loss_action=lambda cycle: losses[cycle - 1],
            rearm_after_verified_hold=lambda cycle: None,
        )

    def test_clean_success(self):
        result = self.supervise([{"success": True}], [True], [None])
        self.assertTrue(result["success"])

    def test_relocation_retry(self):
        result = self.supervise([{"success": False, "failure_stage": "RELOCATION_ERROR"},
                                 {"success": True}], [False, True], [None, None])
        self.assertTrue(result["success"]); self.assertEqual(result["outer_cycles"], 2)

    def test_goal_verification_retry(self):
        result = self.supervise([{"success": True}, {"success": True}], [False, True], [None, None])
        self.assertTrue(result["success"]); self.assertEqual(result["goal_verifications"], [False, True])

    def test_secondary_loss_rearms(self):
        result = self.supervise([{"success": True}, {"success": True}], [False, True],
                                ["recover_object", None])
        self.assertTrue(result["success"]); self.assertEqual(result["rearmed_loss_episodes"], 2)

    def test_permanent_relocation_failure_is_bounded(self):
        result = self.supervise([{"success": False, "failure_stage": "RELOCATION_ERROR"}] * 3,
                                [], [None] * 3)
        self.assertFalse(result["success"]); self.assertEqual(result["outer_cycles"], 3)


if __name__ == "__main__": unittest.main()
