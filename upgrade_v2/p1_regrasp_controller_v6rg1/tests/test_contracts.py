from __future__ import annotations
import math, unittest
from upgrade_v2.p1_regrasp_controller_v6rg1.tracking_controller import above_target, grasp_target, xy_error, z_error, preclose_ok, next_attempt, in_workspace
from upgrade_v2.p1_regrasp_controller_v6rg1.candidate_single_attempt import MAX_ATTEMPTS as A1
from upgrade_v2.p1_regrasp_controller_v6rg1.candidate_one_retry import MAX_ATTEMPTS as A2
from upgrade_v2.p1_regrasp_controller_v6rg1.state_contract import CONTROLLER_MAY_READ_CASE_ID

class MathTests(unittest.TestCase):
    def test_dynamic_targets(self):
        o=[0.1, -0.2, 0.5]
        self.assertEqual(above_target(o)[2], 0.66)
        self.assertEqual(grasp_target(o)[2], 0.63)
        o2=[0.2,-0.2,0.5]
        self.assertNotEqual(above_target(o), above_target(o2))
    def test_preclose_gate(self):
        self.assertFalse(preclose_ok([0,0,0.63],[0,0,0.5],[0,0,0],2))
        self.assertTrue(preclose_ok([0,0,0.63],[0,0,0.5],[0,0,0],3))
        self.assertFalse(preclose_ok([0.02,0,0.63],[0,0,0.5],[0,0,0],3))
    def test_speed_gate(self):
        self.assertFalse(preclose_ok([0,0,0.63],[0,0,0.5],[0.1,0,0],3))
    def test_attempts(self):
        self.assertEqual(A1,1); self.assertEqual(A2,2)
        self.assertIsNone(next_attempt(1,1))
        self.assertEqual(next_attempt(1,2),2)
        self.assertIsNone(next_attempt(2,2))
    def test_no_third(self):
        self.assertIsNone(next_attempt(2,2))
    def test_workspace(self):
        self.assertFalse(in_workspace([5,0,0.5]))
    def test_no_case_id(self):
        self.assertFalse(CONTROLLER_MAY_READ_CASE_ID)

if __name__=='__main__':
    unittest.main()
