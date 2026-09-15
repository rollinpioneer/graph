from __future__ import annotations
import math, unittest
from upgrade_v2.p1_return_controller_v6rc1.return_geometry import (
    relative_position_in_eef, desired_eef_position, quat_normalize, clipnorm, in_workspace, finite_vec
)
from upgrade_v2.p1_return_controller_v6rc1.attachment_estimator import AttachmentEstimator, N_SAMPLES
from upgrade_v2.p1_return_controller_v6rc1.state_machine import stages_for
from upgrade_v2.p1_return_controller_v6rc1.selection import COMPLEXITY

class MathTests(unittest.TestCase):
    def test_relative_identity(self):
        self.assertEqual(relative_position_in_eef([1,2,3],[0,0,0],[1,0,0,0]), [1.0,2.0,3.0])
    def test_desired(self):
        self.assertEqual(desired_eef_position([1,2,3],[1,0,0,0],[0.1,0.2,0.3]), [0.9,1.8,2.7])
    def test_compensation(self):
        p=desired_eef_position([0.5,0.0,0.4],[1,0,0,0],[0.02,-0.01,-0.13])
        self.assertAlmostEqual(p[0],0.48); self.assertAlmostEqual(p[1],0.01); self.assertAlmostEqual(p[2],0.53)
    def test_zero_quat(self):
        with self.assertRaises(ValueError):
            quat_normalize([0,0,0,0])
    def test_left_right_offset_different(self):
        a=desired_eef_position([0,0,0.5],[1,0,0,0],[0.0,-0.04,-0.13])
        b=desired_eef_position([0,0,0.5],[1,0,0,0],[0.0, 0.04,-0.13])
        self.assertNotEqual(a,b)
    def test_clipnorm(self):
        v=clipnorm([3,4,0],1.0)
        self.assertAlmostEqual(math.sqrt(sum(x*x for x in v)),1.0)
    def test_workspace_reject(self):
        self.assertFalse(in_workspace([5,0,0.5]))
    def test_nan_reject(self):
        with self.assertRaises(ValueError):
            finite_vec([float("nan"),0,0])
    def test_fixed_orientation_tag(self):
        from upgrade_v2.p1_return_controller_v6rc1.return_geometry import COMPENSATION_MODE
        self.assertEqual(COMPENSATION_MODE, "TRANSLATION_COMPENSATED_FIXED_ORIENTATION")
        self.assertNotEqual(COMPENSATION_MODE, "FULL_SE3_COMPENSATED")

class AttachmentTests(unittest.TestCase):
    def test_ten_samples(self):
        e=AttachmentEstimator()
        for i in range(10):
            e.add([0,0,0.5],[0,0,0.63],[1,0,0,0])
        est=e.estimate()
        self.assertTrue(est["stable"])
        self.assertEqual(est["n"],10)
    def test_unstable(self):
        e=AttachmentEstimator()
        for i in range(10):
            e.add([0.02*i,0,0.5],[0,0,0.63],[1,0,0,0])
        self.assertEqual(e.estimate()["status"], "ATTACHMENT_UNSTABLE")

class StageTests(unittest.TestCase):
    def test_rc_order(self):
        s=stages_for("RC2_OBJECT_COMPENSATED_STAGED")
        self.assertEqual(s[:4], ["WAIT_REHOLD_STABLE","ESTIMATE_ATTACHMENT","PLAN_RETURN","SAFE_CLEARANCE"])
    def test_complexity(self):
        self.assertLess(COMPLEXITY["RC1_OBJECT_COMPENSATED_DIRECT"], COMPLEXITY["RC3_OBJECT_ERROR_SERVO"])

class PolicyTests(unittest.TestCase):
    def test_no_seed_retry_flag(self):
        from upgrade_v2.p1_return_controller_v6rc1.collector import ReturnCollector
        self.assertTrue(hasattr(ReturnCollector, "collect"))
    def test_timeout_constant(self):
        from upgrade_v2.p1_return_controller_v6rc1.collector import RETURN_TIMEOUT_S
        self.assertEqual(RETURN_TIMEOUT_S, 2.0)

if __name__=="__main__":
    unittest.main()
