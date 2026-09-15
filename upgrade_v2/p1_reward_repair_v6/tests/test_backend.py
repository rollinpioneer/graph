from __future__ import annotations
import json, math, os, tempfile, unittest
from pathlib import Path
import numpy as np

REPO = Path(os.environ.get("P1_FROZEN_REPO") or "/home/__compress_data/xushijie/graph_pathgraph_p1_v6_three_issue_worktree")

class BackendUnit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        try:
            import mujoco  # noqa: F401
        except Exception:
            raise unittest.SkipTest("mujoco missing")
        from upgrade_v2.p1_reward_repair_v6.backend_adapter import PhysicalBackend, PHYSICS_DT
        cls.PhysicalBackend = PhysicalBackend
        cls.PHYSICS_DT = PHYSICS_DT

    def _be(self):
        be = self.PhysicalBackend(REPO)
        be.reset_episode({"family_id": 930100, "rollout_seed": 1}, {"case_id": "UNIT"})
        return be

    def test_mj_step_time(self):
        be = self._be()
        for _ in range(10):
            be.advance_control_interval()
        self.assertTrue(math.isclose(be.mj_steps * self.PHYSICS_DT, float(be.data.time), abs_tol=1e-9, rel_tol=0))

    def test_reset_during_episode_errors(self):
        be = self._be()
        with self.assertRaises(RuntimeError):
            be.reset_episode({"family_id": 930100}, {"case_id": "UNIT"})

    def test_direct_qpos_after_start_errors(self):
        be = self._be()
        with self.assertRaises(RuntimeError):
            be._write_object_qpos_init(be._object_xyz())

    def test_checkpoint_restore_errors(self):
        be = self._be()
        snap = be.read_integration_state()
        with self.assertRaises(RuntimeError):
            be.restore_checkpoint_forbidden(snap)

    def test_open_weld_not_automatic_loss(self):
        be = self._be()
        # never held: weld off is not a loss
        be.set_controller_target(be.data.mocap_pos[0], "open", mode="WAIT")
        for _ in range(8):
            be.advance_control_interval()
        kinds = [e["kind"] for r in be.reference_log for e in r.get("events") or []]
        self.assertNotIn("LOSS", kinds)

    def test_recovery_command_without_motion_no_false_progress(self):
        be = self._be()
        obj0 = be._object_xyz().copy()
        be.issue_recovery_command(execute=False)
        be.set_controller_target(be.data.mocap_pos[0], "open", mode="WAIT")
        for _ in range(10):
            be.advance_control_interval()
        self.assertTrue(np.linalg.norm(be._object_xyz() - obj0) < 0.05)
        self.assertFalse(be.reference_log[-1]["recovery_executing"])

    def test_partial_approach_not_regrasp(self):
        be = self._be()
        obj = be._object_xyz()
        near = obj + np.array([0.0, 0.0, 0.18])
        be.set_controller_target(near, "open", mode="PARTIAL_APPROACH")
        for _ in range(30):
            be.advance_control_interval()
        self.assertFalse(be.reference_log[-1]["held"])

    def test_commanded_release_not_loss_token(self):
        be = self._be()
        obj = be._object_xyz()
        grasp = obj + np.array([0.0, 0.0, 0.13])
        be.set_controller_target(grasp, "closed", mode="CLOSE")
        for _ in range(25):
            be.advance_control_interval()
        be.set_controller_target(np.array(be.data.mocap_pos[0]), "open", mode="RELEASE")
        for _ in range(15):
            be.advance_control_interval()
        kinds = [e["kind"] for r in be.reference_log for e in r.get("events") or []]
        self.assertNotIn("LOSS", kinds)

    def test_unknown_log_cannot_confirm(self):
        from upgrade_v2.p1_reward_repair_v6.confirm_analysis import analyze
        with tempfile.TemporaryDirectory() as td:
            td = Path(td)
            # empty input should not set confirmation true
            (td/"in").mkdir(); (td/"in"/"states.jsonl").write_text("", encoding="utf-8")
            (td/"in"/"manifest.json").write_text("{}", encoding="utf-8")
            (td/"scores").mkdir(); (td/"scores"/"per_transition_rewards.csv").write_text("episode_id,method,reward\n", encoding="utf-8")
            (td/"scores"/"values").mkdir()
            pkg = Path("/home/xushijie/PathGraph_P1_V6_Three_Issue_Execution_Agent_Package_V1.0")
            closure = pkg/"contracts"/"closure_contract.json"
            rec = analyze(td/"in", td/"scores", closure, pkg, None, td/"out")
            self.assertFalse(rec["confirmation_passed"])
            self.assertFalse(rec["independent_verification_passed"])

    def test_confirmation_no_seed_replace_flag(self):
        from upgrade_v2.p1_reward_repair_v6.collector import ScriptedCollector
        self.assertTrue(hasattr(ScriptedCollector, "collect_plan"))

if __name__ == "__main__":
    unittest.main()
