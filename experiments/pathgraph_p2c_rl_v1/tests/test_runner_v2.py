from __future__ import annotations
import json
import tempfile
import unittest
from pathlib import Path

from p2crl.backends import FakeBackend, FakeModel, close_model
from p2crl.campaign import complete_panel_gate, smoke_gate
from p2crl.campaign_ledger import CampaignLedger
from p2crl.checkpointing import save_milestone
from p2crl.constants_b import FORMAL_STEPS, MILESTONES, SMOKE_STEPS
from p2crl.errors import CheckpointStepMismatch, GateError, ProtocolViolation
from p2crl.io_utils import write_new
from p2crl.optimizer_counter import OptimizerStepCounter
from p2crl.release import ReleaseRejected, validate_release
from p2crl.release_terminal_registry import assert_release_not_tombstoned
from p2crl.training_shape import FORMAL_SHAPE, SMOKE_SHAPE, shape_for_kind
from p2crl.train_job import execute_job
from tests.test_runner_b0 import make_release, mini_plan


class DummyHandle:
    def __init__(self):
        self.removed = False
    def remove(self):
        self.removed = True


class DummyOptimizer:
    def __init__(self):
        self.hooks = []
    def register_step_post_hook(self, hook):
        self.hooks.append(hook)
        return DummyHandle()
    def step(self):
        for h in list(self.hooks):
            h(self, (), {})


class ShapeTests(unittest.TestCase):
    def test_smoke_shape_8x64(self):
        self.assertEqual((SMOKE_SHAPE.n_envs, SMOKE_SHAPE.n_steps), (8, 64))
    def test_formal_shape_8x256(self):
        self.assertEqual((FORMAL_SHAPE.n_envs, FORMAL_SHAPE.n_steps), (8, 256))
    def test_smoke_quantum_512(self):
        self.assertEqual(SMOKE_SHAPE.rollout_quantum, 512)
        self.assertEqual(SMOKE_SHAPE.n_envs * SMOKE_SHAPE.n_steps, SMOKE_STEPS)
    def test_formal_milestones_divisible(self):
        for step in MILESTONES:
            self.assertEqual(step % FORMAL_SHAPE.rollout_quantum, 0)
    def test_buffer_divisible_by_batch(self):
        self.assertEqual(SMOKE_SHAPE.rollout_quantum % SMOKE_SHAPE.batch_size, 0)
        self.assertEqual(FORMAL_SHAPE.rollout_quantum % FORMAL_SHAPE.batch_size, 0)
    def test_smoke_optimizer_theory_20(self):
        self.assertEqual(SMOKE_SHAPE.expected_optimizer_steps(512), 20)
    def test_formal_optimizer_theory_20480(self):
        self.assertEqual(FORMAL_SHAPE.expected_optimizer_steps(FORMAL_STEPS), 20480)
    def test_kind_dispatch(self):
        self.assertIs(shape_for_kind("smoke"), SMOKE_SHAPE)
        self.assertIs(shape_for_kind("formal"), FORMAL_SHAPE)


class OptimizerHookTests(unittest.TestCase):
    def test_post_hook_counts_actual_steps(self):
        opt = DummyOptimizer()
        counter = OptimizerStepCounter()
        counter.attach(opt)
        opt.step(); opt.step()
        self.assertEqual(counter.count, 2)
        counter.detach()
    def test_hook_removed_on_exception(self):
        opt = DummyOptimizer()
        counter = OptimizerStepCounter()
        handle = counter.attach(opt)
        try:
            raise RuntimeError("boom")
        except RuntimeError:
            counter.detach()
        self.assertTrue(handle.removed)
    def test_missing_hook_api_fails(self):
        counter = OptimizerStepCounter()
        with self.assertRaises(Exception):
            counter.attach(object())
    def test_fake_backend_does_not_call_optimizer(self):
        be = FakeBackend()
        model = be.construct(make_release(), [], "TASK_ONLY_ZERO_V1", 1, 8, 64, 256, 10, shape=SMOKE_SHAPE)
        self.assertIsNone(model.policy.optimizer)
        self.assertFalse(be.learn_called)
        self.assertEqual(be.optimizer_step_count, 0)


class CheckpointExactTests(unittest.TestCase):
    def test_mismatch_creates_no_zip(self):
        with tempfile.TemporaryDirectory() as td:
            model = FakeModel()
            model.num_timesteps = 2048
            with self.assertRaises(CheckpointStepMismatch):
                save_milestone(model, Path(td), 512, post_update=True)
            self.assertFalse((Path(td) / "policy_512.zip").exists())
            self.assertEqual(list(Path(td).glob("*")), [])
    def test_step0_initialization_semantics(self):
        with tempfile.TemporaryDirectory() as td:
            rec = save_milestone(FakeModel(), Path(td), 0, post_update=True)
            self.assertEqual(rec["checkpoint_phase"], "INITIALIZATION")
            self.assertFalse(rec["post_update"])
            self.assertEqual(rec["actual_num_timesteps"], 0)
            self.assertEqual(rec["gradient_updates_semantics"], "OPTIMIZER_STEP_CALLS")
    def test_post_update_metadata(self):
        with tempfile.TemporaryDirectory() as td:
            model = FakeModel()
            model.num_timesteps = 512
            model._p2crl_rollout_quantum = 512
            model._p2crl_rollout_iteration_count = 1
            model._p2crl_ppo_epoch_count = 10
            model._p2crl_optimizer_step_count = 20
            rec = save_milestone(model, Path(td), 512, post_update=True)
            self.assertEqual(rec["checkpoint_phase"], "POST_UPDATE")
            self.assertTrue(rec["post_update"])
            self.assertEqual(rec["requested_step"], rec["actual_num_timesteps"])
            self.assertEqual(rec["optimizer_step_count"], 20)
            self.assertEqual(rec["gradient_updates"], 20)


class SmokeGateV2Tests(unittest.TestCase):
    def _trained_smoke(self, root, steps=512, extra=None):
        led = CampaignLedger(root / "c.sqlite3")
        jobs = [{"job_id": f"SMOKE_{i}", "kind": "smoke"} for i in range(5)]
        led.init_campaign("rel", jobs)
        rec0 = None
        for i, job in enumerate(jobs):
            d = root / "smoke" / job["job_id"]
            d.mkdir(parents=True)
            rec = {
                "status": "TRAINING_COMPLETE",
                "environment_steps": steps,
                "rollout_quantum": 512,
                "rollout_iterations": 1,
                "ppo_epoch_count": 10,
                "optimizer_step_count": 20,
                "final_checkpoint_actual_step": steps,
                "backend": "real",
                "invalid_selected": 0,
                "nonfinite": 0,
            }
            if extra:
                rec.update(extra)
            write_new(d / "complete.json", rec)
            write_new(d / "initialization.json", {"policy_sha256": "abc"})
            led.claim(job["job_id"], directory=d, pid=1)
            led.mark_running(job["job_id"], directory=str(d), pid=1)
            led.mark_complete(job["job_id"], environment_steps=steps, gradient_updates=20, directory=str(d))
            rec0 = rec
        return led

    def test_accepts_exact_512(self):
        with tempfile.TemporaryDirectory() as td:
            led = self._trained_smoke(Path(td), 512)
            self.assertTrue(smoke_gate(led))
            led.close()
    def test_rejects_511(self):
        with tempfile.TemporaryDirectory() as td:
            led = self._trained_smoke(Path(td), 511)
            with self.assertRaises(GateError):
                smoke_gate(led)
            led.close()
    def test_rejects_513(self):
        with tempfile.TemporaryDirectory() as td:
            led = self._trained_smoke(Path(td), 513)
            with self.assertRaises(GateError):
                smoke_gate(led)
            led.close()
    def test_rejects_2048(self):
        with tempfile.TemporaryDirectory() as td:
            led = self._trained_smoke(Path(td), 2048, extra={"final_checkpoint_actual_step": 2048})
            with self.assertRaises(GateError):
                smoke_gate(led)
            led.close()
    def test_rejects_optimizer_19(self):
        with tempfile.TemporaryDirectory() as td:
            led = self._trained_smoke(Path(td), 512, extra={"optimizer_step_count": 19})
            with self.assertRaises(GateError):
                smoke_gate(led)
            led.close()


class TombstoneTests(unittest.TestCase):
    def test_tombstoned_release_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            rel = Path(td) / "release.json"
            write_new(rel, make_release())
            from p2crl.io_utils import sha256_file
            sha = sha256_file(rel)
            reg = Path(td) / "reg.json"
            write_new(reg, {"schema": "P2CRL_RELEASE_TOMBSTONE_REGISTRY_V1", "entries": [
                {"attempt_id": "P2CRL_V1_ATTEMPT_01", "release_sha256": sha, "reusable": False, "stop_code": "SMOKE_FAILED_STOP"}
            ]})
            with self.assertRaises(ReleaseRejected) as e:
                assert_release_not_tombstoned(rel, reg)
            self.assertEqual(e.exception.code, "RELEASE_TOMBSTONED")
    def test_v2_schema_accepted_with_shapes(self):
        rel = make_release(schema="P2CRL_CAMPAIGN_RELEASE_V2", attempt_id="P2CRL_V1_ATTEMPT_02", runner_v2_commit="a"*40, runner_commit=None,
                           smoke_shape={"n_envs": 8, "n_steps": 64, "batch_size": 256, "n_epochs": 10},
                           formal_shape={"n_envs": 8, "n_steps": 256, "batch_size": 256, "n_epochs": 10},
                           attempt_root="/tmp/campaign_v1_attempt_02", job_plan_sha256="b2713d61f432b3d9328b81ea3265bb6fc509055816edfee6e5bfcff2b1013bdd")
        # make_release still has runner_commit unless None deleted
        rel.pop("runner_commit", None)
        rel["runner_v2_commit"] = "a" * 40
        self.assertTrue(validate_release(rel, require_active=True, repo=None))
    def test_v2_old_root_rejected(self):
        rel = make_release(schema="P2CRL_CAMPAIGN_RELEASE_V2", attempt_id="P2CRL_V1_ATTEMPT_02",
                           smoke_shape={"n_envs": 8, "n_steps": 64, "batch_size": 256, "n_epochs": 10},
                           formal_shape={"n_envs": 8, "n_steps": 256, "batch_size": 256, "n_epochs": 10},
                           attempt_root="/x/campaign_v1")
        rel.pop("runner_commit", None)
        rel["runner_v2_commit"] = "a" * 40
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(rel, require_active=True, repo=None)
        self.assertEqual(e.exception.code, "OLD_ATTEMPT_ROOT")


class FakeJobShapeTests(unittest.TestCase):
    def test_register_only_uses_smoke_shape(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            job = {"job_id": "SMOKE_0", "kind": "smoke", "method": "TASK_ONLY_ZERO_V1", "run_seed": 1}
            execute_job(release=make_release(), plan_job=job, campaign_root=root, backend="fake", register_only=True)
            ident = json.loads((root / "smoke" / "SMOKE_0" / "run_identity.json").read_text(encoding="utf-8"))
            self.assertEqual(ident["training_shape"]["n_steps"], 64)
            init = json.loads((root / "smoke" / "SMOKE_0" / "initialization.json").read_text(encoding="utf-8"))
            self.assertEqual(init["rollout_quantum"], 512)
            rec = json.loads((root / "smoke" / "SMOKE_0" / "complete.json").read_text(encoding="utf-8"))
            self.assertFalse(rec["learn_called"])
            self.assertEqual(rec["optimizer_step_count"], 0)


class FormalGateOptionalFields(unittest.TestCase):
    def test_untrained_still_fails(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            from p2crl.campaign import execute_campaign
            execute_campaign(release=make_release(), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            led = CampaignLedger(root / "control" / "campaign.sqlite3")
            with self.assertRaises(GateError):
                complete_panel_gate(led)
            led.close()


class CloseModelTests(unittest.TestCase):
    def test_close_model_detaches(self):
        model = FakeModel()
        counter = OptimizerStepCounter()
        opt = DummyOptimizer()
        handle = counter.attach(opt)
        model._p2crl_optimizer_counter = counter
        close_model(model)
        self.assertTrue(handle.removed)

if __name__ == "__main__":
    unittest.main()