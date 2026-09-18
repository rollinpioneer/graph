from __future__ import annotations
import ast
import json
import tempfile
import threading
import unittest
from pathlib import Path
from p2crl.analysis import bootstrap_paired_delta, confirmatory_statistics, panel_is_complete, N_BOOT, BOOT_SEED, MARGIN, CI_LEVEL
from p2crl.backends import FakeBackend, FakeModel, get_backend
from p2crl.campaign import complete_panel_gate, execute_campaign, smoke_gate
from p2crl.campaign_ledger import CampaignLedger, LedgerError
from p2crl.checkpointing import save_milestone, state_dict_digest
from p2crl.constants_b import (
    CAMPAIGN_ID, DATASET_MANIFEST_SHA256, FORMAL_JOB_LIMIT, FORMAL_STEP_LIMIT,
    KNOWN_DEVIATIONS, PLAN_SHA256, PREREGISTRATION_COMMIT, PROTOCOL_SEMANTIC_SHA256,
    PROTOCOL_SHA256, SOURCE_LOCK_SHA256_A, TEST_NAMESPACE,
)
from p2crl.errors import GateError, IncompletePanel, ResourceLimit
from p2crl.freeze_runner import ALLOWED_MODIFY, FORBIDDEN_PREFIXES, assert_allowlist, parse_porcelain
from p2crl.io_utils import hash_json, sha256_file, write_new
from p2crl.release import ReleaseRejected, assert_no_test_payload, validate_release
from p2crl.test_panel import formal_training_complete, run_main_test
from p2crl.stochastic_panel import run_stochastic_panel
from p2crl.critical_state_panel import run_critical_state_panel
from p2crl.data_access import contracts_for_split, load_test_contracts
from p2crl.train_job import execute_job
from p2crl import rng_state

HERE = Path(__file__).resolve().parents[1]
PROTOCOL = HERE / "protocol.json"
STAGING_ART = Path(__file__).resolve().parents[4] / "artifacts"
PLAN = STAGING_ART / "job_plan.json" if (STAGING_ART / "job_plan.json").exists() else None

def make_release(**over):
    rec = {
        "schema": "P2CRL_CAMPAIGN_RELEASE_V1",
        "status": "ACTIVE",
        "campaign_id": CAMPAIGN_ID,
        "protocol_sha256": PROTOCOL_SHA256,
        "protocol_semantic_sha256": PROTOCOL_SEMANTIC_SHA256,
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "runner_commit": "a" * 40,
        "source_lock_sha256": SOURCE_LOCK_SHA256_A,
        "dataset_manifest_sha256": DATASET_MANIFEST_SHA256,
        "plan_sha256": PLAN_SHA256,
        "amendment_sha256": "b" * 64,
        "explicit_user_execution_instruction_ref": "ticket://b0-fixture",
        "activated_at_utc": "2026-09-18T00:00:00Z",
        "formal_job_limit": 60,
        "formal_step_limit": 31457280,
        "smoke_job_limit": 5,
        "smoke_steps_each": 512,
        "parallel_jobs_max": 2,
        "training_release": True,
        "known_deviations": list(KNOWN_DEVIATIONS),
    }
    rec.update(over)
    return rec

def make_amendment(tmp, runner_commit="a"*40):
    rec = {
        "schema": "P2CRL_PREREGISTRATION_IMPLEMENTATION_AMENDMENT_V1",
        "category": "NON_SCIENTIFIC_EXECUTION_RUNNER_COMPLETION",
        "preregistration_commit": PREREGISTRATION_COMMIT,
        "runner_commit": runner_commit,
        "scientific_protocol_changed": False,
        "dataset_changed": False,
        "methods_changed": False,
        "reward_changed": False,
        "mask_changed": False,
        "statistics_changed": False,
        "only_execution_runner_completed": True,
    }
    path = Path(tmp) / "amendment.json"
    write_new(path, rec)
    return path, rec

def mini_plan(n_smoke=5, n_formal=60):
    smoke = [{"job_id": f"SMOKE_{i}", "kind": "smoke", "method": f"M{i}", "run_seed": 1, "environment_step_limit": 512} for i in range(n_smoke)]
    formal = [{"job_id": f"F_{i}", "kind": "formal", "method": f"M{i%5}", "draw": "ABC"[i%3], "policy_seed": 317, "run_seed": 1, "environment_step_limit": 524288} for i in range(n_formal)]
    return {"schema": "P2CRL_PLAN_V1", "smoke_jobs": smoke, "formal_jobs": formal, "formal_job_count": n_formal}

class ReleaseTests(unittest.TestCase):
    def test_schema_reject(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release({"schema": "NOPE"}, require_active=True, repo=None)
        self.assertEqual(e.exception.code, "RELEASE_SCHEMA")
    def test_campaign_id(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(campaign_id="X"), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "CAMPAIGN_ID")
    def test_not_active(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(status="NOT_RELEASED", training_release=False), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "RELEASE_NOT_ACTIVE")
    def test_training_release_false(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(training_release=False), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "RELEASE_NOT_ACTIVE")
    def test_prereg(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(preregistration_commit="0"*40), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "PREREGISTRATION_COMMIT")
    def test_runner_commit_missing(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(runner_commit=None), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "RUNNER_COMMIT")
    def test_known_deviations(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(known_deviations=[]), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "KNOWN_DEVIATIONS")
    def test_formal_job_limit(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(formal_job_limit=59), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "FORMAL_JOB_LIMIT")
    def test_formal_step_limit(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(formal_step_limit=1), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "FORMAL_STEP_LIMIT")
    def test_smoke_job_limit(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(smoke_job_limit=4), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "SMOKE_JOB_LIMIT")
    def test_smoke_steps(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(smoke_steps_each=256), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "SMOKE_STEPS")
    def test_parallel(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(parallel_jobs_max=8), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "PARALLEL_JOBS")
    def test_instruction_missing(self):
        with self.assertRaises(ReleaseRejected) as e:
            validate_release(make_release(explicit_user_execution_instruction_ref=None), require_active=True, repo=None)
        self.assertEqual(e.exception.code, "EXPLICIT_INSTRUCTION")
    def test_inactive_when_not_required(self):
        self.assertTrue(validate_release(make_release(status="NOT_RELEASED", training_release=False), require_active=False, repo=None))
    def test_protocol_file_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "p.json"; p.write_text("{}", encoding="utf-8")
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(make_release(), protocol_path=p, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "PROTOCOL_HASH")
    def test_plan_file_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "plan.json"; p.write_text("{}", encoding="utf-8")
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(make_release(), plan_path=p, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "PLAN_HASH")
    def test_dataset_file_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "d.json"; p.write_text("{}", encoding="utf-8")
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(make_release(), dataset_manifest_path=p, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "DATASET_MANIFEST_HASH")
    def test_source_lock_hash(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "s.json"; p.write_text("{}", encoding="utf-8")
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(make_release(), source_lock_path=p, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "SOURCE_LOCK_HASH")
    def test_amendment_category(self):
        with tempfile.TemporaryDirectory() as td:
            path, rec = make_amendment(td); rec["category"] = "NOPE"; path.unlink(); write_new(path, rec)
            rel = make_release(); rel["amendment_sha256"] = sha256_file(path)
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(rel, amendment_path=path, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "AMENDMENT_CATEGORY")
    def test_amendment_science(self):
        with tempfile.TemporaryDirectory() as td:
            path, rec = make_amendment(td); rec["reward_changed"] = True; path.unlink(); write_new(path, rec)
            rel = make_release(); rel["amendment_sha256"] = sha256_file(path)
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(rel, amendment_path=path, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "AMENDMENT_SCIENCE")
    def test_amendment_scope(self):
        with tempfile.TemporaryDirectory() as td:
            path, rec = make_amendment(td); rec["only_execution_runner_completed"] = False; path.unlink(); write_new(path, rec)
            rel = make_release(); rel["amendment_sha256"] = sha256_file(path)
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(rel, amendment_path=path, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "AMENDMENT_SCOPE")
    def test_amendment_runner(self):
        with tempfile.TemporaryDirectory() as td:
            path, rec = make_amendment(td, runner_commit="c"*40)
            rel = make_release(); rel["amendment_sha256"] = sha256_file(path)
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(rel, amendment_path=path, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "AMENDMENT_RUNNER_COMMIT")
    def test_amendment_hash(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = make_amendment(td)
            with self.assertRaises(ReleaseRejected) as e:
                validate_release(make_release(amendment_sha256="0"*64), amendment_path=path, require_active=True, repo=None)
            self.assertEqual(e.exception.code, "AMENDMENT_HASH")
    def test_good_amendment(self):
        with tempfile.TemporaryDirectory() as td:
            path, _ = make_amendment(td)
            rel = make_release(); rel["amendment_sha256"] = sha256_file(path)
            self.assertTrue(validate_release(rel, amendment_path=path, require_active=True, repo=None))
    def test_real_protocol_hash_if_present(self):
        if not PROTOCOL.exists():
            self.skipTest("protocol missing")
        self.assertEqual(sha256_file(PROTOCOL), PROTOCOL_SHA256)
        from p2crl.contracts import load_protocol
        self.assertEqual(hash_json(load_protocol(PROTOCOL)), PROTOCOL_SEMANTIC_SHA256)
    def test_real_plan_hash_if_present(self):
        if PLAN is None or not Path(PLAN).exists():
            self.skipTest("plan missing")
        self.assertEqual(sha256_file(PLAN), PLAN_SHA256)
    def test_reject_before_model(self):
        be = FakeBackend()
        with self.assertRaises(ReleaseRejected):
            be.construct(make_release(status="NOT_RELEASED", training_release=False), [], "TASK_ONLY_ZERO_V1", 1, 8, 256, 256, 10)
        self.assertEqual(be.models_constructed, 0)
        self.assertFalse(be.learn_called)
    def test_active_allows_fake_construct(self):
        be = FakeBackend()
        model = be.construct(make_release(), [], "TASK_ONLY_ZERO_V1", 1, 8, 256, 256, 10)
        self.assertIsInstance(model, FakeModel)
        self.assertFalse(model.learn_called)
        with self.assertRaises(RuntimeError):
            model.learn(total_timesteps=1)

class LedgerTests(unittest.TestCase):
    def _led(self, n_smoke=5, n_formal=60):
        td = tempfile.TemporaryDirectory()
        led = CampaignLedger(Path(td.name) / "c.sqlite3")
        jobs = mini_plan(n_smoke, n_formal)
        led.init_campaign("rel", jobs["smoke_jobs"] + jobs["formal_jobs"])
        return td, led, jobs
    def test_init_pending(self):
        td, led, jobs = self._led(2, 3); self.addCleanup(td.cleanup)
        self.assertEqual(len(led.jobs(status="PENDING")), 5)
    def test_claim_once(self):
        td, led, jobs = self._led(1, 1); self.addCleanup(td.cleanup)
        led.claim("SMOKE_0")
        with self.assertRaises(LedgerError):
            led.claim("SMOKE_0")
    def test_claim_unknown(self):
        td, led, jobs = self._led(1, 1); self.addCleanup(td.cleanup)
        with self.assertRaises(LedgerError):
            led.claim("NOPE")
    def test_concurrent_claim(self):
        td, led, jobs = self._led(1, 1); self.addCleanup(td.cleanup)
        err = []
        def go():
            try:
                CampaignLedger(led.path).claim("SMOKE_0")
            except Exception as e:
                err.append(e)
        t1 = threading.Thread(target=go); t2 = threading.Thread(target=go)
        t1.start(); t2.start(); t1.join(); t2.join()
        row = CampaignLedger(led.path).job("SMOKE_0")
        self.assertEqual(row["status"], "CLAIMED")
        self.assertEqual(sum(1 for e in err if isinstance(e, LedgerError)), 1)
    def test_smoke_budget(self):
        td, led, jobs = self._led(5, 1); self.addCleanup(td.cleanup)
        for i in range(5):
            led.claim(f"SMOKE_{i}")
        led.init_campaign("rel", [{"job_id": "SMOKE_X", "kind": "smoke"}])
        with self.assertRaises(LedgerError):
            led.claim("SMOKE_X")
    def test_formal_budget(self):
        td, led, jobs = self._led(0, 60); self.addCleanup(td.cleanup)
        for i in range(60):
            led.claim(f"F_{i}")
        led.init_campaign("rel", [{"job_id": "F_X", "kind": "formal"}])
        with self.assertRaises(LedgerError):
            led.claim("F_X")
    def test_started_consumes_budget(self):
        td, led, jobs = self._led(0, 1); self.addCleanup(td.cleanup)
        led.claim("F_0"); led.mark_failed("F_0", "boom")
        self.assertEqual(led.campaign()["formal_jobs_started"], 1)
    def test_fail_stops_claim(self):
        td, led, jobs = self._led(0, 2); self.addCleanup(td.cleanup)
        led.claim("F_0"); led.mark_failed("F_0", "boom")
        with self.assertRaises(LedgerError):
            led.claim("F_1")
    def test_running_can_complete_after_fail(self):
        td, led, jobs = self._led(0, 2); self.addCleanup(td.cleanup)
        led.claim("F_0"); led.mark_running("F_0")
        led.claim("F_1"); led.mark_running("F_1")
        led.mark_failed("F_0", "boom")
        led.mark_complete("F_1", environment_steps=0, gradient_updates=0)
        self.assertEqual(led.job("F_1")["status"], "COMPLETE")
    def test_no_retry_failed(self):
        td, led, jobs = self._led(0, 1); self.addCleanup(td.cleanup)
        led.claim("F_0"); led.mark_failed("F_0", "boom")
        with self.assertRaises(LedgerError):
            led.claim("F_0")
    def test_no_retry_complete(self):
        td, led, jobs = self._led(0, 1); self.addCleanup(td.cleanup)
        led.claim("F_0"); led.mark_complete("F_0")
        with self.assertRaises(LedgerError):
            led.claim("F_0")
    def test_transitions(self):
        td, led, jobs = self._led(1, 0); self.addCleanup(td.cleanup)
        led.claim("SMOKE_0")
        n = led.con.execute("select count(*) from transitions").fetchone()[0]
        self.assertGreaterEqual(n, 1)
    def test_release_mismatch(self):
        td, led, jobs = self._led(1, 0); self.addCleanup(td.cleanup)
        with self.assertRaises(LedgerError):
            led.init_campaign("other", jobs["smoke_jobs"])
    def test_step_budget(self):
        td, led, jobs = self._led(0, 1); self.addCleanup(td.cleanup)
        led.claim("F_0")
        with self.assertRaises(LedgerError):
            led.mark_complete("F_0", environment_steps=FORMAL_STEP_LIMIT + 1)
    def test_wal(self):
        td, led, jobs = self._led(1, 0); self.addCleanup(td.cleanup)
        mode = led.con.execute("pragma journal_mode").fetchone()[0]
        self.assertEqual(str(mode).lower(), "wal")
    def test_blocked(self):
        td, led, jobs = self._led(1, 0); self.addCleanup(td.cleanup)
        led.claim("SMOKE_0"); led.mark_blocked("SMOKE_0", "x")
        self.assertEqual(led.job("SMOKE_0")["status"], "BLOCKED")
    def test_job_state_machine_order(self):
        td, led, jobs = self._led(0, 1); self.addCleanup(td.cleanup)
        self.assertEqual(led.job("F_0")["status"], "PENDING")
        led.claim("F_0"); self.assertEqual(led.job("F_0")["status"], "CLAIMED")
        led.mark_running("F_0"); self.assertEqual(led.job("F_0")["status"], "RUNNING")
        led.mark_complete("F_0", environment_steps=0); self.assertEqual(led.job("F_0")["status"], "COMPLETE")

class CampaignDrillTests(unittest.TestCase):
    def test_not_released_rejects_without_ledger(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"
            with self.assertRaises(ReleaseRejected):
                execute_campaign(release=make_release(status="NOT_RELEASED", training_release=False), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            self.assertFalse((root / "control" / "campaign.sqlite3").exists())
            self.assertTrue((root / "control" / "reject.json").exists())
    def test_active_fake_registers_without_learn(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td) / "camp"
            report = execute_campaign(release=make_release(), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            self.assertFalse(report["learn_called"])
            self.assertEqual(report["gradient_updates"], 0)
            self.assertEqual(report["formal_steps"], 0)
            self.assertEqual(report["smoke_steps"], 0)
            self.assertEqual(report["smoke_registered"], 5)
            self.assertEqual(report["formal_registered"], 60)
            self.assertTrue(report["test_refused"])
            led = CampaignLedger(root / "control" / "campaign.sqlite3")
            self.assertEqual(len(led.jobs(kind="smoke")), 5)
            self.assertEqual(len(led.jobs(kind="formal")), 60)
            self.assertTrue(all(j["environment_steps"] == 0 for j in led.jobs()))
            led.close()
    def test_fake_backend_cannot_train(self):
        with self.assertRaises(RuntimeError):
            FakeBackend().train_segment(FakeModel(), 8)
    def test_get_backend_names(self):
        self.assertEqual(get_backend("fake").name, "fake")
        self.assertEqual(get_backend("real").name, "real")
    def test_execute_job_register_only(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            led = CampaignLedger(root / "c.sqlite3")
            job = {"job_id": "SMOKE_0", "kind": "smoke", "method": "TASK_ONLY_ZERO_V1", "run_seed": 1}
            led.init_campaign("rel", [job])
            execute_job(release=make_release(), plan_job=job, campaign_root=root, ledger=led, backend="fake", register_only=True)
            rec = json.loads((root / "smoke" / "SMOKE_0" / "complete.json").read_text(encoding="utf-8"))
            self.assertEqual(rec["status"], "REGISTERED_ZERO_GRADIENT")
            self.assertFalse(rec["learn_called"])
            self.assertEqual(rec["environment_steps"], 0)
            led.close()
    def test_register_only_rejects_real_backend(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ReleaseRejected):
                execute_job(release=make_release(), plan_job={"job_id": "X", "kind": "smoke"}, campaign_root=td, backend="real", register_only=True)

class GateTests(unittest.TestCase):
    def test_smoke_gate_untrained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            execute_campaign(release=make_release(), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            led = CampaignLedger(root / "control" / "campaign.sqlite3")
            with self.assertRaises(GateError):
                smoke_gate(led)
            led.close()
    def test_complete_panel_untrained(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            execute_campaign(release=make_release(), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            led = CampaignLedger(root / "control" / "campaign.sqlite3")
            with self.assertRaises(GateError):
                complete_panel_gate(led)
            ok, reason = formal_training_complete(led)
            self.assertFalse(ok)
            led.close()
    def test_test_panel_refuses_incomplete(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            execute_campaign(release=make_release(), plan=mini_plan(), campaign_root=root, backend="fake", zero_gradient_drill=True)
            led = CampaignLedger(root / "control" / "campaign.sqlite3")
            with self.assertRaises(IncompletePanel):
                run_main_test(ledger=led, data_root=root, campaign_root=root, backend="fake")
            with self.assertRaises(IncompletePanel):
                run_stochastic_panel(ledger=led, campaign_root=root, backend="fake")
            with self.assertRaises(IncompletePanel):
                run_critical_state_panel(ledger=led, campaign_root=root, backend="fake")
            led.close()

class IsolationTests(unittest.TestCase):
    def test_split_test_blocked(self):
        with tempfile.TemporaryDirectory() as td:
            fam = Path(td) / "dataset"; fam.mkdir(parents=True)
            write_new(fam / "families.json", {"families": [{"split": "test", "family_id": 1325001, "left": {}, "right": {}}]})
            with self.assertRaises(ReleaseRejected) as e:
                contracts_for_split(td, "test")
            self.assertEqual(e.exception.code, "TEST_CONTAMINATION_PROTOCOL_VIOLATION")
    def test_load_test_requires_allow(self):
        with tempfile.TemporaryDirectory() as td:
            fam = Path(td) / "dataset"; fam.mkdir(parents=True)
            write_new(fam / "families.json", {"families": []})
            with self.assertRaises(ReleaseRejected) as e:
                load_test_contracts(td, allow=False)
            self.assertEqual(e.exception.code, "TEST_CONTAMINATION_PROTOCOL_VIOLATION")
    def test_assert_no_test_payload_split(self):
        with self.assertRaises(ReleaseRejected):
            assert_no_test_payload({"split": "test", "family_id": 1})
    def test_assert_no_test_payload_namespace(self):
        with self.assertRaises(ReleaseRejected):
            assert_no_test_payload({"split": "train_A", "family_id": TEST_NAMESPACE})
    def test_train_family_ok(self):
        assert_no_test_payload({"split": "train_A", "family_id": 1321000})
    def test_rng_capture_stable(self):
        a = rng_state.capture(); b = rng_state.capture()
        self.assertEqual(a["python"], b["python"])
    def test_validation_subprocess_stub(self):
        from p2crl.validation_worker import run_isolated_validation
        with tempfile.TemporaryDirectory() as td:
            before = rng_state.capture()
            run_isolated_validation(checkpoint=Path(td)/"x.zip", data_root=td, out_dir=Path(td)/"val", method="TASK_ONLY_ZERO_V1", backend="fake")
            after = rng_state.capture()
            self.assertEqual(before, after)
            self.assertTrue((Path(td)/"val"/"summary.json").exists())

class CheckpointTests(unittest.TestCase):
    def test_post_update_true(self):
        with tempfile.TemporaryDirectory() as td:
            rec = save_milestone(FakeModel(3), Path(td), 0, post_update=True)
            self.assertFalse(rec["post_update"])
            self.assertEqual(rec["checkpoint_phase"], "INITIALIZATION")
            self.assertEqual(rec["num_timesteps"], 0)
            self.assertEqual(rec["requested_step"], rec["actual_num_timesteps"])
            self.assertTrue((Path(td) / "policy_0.zip").exists())
    def test_post_update_false_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(RuntimeError):
                save_milestone(FakeModel(), Path(td), 32768, post_update=False)
    def test_digest_stable(self):
        self.assertEqual(state_dict_digest(FakeModel(1)), state_dict_digest(FakeModel(1)))
        self.assertNotEqual(state_dict_digest(FakeModel(1)), state_dict_digest(FakeModel(2)))
    def test_milestones_constant(self):
        from p2crl.constants_b import MILESTONES
        self.assertEqual(MILESTONES, [0, 32768, 65536, 131072, 262144, 524288])

class AnalysisTests(unittest.TestCase):
    def test_missing_panel(self):
        miss = panel_is_complete({"formal_jobs": 59, "validation_panels": 360, "main_rows": 30720, "stochastic_rows": 15360})
        self.assertIn("formal_jobs", miss)
    def test_incomplete_no_pass(self):
        rec = confirmatory_statistics([], {"formal_jobs": 0, "validation_panels": 0, "main_rows": 0, "stochastic_rows": 0})
        self.assertEqual(rec["status"], "INCOMPLETE_NO_CONFIRMATORY_PASS")
        self.assertFalse(rec["passed"])
    def test_complete_counts_empty_records_still_computes(self):
        rec = confirmatory_statistics([], {"formal_jobs": 60, "validation_panels": 360, "main_rows": 30720, "stochastic_rows": 15360})
        self.assertIn("RW_V2_minus_GEOM", rec)
        self.assertFalse(rec["global_confirmation_passed"])
    def _toy_records(self):
        recs = []
        for draw in "ABC":
            for seed in (317, 331):
                for motif in ("PRECEDENCE", "SHARED_PREREQUISITE"):
                    for fam in (1, 2):
                        for side in ("left", "right"):
                            for method, suc in (("PATHGRAPH_REMAINING_WORK_PBRS_V2", 1), ("GEOM_COUNT_EVENTS_PBRS_V1", 0), ("FLAT_CONTRACT_PBRS_V1", 0)):
                                recs.append({"draw": draw, "policy_seed": seed, "motif": motif, "family_id": fam, "side": side, "method": method, "success": suc})
        return recs
    def test_bootstrap_deterministic(self):
        recs = self._toy_records()
        a = bootstrap_paired_delta(recs, "PATHGRAPH_REMAINING_WORK_PBRS_V2", "GEOM_COUNT_EVENTS_PBRS_V1", n_boot=16, seed=BOOT_SEED)
        b = bootstrap_paired_delta(recs, "PATHGRAPH_REMAINING_WORK_PBRS_V2", "GEOM_COUNT_EVENTS_PBRS_V1", n_boot=16, seed=BOOT_SEED)
        self.assertEqual(a, b)
    def test_bootstrap_seed_changes(self):
        recs = []
        for i, draw in enumerate("ABC"):
            for seed in (317, 331, 347, 359):
                for motif in ("PRECEDENCE", "SHARED_PREREQUISITE", "ALTERNATIVE_COST", "INVALIDATION_RECOVERY"):
                    for fam in (1, 2, 3):
                        for side in ("left", "right"):
                            rw = 1 if (i + seed + fam) % 3 else 0
                            geom = 1 if (seed + fam) % 5 == 0 else 0
                            recs.append({"draw": draw, "policy_seed": seed, "motif": motif, "family_id": fam, "side": side, "method": "PATHGRAPH_REMAINING_WORK_PBRS_V2", "success": rw})
                            recs.append({"draw": draw, "policy_seed": seed, "motif": motif, "family_id": fam, "side": side, "method": "GEOM_COUNT_EVENTS_PBRS_V1", "success": geom})
        a = bootstrap_paired_delta(recs, "PATHGRAPH_REMAINING_WORK_PBRS_V2", "GEOM_COUNT_EVENTS_PBRS_V1", n_boot=32, seed=1)
        b = bootstrap_paired_delta(recs, "PATHGRAPH_REMAINING_WORK_PBRS_V2", "GEOM_COUNT_EVENTS_PBRS_V1", n_boot=32, seed=2)
        self.assertNotEqual((a["ci_low"], a["ci_high"]), (b["ci_low"], b["ci_high"]))
    def test_margin_constant(self):
        self.assertEqual(MARGIN, 0.03); self.assertEqual(CI_LEVEL, 0.975)
        self.assertEqual(N_BOOT, 20000); self.assertEqual(BOOT_SEED, 2026091806)
    def test_bootstrap_spec_present(self):
        t = (HERE / "p2crl" / "analysis.py").read_text(encoding="utf-8")
        self.assertIn("20000", t)

class AllowlistAstTests(unittest.TestCase):
    def test_allowlist_cli(self):
        self.assertIn("experiments/pathgraph_p2c_rl_v1/p2crl/cli.py", ALLOWED_MODIFY)
    def test_allowlist_forbids_protocol(self):
        self.assertTrue(any("protocol.json" in x for x in FORBIDDEN_PREFIXES))
    def test_allowlist_accepts_new_modules(self):
        assert_allowlist(["experiments/pathgraph_p2c_rl_v1/p2crl/campaign.py"])
    def test_allowlist_rejects_rm(self):
        with self.assertRaises(RuntimeError):
            assert_allowlist(["experiments/pathgraph_p2c_rm_v1/p2crm/mask_contract.py"])
    def test_ast_no_learn_in_campaign(self):
        tree = ast.parse((HERE / "p2crl" / "campaign.py").read_text(encoding="utf-8"))
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "learn"]
        self.assertEqual(hits, [])
    def test_ast_no_learn_in_train_job(self):
        tree = ast.parse((HERE / "p2crl" / "train_job.py").read_text(encoding="utf-8"))
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "learn"]
        self.assertEqual(hits, [])
    def test_ast_no_learn_in_cli(self):
        tree = ast.parse((HERE / "p2crl" / "cli.py").read_text(encoding="utf-8"))
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "learn"]
        self.assertEqual(hits, [])
    def test_ast_learn_only_real_backend(self):
        src = (HERE / "p2crl" / "backends.py").read_text(encoding="utf-8")
        tree = ast.parse(src)
        hits = [n.lineno for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and n.func.attr == "learn"]
        self.assertTrue(hits)
        for ln in hits:
            window = "\n".join(src.splitlines()[max(0, ln-20):ln+1])
            self.assertTrue("train_segment" in window or "FakeModel" in window)
    def test_cli_refuses_string(self):
        self.assertIn("execute-campaign refused", (HERE / "p2crl" / "cli.py").read_text(encoding="utf-8"))
    def test_write_new_exclusive(self):
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "x.json"; write_new(p, {"a": 1})
            with self.assertRaises(FileExistsError):
                write_new(p, {"a": 2})
    def test_known_deviation_exact(self):
        self.assertEqual(KNOWN_DEVIATIONS, ["CANDIDATE_WORKTREE_UNWRAP_MASKED_SMOKE_FOLLOWS_RESULT_BASE"])
    def test_formal_limits(self):
        self.assertEqual(FORMAL_JOB_LIMIT, 60)
        self.assertEqual(FORMAL_STEP_LIMIT, 31457280)

class SamplerMaskPbrsTests(unittest.TestCase):
    def test_run_seed_method_independent(self):
        from p2crl.sampler import run_seed
        self.assertEqual(run_seed("A", 317), run_seed("A", 317))
        self.assertNotEqual(run_seed("A", 317), run_seed("B", 317))
    def test_sampler_seed_rank(self):
        from p2crl.sampler import sampler_seed
        self.assertNotEqual(sampler_seed(1, 0), sampler_seed(1, 1))
    def test_terminal_pbrs(self):
        try:
            from p2crm.potentials_v2 import shaped_training_reward
        except Exception:
            self.skipTest("p2crm missing")
        r, b = shaped_training_reward(1.0, -0.5, 0.3, gamma=0.99, beta=1.0, terminated=True)
        self.assertAlmostEqual(b, 0.99 * 0.0 - (-0.5))
        self.assertAlmostEqual(r, 1.0 + b)
    def test_mask_length(self):
        try:
            from p2crm.mask_contract import legal_mask_bool
            from p2cq_research.environment import SkillEnv
            from p2crl.dataset_registry import build_family
            from p2crl.contracts import load_protocol
        except Exception:
            self.skipTest("env missing")
        p = load_protocol(PROTOCOL)
        fam = build_family(p, "integration", "PRECEDENCE", 0)
        env = SkillEnv(fam["left"]); env.reset()
        mask = legal_mask_bool(env)
        self.assertEqual(len(mask), 37)
        self.assertTrue(bool(mask[0]))
    def test_block_sampler_fingerprint(self):
        class C:
            def __init__(self, motif, family_id):
                self.motif = motif; self.family_id = family_id
        from p2crl.sampler import BlockSampler
        from p2crl import MOTIFS
        contracts = []; fid = 0
        for _ in range(8):
            for m in MOTIFS:
                contracts.append(C(m, fid)); contracts.append(C(m, fid)); fid += 1
        s1 = BlockSampler(contracts, 7); s2 = BlockSampler(contracts, 7)
        self.assertEqual(s1.fingerprint(), s2.fingerprint())
        s1.next_index(); s2.next_index()
        self.assertEqual(s1.fingerprint(), s2.fingerprint())

class FreezeStatusTests(unittest.TestCase):
    def test_status_string(self):
        t = (HERE / "p2crl" / "freeze_runner.py").read_text(encoding="utf-8")
        self.assertIn("RUNNER_FROZEN_WAITING_FOR_EXPLICIT_CAMPAIGN_EXECUTION", t)
        self.assertIn("training_release", t)
        self.assertIn("smoke_jobs", t)
    def test_hashes_constants(self):
        self.assertEqual(PROTOCOL_SHA256, "cb2961a67e31c62fff91597b8d4ff357bd950d574381c9326667385734af46e8")
        self.assertEqual(PLAN_SHA256, "b2713d61f432b3d9328b81ea3265bb6fc509055816edfee6e5bfcff2b1013bdd")
        self.assertEqual(DATASET_MANIFEST_SHA256, "04a58235b83c30416b0dd82d3e61450c88643b1559279392dcbd7c6516b01c1f")
        self.assertEqual(SOURCE_LOCK_SHA256_A, "9e29cc5d0f88f5480fc505523a75793e1665460f2651888cc4f2351bc2ccfe33")
    def test_monitoring_disk(self):
        from p2crl.monitoring import assert_disk, campaign_bytes
        with tempfile.TemporaryDirectory() as td:
            self.assertEqual(campaign_bytes(td), 0)
            self.assertEqual(assert_disk(td, 0), 0)
    def test_resource_limit(self):
        from p2crl.monitoring import assert_disk
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ResourceLimit):
                assert_disk(td, extra=10**18)
    def test_cli_subcommands_present(self):
        text = (HERE / "p2crl" / "cli.py").read_text(encoding="utf-8")
        for cmd in ("freeze-runner", "validate-release", "execute-campaign", "campaign-status", "finalize-campaign"):
            self.assertIn(cmd, text)
    def test_no_on_rollout_end_milestone(self):
        self.assertNotIn("_on_rollout_end", (HERE / "p2crl" / "train_job.py").read_text(encoding="utf-8"))
    def test_package_preflight_stays_false(self):
        t = (HERE / "p2crl" / "freeze_runner.py").read_text(encoding="utf-8")
        self.assertIn("package_preflight_passed", t)
        self.assertIn("False", t)
    def test_parse_porcelain_keeps_unstaged_first_path(self):
        text = (
            " M experiments/pathgraph_p2c_rl_v1/p2crl/checkpointing.py\n"
            " M experiments/pathgraph_p2c_rl_v1/p2crl/cli.py\n"
            "?? experiments/pathgraph_p2c_rl_v1/p2crl/freeze_runner.py\n"
        )
        items = parse_porcelain(text)
        self.assertIn("experiments/pathgraph_p2c_rl_v1/p2crl/checkpointing.py", items)
        self.assertNotIn("xperiments/pathgraph_p2c_rl_v1/p2crl/checkpointing.py", items)
        self.assertEqual(items["experiments/pathgraph_p2c_rl_v1/p2crl/checkpointing.py"], "M")
        assert_allowlist(items)
    def test_git_helper_does_not_strip_leading_space(self):
        src = (HERE / "p2crl" / "freeze_runner.py").read_text(encoding="utf-8")
        self.assertIn(".rstrip(\"\\n\")", src)
        self.assertNotIn("text=True).strip()", src)
    def test_inventory_csv_uses_lf(self):
        src = (HERE / "p2crl" / "freeze_runner.py").read_text(encoding="utf-8")
        self.assertIn("lineterminator=\"\\n\"", src)

if __name__ == "__main__":
    unittest.main()
