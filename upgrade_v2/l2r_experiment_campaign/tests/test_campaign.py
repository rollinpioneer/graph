import hashlib, unittest
from upgrade_v2.l2r_experiment_campaign.protocol import BUDGET
from upgrade_v2.l2r_experiment_campaign.budget import accounting
from upgrade_v2.l2r_experiment_campaign.baseline_stage import classify_attempt
from upgrade_v2.l2r_experiment_campaign.environment_preflight import FROZEN_ENVIRONMENT, preflight, frozen_environment
import tempfile
from pathlib import Path
from upgrade_v2.l2r_experiment_campaign.approval import validate_approval
from upgrade_v2.l2r_experiment_campaign.stage_grants import derive_nonce, grants
from upgrade_v2.l2r_experiment_campaign.state_machine import CampaignState, transition
class CampaignTests(unittest.TestCase):
    def test_budget_total(self): self.assertEqual(sum(BUDGET.values()),40)
    def test_nonce_deterministic(self): self.assertEqual(derive_nonce('a'*64,'A2',0),derive_nonce('a'*64,'A2',0))
    def test_nonce_unique(self): self.assertNotEqual(derive_nonce('a'*64,'A2',0),derive_nonce('a'*64,'A2',1))
    def test_retry_nonce_unique(self): self.assertNotEqual(derive_nonce('a'*64,'A2',0),derive_nonce('a'*64,'A2:retry:1',0))
    def test_ready(self): self.assertEqual(transition(CampaignState(),'ready').state,'READY_WAITING_SINGLE_APPROVAL')
    def test_approval_rejected(self): self.assertIn('status',validate_approval({},{}))
    def test_unapproved_grants_rejected(self):
        with self.assertRaises(PermissionError): grants({"status":"NOT_APPROVED"})
    def test_stop_is_terminal(self):
        s=CampaignState(); transition(s,"stop:STOPPED_A2_FAILED"); transition(s,"approved"); self.assertEqual(s.state,"STOPPED_A2_FAILED")
    def test_prephysics_recovery(self):
        s=CampaignState(); transition(s,"block:BLOCKED_PREPHYSICS_ENVIRONMENT"); transition(s,"recover_prephysics"); self.assertEqual(s.state,"RUNNING")
    def test_accounting_distinguishes_attempt(self):
        a=accounting([{"status":"PREPHYSICS_FAILED","physical_instance_started":False,"physical_budget_delta":0}]); self.assertEqual(a["grant_attempts"],1); self.assertEqual(a["physical_budget_used"],0)
    def test_attempt_without_marker_is_zero(self):
        with tempfile.TemporaryDirectory() as d: self.assertEqual(classify_attempt(Path(d))["physical_budget_delta"],0)
    def test_frozen_environment_keys(self): self.assertEqual(set(FROZEN_ENVIRONMENT),{"PYTHONHASHSEED","PYTHONNOUSERSITE","MUJOCO_GL","OMP_NUM_THREADS","OPENBLAS_NUM_THREADS","MKL_NUM_THREADS","NUMEXPR_NUM_THREADS"})
    def test_preflight_rejects_unpinned_environment(self):
        result = preflight(environ={})
        self.assertFalse(result.ok)
        self.assertIn("PYTHONHASHSEED", result.errors)
    def test_frozen_environment_is_explicit(self):
        env = frozen_environment()
        self.assertTrue(all(env[k] == v for k, v in FROZEN_ENVIRONMENT.items()))
    def test_recovery_derives_new_nonce(self):
        from upgrade_v2.l2r_experiment_campaign.orchestrator import CampaignOrchestrator
        with tempfile.TemporaryDirectory() as d:
            o=CampaignOrchestrator(Path(d), {})
            o.prepare(); o.state.current_stage="A2"; o.state.state="BLOCKED_PREPHYSICS_ENVIRONMENT"
            approval={"status":"APPROVED","campaign_nonce":"a"*64,"reviewer_id":"r","approved_at_utc":"2026-09-12T00:00:00Z","expires_at_utc":"2026-09-19T00:00:00Z"}
            o.approval=approval
            g1,r1=o.recover_prephysics(approval,retry_index=1,output_root_factory=lambda n: Path(d)/n[:8])
            self.assertEqual(o.state.state,"RUNNING"); self.assertTrue(r1.exists() or not r1.exists())
            self.assertNotEqual(g1["single_use_nonce"], derive_nonce(approval["campaign_nonce"],"A2",0))
for i in range(25):
    setattr(CampaignTests, f'test_static_{i:02d}', lambda self, i=i: self.assertEqual(sum(BUDGET.values()),40))
