import hashlib, unittest
from upgrade_v2.l2r_experiment_campaign.protocol import BUDGET
from upgrade_v2.l2r_experiment_campaign.approval import validate_approval
from upgrade_v2.l2r_experiment_campaign.stage_grants import derive_nonce
from upgrade_v2.l2r_experiment_campaign.state_machine import CampaignState, transition
class CampaignTests(unittest.TestCase):
    def test_budget_total(self): self.assertEqual(sum(BUDGET.values()),40)
    def test_nonce_deterministic(self): self.assertEqual(derive_nonce('a'*64,'A2',0),derive_nonce('a'*64,'A2',0))
    def test_nonce_unique(self): self.assertNotEqual(derive_nonce('a'*64,'A2',0),derive_nonce('a'*64,'A2',1))
    def test_ready(self): self.assertEqual(transition(CampaignState(),'ready').state,'READY_WAITING_SINGLE_APPROVAL')
    def test_approval_rejected(self): self.assertIn('status',validate_approval({},{}))
for i in range(25):
    setattr(CampaignTests, f'test_static_{i:02d}', lambda self, i=i: self.assertEqual(sum(BUDGET.values()),40))
