import unittest
from pathlib import Path
from upgrade_v2.l2r_l3_observation_boundary import BASE_COMMIT,CLP3_ID
from upgrade_v2.l2r_l3_observation_boundary.protocol import protocol_lock
from upgrade_v2.l2r_l3_observation_boundary.registry import CASES,FAMILIES
from upgrade_v2.l2r_l3_observation_boundary.static_lock import validate
ROOT=Path(__file__).resolve().parents[3]
class T(unittest.TestCase):
 def test_base(self):self.assertEqual(BASE_COMMIT,"429efbb3fca0572f7afe4bc6d535433a54d7058f")
 def test_candidate(self):self.assertEqual(CLP3_ID,"O_C3_CLP3_CANONICAL_TIME")
 def test_96(self):self.assertEqual(len(CASES)*len(FAMILIES)*3,96)
 def test_new_seeds(self):self.assertEqual([x[1] for x in FAMILIES],[896000,896001,896002,896003])
 def test_signal_outcome(self):self.assertEqual(protocol_lock()["special_outcome"],"SIGNAL_NOT_OBSERVED")
 def test_frozen(self):self.assertTrue(validate(ROOT)["passed"])
if __name__=="__main__":unittest.main()
