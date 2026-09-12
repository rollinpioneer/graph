import tempfile, unittest
from pathlib import Path
from upgrade_v2.l2r_canonical_time_confirmation.conformance import run
from upgrade_v2.l2r_canonical_time_confirmation.guard import CanonicalTimeGuard, MAXIMUM_PHYSICAL_GAP_NS
from upgrade_v2.l2r_canonical_time_confirmation.registry import CASES,FAMILIES

ROOT=Path(__file__).resolve().parents[3]


class CLP3Tests(unittest.TestCase):
    def test_integer_boundary(self): self.assertEqual(MAXIMUM_PHYSICAL_GAP_NS,100_000_000)
    def test_minimum_distinct_time(self): self.assertEqual(CanonicalTimeGuard.minimum_physical_gap_ns,1)
    def test_conformance_14(self):
        with tempfile.TemporaryDirectory() as d: result=run(ROOT,Path(d)/"result.json")
        self.assertEqual((result["status"],result["cases_passed"],result["cases_total"]),("PASS",14,14))
    def test_no_mujoco(self):
        with tempfile.TemporaryDirectory() as d: result=run(ROOT,Path(d)/"result.json")
        self.assertFalse(result["mujoco_imported"]); self.assertEqual(result["physical_executions"],0)
    def test_physical_registry_24(self): self.assertEqual(len(CASES)*len(FAMILIES),24)
    def test_new_family_seeds(self): self.assertEqual([r[1] for r in FAMILIES],[890000,890001,890002,890003])


if __name__=="__main__": unittest.main()
