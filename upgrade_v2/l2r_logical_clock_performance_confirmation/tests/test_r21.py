from pathlib import Path
import subprocess, tempfile, unittest

from upgrade_v2.l2r_logical_clock_performance_confirmation.conformance import run
from upgrade_v2.l2r_logical_clock_performance_confirmation.protocol import FROZEN_BLOBS, protocol_lock
from upgrade_v2.l2r_logical_clock_performance_confirmation.registry import CASES, FAMILIES, registry

ROOT=Path(__file__).resolve().parents[3]


class R21Tests(unittest.TestCase):
    def test_32_rollouts(self): self.assertEqual(len(CASES)*len(FAMILIES),32)
    def test_four_new_families(self): self.assertEqual([x[1] for x in FAMILIES],[888000,888001,888002,888003])
    def test_seeds_unique(self): self.assertEqual(len({base+i for _,_,base in FAMILIES for i in range(8)}),32)
    def test_fixed_cases(self): self.assertEqual([c.case_id.split('_',1)[0] for c in CASES],["T1","T2","T3","T4","T7","T9","T11","T12"])
    def test_no_replacement(self): self.assertFalse(registry()["seed_replacement_allowed"])
    def test_thresholds_frozen(self): self.assertEqual(protocol_lock()["performance_gates"]["overall_accuracy_min"],71/72)
    def test_l3_closed(self): self.assertFalse(protocol_lock()["l3_entry_allowed"])
    def test_physical_reference_frozen(self): self.assertIn("physical_reference",FROZEN_BLOBS)
    def test_time_scoring_frozen(self): self.assertEqual(FROZEN_BLOBS["temporal_scoring"],"e9182eff3e277017ebee8ad05a5bf3e9120871df")
    def test_guard_blob(self):
        got=subprocess.run(["git","-C",str(ROOT),"hash-object","upgrade_v2/l2r_logical_observation_clock/guard.py"],capture_output=True,text=True,check=True).stdout.strip()
        self.assertEqual(got,FROZEN_BLOBS["guard"])
    def test_conformance_localizes_nominal_boundary_failure(self):
        with tempfile.TemporaryDirectory() as d:
            result=run(ROOT,Path(d)/"result.json")
        self.assertEqual((result["status"],result["cases_passed"],result["cases_total"]),("FAIL",11,12))
        failed=[row["case_id"] for row in result["cases"] if not row["passed"]]
        self.assertEqual(failed,["C04_nominal_100_boundary_confirms"])
    def test_conformance_zero_physics(self):
        with tempfile.TemporaryDirectory() as d: result=run(ROOT,Path(d)/"result.json")
        self.assertEqual((result["mujoco_imported"],result["physical_executions"]),(False,0))


if __name__=="__main__": unittest.main()
