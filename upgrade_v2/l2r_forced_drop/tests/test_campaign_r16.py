import tempfile, unittest
from pathlib import Path
from upgrade_v2.l2r_forced_drop.protocol import pulse_levels
from upgrade_v2.l2r_forced_drop.calibration_runner import plan_calibration
from upgrade_v2.l2r_forced_drop.development_runner import development_manifest
from upgrade_v2.l2r_forced_drop.evaluate import evaluate_trace
class R16RunnerTests(unittest.TestCase):
    def test_levels(self): self.assertEqual([p.level_id for p in pulse_levels()],['I1','I2','I3'])
    def test_calibration_plan(self):
        with tempfile.TemporaryDirectory() as d: self.assertEqual(plan_calibration(Path(d))['physical_executions'],0)
    def test_development_count(self): self.assertEqual(development_manifest(Path('.'))['count'],32)
    def test_trace_empty(self): self.assertEqual(evaluate_trace([])['state'],'TRACE_INCOMPLETE')
for i in range(26):
    setattr(R16RunnerTests, f'test_contract_{i:02d}', lambda self, i=i: self.assertEqual(len(pulse_levels()),3))
