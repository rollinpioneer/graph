import tempfile,unittest
from pathlib import Path
from upgrade_v2.l2r_forced_drop.development_runner_v2 import run_development_v2
from upgrade_v2.l2r_forced_drop.evaluate_v2 import evaluate_v2
class V2PipelineTests(unittest.TestCase):
    def test_development_requires_calibration_selection(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'CALIBRATION_SELECTION_REQUIRED'): run_development_v2(Path(d)/'dev',selected_level=None)
    def test_evaluation_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            ref=Path(d)/'ref'; ref.mkdir(); (ref/'generator_gate.json').write_text('{"candidate_evaluation_allowed":false}')
            with self.assertRaisesRegex(RuntimeError,'GENERATOR_GATE_REQUIRED'): evaluate_v2(ref,Path(d)/'eval')
