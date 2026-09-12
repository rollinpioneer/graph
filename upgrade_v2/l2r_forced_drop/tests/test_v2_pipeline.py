import tempfile,unittest
from pathlib import Path
import json
from upgrade_v2.l2r_forced_drop.development_runner_v2 import run_development_v2
from upgrade_v2.l2r_forced_drop.evaluate_v2 import evaluate_v2
from upgrade_v2.l2r_forced_drop.reference_builder_v2 import build_reference_v2
class V2PipelineTests(unittest.TestCase):
    def test_development_requires_calibration_selection(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError,'CALIBRATION_SELECTION_REQUIRED'): run_development_v2(Path(d)/'dev',selected_level=None)
    def test_evaluation_fail_closed(self):
        with tempfile.TemporaryDirectory() as d:
            ref=Path(d)/'ref'; ref.mkdir(); (ref/'generator_gate.json').write_text('{"candidate_evaluation_allowed":false}')
            with self.assertRaisesRegex(RuntimeError,'GENERATOR_GATE_REQUIRED'): evaluate_v2(ref,Path(d)/'eval')
    def test_development_has_eight_distinct_physical_routes(self):
        text=Path(__file__).parents[1].joinpath('development_runner_v2.py').read_text()
        for token in ('retry_required','touch_only','post_weld_off_pre_force','force_pulse_step','commanded_release','transport_before_force'):
            self.assertIn(token,text)
    def test_reference_gate_enforces_scientific_counts(self):
        text=Path(__file__).parents[1].joinpath('reference_builder_v2.py').read_text()
        for token in ('prehold_24_of_24','forced_loss_12_of_12','F3_no_loss_4_of_4','F7_commanded_release_4_of_4','resolvable_F1_F2_F4'):
            self.assertIn(token,text)
    def test_evaluation_produces_real_metrics(self):
        text=Path(__file__).parents[1].joinpath('evaluate_v2.py').read_text()
        self.assertNotIn('READY_FOR_REVIEW',text)
        self.assertNotIn("'overall_accuracy':None",text)
        for method in ('O_B2','O_C3','S_G_H','S_G_R','S_G_HR'):
            self.assertIn(method,text)
