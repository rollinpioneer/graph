import unittest
from pathlib import Path

class ReferenceV2SourceTests(unittest.TestCase):
    def test_no_world_z_placeholder(self):
        for name in ('geometry_reference_v2.py','contact_reference_v2.py','trace_recorder_v2.py','calibration_runner_v2.py'):
            text=Path(__file__).parents[1].joinpath(name).read_text()
            self.assertNotIn('object_world_z < 0.45', text)
    def test_force_uses_model_mass_and_rotation(self):
        text=Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertIn('model.body_mass', text)
        self.assertIn('xmat[body].reshape(3,3)', text)
