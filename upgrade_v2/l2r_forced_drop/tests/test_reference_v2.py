import ast
import unittest
from pathlib import Path

class ReferenceV2StaticTests(unittest.TestCase):
    def test_prehold_anchor_is_before_lift(self):
        text = Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertEqual(text.count('z0=float(sim.object_xyz[2])'), 1)
        self.assertLess(text.index('z0=float(sim.object_xyz[2])'), text.index("sim.perform('lift')"))

    def test_reference_modules_parse(self):
        for name in ('geometry_reference_v2.py','contact_reference_v2.py','trace_recorder_v2.py','calibration_runner_v2.py'):
            ast.parse(Path(__file__).parents[1].joinpath(name).read_text())
