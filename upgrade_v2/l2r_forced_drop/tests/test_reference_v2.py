import unittest
from pathlib import Path
from upgrade_v2.l2r_forced_drop.physical_reference import evaluate_loss_trace, evaluate_no_force_control

class ReferenceV2SourceTests(unittest.TestCase):
    def test_no_world_z_placeholder(self):
        for name in ('geometry_reference_v2.py','contact_reference_v2.py','trace_recorder_v2.py','calibration_runner_v2.py'):
            text=Path(__file__).parents[1].joinpath(name).read_text()
            self.assertNotIn('object_world_z < 0.45', text)
    def test_force_uses_model_mass_and_rotation(self):
        text=Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertIn('model.body_mass', text)
        self.assertIn('xmat[body].reshape(3,3)', text)

    def test_loss_uses_absolute_row_time_and_nested_numeric_health(self):
        rows=[]
        for i in range(12):
            rows.append({'time':10+i*.01,'outside_capture':True,'support_force_ratio_mg':0.0,
                         'numeric_health':{'object_speed_mps':1.0,'object_angular_speed_rads':2.0,'object_z_m':.5}})
        result=evaluate_loss_trace(rows,pre_hold_verified=True,force_start_time=9.9)
        self.assertAlmostEqual(result['loss_confirmed_time_abs'],10.09)
        self.assertAlmostEqual(result['loss_confirmed_delay_from_force_s'],.19)

    def test_no_force_control_requires_capture_and_support(self):
        good=[{'time':i*.01,'inside_capture':True,'support_force_ratio_mg':1.0} for i in range(60)]
        self.assertTrue(evaluate_no_force_control(good)['passed'])
        good[-1]['support_force_ratio_mg']=0.0
        self.assertFalse(evaluate_no_force_control(good)['passed'])

    def test_prehold_rows_are_raw_until_summary(self):
        text=Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertIn('pre_hold_verified=False',text)
        self.assertIn('pre_hold_summary.json',text)

    def test_family_and_rollout_seeds_are_separate(self):
        text=Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertIn('family_seed: int, rollout_seed: int',text)
        self.assertIn('family_seed, rollout_seed',text)

    def test_force_rows_are_captured_inside_physics_callback(self):
        text=Path(__file__).parents[1].joinpath('calibration_runner_v2.py').read_text()
        self.assertIn('event == "physics_step"',text)
        self.assertNotIn('for _ in pulse_rows',text)
