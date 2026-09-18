from __future__ import annotations

import ast
import unittest
from unittest import mock

import numpy as np

from p2crl import evaluation_repair as repair


class StrictInputTests(unittest.TestCase):
    def _fixture(self):
        checkpoints = [
            {
                "job_id": f"A_s317_{index}",
                "method": method,
                "draw": "A",
                "policy_seed": "317",
            }
            for index, method in enumerate(repair.METHODS)
        ]
        cases = [
            {"family_id": 1, "side": side}
            for side in ("left", "right")
        ]
        records = []
        for checkpoint in checkpoints:
            for case in cases:
                records.append({
                    "job_id": checkpoint["job_id"],
                    "method": checkpoint["method"],
                    "draw": "A",
                    "policy_seed": "317",
                    "family_id": "1",
                    "side": case["side"],
                    "invalid_actions": "0",
                    "nonfinite": "0",
                    "terminated": "1",
                    "truncated": "0",
                })
        return checkpoints, cases, records

    def test_empty_records_hard_fail(self):
        with self.assertRaisesRegex(repair.EvaluationRepairError, "EMPTY_RECORDS"):
            repair._require_nonempty([])

    def test_duplicate_episode_key_hard_fail(self):
        row = {"job_id": "j", "family_id": "1", "side": "left"}
        with self.assertRaisesRegex(
            repair.EvaluationRepairError, "DUPLICATE_EPISODE_KEY"
        ):
            repair._require_unique([row, dict(row)], ("job_id", "family_id", "side"))

    def test_missing_row_hard_fail(self):
        checkpoints, cases, records = self._fixture()
        with mock.patch.object(repair, "MAIN_EXPECTED", 10):
            with self.assertRaisesRegex(repair.EvaluationRepairError, "MISSING_ROWS"):
                repair.validate_main_records(records[:-1], checkpoints, cases)

    def test_incomplete_method_panel_hard_fail(self):
        checkpoints, cases, records = self._fixture()
        records[0]["method"] = repair.GEOM
        with mock.patch.object(repair, "MAIN_EXPECTED", 10):
            with self.assertRaisesRegex(
                repair.EvaluationRepairError, "INCOMPLETE_METHOD_PANEL"
            ):
                repair.validate_main_records(records, checkpoints, cases)

    def test_complete_small_panel_passes(self):
        checkpoints, cases, records = self._fixture()
        with mock.patch.object(repair, "MAIN_EXPECTED", 10):
            repair.validate_main_records(records, checkpoints, cases)
    def test_critical_wrong_key_cannot_replace_missing_row(self):
        checkpoints = [
            {
                "job_id": f"job_{index}",
                "method": method,
                "draw": "A",
                "policy_seed": "317",
            }
            for index, method in enumerate(repair.METHODS)
        ]
        cases = [{"family_id": 1, "side": "left"}]
        records = [
            {
                "job_id": checkpoint["job_id"],
                "method": checkpoint["method"],
                "draw": "A",
                "policy_seed": "317",
                "family_id": "1",
                "side": "left",
                "prefix_index": str(prefix_index),
            }
            for checkpoint in checkpoints
            for prefix_index in range(16)
        ]
        records[0]["family_id"] = "999"
        with mock.patch.object(repair, "CRITICAL_EXPECTED", len(records)):
            with self.assertRaisesRegex(repair.EvaluationRepairError, "MISSING_ROWS"):
                repair.validate_critical_records(records, checkpoints, cases)



class RegisteredPrefixTests(unittest.TestCase):
    def test_registered_action_executes_without_mask_precheck(self):
        env = mock.Mock()
        env.terminated.return_value = False
        with mock.patch.object(repair, "SkillEnv", return_value=env):
            observed = repair._run_registered_prefix({}, [36])

        self.assertIs(observed, env)
        env.reset.assert_called_once_with()
        env.step.assert_called_once_with(36)

    def test_registered_action_range_hard_fail(self):
        env = mock.Mock()
        env.terminated.return_value = False
        with mock.patch.object(repair, "SkillEnv", return_value=env):
            with self.assertRaisesRegex(
                repair.EvaluationRepairError, "CRITICAL_PREFIX_ACTION_RANGE"
            ):
                repair._run_registered_prefix({}, [37])

        env.step.assert_not_called()


class BootstrapTests(unittest.TestCase):
    @staticmethod
    def _reference(delta_geom, delta_flat, n_boot):
        rng = np.random.RandomState(repair.BOOTSTRAP_SEED)
        geom = []
        flat = []
        for _ in range(n_boot):
            draws = [int(value) for value in rng.randint(0, 3, size=3)]
            seeds = {
                draw: [int(value) for value in rng.randint(0, 4, size=4)]
                for draw in range(3)
            }
            families = {
                motif: [int(value) for value in rng.randint(0, 64, size=64)]
                for motif in range(4)
            }
            values_geom = []
            values_flat = []
            for motif in range(4):
                motif_geom = []
                motif_flat = []
                for draw in draws:
                    for seed in seeds[draw]:
                        for family in families[motif]:
                            motif_geom.append(delta_geom[draw, seed, motif, family])
                            motif_flat.append(delta_flat[draw, seed, motif, family])
                values_geom.append(float(np.mean(motif_geom)))
                values_flat.append(float(np.mean(motif_flat)))
            geom.append(float(np.mean(values_geom)))
            flat.append(float(np.mean(values_flat)))
        return np.asarray(geom), np.asarray(flat)

    def test_vectorized_scores_match_frozen_resampling_order(self):
        rng = np.random.RandomState(7)
        delta_geom = rng.randn(3, 4, 4, 64)
        delta_flat = rng.randn(3, 4, 4, 64)
        n_boot = 32
        expected_geom, expected_flat = self._reference(
            delta_geom, delta_flat, n_boot
        )
        with mock.patch.object(repair, "BOOTSTRAP_REPLICATES", n_boot):
            observed_geom, observed_flat = repair._bootstrap_two_deltas(
                delta_geom, delta_flat
            )
        np.testing.assert_allclose(observed_geom, expected_geom, atol=1e-14)
        np.testing.assert_allclose(observed_flat, expected_flat, atol=1e-14)

    def test_ci_interpretation_is_locked(self):
        self.assertEqual(repair.LOWER_QUANTILE, 0.025)
        self.assertEqual(repair.UPPER_QUANTILE, 0.975)
        self.assertEqual(repair.BOOTSTRAP_REPLICATES, 200000)


class TrainingProhibitionTests(unittest.TestCase):
    def test_no_training_calls_in_repair_module(self):
        source_path = repair.Path(repair.__file__)
        tree = ast.parse(source_path.read_text(encoding="utf-8"))
        forbidden = []
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            if isinstance(node.func, ast.Attribute) and node.func.attr == "learn":
                forbidden.append(("learn", node.lineno))
            if (
                isinstance(node.func, ast.Attribute)
                and node.func.attr == "step"
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "optimizer"
            ):
                forbidden.append(("optimizer.step", node.lineno))
        self.assertEqual(forbidden, [])


if __name__ == "__main__":
    unittest.main()
