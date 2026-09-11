from __future__ import annotations

import csv
import json
import unittest
from pathlib import Path

from upgrade_v2.l2r_geometry_events import o_interface
from upgrade_v2.l2r_geometry_events.adapters import assert_no_forbidden_inputs
from upgrade_v2.l2r_geometry_events.state_machine import (
    METHOD_G_H,
    METHOD_G_R,
    METHOD_G_HR,
    GeometryEventStateMachine,
)
from upgrade_v2.l2r_task_context import online_interface_repair

from .helpers import PROTOCOL, sample

DATA_ROOT = Path(
    "/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/"
    "upgrade_v2/task_context_l2rar2_v1/data_attach_relpose_repair_v1"
)
ROUND10 = Path(
    "/home/__compress_data/xushijie/graph_l2ra_r2_worktree/artifacts/pathgraph_sarm/"
    "upgrade_v2/task_context_l2rar2_v1/rounds/l2rar2_10_online_interface_repair"
)


def run(method: str, samples):
    machine = GeometryEventStateMachine(method, PROTOCOL, "rollout", "root")
    return machine.run(samples), machine


class OFrozenInterfaceTests(unittest.TestCase):
    def test_o_b2_uses_repaired_interface(self):
        self.assertIs(o_interface.run_repaired_interface, online_interface_repair.run_repaired_interface)
        self.assertEqual(o_interface.O_METHOD_TO_CANDIDATE["O_B2"], "B_count2")

    def test_o_c3_uses_repaired_interface(self):
        self.assertIs(o_interface.run_repaired_interface, online_interface_repair.run_repaired_interface)
        self.assertEqual(o_interface.O_METHOD_TO_CANDIDATE["O_C3"], "C3_vector_rho035")

    @unittest.skipUnless(DATA_ROOT.is_dir(), "frozen cache not available")
    def test_touch_request_does_not_retry(self):
        meta = json.loads(
            (DATA_ROOT / "rollouts/L2RAR2_REPAIR_00_840000/K2_touch_request_completes_without_hold/metadata.json").read_text()
        )
        result = o_interface.run_o_method(meta, "O_B2")
        self.assertEqual(result["decision"]["selected_action"], "none")

    def test_round10_row_key_mapping(self):
        frozen = [
            {
                "candidate_id": "B_count2",
                "rollout_id": "r1",
                "selected_action": "none",
                "selected_time": "",
                "selected_reason": "no_emergency_condition",
                "correct": "True",
            }
        ]
        decisions = [
            {
                "method_id": "O_B2",
                "candidate_id": "B_count2",
                "rollout_id": "r1",
                "selected_action": "none",
                "selected_time": None,
                "selected_reason": "no_emergency_condition",
                "correct": True,
            }
        ]
        rows = o_interface.parity_rows(frozen, decisions)
        self.assertTrue(rows[0]["matched"])
        self.assertEqual(o_interface.parity_metrics(rows)["matched"], 1)

    @unittest.skipUnless(ROUND10.is_dir(), "round-10 artefacts not available")
    def test_round10_metrics_match(self):
        with (ROUND10 / "interface_repair_comparison.csv").open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        with (ROUND10 / "interface_repair_metrics.csv").open(newline="", encoding="utf-8") as handle:
            frozen = {row["candidate_id"]: row for row in csv.DictReader(handle)}
        self.assertEqual(len(rows), 64)
        for candidate_id, frozen_row in frozen.items():
            subset = [row for row in rows if row["candidate_id"] == candidate_id]
            k1 = [row for row in subset if row["case_id"].startswith("K1")]
            self.assertAlmostEqual(
                sum(row["correct"] == "True" for row in k1) / len(k1),
                float(frozen_row["K1_recall"]),
                places=9,
            )
            accuracy = sum(row["correct"] == "True" for row in subset) / len(subset)
            self.assertAlmostEqual(accuracy, float(frozen_row["labeled_event_accuracy"]), places=9)


class AblationDefinitionTests(unittest.TestCase):
    def test_g_h_does_not_use_relative_loss(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
            sample(0.05, 1, [0.0, 0.0, 0.53], [0.0, 0.0, 0.6], True, "closed"),
            sample(0.10, 2, [0.0, 0.0, 0.53], [0.0, 0.0, 0.6], True, "closed"),
        ]
        for index in range(3, 8):
            samples.append(
                sample(
                    index * 0.05,
                    index,
                    [0.0, 0.0, 0.53 + (index - 2) * 0.005],
                    [0.0, 0.0, 0.6],
                    True,
                    "closed",
                )
            )
        records, machine = run(METHOD_G_H, samples)
        self.assertTrue(machine.ever_held)
        self.assertTrue(all(row["loss_source"] is None for row in records))
        self.assertTrue(all(row["selected_action"] != "recover_object" for row in records))

    def test_g_r_does_not_use_height_entry(self):
        samples = [
            sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open"),
        ]
        for index in range(1, 8):
            samples.append(
                sample(index * 0.05, index, [0.0, 0.0, 0.5 + index * 0.01], [0.0, 0.0, 0.6], True, "closed")
            )
        _, machine = run(METHOD_G_R, samples)
        self.assertFalse(machine.ever_held)

    def test_g_hr_has_two_entry_routes(self):
        low = [sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open")]
        for index in range(1, 4):
            low.append(
                sample(
                    index * 0.05,
                    index,
                    [0.0, 0.0, 0.5 + index * 0.005],
                    [0.0, 0.0, 0.6 + index * 0.005],
                    True,
                    "closed",
                )
            )
        _, low_machine = run(METHOD_G_HR, low)
        self.assertEqual(low_machine.hold_entry_route, "LOW_HEIGHT_CO_MOTION")

        # Entry A exists for slow cumulative lifts where each step stays below
        # the per-interval motion gate.
        lift = [sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open")]
        for index in range(1, 9):
            lift.append(
                sample(
                    index * 0.05,
                    index,
                    [0.0, 0.0, 0.5 + index * 0.003],
                    [0.0, 0.0, 0.6 + index * 0.003],
                    True,
                    "closed",
                )
            )
        _, lift_machine = run(METHOD_G_HR, lift)
        self.assertEqual(lift_machine.hold_entry_route, "LIFT_COUPLED")

    def test_hold_route_count_uses_transitions(self):
        samples = [sample(0.0, 0, [0.0, 0.0, 0.5], [0.0, 0.0, 0.6], False, "open")]
        for index in range(1, 10):
            samples.append(
                sample(
                    index * 0.05,
                    index,
                    [0.0, 0.0, 0.5 + index * 0.006],
                    [0.0, 0.0, 0.6 + index * 0.006],
                    True,
                    "closed",
                )
            )
        records, _ = run(METHOD_G_R, samples)
        self.assertEqual(sum(1 for row in records if row["hold_transition_event"]), 1)

    def test_forbidden_candidate_fields_rejected(self):
        for key in ("weld_state", "case_id", "expected_action", "events"):
            with self.assertRaises(AssertionError):
                assert_no_forbidden_inputs({key: 1})


if __name__ == "__main__":
    unittest.main()
