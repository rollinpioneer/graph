from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path

from upgrade_v2.l2r_reproducible_baseline.hash_domains import (
    compare_callback_return_pair,
    pair_callback_return_states,
    physical_state_sha256,
    record_sha256,
    semantic_state_sha256,
)
from upgrade_v2.l2r_reproducible_baseline.protocol import load_protocol, make_protocol


class HashDomainTests(unittest.TestCase):
    def row(self, **changes):
        value = {
            "sequence": 1,
            "sampling_point": "action_end_callback",
            "action": "observe_scene",
            "action_index": 1,
            "control_index": 2,
            "time_hex": float(0.1).hex(),
            "arrays": {"qpos": {"shape": [1], "dtype": "<f8", "byte_length": 8, "sha256": "q"}, "qacc_warmstart": {"shape": [1], "dtype": "<f8", "byte_length": 8, "sha256": "w"}},
            "object_xyz": [1.0, 2.0, 3.0], "gripper_xyz": [0.0, 0.0, 0.0], "object_qpos": [1.0], "object_qvel": [0.0],
            "rng_state": {"state": 1}, "events": [{"event": "start"}], "attempt_lifecycle": {"phase": "active"},
            "simulator_flags": {"attached": False},
        }
        value.update(changes)
        return value

    def pair(self, index=1):
        left = self.row(sequence=index * 2 - 1, action_index=index)
        right = copy.deepcopy(left)
        right.update(sequence=index * 2, sampling_point="after_perform_return")
        return left, right

    # Hash-domain contract (1-12)
    def test_record_hash_includes_sequence(self):
        self.assertNotEqual(record_sha256(self.row()), record_sha256(self.row(sequence=2)))

    def test_record_hash_includes_sampling_point(self):
        self.assertNotEqual(record_sha256(self.row()), record_sha256(self.row(sampling_point="after_perform_return")))

    def test_physical_hash_ignores_sequence(self):
        self.assertEqual(physical_state_sha256(self.row()), physical_state_sha256(self.row(sequence=9)))

    def test_physical_hash_ignores_sampling_point(self):
        self.assertEqual(physical_state_sha256(self.row()), physical_state_sha256(self.row(sampling_point="after_perform_return")))

    def test_semantic_hash_ignores_record_identity(self):
        self.assertEqual(semantic_state_sha256(self.row()), semantic_state_sha256(self.row(sequence=9, sampling_point="after_perform_return")))

    def test_physical_hash_changes_on_qpos_bit(self):
        changed = self.row(arrays={**self.row()["arrays"], "qpos": {"shape": [1], "dtype": "<f8", "byte_length": 8, "sha256": "q2"}})
        self.assertNotEqual(physical_state_sha256(self.row()), physical_state_sha256(changed))

    def test_physical_hash_changes_on_warmstart(self):
        changed = self.row(arrays={**self.row()["arrays"], "qacc_warmstart": {"shape": [1], "dtype": "<f8", "byte_length": 8, "sha256": "w2"}})
        self.assertNotEqual(physical_state_sha256(self.row()), physical_state_sha256(changed))

    def test_physical_hash_changes_on_rng(self):
        self.assertNotEqual(physical_state_sha256(self.row()), physical_state_sha256(self.row(rng_state={"state": 2})))

    def test_semantic_hash_changes_on_events(self):
        self.assertNotEqual(semantic_state_sha256(self.row()), semantic_state_sha256(self.row(events=[{"event": "changed"}])))

    def test_semantic_hash_changes_on_lifecycle(self):
        self.assertNotEqual(semantic_state_sha256(self.row()), semantic_state_sha256(self.row(attempt_lifecycle={"phase": "done"})))

    def test_semantic_hash_changes_on_flags(self):
        self.assertNotEqual(semantic_state_sha256(self.row()), semantic_state_sha256(self.row(simulator_flags={"attached": True})))

    def test_legacy_state_hash_is_record_alias(self):
        left, right = self.pair()
        left["state_sha256"] = record_sha256(left)
        self.assertEqual(left["state_sha256"], record_sha256(left))
        self.assertNotEqual(record_sha256(left), record_sha256(right))

    # Callback-return gate (13-24)
    def test_pair_eight_action_end_return_rows(self):
        rows = [item for i in range(1, 9) for item in self.pair(i)]
        result = pair_callback_return_states(rows)
        self.assertTrue(result["pairing_complete"])
        self.assertEqual(len(result["pairs"]), 8)

    def test_pair_requires_same_action_index(self):
        left, right = self.pair(); right["action_index"] = 2
        self.assertFalse(pair_callback_return_states([left, right])["pairing_complete"])

    def test_pair_requires_same_action(self):
        left, right = self.pair(); right["action"] = "lift"
        self.assertFalse(pair_callback_return_states([left, right])["pairing_complete"])

    def test_pair_requires_adjacent_sequence(self):
        left, right = self.pair(); right["sequence"] = 4
        self.assertFalse(pair_callback_return_states([left, right])["pairing_complete"])

    def test_identical_state_different_sampling_point_passes(self):
        left, right = self.pair()
        result = compare_callback_return_pair(left, right)
        self.assertTrue(result["physical_exact"] and result["semantic_exact"])
        self.assertFalse(result["record_exact"])

    def test_qpos_change_fails_physical_gate(self):
        left, right = self.pair(); right["object_qpos"] = [2.0]
        self.assertFalse(compare_callback_return_pair(left, right)["physical_exact"])

    def test_rng_change_fails_physical_gate(self):
        left, right = self.pair(); right["rng_state"] = {"state": 2}
        self.assertFalse(compare_callback_return_pair(left, right)["physical_exact"])

    def test_event_change_fails_semantic_gate(self):
        left, right = self.pair(); right["events"] = [{"event": "changed"}]
        self.assertFalse(compare_callback_return_pair(left, right)["semantic_exact"])

    def test_lifecycle_change_fails_semantic_gate(self):
        left, right = self.pair(); right["attempt_lifecycle"] = {"phase": "done"}
        self.assertFalse(compare_callback_return_pair(left, right)["semantic_exact"])

    def test_missing_return_row_fails_pairing(self):
        left, right = self.pair()
        self.assertFalse(pair_callback_return_states([left])["pairing_complete"])

    def test_duplicate_return_row_fails_pairing(self):
        left, right = self.pair()
        self.assertFalse(pair_callback_return_states([left, right, copy.deepcopy(right)])["pairing_complete"])

    def test_main_gate_count_is_16(self):
        self.assertEqual(make_protocol()["expected_main_gate_count"], 16)

    # Cross-run domain reporting (25-31)
    def test_cross_run_identity_difference_fails(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        self.assertEqual(_state_mismatch(self.row(), self.row(sequence=2), 0)["domain"], "IDENTITY")

    def test_cross_run_physical_difference_fails(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        self.assertEqual(_state_mismatch(self.row(), self.row(object_qpos=[2.0]), 0)["domain"], "PHYSICAL")

    def test_cross_run_semantic_difference_fails(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        self.assertEqual(_state_mismatch(self.row(), self.row(events=[{"event": "changed"}]), 0)["domain"], "SEMANTIC")

    def test_cross_run_record_difference_fails(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        left, right = self.row(), self.row()
        left["record_sha256"], right["record_sha256"] = "left", "right"
        self.assertEqual(_state_mismatch(left, right, 0)["domain"], "RECORD")

    def test_first_mismatch_domain_physical(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        self.assertEqual(_state_mismatch(self.row(), self.row(rng_state={"state": 2}), 0)["domain"], "PHYSICAL")

    def test_first_mismatch_domain_semantic(self):
        from upgrade_v2.l2r_reproducible_baseline.comparison import _state_mismatch
        self.assertEqual(_state_mismatch(self.row(), self.row(simulator_flags={"attached": True}), 0)["domain"], "SEMANTIC")

    def test_exact_tolerance_remains_zero(self):
        self.assertEqual(make_protocol()["comparison"]["numeric_atol"], 0.0)
        self.assertEqual(make_protocol()["comparison"]["numeric_rtol"], 0.0)

    # Version and authorization isolation (32-38)
    def test_v1_authorization_rejected_by_v2(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            value = make_protocol(); value["schema"] = "l2rar2_r14b_protocol_lock_v2"
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(path)

    def test_v1_nonce_journal_not_reused(self):
        self.assertEqual(make_protocol()["physical_authorization_total"], 0)

    def test_v2_A_authorization_zero_in_template(self):
        self.assertEqual(make_protocol()["stages"]["A"]["authorized_instances"], 0)

    def test_v2_B_C_R16_zero(self):
        protocol = make_protocol()
        self.assertEqual(protocol["stages"]["B"]["authorized_instances"], 0)
        self.assertEqual(protocol["stages"]["C"]["authorized_instances"], 0)
        self.assertFalse(protocol["confirmation_run"])

    def test_old_A1_accounting_preserved(self):
        self.assertEqual(make_protocol()["scientific_status"], "L2RAR2_PARTIAL_KEEP_G1")

    def test_A2_requires_new_runner_commit(self):
        protocol = make_protocol()
        self.assertEqual(protocol["base_commit"], "45a3544c68c666ac09796407ae4aee31717bd453")

    def test_A2_requires_new_protocol_hash(self):
        self.assertNotIn("protocol_sha256", make_protocol())


if __name__ == "__main__":
    unittest.main()
