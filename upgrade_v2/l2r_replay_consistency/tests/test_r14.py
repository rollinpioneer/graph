from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

from upgrade_v2.l2r_replay_consistency.authorization import AuthorizationDenied, consume_authorization, validate_authorization
from upgrade_v2.l2r_replay_consistency.comparison import compare_callback_to_return, state_health
from upgrade_v2.l2r_replay_consistency.protocol import load_protocol


class ProtocolTests(unittest.TestCase):
    def test_other_stages_must_remain_zero(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "protocol.json"
            value = {
                "schema": "l2rar2_r14_ordinary_replay_protocol_v1",
                "stage": "R14_ORDINARY_REPLAY",
                "authorized_instances_per_authorization": 1,
                "case_id": "K3_normal_hold_pause_resume",
                "root_family_id": "L2RAR2_REPAIR_00_840000",
                "rollout_seed": 84100002,
                "automatic_retry": False,
                "ordinary_failure_action": "STOP_AFTER_EXECUTION_1",
                "other_stage_authorized_instances": {
                    "R14_INSTRUMENTED_REPLAY": 1,
                    "R16_CALIBRATION": 0,
                    "R16_DEVELOPMENT": 0,
                },
                "tolerances": {"rtol": 0, "time_atol": 1e-12, "geometry_atol": 1e-12, "mocap_atol": 2e-9},
            }
            path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(ValueError):
                load_protocol(path)


class AuthorizationTests(unittest.TestCase):
    def _authorization(self, output_root: Path) -> dict[str, object]:
        return {
            "schema": "l2rar2_r14_execution_authorization_v1",
            "status": "AUTHORIZED",
            "stage": "R14_ORDINARY_REPLAY",
            "authorized_instances": 1,
            "case": "K3_normal_hold_pause_resume",
            "root_family_id": "L2RAR2_REPAIR_00_840000",
            "rollout_seed": 84100002,
            "automatic_retry": False,
            "on_any_main_gate_mismatch": "STOP_AFTER_EXECUTION_1",
            "instrumented_replay_authorized_instances": 0,
            "r16_calibration_authorized_instances": 0,
            "r16_development_authorized_instances": 0,
            "agent_self_authorization_prohibited": True,
            "runner_commit": "abc",
            "runner_file_hashes": {"runner": "hash"},
            "protocol_sha256": "protocol",
            "output_root": str(output_root.resolve()),
            "single_use_nonce": "n" * 32,
            "expires_at_utc": (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat(),
            "authorization_id": "auth-1",
            "approved_at_utc": datetime.now(timezone.utc).isoformat(),
            "reviewer_id": "human-user",
        }

    @mock.patch("upgrade_v2.l2r_replay_consistency.authorization.runner_file_hashes", return_value={"runner": "hash"})
    @mock.patch("upgrade_v2.l2r_replay_consistency.authorization.current_commit", return_value="abc")
    def test_exact_single_instance_authorization(self, unused_commit: mock.Mock, unused_hashes: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            auth_path = root / "auth.json"
            auth_path.write_text(json.dumps(self._authorization(output)), encoding="utf-8")
            auth = validate_authorization(auth_path, repo=root, protocol_sha256="protocol", requested_output_root=output)
            consume_authorization(output, auth, auth_path)
            self.assertTrue((output / "authorization_consumption.json").is_file())
            with self.assertRaises(AuthorizationDenied):
                consume_authorization(output, auth, auth_path)

    @mock.patch("upgrade_v2.l2r_replay_consistency.authorization.runner_file_hashes", return_value={"runner": "hash"})
    @mock.patch("upgrade_v2.l2r_replay_consistency.authorization.current_commit", return_value="abc")
    def test_instrumented_permission_is_rejected(self, unused_commit: mock.Mock, unused_hashes: mock.Mock) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "output"
            value = self._authorization(output)
            value["instrumented_replay_authorized_instances"] = 1
            auth_path = root / "auth.json"
            auth_path.write_text(json.dumps(value), encoding="utf-8")
            with self.assertRaises(AuthorizationDenied):
                validate_authorization(auth_path, repo=root, protocol_sha256="protocol", requested_output_root=output)


class StateTests(unittest.TestCase):
    def _state(self) -> dict[str, object]:
        return {
            "time": 0.0,
            "qpos": [0.0, 1.0],
            "qvel": [0.0],
            "qacc_warmstart": [0.0],
            "mocap_pos": [[0.0, 0.0, 0.0]],
            "mocap_quat": [[1.0, 0.0, 0.0, 0.0]],
            "eq_active": [0],
            "model_eq_data": [[0.0, 0.0]],
            "rng_state_sha256": "same",
            "object_xyz": [0.0, 0.0, 0.0],
            "gripper_xyz": [0.0, 0.0, 0.0],
            "contact_pair_summary": [],
        }

    def test_healthy_state_and_callback_return_identity(self) -> None:
        state = self._state()
        self.assertTrue(state_health([state])["passed"])
        self.assertTrue(compare_callback_to_return([state], [dict(state)])["passed"])

    def test_non_finite_state_fails(self) -> None:
        state = self._state()
        state["qvel"] = [float("nan")]
        self.assertFalse(state_health([state])["passed"])


if __name__ == "__main__":
    unittest.main()
