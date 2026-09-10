from __future__ import annotations

import importlib
import sys
import unittest
from pathlib import Path
from unittest.mock import Mock

from upgrade_v2.l2r_execution_audit.policy import PhysicalExecutionDenied


class RepositoryGuardTests(unittest.TestCase):
    def test_physics_probe_import_is_lazy_and_all_entries_deny_before_factory(self) -> None:
        before = set(sys.modules)
        module = importlib.import_module("upgrade_v2.l2r_loss_observability.physics_probe")
        self.assertNotIn("mujoco", set(sys.modules) - before)
        fake_factory = Mock()
        with self.assertRaises(PhysicalExecutionDenied):
            module.run_physics_probe(Path("missing"), Path("missing"), 8, True, Path("missing"))
        with self.assertRaises(PhysicalExecutionDenied):
            module._ordinary_replay(fake_factory, 0, [], 0)
        with self.assertRaises(PhysicalExecutionDenied):
            module._instrumented_replay(fake_factory, 0, [], 0)
        fake_factory.assert_not_called()

    def test_cli_help_does_not_import_physics_chain(self) -> None:
        before = set(sys.modules)
        cli = importlib.import_module("upgrade_v2.l2r_loss_observability.cli")
        with self.assertRaises(SystemExit) as ctx:
            cli.parser().parse_args(["--help"])
        self.assertEqual(ctx.exception.code, 0)
        self.assertNotIn("mujoco", set(sys.modules) - before)

    def test_pure_helpers_do_not_relabel_loss(self) -> None:
        from upgrade_v2.l2r_execution_audit.event_boundary_audit import first_saved_contact_loss

        rows = [
            {"time": 0.0, "contact_present": True},
            {"time": 0.1, "contact_present": None},
            {"time": 0.2, "contact_present": False},
        ]
        result = first_saved_contact_loss(rows, 0.1, 0.2)
        self.assertEqual(result["status"], "NOT_OBSERVED_IN_AVAILABLE_WINDOW")
        self.assertFalse(result["physical_loss_verified"])


if __name__ == "__main__":
    unittest.main()
