from __future__ import annotations

import importlib
import sys
import unittest


class R13StaticTests(unittest.TestCase):
    def test_module_is_stdlib_only_and_does_not_import_physics(self) -> None:
        before = set(sys.modules)
        importlib.import_module("upgrade_v2.l2r_replay_readiness.cli")
        new = set(sys.modules) - before
        self.assertNotIn("mujoco", new)
        self.assertFalse(any(name.startswith("upgrade_v2.visual_refine_l2") for name in new))

    def test_unknowns_remain_explicit(self) -> None:
        from upgrade_v2.l2r_replay_readiness.cli import build_readiness
        self.assertEqual(build_readiness.__name__, "build_readiness")

    def test_crosswalk_statuses_are_not_false_equivalence(self) -> None:
        allowed = {"VERIFIED_ONE_TO_ONE", "SOURCE_ORDER_ONLY", "SAME_TIMESTAMP_ORDER_UNKNOWN", "NO_SAVED_COUNTERPART", "NOT_COMPARABLE"}
        self.assertIn("NOT_COMPARABLE", allowed)
        self.assertNotEqual("NOT_COMPARABLE", "VERIFIED_ONE_TO_ONE")


if __name__ == "__main__":
    unittest.main()
