"""Focused regression tests for the U2 audit-correction handoff."""

from __future__ import annotations

import unittest

from upgrade_v2.u2_handoff_patch.cli import build_parser
from upgrade_v2.u2_handoff_patch.primitives import incoming_segment_return, match_events


class BoundaryMatchingTests(unittest.TestCase):
    def test_exact_episode_local_match(self) -> None:
        result = match_events([2, 9], [2, 9], tolerance=1)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (2, 0, 0))
        self.assertEqual(result["pairs"], [(2, 2), (9, 9)])

    def test_cross_episode_events_do_not_share_a_matching_pool(self) -> None:
        # The old concatenated evaluator could match A:t=3 to B:t=0.  Calling
        # the matcher per episode leaves both events unmatched, as required.
        episode_a = match_events([3], [], tolerance=1)
        episode_b = match_events([], [0], tolerance=1)
        self.assertEqual(episode_a["tp"] + episode_b["tp"], 0)
        self.assertEqual(episode_a["fp"] + episode_b["fp"], 1)
        self.assertEqual(episode_a["fn"] + episode_b["fn"], 1)

    def test_input_order_does_not_change_match_score(self) -> None:
        ordered = match_events([1, 4, 7], [2, 5, 8], tolerance=1)
        shuffled = match_events([7, 1, 4], [8, 2, 5], tolerance=1)
        self.assertEqual(ordered, shuffled)


class RewardAttributionTests(unittest.TestCase):
    def test_incoming_segments_conserve_stored_potential_difference(self) -> None:
        phi = [0.0, 0.2, 0.5, 0.8]
        returns = [
            incoming_segment_return(phi, 0, 1),
            incoming_segment_return(phi, 2, 2),
            incoming_segment_return(phi, 3, 3),
        ]
        self.assertAlmostEqual(sum(returns), phi[-1] - phi[0])
        self.assertAlmostEqual(returns[1], 0.3)
        self.assertAlmostEqual(returns[2], 0.3)


class CliShapeTests(unittest.TestCase):
    def test_required_commands_are_exposed(self) -> None:
        parser = build_parser()
        commands = {action.dest for action in parser._actions}
        self.assertIn("command", commands)
        for name in ("recompute-boundaries", "recompute-reward", "export-u3-train", "freeze-handoff"):
            with self.subTest(name=name):
                self.assertIn(name, parser._subparsers._group_actions[0].choices)


if __name__ == "__main__":
    unittest.main()
