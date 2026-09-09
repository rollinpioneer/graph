import unittest

from upgrade_v2.l2r_hold_evidence.attempt_lifecycle import AttemptLifecycle


class AttemptLifecycleTests(unittest.TestCase):
    def test_attempt_id_is_monotonic_and_end_is_one_shot(self):
        lifecycle = AttemptLifecycle()
        self.assertEqual(lifecycle.snapshot()["attempt_id"], 0)
        self.assertEqual(lifecycle.begin("acquiring"), 1)
        lifecycle.set_phase("settling")
        lifecycle.end("segment_complete")
        ended = lifecycle.snapshot()
        self.assertEqual(ended["attempt_id"], 1)
        self.assertEqual(ended["attempt_phase"], "ended")
        self.assertTrue(ended["attempt_end"])
        self.assertFalse(ended["attempt_active"])
        lifecycle.begin_next_cycle()
        cleared = lifecycle.snapshot()
        self.assertFalse(cleared["attempt_end"])
        self.assertIsNone(cleared["attempt_end_reason"])
        self.assertEqual(cleared["attempt_phase"], "inactive")
        self.assertEqual(lifecycle.begin("acquiring"), 2)

    def test_end_reason_cannot_encode_physical_result(self):
        lifecycle = AttemptLifecycle()
        lifecycle.begin()
        with self.assertRaises(ValueError):
            lifecycle.end("missed_grasp")
        with self.assertRaises(ValueError):
            lifecycle.end("object_attached")

    def test_end_cannot_be_emitted_twice_in_one_cycle(self):
        lifecycle = AttemptLifecycle()
        lifecycle.begin()
        lifecycle.end("release")
        with self.assertRaises(RuntimeError):
            lifecycle.end("release")

    def test_release_can_close_an_already_inactive_attempt(self):
        lifecycle = AttemptLifecycle()
        lifecycle.begin()
        lifecycle.end("segment_complete")
        lifecycle.begin_next_cycle()
        lifecycle.end("release")
        state = lifecycle.snapshot()
        self.assertTrue(state["attempt_end"])
        self.assertEqual(state["attempt_end_reason"], "release")


if __name__ == "__main__":
    unittest.main()
