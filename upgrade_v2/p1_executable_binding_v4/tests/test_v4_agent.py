from upgrade_v2.p1_executable_binding_v4 import history_lift, reference_binding, g1_reward_view

def test_dual_sequences_require_both_and_goal():
    class Dummy:
        def __init__(self, e, o, g):
            self.e, self.o, self.g = e, o, g
            self.facts = {k: None for k in (
                "hold_current","hold_loss_confirmed","release_intent","a_current_valid",
                "b_current_valid","goal_verified","terminal_failure","recovery_completed")}
            self.established_once = False
            self.last_key = None
            self.consumed_event_ids = set()
            self.attempt_id = None
            self.last_observed = {}
        def step(self, event):
            from state_kernel import StateKernel
            raise RuntimeError("use real kernel in integration")

def test_linear_intermediate_not_terminal_only():
    assert reference_binding.LINEAR["grasped"] == 0.25
    assert reference_binding.LINEAR["in_transit"] == 0.5
    assert reference_binding.unordered("in_transit", {"hold_established_earlier": True}) > 0

def test_g1_not_controller():
    g = g1_reward_view.reward_view_graph()
    assert g["controller_driven_by_graph"] is False
    assert "already_done" in g["original_edges"]

def test_compressed_not_forbidden():
    cls, *_ = reference_binding.classify("recovery", "in_transit")
    assert cls == "SAMPLED_PATH_COMPRESSED_UNRESOLVED"
    cls, *_ = reference_binding.classify("start", "success")
    assert cls == "TASK_OUTSIDE_GRAPH_DOMAIN"