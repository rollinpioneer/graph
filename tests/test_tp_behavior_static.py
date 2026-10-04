import subprocess
import sys

from cp_disr.analysis import tp_behavior_static as m

TOK = {"ontop", "inside", "open", "covered", "saturated", "cooked", "toggled_on", "filled", "contains"}
ok = lambda t: t in TOK  # noqa: E731

SYM = """
class SymbolicSemanticActionPrimitiveSet(IntEnum):
    _init_ = "value __doc__"
    OPEN = auto(), "Open an object"
    PLACE_ON_TOP = auto(), "Place on top"
    WIPE = auto(), "Wipe"
    GHOST = auto(), "never mapped"

class SymbolicSemanticActionPrimitives(Base):
    def __init__(self, env, robot):
        self.controller_functions = {
            SymbolicSemanticActionPrimitiveSet.OPEN: self._open,
            SymbolicSemanticActionPrimitiveSet.PLACE_ON_TOP: self._place_on_top,
            SymbolicSemanticActionPrimitiveSet.WIPE: self._wipe,
        }
    def _open(self, obj):
        yield from self._open_or_close(obj, True)
    def _open_or_close(self, obj, should_open):
        if object_states.Open not in obj.states:
            raise ActionPrimitiveError(ActionPrimitiveError.Reason.PRE_CONDITION_ERROR, "no")
        obj.states[object_states.Open].set_value(should_open)
    def _place_on_top(self, obj):
        yield from self._place_with_predicate(obj, object_states.OnTop)
    def _place_with_predicate(self, obj, predicate):
        pass
    def _wipe(self, obj):
        obj_in_hand = self.robot
        obj.states[object_states.Covered].set_value(system, False)
"""


def test_flatten_goal_tracks_polarity_ops_and_unknown_heads():
    goal = ["and", ["ontop", "?a", "?b"], ["not", ["covered", "?a", "?d"]], ["forall", ["?x", "-", "t"], ["inside", "?x", "?b"]], ["mystery", "?q"]]
    acc, ops, unknown = m.flatten_goal(goal, True, token_ok=ok)
    assert ("ontop", True) in acc and ("covered", False) in acc and ("inside", True) in acc
    assert set(ops) == {"and", "not", "forall"} and unknown == ["mystery"]


def test_primitive_extraction_follows_helpers_and_records_diff():
    inv = m.extract_primitives(SYM, "def _grasp(self):\n    pass\n")
    p = inv["primitives"]
    assert set(p) == {"OPEN", "PLACE_ON_TOP", "WIPE", "GHOST"} and p["GHOST"]["facts"] is None
    assert {"state": "Open", "value": "param"} in p["OPEN"]["facts"]["set_value_writes"] and "_open_or_close" in p["OPEN"]["facts"]["helpers_followed"]
    assert "PRE_CONDITION_ERROR" in p["OPEN"]["facts"]["error_reasons"]
    assert p["PLACE_ON_TOP"]["facts"]["placement_predicates"] == ["OnTop"]
    assert {"state": "Covered", "value": False} in p["WIPE"]["facts"]["set_value_writes"]
    d = inv["diff_against_card_expectation"]
    assert "GRASP" in d["expected_not_in_source"] and "GHOST" in d["in_source_not_expected"] and d["identical"] is False
    w = m.primitive_writes(inv)
    assert ("Open", "param") in w and ("OnTop", True) in w and ("Covered", False) in w


def test_static_coverage_classes():
    pred2state = {"OnTop": "OnTop", "Open": "Open", "Covered": "Covered", "Cooked": "Cooked", "Filled": "Filled"}
    writes = {("Open", "param"), ("OnTop", True), ("Covered", False)}
    rule_states = {"Cooked": [True]}
    base = {"parse_ok": True, "goal_unknown_heads": []}
    f = lambda leaves: m.coverage_for(dict(base, goal_leaves=leaves), pred2state, writes, rule_states)[0]  # noqa: E731
    assert f(["ontop", "open", "not covered"]) == "FULL"
    assert f(["ontop", "covered"]) == "PARTIAL"
    assert f(["covered", "filled"]) == "UNSUPPORTED"
    assert f(["cooked"]) == "FULL"
    assert f(["grasped"]) == "UNKNOWN"
    assert m.coverage_for({"parse_ok": False}, pred2state, writes, rule_states)[0] == "UNKNOWN"
    assert f(["not open"]) == "FULL"          # a 'param' write covers both polarities


def test_canonical_state_is_order_invariant_and_sensitive_to_content():
    objs = {"a.n.01": ["a.n.01_1", "a.n.01_2"], "b.n.01": ["b.n.01_1"]}
    init = [["ontop", "a.n.01_1", "b.n.01_1"], ["open", "b.n.01_1"]]
    goal = ["and", ["inside", "?a", "?b"], ["open", "?b"]]
    _, h1, g1 = m.canonical_state(objs, init, goal)
    _, h2, g2 = m.canonical_state({"b.n.01": ["b.n.01_1"], "a.n.01": ["a.n.01_2", "a.n.01_1"]}, list(reversed(init)), ["and", ["open", "?b"], ["inside", "?a", "?b"]])
    assert (h1, g1) == (h2, g2)
    _, h3, _ = m.canonical_state(objs, init[:1], goal)
    assert h3 != h1
    rows = [{"_parsed": (objs, init, goal)}]
    assert m.permutation_invariance(rows)["mismatches"] == 0


def test_token_class_names_match_the_official_predicate_classes():
    assert [m.token_class_name(t) for t in ("ontop", "toggled_on", "on_fire", "nextto", "covered")] == ["OnTop", "ToggledOn", "OnFire", "NextTo", "Covered"]


def test_extract_rule_classes_flags_engine_imports():
    r = m.extract_rule_classes("import torch as th\nimport omnigibson as og\nclass BaseTransitionRule: pass\nclass SlicingRule(BaseTransitionRule):\n    '''slice things'''\n")
    assert r["imports_omnigibson"] and r["imports_torch"] and [c["name"] for c in r["rule_classes"]] == ["BaseTransitionRule", "SlicingRule"]


def test_module_import_pulls_no_simulator_stack():
    code = "import sys; import cp_disr.analysis.tp_behavior_static as x; print([k for k in ('omnigibson','isaacsim','torch','robosuite','mujoco') if k in sys.modules])"
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.stdout.strip() == "[]", out.stdout + out.stderr
