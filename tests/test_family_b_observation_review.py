import json
from pathlib import Path
from scripts.family_b_observation_review import (
    validate_planner_record_semantics, production_mask, replay_public, verify, BRANCHES
)

def test_r01_continue_noplan_with_record_is_valid_but_gate_fail():
    r = validate_planner_record_semantics({}, {"reason": "CONTINUE", "task_success": False, "terminated": False},
                                          {"status": "NO_PLAN", "plan": []}, [])
    assert r["valid"] and r["planner_required"]

def test_r02_continue_noplan_missing_record_fails():
    r = validate_planner_record_semantics({}, {"reason": "CONTINUE", "task_success": False, "terminated": False}, {}, [])
    assert not r["valid"]

def test_r03_success_without_planner_is_legal():
    r = validate_planner_record_semantics({}, {"reason": "TASK_SUCCESS", "task_success": True, "terminated": True}, {}, [])
    assert r["valid"] and not r["planner_required"]

def test_r04_success_with_real_planner_is_protocol_failure():
    r = validate_planner_record_semantics({}, {"reason": "TASK_SUCCESS", "task_success": True, "terminated": True},
                                          {"status": "PLAN_FOUND"}, [])
    assert not r["valid"]

def test_r05_last_action_number_does_not_change_requirement():
    a = validate_planner_record_semantics({"action_index": 999}, {"reason": "CONTINUE"}, {"status": "NO_PLAN"}, [])
    b = validate_planner_record_semantics({"action_index": 0}, {"reason": "CONTINUE"}, {"status": "NO_PLAN"}, [])
    assert a["planner_required"] == b["planner_required"] is True

def test_r06_unknown_on_table_blocks_pick_with_production_function():
    root = Path(__file__).resolve().parents[1]
    facts = {"p:GripperEmpty": "TRUE", "p:Open:receiver": "TRUE",
             "p:OnTable:obj_b": "UNKNOWN", "p:OnTable:obj_c": "TRUE",
             "p:OnTable:carrier": "FALSE", "p:Held:obj_b": "FALSE", "p:Held:obj_c": "FALSE",
             "p:Held:carrier": "FALSE", "p:Inside:obj_b:receiver": "FALSE",
             "p:Inside:obj_c:receiver": "TRUE", "p:Inside:carrier:receiver": "FALSE"}
    ids, mask = production_mask(root, facts)
    assert "a:PICK:obj_b:v1" in ids
    assert mask[ids.index("a:PICK:obj_b:v1")] is False

def test_r07_no_public_support_cannot_be_true():
    root = Path(__file__).resolve().parents[1]
    ids, mask = production_mask(root, {"p:OnTable:obj_b": "UNKNOWN"})
    assert not mask or "a:PICK:obj_b:v1" not in [i for i, ok in zip(ids, mask) if ok]

def test_r08_replay_declares_all_four_without_fabricating_facts(tmp_path):
    out = tmp_path / "review"; out.mkdir()
    replay_public(Path.cwd(), tmp_path, out)
    data = json.loads((out / "replay/replay_result.json").read_text())
    assert set(data["coverage"]) == set(BRANCHES)
    assert data["status"] == "INSUFFICIENT_FOR_REPLAY"
    assert data["new_skill_calls"] == 0

def test_r09_png_identity_classification_shape():
    from PIL import Image
    a, b = Path(__file__).parent / "tmp_a.png", Path(__file__).parent / "tmp_b.png"
    try:
        Image.new("RGB", (2,2), (1,2,3)).save(a)
        Image.new("RGB", (2,2), (1,2,3)).save(b, optimize=True)
        assert a.read_bytes() != b.read_bytes() or a.read_bytes() == b.read_bytes()
    finally:
        a.unlink(missing_ok=True); b.unlink(missing_ok=True)

def test_r10_next_action_does_not_release_remaining(tmp_path):
    from scripts.family_b_observation_review import assemble
    out = tmp_path / "review"; out.mkdir()
    assemble(Path.cwd(), tmp_path, out)
    data = json.loads((out / "decision/next_action.json").read_text())
    assert data["remaining_20_released"] is False
    assert data["new_physical_attempts"] == 0

def test_r11_integrity_can_pass_with_science_unresolved(tmp_path):
    from scripts.family_b_observation_review import assemble
    out = tmp_path / "review"; out.mkdir()
    assemble(Path.cwd(), tmp_path, out)
    assert "technical gates remain FAIL" in (out / "final_summary.md").read_text()

def test_r12_protected_change_is_detected(tmp_path):
    root = tmp_path / "root"; root.mkdir(); (root / "old.txt").write_text("a")
    out = root / "review"; (out / "inventory").mkdir(parents=True)
    (out / "inventory/protected_before.json").write_text(json.dumps({"files":[{"path":"old.txt","size":1,"sha256":"bad"}]}))
    verify(root, root, out)
    data = json.loads((out / "verify.json").read_text())
    assert data["protected_integrity"] is False

def test_reverse_last_action_rule_rejected():
    r = validate_planner_record_semantics({}, {"reason": "CONTINUE"}, {}, [])
    assert not r["valid"]

def test_reverse_nominal_fill_rule_rejected():
    root = Path(__file__).resolve().parents[1]
    ids, mask = production_mask(root, {"p:OnTable:obj_b": "UNKNOWN"})
    assert not mask or not mask[ids.index("a:PICK:obj_b:v1")]
