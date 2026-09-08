from upgrade_v2.visual_refine_l2.io import find_forbidden_keys


def test_online_schema_rejects_nested_oracle_fields() -> None:
    clean = {"front_rgb": "frame.jpg", "history": [{"contact_present": True}], "action": "lift"}
    assert find_forbidden_keys(clean) == []
    leaked = {"history": [{"qpos": [0.0]}], "future_outcome": True, "scenario": "slip"}
    findings = find_forbidden_keys(leaked)
    assert "history[0].qpos" in findings
    assert "future_outcome" in findings
    assert "scenario" in findings
