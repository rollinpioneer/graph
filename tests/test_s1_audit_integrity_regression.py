import json
from pathlib import Path

def test_review_manifest_preserves_budget_and_training_lock():
    out=Path("runs/final_master/S1/20260928T154311Z_bea4bf0d/engineering_review_f0d3e18")
    before=json.loads((out/"budget_before.json").read_text())
    after=json.loads((out/"budget_after.json").read_text())
    result=json.loads((out/"reviewed_eligibility.json").read_text())
    assert before == after
    assert result["tp_training_authorized"] is False
    assert result["new_physical_episodes_authorized"] == 0

def test_audit_has_no_live_execution_entrypoints():
    text=Path("src/cp_disr/analysis/s1_evidence_review.py").read_text()
    assert "from cp_disr.runtime" not in text
    assert "make_env(" not in text
