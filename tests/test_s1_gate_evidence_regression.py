import csv
from pathlib import Path

def test_gate_review_requires_independent_evidence():
    p=Path("runs/final_master/S1/20260928T154311Z_bea4bf0d/engineering_review_f0d3e18/audit/gates/gate_evidence_matrix.csv")
    rows=list(csv.DictReader(p.open()))
    assert {r["gate"] for r in rows} == {"E1","E2","E3","E4","E5","E6"}
    assert all(r["reviewed_status"] == "NOT_ESTABLISHED" for r in rows)

def test_old_witnesses_are_not_reclassified_as_success():
    p=Path("runs/final_master/S1/20260928T154311Z_bea4bf0d/engineering_review_f0d3e18/audit/runner/witness_reclassification.csv")
    rows=list(csv.DictReader(p.open()))
    assert len(rows)==8
    assert all(r["reviewed_classification"] == "INCOMPLETE_CONTINUATION" for r in rows)
