from cp_disr.analysis.s1_evidence_review import evaluate_gate_evidence

def test_G01_provider_without_truth_does_not_pass():
    r=evaluate_gate_evidence({"relation_adjudications":[],"consequence_comparisons":[],"utility_records":[]})
    assert all(r[k]["status"]=="NOT_ESTABLISHED" for k in ("E1","E4","E5","E6"))

def test_G02_valid_rows_without_pairing_do_not_pass_E4():
    b={"witness_records":[{"protocol_complete":True,"full_episode_completed":True,"independent_evaluator":False,"paired_restore_verified":False,"eligible_for_e4":False}],"consequence_comparisons":[]}
    assert evaluate_gate_evidence(b)["E4"]["status"]=="NOT_ESTABLISHED"

def test_G03_same_case_repeat_not_independent():
    b={"relation_adjudications":[{"case_id":"same","independent_adjudication":True,"truth_status":"ADJUDICATED"},{"case_id":"same","independent_adjudication":True,"truth_status":"ADJUDICATED"}]}
    assert evaluate_gate_evidence(b)["E1"]["status"]=="NOT_ESTABLISHED"

def test_G04_missing_fields_do_not_pass():
    assert evaluate_gate_evidence({"relation_adjudications":[{"case_id":"x"}]})["E1"]["status"]=="NOT_ESTABLISHED"

def test_G05_synthetic_fixture_separate():
    b={"utility_records":[{"truth_adjudicated":True,"utility_adjudicated":True,"opportunity_classified":True,"synthetic_unit_fixture":True}]}
    assert evaluate_gate_evidence(b)["E6"]["status"]=="NOT_ESTABLISHED"
