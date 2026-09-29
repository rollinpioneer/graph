from cp_disr.analysis.s1_evidence_review import evaluate_gate_evidence

def test_P01_alignment_requires_actual_comparison():
    b={"alignment_records":[{"case_id":"c1","candidate_id_alignment":True,"mask_alignment":True,"goal_alignment":True,"node_alignment":True,"actual_comparison":False},{"case_id":"c2","candidate_id_alignment":True,"mask_alignment":True,"goal_alignment":True,"node_alignment":True,"actual_comparison":False}]}
    assert evaluate_gate_evidence(b)["E2"]["status"]=="NOT_ESTABLISHED"

def test_P02_gradients_require_separate_sources():
    b={"gradient_records":[{"case_id":"c1","relation_gradient_measured":True,"patch_gradient_measured":True,"separate_sources":False,"relative_response":1.0},{"case_id":"c2","relation_gradient_measured":True,"patch_gradient_measured":True,"separate_sources":False,"relative_response":1.0}]}
    assert evaluate_gate_evidence(b)["E3"]["status"]=="NOT_ESTABLISHED"

def test_P04_offline_rescore_is_failure_closed(tmp_path):
    from cp_disr.analysis.s1_revision import offline_rescore_planner
    r=offline_rescore_planner(tmp_path,"case",tmp_path)
    assert callable(offline_rescore_planner)
    assert r["runtime_constructed"] is False and r["environment_episodes"] == 0
