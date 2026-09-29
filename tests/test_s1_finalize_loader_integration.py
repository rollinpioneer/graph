import json, shutil
from pathlib import Path
from cp_disr.analysis.s1_integration import load_gate_evidence_bundle, sha256_file
from cp_disr.analysis.s1_revision_resume import finalize_eligibility

def table(path, rows):
    path.write_text(json.dumps(rows), encoding="utf-8")
    return sha256_file(path)

def make_manifest(tmp_path, complete=True):
    names=("relation_adjudications","alignment_records","gradient_records","witness_records","consequence_comparisons","utility_records")
    tables={}
    rows={
        "relation_adjudications":[{"case_id":"c1","independent_adjudication":True,"truth_status":"ADJUDICATED"},
                                  {"case_id":"c2","independent_adjudication":True,"truth_status":"ADJUDICATED"}],
        "alignment_records":[{"case_id":"c1","candidate_id_alignment":True,"mask_alignment":True,"goal_alignment":True,"node_alignment":True,"actual_comparison":True},
                             {"case_id":"c2","candidate_id_alignment":True,"mask_alignment":True,"goal_alignment":True,"node_alignment":True,"actual_comparison":True}],
        "gradient_records":[{"case_id":"c1","relation_gradient_measured":True,"patch_gradient_measured":True,"separate_sources":True,"relative_response":1.0},
                            {"case_id":"c2","relation_gradient_measured":True,"patch_gradient_measured":True,"separate_sources":True,"relative_response":1.0}],
        "witness_records":[{"case_id":"c1","protocol_complete":True,"full_episode_completed":True,"independent_evaluator":True,"paired_restore_verified":True,"eligible_for_e4":True},
                           {"case_id":"c2","protocol_complete":True,"full_episode_completed":True,"independent_evaluator":True,"paired_restore_verified":True,"eligible_for_e4":True}],
        "consequence_comparisons":[{"reliable":True,"outcome_different":True,"contract_ranking_comparable":True}],
        "utility_records":[{"truth_adjudicated":True,"utility_adjudicated":True,"opportunity_classified":True,"natural":True,"provider":True}],
    }
    for name in names:
        if complete or name!="utility_records":
            path=tmp_path/f"{name}.json"; tables[name]={"path":str(path),"sha256":""}
            tables[name]["sha256"]=table(path,rows[name])
    manifest={"schema_version":"s1_gate_evidence_bundle_v1","source_revision_index":1,
              "evidence_context":"ENGINEERING_TEST","tables":tables}
    m=tmp_path/"evidence_manifest.json"; m.write_text(json.dumps(manifest),encoding="utf-8")
    return m

def test_loader_validates_hash_and_all_tables(tmp_path):
    m=make_manifest(tmp_path)
    b=load_gate_evidence_bundle(tmp_path,m)
    assert not b["missing_evidence_tables"] and not b["hash_mismatches"]
    assert len(b["relation_adjudications"])==2 and b["scientific_admissible"] is False

def test_loader_missing_table_is_explicit(tmp_path):
    m=make_manifest(tmp_path,False)
    b=load_gate_evidence_bundle(tmp_path,m)
    assert "utility_records" in b["missing_evidence_tables"]

def test_finalize_uses_disk_loader_and_test_data_is_not_scientific(tmp_path):
    out=tmp_path/"out"; out.mkdir(); m=make_manifest(out)
    (out/"evidence_manifest.json").write_text(m.read_text())
    result=finalize_eligibility(tmp_path,out)
    assert result["status"]=="COMPLETE"
    assert result["scientific_admissible"] is False
    assert result["eligibility_decision"]=="NOT_ELIGIBLE_AFTER_SINGLE_REVISION"
    assert result["gates"]["E1"] == "PASS"

def test_finalize_missing_manifest_is_pending(tmp_path):
    out=tmp_path/"out"; out.mkdir()
    result=finalize_eligibility(tmp_path,out)
    assert result["status"]=="STOPPED" and result["eligibility_decision"]=="PENDING_SAME_REVISION"

def test_finalize_stopped_stage_stays_pending(tmp_path):
    out=tmp_path/"out"; out.mkdir()
    (out/"stage_manifest.json").write_text(json.dumps({"status":"STOPPED","stop_reason":"PROVIDER_STOPPED"}))
    result=finalize_eligibility(tmp_path,out)
    assert result["status"]=="STOPPED" and result["eligibility_decision"]=="PENDING_SAME_REVISION"


def test_finalize_research_evidence_can_reach_eligible_branch(tmp_path):
    out = tmp_path / "out"; out.mkdir()
    m = make_manifest(out)
    doc = json.loads(m.read_text())
    doc["evidence_context"] = "RESEARCH"
    m.write_text(json.dumps(doc), encoding="utf-8")
    (out / "evidence_manifest.json").write_text(m.read_text())
    result = finalize_eligibility(tmp_path, out)
    assert result["scientific_admissible"] is True
    assert result["eligibility_decision"] == "ELIGIBLE_AFTER_SINGLE_REVISION"
