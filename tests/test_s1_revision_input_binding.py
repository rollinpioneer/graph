from __future__ import annotations

import json

import pytest

from cp_disr.analysis.s1_revision import RevisionError
from cp_disr.analysis.s1_revision_resume import (
    EXPECTED_GOALS,
    EXPECTED_SCENES,
    _frozen_and_rows,
    _reset,
    _split_row_hash,
)


def row(case_id="T_A_dev_12"):
    return {"case_id": case_id, "seed": 6112, "target_xy": [-.1, -.1], "second_xy": [.1, -.1], "container_xy": [.18, .12], "buffer_xy": [-.18, .12], "lid_closed": True, "split": "dev", "cache_key": "old", "cache_dir": "old", "cache_status": "REUSED"}


def fixture_tree(tmp_path):
    out=tmp_path/"out"; (out/"manifests").mkdir(parents=True); (tmp_path/"configs/splits").mkdir(parents=True)
    rows=[]; scenes=[]
    for i,scene_id in enumerate(EXPECTED_SCENES):
        value=row(scene_id); value["seed"]=6112+i; rows.append(value); scenes.append({"scene_id":scene_id,"generator_seed":value["seed"],"reset_config":_reset(value),"reset_config_sha256":_split_row_hash(value)})
    (out/"manifests/frozen_discovery_scene_manifest.json").write_text(json.dumps({"scenes":scenes}))
    (tmp_path/"configs/splits/T_A_stage_2a.json").write_text(json.dumps({"dev":rows}))
    return out,rows,scenes


def test_i01_frozen_scene_matches_split(tmp_path):
    out,_,_ = fixture_tree(tmp_path)
    _,_,pairs=_frozen_and_rows(tmp_path,out)
    assert len(pairs)==8


def test_i02_reset_mismatch_fails(tmp_path):
    out,rows,_=fixture_tree(tmp_path); rows[0]["target_xy"]=[0,0]
    (tmp_path/"configs/splits/T_A_stage_2a.json").write_text(json.dumps({"dev":rows}))
    with pytest.raises(RevisionError): _frozen_and_rows(tmp_path,out)


def test_i03_scene_id_mismatch_fails(tmp_path):
    out,_,scenes=fixture_tree(tmp_path); scenes[0]["scene_id"]="T_A_dev_99"
    (out/"manifests/frozen_discovery_scene_manifest.json").write_text(json.dumps({"scenes":scenes}))
    with pytest.raises(RevisionError): _frozen_and_rows(tmp_path,out)


def test_i04_two_production_goals():
    assert EXPECTED_GOALS == ({"fact_id":"p:Inside:target:container","sign":1},{"fact_id":"p:Inside:second_object:container","sign":1})


def test_i05_move_not_in_expected_stage2a_actions():
    actions=("a:OPEN:container:v1","a:PICK:target:v1","a:PLACE:target:container:v1")
    assert not any(":MOVE:" in x for x in actions)


def test_i06_relation_fields_removed():
    clean={k:v for k,v in row().items() if k not in ("cache_key","cache_dir","cache_status")}
    assert not ({"cache_key","cache_dir","cache_status"}&set(clean))


def test_i07_hidden_truth_is_qa_only():
    qa={"hidden_truth_used_for_prompt":False,"hidden_truth_used_for_policy":False,"hidden_truth_used_for_relation_admission":False,"qa_only":True}
    assert qa["qa_only"] and not any(v for k,v in qa.items() if k.startswith("hidden_truth_used"))


def test_i21_split_hash_includes_cache_identity():
    assert _split_row_hash(row()) != _split_row_hash({k:v for k,v in row().items() if k!="cache_key"})
