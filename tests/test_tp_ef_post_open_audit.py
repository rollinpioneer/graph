from pathlib import Path

import pytest

from cp_disr.analysis import tp_ef_post_open_audit as a
from cp_disr.analysis import tp_ef_protocol_review as pr
from cp_disr.facts import Truth

ROOT = Path(__file__).resolve().parents[1]
INITIAL = {"p:AtBuffer:second_object:buffer": "FALSE", "p:AtBuffer:target:buffer": "FALSE", "p:GripperEmpty": "TRUE",
           "p:Held:second_object": "FALSE", "p:Held:target": "FALSE", "p:Inside:second_object:container": "FALSE",
           "p:Inside:target:container": "FALSE", "p:OnTable:second_object": "TRUE", "p:OnTable:target": "TRUE", "p:Open:container": "FALSE"}


@pytest.fixture(scope="module")
def tpl():
    return pr.build_template(ROOT)


def test_guard_refuses_test_and_holdout_paths(tmp_path):
    g = a.Guard()
    for bad in ("runs/x/tb_indep_holdout_6129/f.json", "configs/splits/T_B_stage_2a_test30.json", "runs/final_test_id/x.json"):
        with pytest.raises(PermissionError):
            g.read(bad)
    assert len(g.refused) == 3 and g.opened == 0


def test_hypothetical_post_open_routes_are_tied_and_legal(tpl):
    reg = [{"scene_id": "c", "facts": INITIAL}]
    values, nd = a.hypothetical_post_open(tpl, reg)
    audit = a.contract_audit(tpl, values, nd)
    assert nd == 1 and audit["pick_target_legal"] and audit["pick_second_legal"]
    assert audit["routes"]["T"]["fully_legal_reaches_goal"] and audit["routes"]["S"]["fully_legal_reaches_goal"]
    assert audit["classification"] == "CONTRACT_TIED_SCRIPTABLE" and audit["min_skills_after_candidate"] == {"PICK_target": 4, "PICK_second": 4}
    assert "closed-container" in audit["scope_limitation"]


def test_relation_universe_excludes_direct_redundancy(tpl):
    values = {k: Truth(x) for k, x in INITIAL.items()}
    uni = a.relation_universe(tpl, [pr.IDS["pt"], pr.IDS["ps"]])
    red = [r for r in uni if r["contract_redundant"]]
    assert any(r["source"] == pr.IDS["open"] and r["target"] == pr.IDS["plt"] and r["effect_fact_ref"] == "p:Open:container" for r in red)
    rel = a.relation_audit(tpl, values)
    assert rel["non_redundant_extra_contract"] == len(uni) - len(red) and rel["provider_calls"] == 0


def test_request_is_budget_only_and_capped():
    inv = {"qualifying_snapshot_count": 0, "discovery1_open_confirmed_distinct_cases": ["T_B_dev_03", "T_B_dev_05", "T_B_dev_18"]}
    audit = {"classification": "CONTRACT_TIED_SCRIPTABLE"}
    rel = {"entry_exists": True, "non_redundant_touching_candidate_pair": 5, "candidate_relative_difference_exists": True}
    md = a.request_md(inv, audit, rel)
    assert "budget request only" in md and "Setup-only episodes: **2**" in md and "continuation skills = 0" in md
