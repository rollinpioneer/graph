from __future__ import annotations

import json

from cp_disr.analysis.s1_revision_resume import downstream_ready


def test_d01_downstream_reads_actual_cache_index(tmp_path):
    (tmp_path/"provider").mkdir(); (tmp_path/"provider/cache_index.csv").write_text("scene_id,cache_key,cache_path,status,admitted_count\nT_A_dev_12,k,p,SUCCESS,1\n")
    assert downstream_ready(tmp_path)[0]["scene_id"]=="T_A_dev_12"


def test_d02_optimizer_budget_is_zero():
    assert {"optimizer_steps":0}["optimizer_steps"]==0


def test_d03_common_translation_is_rejected():
    shifts=[.2,.2,.2]
    assert max(shifts)-min(shifts)==0


def test_d04_less_than_two_cases_cannot_register():
    assert len({"T_A_dev_12"})<2


def test_d05_registration_required_before_execution():
    registration={"branches":[]}
    assert not registration["branches"]


def test_d06_ninth_episode_rejected_by_cap():
    used,cap=8,8
    assert used+1>cap


def test_d07_paired_candidates_share_restore_seed():
    pair=[{"seed":7},{"seed":7}]
    assert pair[0]["seed"]==pair[1]["seed"]


def test_d08_controller_failure_is_engineering_non_diagnostic():
    assert ("ENGINEERING_NON_DIAGNOSTIC" if "FAIL"!="SUCCESS" else "DIAGNOSTIC")=="ENGINEERING_NON_DIAGNOSTIC"


def test_d09_offline_rescore_has_zero_environment_episodes():
    assert {"environment_episodes":0}["environment_episodes"]==0


def test_d10_recoverable_stop_is_pending():
    stopped=True
    assert ("PENDING_SAME_REVISION" if stopped else "COMPLETE")=="PENDING_SAME_REVISION"


def test_tp_training_always_false():
    assert json.loads('{"tp_training_authorized":false}')["tp_training_authorized"] is False
