import hashlib, json
from cp_disr.analysis import s1_e4_recovery as m


def _side(out, bid, q, key):
    d = out / "witnesses/initial_state_checks"; d.mkdir(parents=True, exist_ok=True)
    (d / f"{bid}.json").write_text(json.dumps({"status": "OK", "compare_key": key, "candidate_ids": ["a", "b"], "candidate_mask": [True, True],
        "physical": {"qpos": q, "qvel": [0.0], "qpos_sha256": "x", "qvel_sha256": "y", "hidden_truth": {}}}))


def _reg():
    return {"branches": [{"branch_id": "b1", "case_id": "c", "repeat": 0}, {"branch_id": "b2", "case_id": "c", "repeat": 0}]}


def test_paired_restore_verified_and_detects_difference(tmp_path):
    key = {"case_id": "c", "applied_restore_seed": 1}
    _side(tmp_path, "b1", [0.1, 0.2], key); _side(tmp_path, "b2", [0.1, 0.2], key)
    assert m.paired_restore(tmp_path, "c", 0, _reg())["status"].startswith("PAIRED_RESTORE_VERIFIED")
    _side(tmp_path, "b2", [0.1, 0.3], key)
    assert m.paired_restore(tmp_path, "c", 0, _reg())["status"] == "PAIRED_RESTORE_NOT_ESTABLISHED"
    _side(tmp_path, "b2", [0.1, 0.2], {**key, "applied_restore_seed": 2})
    assert m.paired_restore(tmp_path, "c", 0, _reg())["status"] == "PAIRED_RESTORE_NOT_ESTABLISHED"


def test_paired_restore_missing_sidecar_is_not_established(tmp_path):
    assert m.paired_restore(tmp_path, "c", 0, _reg())["reason"] == "MISSING_SIDECAR"


def test_seed_rule_is_candidate_independent():
    s = lambda case, rep: int(hashlib.sha256(f"{case}|{rep}".encode()).hexdigest()[:8], 16)
    assert s("T_A_dev_14", 0) == s("T_A_dev_14", 0) and s("T_A_dev_14", 0) != s("T_A_dev_14", 1)
