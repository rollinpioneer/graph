"""Offline tests for scripts/final_tb_holdout_runner.py (CP-DISR-TB-INDEP-HOLDOUT-01): 0 environment, 0 episode, 0 GPU."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("holdout_runner", ROOT / "scripts/final_tb_holdout_runner.py")
R = importlib.util.module_from_spec(spec)
spec.loader.exec_module(R)

ORDER = ["T_B_test_%02d" % i for i in range(30)]
SLOTS = [("TB-M3", "CORE", "B1-K", 1), ("TB-M5", "CORE", "B2", 1), ("TB-M6", "CORE", "B1-K+E", 0), ("TB-C1", "CORE", "B1-K+E", 1),
         ("TB-M2", "HISTORICAL_EXTENSION", "B1-K", 0), ("TB-M4", "HISTORICAL_EXTENSION", "B2", 0), ("TB-M1", "SYSTEM_REFERENCE", "B0", 0)]


def manifest():
    models = [{"slot": s, "tier": t, "method": m, "seed": sd, "plan_run": "R-" + s, "execution_class": R.CURRENT if s in R.CLASS_ORDER[R.CURRENT] else R.ARCHIVED,
               "final_checkpoint": {"path": "/x/%s.pt" % s, "sha256": "ab" * 32, "bytes": 1}} for s, t, m, sd in SLOTS]
    return {"models": models, "test_cases": {"order": ORDER}}


def row(case, success=True, G=1.0, steps=2, reason="TASK_SUCCESS", ttf=10.0):
    acts = [{"decision_index": i, "candidate_id": "c%d" % i, "clock_start": float(i), "clock_end": float(i + 1), "duration": 1.0, "reward": 0.0, "weight": 1.0,
             "controller_exit": "ok", "terminated": False, "truncated": False, "reason": None, "success_confirmed": success and i == steps - 1} for i in range(steps)]
    return {"case_id": case, "success": success, "G": G, "steps": steps, "reason": reason, "success_seconds": 1.0 if success else None, "source_n": 0, "prior_mode": "absent",
            "episode_start_clock_s": 0.0, "time_to_first_confirmed_success_s": ttf if success else None, "first_confirmed_success_decision_index": steps - 1 if success else None,
            "episode_end_elapsed_s": float(steps), "actions": acts, "inter_step_clock_gap_s": 0.0, "no_transition_exit": None, "outcome_class": "NORMAL_POLICY_OUTCOME"}


def wrow(slot, k, **kw):
    cls = R.CURRENT if slot in R.CLASS_ORDER[R.CURRENT] else R.ARCHIVED
    return {"slot_id": "%s|%s" % (slot, ORDER[k]), "slot": slot, "case_id": ORDER[k], "case_index": k, "worker": "A" if cls == R.CURRENT else "B", "class": cls, "row": row(ORDER[k], **kw)}


def test_derive_split_strips_cache_and_rejects_pointers():
    dev = {"train": [{"case_id": "t0", "cache_dir": "x", "cache_key": "k", "seed": 1}], "dev": [{"case_id": "d0", "cache_status": "s", "seed": 2}]}
    test = [{"case_id": c, "seed": i} for i, c in enumerate(ORDER)]
    sp = R.derive_no_prior_test_split(dev, test)
    assert [r["case_id"] for r in sp["test"]] == ORDER and not any(k in json.dumps(sp) for k in R.CACHE_KEYS)
    bad = [dict(test[0], cache_dir="runs/vlm_cache/test/x")] + test[1:]
    with pytest.raises(ValueError):
        R.derive_no_prior_test_split(dev, bad)


def test_forbidden_paths_and_stop_exception_type():
    for p in ("a/vlm_cache/test/x.json", "r/relation_truth/a", "configs/splits/T_B_stage_2a_v11.json", "x/test_metrics.json"):
        assert R.is_forbidden_path(p)
    assert not R.is_forbidden_path("configs/splits/T_B_stage_2a_test30.json")
    assert not R.is_forbidden_path("/home/x/src/cp_disr/vlm_cache_pipeline.py")  # code, not a cache
    assert not R.is_forbidden_path("/home/x/runs/final_master/2.1.1/launch/y_tb_indep_holdout/worker_current/progress.json")
    assert not issubclass(R.ReleaseStopRequested, Exception) and issubclass(R.ReleaseStopRequested, BaseException)


def test_exact_sign_p_and_paired():
    assert R.exact_sign_p(0, 0) == 1.0 and R.exact_sign_p(3, 3) == 1.0
    assert abs(R.exact_sign_p(0, 5) - 2 / 32) < 1e-12
    a = [row(c, success=(i % 2 == 0), G=float(i % 2)) for i, c in enumerate(ORDER)]
    b = [row(c, success=True, G=1.0) for c in ORDER]
    p = R.paired(a, b, "a", "b", "t")
    assert p["n_paired_cases"] == 30 and p["a_success_b_fail"] == 0 and p["a_fail_b_success"] == 15 and p["both_success"] == 15
    assert abs(p["mean_delta_G_a_minus_b"] - (-0.5)) < 1e-12


def test_model_stats_time_only_over_successes():
    rows = [row(ORDER[0], True, 1.0, 2, ttf=8.0), row(ORDER[1], False, 0.0, 3, reason="DEADLINE"), row(ORDER[2], True, 0.5, 1, ttf=4.0)]
    meta = manifest()["models"][0]
    st = R.model_stats("TB-M3", meta, rows, 30, 0)
    assert st["success"]["text"] == "2/3" and st["time_to_first_confirmed_success_s"]["mean"] == 6.0 and st["time_to_first_confirmed_success_s"]["median"] == 6.0
    assert st["failure_reason_distribution"] == {"DEADLINE": 1} and st["counts"]["not_executed"] == 27


def test_account_statuses_infra_stop_and_dead_worker(tmp_path):
    man = manifest()
    rows = [wrow("TB-M3", k) for k in range(30)] + [wrow("TB-M5", k) for k in range(3)]
    events = [{"kind": "INFRASTRUCTURE_FAILURE", "slot_id": "TB-M5|%s" % ORDER[3], "slot": "TB-M5", "case_id": ORDER[3]}]
    acc, synth, ntech, _ = R.account(tmp_path, man, rows, events, {"A": "INFRASTRUCTURE_FAILURE", "B": "STOPPED_BY_RELEASE_FLAG"})
    st = {x["slot_id"]: x["status"] for x in acc["slots"]}
    assert acc["planned_slots"] == 210 and acc["status_counts"]["MEASURED"] == 33 and acc["status_counts"]["TECHNICAL_NOT_MEASURED"] == 1
    assert st["TB-M5|%s" % ORDER[3]] == "TECHNICAL_NOT_MEASURED" and st["TB-M5|%s" % ORDER[4]] == "NOT_EXECUTED_RELEASE_STOPPED" and not acc["all_planned_slots_measured"]
    # a worker that vanished (state still RUNNING) -> first missing slot is TECHNICAL (never a policy failure), the rest NOT_EXECUTED
    acc2, synth2, _, _ = R.account(tmp_path, man, [wrow("TB-M3", k) for k in range(5)], [], {"A": "RUNNING", "B": "NOT_STARTED"})
    st2 = {x["slot_id"]: x["status"] for x in acc2["slots"]}
    assert st2["TB-M3|%s" % ORDER[5]] == "TECHNICAL_NOT_MEASURED" and st2["TB-M3|%s" % ORDER[6]] == "NOT_EXECUTED_RELEASE_STOPPED" and len(synth2) == 1


def test_full_completion_is_all_measured(tmp_path):
    man = manifest()
    rows = [wrow(s, k) for s, *_ in SLOTS for k in range(30)]
    acc, _, _, _ = R.account(tmp_path, man, rows, [], {"A": "FINISHED_ALL_PLANNED", "B": "FINISHED_ALL_PLANNED"})
    assert acc["status_counts"] == {"MEASURED": 210} and acc["all_planned_slots_measured"]


def test_collect_on_partial_release_dir_never_crashes_and_prints_no_scores(tmp_path, capsys):
    man = manifest()
    man.update({"card": R.CARD, "bindings": {}})
    (tmp_path / "test_release_manifest.json").write_text(json.dumps(man))
    wd = tmp_path / "worker_current"
    wd.mkdir()
    for k in range(4):
        R.append_jsonl(wd / "per_episode_eval_row_v2.jsonl", wrow("TB-M3", k, G=0.123456))
    (wd / "worker_state.json").write_text(json.dumps({"state": "STOPPED_BY_RELEASE_FLAG"}))
    assert R.cmd_collect(SimpleNamespace(release=str(tmp_path))) == 0
    out = capsys.readouterr().out
    assert "0.123456" not in out
    acc = json.loads((tmp_path / "holdout_accounting.json").read_text())
    assert acc["status_counts"]["MEASURED"] == 4 and (tmp_path / "core_results.json").is_file() and (tmp_path / "holdout_summary.md").is_file()
    assert len((tmp_path / "per_episode_eval_row_v2.jsonl").read_text().splitlines()) == 4


def test_append_jsonl_and_fsync_probe(tmp_path):
    p = tmp_path / "x.jsonl"
    R.append_jsonl(p, {"a": 1})
    R.append_jsonl(p, {"a": 2})
    assert [r["a"] for r in R.read_jsonl(p)] == [1, 2]
    assert R.fsync_probe(tmp_path / "newdir/sub")
    blocker = tmp_path / "file"
    blocker.write_text("x")
    with pytest.raises(Exception):
        R.fsync_probe(blocker / "sub")


def test_runner_source_has_no_training_provider_or_retry_and_is_not_a_src_change():
    src = (ROOT / "scripts/final_tb_holdout_runner.py").read_text()
    for needle in ("torch.optim", "optimizer.step", "import requests", "dashscope", "backward(", "while True"):
        assert needle not in src.replace("FORBIDDEN", "")
    assert "evaluate_checkpoint_recorded" in src and "no_prior_case_gate" in src
