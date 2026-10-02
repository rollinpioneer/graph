"""CP-DISR-TB-SMOKE-SELECTOR-REPAIR-R2-1 offline tests (CPU only; engineering-only data).

The tests import the ACTUAL repaired `final_tb.ScriptedSelector` and the ACTUAL `Collector.step`; neither is mocked.
Only the physical boundary is replaced (clock, executor, observation/perception/verifier/evaluator/safety, successor
builder).  The policy is the production Policy on the real T_B template.  No MuJoCo/EGL object is constructed and no
optimizer step is taken.  The mutation tests copy the source tree, restore the OLD selector and require the first
check to fail on the mutant and pass on the unmutated control copy.
"""
from __future__ import annotations

import inspect
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from unittest import mock

import pytest

from cp_disr import final_tb
from cp_disr.common import BindingError
from tests.helpers import final_tb_checks as chk
from tests.helpers import smoke_r2_checks as r2

ROOT = r2.ROOT
HEAD_A, HEAD_B = "a" * 40, "b" * 40


# 1-8 ------------------------------------------------------------------------------------------------
def test_01_stale_target_next_value_with_real_collector(tmp_path):
    out = r2.check_stale_target_next_value(ROOT, tmp_path)
    assert out["executor_calls"] == 1 and out["forward_calls"] == 2


def test_02_five_action_script_cursor_only_advances_in_the_explicit_loop(tmp_path):
    assert r2.check_five_action_script_cursor(ROOT, tmp_path)["executor_calls"] == 5


def test_03_masked_or_missing_target_fails_closed_with_zero_executions(tmp_path):
    seen = r2.check_masked_missing_target_fails_closed(ROOT, tmp_path)
    assert set(seen) == {"masked", "missing", "none"}


def test_04_terminal_and_no_candidate_states_keep_original_semantics(tmp_path):
    out = r2.check_terminal_and_no_candidate_semantics(ROOT, tmp_path)
    assert out["no_legal_now"] == "NO_SAFE_CANDIDATES"


def test_05_stale_action_still_legal_causes_no_extra_execution(tmp_path):
    assert r2.check_stale_action_still_legal(ROOT, tmp_path)["select_events"] == 1


def test_06_nonfinite_policy_values_are_never_hidden(tmp_path):
    out = r2.check_nonfinite_policy_not_masked(ROOT, tmp_path)
    assert out["current_forward_nan"]["executor_calls"] == 0 and out["next_value_probe_nan"]["executor_calls"] == 1


def test_07_output_logits_distribution_value_hidden_are_unchanged(tmp_path):
    assert r2.check_output_fields_unchanged(ROOT, tmp_path)["n_legal"] == 3


def test_08_post_execution_failure_keeps_real_call_count_stage_and_traceback(tmp_path):
    out = r2.check_post_execution_failure_is_recorded(ROOT, tmp_path)
    assert out == {"stage_a": "NEXT_SNAPSHOT_BUILD", "stage_b": "NEXT_STATE_VALUE_PROBE"}


# 9 mutation: restore the OLD selector in a temporary copy -------------------------------------------
def _tree(tmp_path, name):
    tree = tmp_path / name
    tree.mkdir()
    for item in ROOT.iterdir():
        if item.name in {"src", "tests", "__pycache__"}:
            continue
        os.symlink(item, tree / item.name)
    shutil.copytree(ROOT / "src", tree / "src", ignore=shutil.ignore_patterns("__pycache__"))
    (tree / "tests").mkdir()
    shutil.copy(ROOT / "tests/__init__.py", tree / "tests/__init__.py")
    shutil.copytree(ROOT / "tests/helpers", tree / "tests/helpers", ignore=shutil.ignore_patterns("__pycache__"))
    os.symlink(ROOT / "tests/fixtures", tree / "tests/fixtures")
    return tree


def _restore_old_selector(tree):
    path = tree / "src/cp_disr/final_tb.py"
    text = path.read_text()
    start, end = text.index("class _SelectionOnlyOutput:"), text.index("class CountingExecutor:")
    old = (ROOT / "tests/fixtures/r2_selector_original.txt").read_text().replace("class ScriptedSelector:", "class _OriginalScriptedSelector:")
    shim = ("\n\nclass ScriptedSelector(_OriginalScriptedSelector):\n"
            "    def __init__(self, policy, recorder=None):\n        super().__init__(policy)\n\n\n")
    path.write_text(text[:start] + old + shim + text[end:])


def _run(tree, check):
    code = ("import sys,tempfile;sys.path.insert(0,%r);from tests.helpers import smoke_r2_checks as c;"
            "from pathlib import Path\nwith tempfile.TemporaryDirectory() as t:\n    c.%s(Path(%r),t)\n") % (str(tree), check, str(tree))
    env = dict(os.environ, PYTHONPATH=str(tree / "src"))
    return subprocess.run([sys.executable, "-c", code], cwd=str(tree), env=env, capture_output=True, text=True, timeout=280)


def test_09_old_selector_in_temp_copy_fails_check_1_and_repaired_passes(tmp_path):
    control = _run(_tree(tmp_path, "control"), "check_stale_target_next_value")
    assert control.returncode == 0, "repaired copy must pass:\n" + control.stderr[-2500:]
    mutant_tree = _tree(tmp_path, "mutant")
    _restore_old_selector(mutant_tree)
    assert "_OriginalScriptedSelector" in (mutant_tree / "src/cp_disr/final_tb.py").read_text()
    mutant = _run(mutant_tree, "check_stale_target_next_value")
    assert mutant.returncode != 0, "the OLD selector must fail the stale-target/next-V check"
    assert "ValueError" in mutant.stderr and "next_v" in mutant.stderr, mutant.stderr[-2500:]
    assert "_OriginalScriptedSelector" not in (ROOT / "src/cp_disr/final_tb.py").read_text()  # production file stays repaired


# 10 smoke pair gating -------------------------------------------------------------------------------
def _reference_run(tmp_path, transitions=1000, size=1 << 20):
    ref = tmp_path / "reference_run"
    (ref / "checkpoints").mkdir(parents=True)
    (ref / "job_summary.json").write_text(json.dumps({"valid_transitions": transitions, "complete_updates": 1}))
    (ref / "checkpoints" / "model.pt").write_bytes(b"\0" * size)
    return ref


def _pass_pair(tmp_path, head, name):
    out = tmp_path / name
    r2.register_for_test(ROOT, out, head)
    res1, _ = r2.run_smoke_with_doubles(ROOT, out, 1, head, tmp_path / (name + "_d1"))
    assert res1["episode_verdict"] == "PASS", res1.get("failure") or res1["steps"]
    assert final_tb.write_smoke_receipt(out) == {"final": False, "verdicts": {1: "PASS"}}
    res2, _ = r2.run_smoke_with_doubles(ROOT, out, 2, head, tmp_path / (name + "_d2"))
    assert res2["episode_verdict"] == "PASS", res2.get("failure") or res2["steps"]
    receipt = final_tb.write_smoke_receipt(out)
    assert receipt["label"] == "TB_RUNTIME_SMOKE_PASS"
    return out, receipt, (res1, res2)


def test_10a_first_smoke_failure_blocks_second_smoke_and_training(tmp_path):
    out = tmp_path / "launch"
    r2.register_for_test(ROOT, out, HEAD_A)
    res, _ = r2.run_smoke_with_doubles(ROOT, out, 1, HEAD_A, tmp_path / "d", builder_fault="raise")
    assert res["episode_verdict"].startswith("FAIL")
    with pytest.raises(BindingError, match="only if the first passed"):
        r2.run_smoke_with_doubles(ROOT, out, 2, HEAD_A, tmp_path / "d2")
    receipt = final_tb.write_smoke_receipt(out)
    assert receipt["label"] == "TB_LAUNCH_BLOCKED_RUNTIME" and receipt["cases"]["1"]["failure_stage"] == "NEXT_SNAPSHOT_BUILD"
    with pytest.raises(BindingError):
        final_tb.run_release(out, "train", HEAD_A, "R-TB-E-0", reference_runs=[_reference_run(tmp_path)], free_bytes_fn=lambda p: 10 ** 13)
    # the failed case is charged once and can never be retried
    with pytest.raises(BindingError, match="already attempted"):
        r2.run_smoke_with_doubles(ROOT, out, 1, HEAD_A, tmp_path / "d3")


def test_10b_two_passes_on_the_same_prep_release_training_only_with_storage(tmp_path):
    out, receipt, (res1, res2) = _pass_pair(tmp_path, HEAD_A, "launch")
    c1, c2 = res1["counters"], res2["counters"]
    for c in (c1, c2):  # real executor entered/returned equals the five scripted actions and the transitions returned
        assert c["executor_calls_entered"] == c["executor_calls_returned"] == c["transitions_returned"] == 5
        assert c["budget_charged_this_attempt"]["skill_calls"] == 5 and c["budget_charged_this_attempt"]["episode_attempts"] == 1
    assert receipt["prep_commit"] == HEAD_A and receipt["card"] == final_tb.R2_CARD
    for index in ("1", "2"):  # both case receipts bound by path + hash
        row = receipt["cases"][index]
        assert Path(row["path"]).is_file() and row["sha256"] == final_tb.sha256_file(row["path"])
    ref = _reference_run(tmp_path)
    with pytest.raises(BindingError, match="STORAGE_BUDGET_NOT_PASSED"):  # nothing deleted, nothing lowered
        final_tb.run_release(out, "train", HEAD_A, "R-TB-E-0", reference_runs=[ref], free_bytes_fn=lambda p: 1 << 20)
    assert json.loads((out / "storage_budget.json").read_text())["passed"] is False
    with pytest.raises(BindingError):  # another prep can not use this receipt
        final_tb.run_release(out, "train", HEAD_B, "R-TB-E-0", free_bytes_fn=lambda p: 10 ** 13)
    token = final_tb.run_release(out, "train", HEAD_A, "R-TB-E-0", free_bytes_fn=lambda p: 10 ** 13)
    evidence = json.loads(Path(token).read_text())["evidence"]
    assert evidence["smoke_receipt_sha256"] == final_tb.sha256_file(out / "smoke_receipt.json") and "storage" in evidence
    assert final_tb.verify_token(str(token), out, "train", HEAD_A, "R-TB-E-0")
    with pytest.raises(BindingError):  # token is single-plan
        final_tb.verify_token(str(token), out, "train", HEAD_A, "R-TB-DK-1")
    with pytest.raises(BindingError):  # no second token for the same plan
        final_tb.run_release(out, "train", HEAD_A, "R-TB-E-0", free_bytes_fn=lambda p: 10 ** 13)
    with pytest.raises(BindingError):  # B2 waits for the +E first-update gate
        final_tb.run_release(out, "train", HEAD_A, "R-TB-DK-1", free_bytes_fn=lambda p: 10 ** 13)
    with pytest.raises(BindingError):  # B1-K waits for B2
        final_tb.run_release(out, "train", HEAD_A, "R-TB-K-1", free_bytes_fn=lambda p: 10 ** 13)
    tok = final_tb.run_release(out, "train", HEAD_A, "R-TB-DK-1", predecessor_blocked_reason="+E binding issue (test)",
                               free_bytes_fn=lambda p: 10 ** 13)
    assert "predecessor_blocked" in json.loads(Path(tok).read_text())["evidence"]
    final_tb.Ledger(out).reserve("R-TB-DK-1", "dk", tmp_path / "rdk", 1, 5)
    with pytest.raises(BindingError):  # a plan that is running (or ran) is never bypassed
        final_tb.run_release(out, "train", HEAD_A, "R-TB-K-1", predecessor_blocked_reason="x", free_bytes_fn=lambda p: 10 ** 13)
    # the live gate counts the bytes that unfinished plans still need; a shrunken disk blocks the next release
    live = final_tb.storage_gate(out, free_bytes_fn=lambda p: 10 ** 13)["live_check"]
    assert live["passed"] and set(live["per_plan_remaining_estimate"]) == set(final_tb.BASE_PLAN_TABLE)
    with pytest.raises(BindingError, match="STORAGE_BUDGET_NOT_PASSED"):
        final_tb.storage_gate(out, free_bytes_fn=lambda p: 1 << 20)
    assert all(json.loads(line)["time"] for line in (out / "storage_checks.jsonl").read_text().splitlines())


def test_10c_receipts_of_another_prep_or_registration_cannot_be_stitched(tmp_path):
    out_a, receipt_a, _ = _pass_pair(tmp_path, HEAD_A, "launch_a")
    # a second registration needs its own parent directory: the R2 smoke budget is issued exactly once per parent
    out_b = tmp_path / "other_parent" / "launch_b"
    r2.register_for_test(ROOT, out_b, HEAD_B)
    for name in ("smoke_receipt.json", "smoke_case1.json", "smoke_case2.json", "smoke_case1_events.jsonl", "smoke_case2_events.jsonl",
                 "smoke_budget.json"):
        shutil.copy(out_a / name, out_b / name)
    for prep in (HEAD_A, HEAD_B):
        with pytest.raises(BindingError):
            final_tb.verify_smoke_receipt(out_b, prep)
    with pytest.raises(BindingError):
        final_tb.run_release(out_b, "train", HEAD_B, "R-TB-E-0", reference_runs=[_reference_run(tmp_path)], free_bytes_fn=lambda p: 10 ** 13)
    # tampering with a bound case receipt after the PASS receipt also breaks verification
    case1 = out_a / "smoke_case1.json"
    doc = json.loads(case1.read_text())
    doc["engineering_test_tamper"] = True
    case1.write_text(json.dumps(doc))
    with pytest.raises(BindingError, match="missing or changed"):
        final_tb.verify_smoke_receipt(out_a, HEAD_A)


# 11 budget ------------------------------------------------------------------------------------------
def test_11_budgets_accumulate_and_re_registration_cannot_zero_history(tmp_path):
    assert final_tb.SMOKE_CAPS_R2 == {"episode_attempts": 2, "construction_attempts": 4, "explicit_resets": 4, "skill_calls": 23}
    assert final_tb.SMOKE_CAPS_CUMULATIVE == {"episode_attempts": 3, "construction_attempts": 6, "explicit_resets": 6, "skill_calls": 24}
    assert final_tb.SMOKE_PRIOR_CHARGED == {"episode_attempts": 1, "construction_attempts": 2, "explicit_resets": 2, "skill_calls": 1}
    doc = final_tb.budget_reconciliation(final_tb.SMOKE_PRIOR_CHARGED)
    assert doc["cumulative_hard_caps"] == final_tb.SMOKE_CAPS_CUMULATIVE and doc["refund"] == "NONE"
    assert doc["newly_authorised_by_r2"] == {"episode_attempts": 1, "construction_attempts": 2, "explicit_resets": 2, "skill_calls": 0}
    assert all(v["status"] == "TRANSFERRED_TO_R2" for v in doc["parent_unused_before_r2"].values())
    assert doc["expected_full_revalidation_skill_calls"] == 10 and doc["rl_attempts"]["r2_not_additional"] == 3
    with pytest.raises(BindingError):  # zeroed history would silently grant extra budget
        final_tb.budget_reconciliation({k: 0 for k in final_tb.SMOKE_PRIOR_CHARGED})
    base = {"caps": dict(final_tb.SMOKE_CAPS_R2), "used": {k: 0 for k in final_tb.SMOKE_CAPS_R2}, "episodes": [],
            "prior_charged": dict(final_tb.SMOKE_PRIOR_CHARGED), "cumulative_caps": dict(final_tb.SMOKE_CAPS_CUMULATIVE)}
    final_tb.preflight_smoke_budget(base)
    inflated = {**base, "caps": {**base["caps"], "episode_attempts": 3}}
    with pytest.raises(final_tb.BudgetExceeded):  # prior 1 + local 3 > cumulative 3
        final_tb.preflight_smoke_budget(inflated)
    spent = {**base, "used": {**base["used"], "skill_calls": 12}}
    with pytest.raises(final_tb.BudgetExceeded):  # 12 + 12 > the local cap of 23 is refused before any construction
        final_tb.preflight_smoke_budget(spent)
    forgotten = {**base, "prior_charged": {k: 0 for k in base["prior_charged"]}}
    with pytest.raises(final_tb.BudgetExceeded):  # a budget file that forgets the parent's charges is refused
        final_tb.preflight_smoke_budget(forgotten)
    # registration: prior evidence stays byte-identical, only one R2 registration per parent, never inside the prior tree
    prior = ROOT / final_tb.PRIOR_LAUNCH_REL
    before = final_tb.tree_manifest(prior)["digest_sha256"]
    out = tmp_path / "launch" / "first"
    r2.register_for_test(ROOT, out, HEAD_A)
    assert final_tb.tree_manifest(prior)["digest_sha256"] == before
    budget = json.loads((out / "smoke_budget.json").read_text())
    assert budget["caps"] == final_tb.SMOKE_CAPS_R2 and budget["prior_charged"] == final_tb.SMOKE_PRIOR_CHARGED and budget["used"]["skill_calls"] == 0
    with pytest.raises(BindingError, match="already exists"):
        r2.register_for_test(ROOT, tmp_path / "launch" / "second", HEAD_B)
    with pytest.raises(BindingError):
        r2.register_for_test(ROOT, prior / "inside", HEAD_B)
    assert not (prior / "inside").exists()
    amendment = json.loads((out / "amendment_authorization.json").read_text())
    assert amendment["card"] == final_tb.R2_CARD and amendment["parent_card"] == final_tb.PARENT_CARD
    assert amendment["case_order"] == ["T_B_dev_00", "T_B_dev_04"] and "Family B" in amendment["not_authorized"]
    auth = json.loads((out / "authorization.json").read_text())
    assert auth["prior_evidence_tree_digest_sha256"] == before and auth["card"] == final_tb.R2_CARD
    # the smoke pair on this registration charges 2 episodes / 10 skills and never exceeds the cumulative caps
    res1, _ = r2.run_smoke_with_doubles(ROOT, out, 1, HEAD_A, tmp_path / "d1")
    res2, _ = r2.run_smoke_with_doubles(ROOT, out, 2, HEAD_A, tmp_path / "d2")
    used = json.loads((out / "smoke_budget.json").read_text())["used"]
    assert used["episode_attempts"] == 2 and used["skill_calls"] == 10
    for key, cap in final_tb.SMOKE_CAPS_CUMULATIVE.items():
        assert final_tb.SMOKE_PRIOR_CHARGED[key] + used[key] <= cap
    assert res1["counters"]["internal_reset_calls_measured"] in (0, "NOT_MEASURED")  # measured separately, never a guess


def test_11b_prior_reconciliation_states_only_what_the_old_files_show():
    doc, facts = final_tb.reconcile_prior(ROOT / final_tb.PRIOR_LAUNCH_REL)
    assert doc["previous_smoke_label"] == "TB_LAUNCH_BLOCKED_RUNTIME"
    assert (doc["previous_episode_attempts_charged"], doc["previous_construction_attempts_charged"],
            doc["previous_explicit_resets_charged"], doc["previous_skill_budget_charged"]) == (1, 2, 2, 1)
    assert doc["previous_completed_transitions"] == 0
    assert doc["previous_actual_executor_calls"] == "UNVERIFIED_FROM_PERSISTED_RECEIPT"
    assert doc["previous_failure_phase"] == "UNVERIFIED_FROM_PERSISTED_TRACEBACK"
    assert doc["code_defect_confirmed"] == "SCRIPT_TARGET_APPLIED_DURING_NEXT_VALUE_PROBE"
    assert doc["cpu_controlled_reproduction_phase"] == "AFTER_ONE_FAKE_EXECUTION_DURING_NEXT_VALUE_PROBE"
    assert doc["interpretation"]["empty_steps_or_no_transition_means_executor_not_called"] is False
    assert doc["interpretation"]["refund"] == "NONE" and facts["rl_used"] == 0


def test_11c_old_registration_cannot_release_and_out_is_always_absolute(tmp_path, monkeypatch):
    old = tmp_path / "old"
    old.mkdir()
    (old / "authorization.json").write_text(json.dumps({"card": final_tb.PARENT_CARD, "prep_commit": HEAD_A}))
    with pytest.raises(BindingError, match="only the"):
        final_tb._registered_prep(old)
    seen = {}

    def fake_register(root, out, prep, text, stamp=None, git=None, prior_out=None):
        seen["out"], seen["prior_out"] = out, prior_out
        return {"smoke_token": "x"}

    text = tmp_path / "auth.txt"
    text.write_text("x")
    monkeypatch.chdir(tmp_path)
    with mock.patch.object(final_tb, "run_register", fake_register):
        assert final_tb.main(["register", "--root", str(ROOT), "--out", "relative_out", "--prep-commit", HEAD_A,
                              "--authorization-text-file", str(text), "--prior-out", "prior_rel"]) == 0
    assert Path(seen["out"]).is_absolute() and Path(seen["out"]) == tmp_path / "relative_out"
    assert Path(seen["prior_out"]).is_absolute()
    args = final_tb.build_parser().parse_args(["release", "--out", "o", "--stage", "train", "--plan", "R-TB-E-0", "--reference-run", "r1"])
    assert args.reference_run == ["r1"]


# 12 training path ------------------------------------------------------------------------------------
def test_12_training_uses_the_plain_policy_never_the_scripted_selector(tmp_path):
    from cp_disr import neural, stage2a_v11 as v11
    for fn in (final_tb.run_train, v11.train_job):
        src = inspect.getsource(fn)
        assert "ScriptedSelector" not in src and "install_smoke_instrumentation" not in src
    users = [p.name for p in (ROOT / "src/cp_disr").glob("*.py") if "ScriptedSelector" in p.read_text()]
    assert users == ["final_tb.py"], users
    smoke_users = [name for name, fn in inspect.getmembers(final_tb, inspect.isfunction)
                   if name != "install_smoke_instrumentation" and "install_smoke_instrumentation" in inspect.getsource(fn)]
    assert smoke_users == ["run_smoke"], smoke_users
    ctx = chk.make_context(tmp_path, "R-TB-E-0")
    obs = chk.run_until_first_step(ROOT, ctx)  # the real train_job up to the first Collector.step
    assert obs["reached_first_step"] and obs["collector_policy_type"] is neural.Policy


def test_12b_event_log_is_line_delimited_json_and_never_replaces_the_original_error(tmp_path):
    log = final_tb.SmokeEventLog(tmp_path / "e.jsonl")
    for i in range(3):
        log.emit("X", i=i)
    assert [json.loads(line)["i"] for line in (tmp_path / "e.jsonl").read_text().splitlines()] == [0, 1, 2]
    broken = final_tb.SmokeEventLog(tmp_path / "missing_dir" / "e.jsonl")
    with mock.patch("os.fsync", side_effect=OSError("disk full")):
        broken.emit("Y")  # a logging failure is recorded, not raised
    assert broken.errors and "disk full" in broken.errors[0]
