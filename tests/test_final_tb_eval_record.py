"""Offline tests for src/cp_disr/final_tb_eval_record.py (CP-DISR-TB-EVAL-PREP-LITE-01).

0 environment, 0 episode, 0 GPU, 0 optimizer.  The recorder and the OLD production loop
(`stage2a_v11.eval_episodes`) are both driven by the same scripted collector so that the old row fields
are compared against the real old code, not against a transcription.
"""
from __future__ import annotations

import ast
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest

from cp_disr import final_tb_eval_record as rec

ROOT = Path(__file__).resolve().parents[1]
BASELINE = "241159c1ceca3b53f0ba8423879524c6b2668e07"
PROTECTED = (
    "src/cp_disr/stage2a_v11.py", "src/cp_disr/collector.py", "src/cp_disr/neural.py", "src/cp_disr/torch_rl.py",
    "src/cp_disr/rl.py", "src/cp_disr/persistence.py", "src/cp_disr/phase_a_v12.py", "src/cp_disr/final_tb.py",
    "src/cp_disr/platforms/libero/runtime_factory.py", "src/cp_disr/platforms/libero/skill_executor.py",
    "src/cp_disr/platforms/libero/verifier.py", "src/cp_disr/platforms/libero/task_evaluator.py",
    "src/cp_disr/platforms/libero/safety.py", "src/cp_disr/platforms/libero/clock.py",
)


# ----------------------------------------------------------------------------- scripted fakes
class Snap:
    def __init__(self, decision_id, clock, env_id="e0", episode_id="ep-1"):
        self.decision_id, self.clock_seconds, self.env_id, self.episode_id = decision_id, clock, env_id, episode_id


class Bundle:
    def __init__(self, start=100.0):
        self.episode_start_seconds = start
        self.environment = SimpleNamespace(close=lambda: None)
        self.template = object()
        self.executor = SimpleNamespace(last={"controller_exit": "NORMAL_TERMINATION"})
        self.starts = []

    def start_case(self, case):
        self.starts.append(case)
        self.episode_start_seconds = 100.0 + 10.0 * len(self.starts)
        return Snap(0, self.episode_start_seconds)


def step(cand, dur, reward=0.0, weight=1.0, success=False, term=False, trunc=False, reason="CONTINUE", extra=0.0, result_dict=False):
    return {"kind": "step", "cand": cand, "dur": dur, "reward": reward, "weight": weight, "success": success,
            "term": term, "trunc": trunc, "reason": reason, "extra": extra, "result_dict": result_dict}


def none(result):
    return {"kind": "none", "result": result}


class Scripted:
    """Collector stand-in: replays a per-case script; clock bookkeeping mirrors the production arithmetic."""

    def __init__(self, bundle, policy, scripts):
        self.bundle, self.scripts, self.queue, self.clock = bundle, scripts, None, None
        self.last_execution = None
        self.after_step = None

    def reset_episode(self, env_id, episode_id):
        case = self.bundle.starts[-1]
        self.queue = list(self.scripts[case])
        self.clock = float(self.bundle.episode_start_seconds)

    def step(self, snapshot, deterministic=False):
        assert deterministic is True
        spec = self.queue.pop(0)
        if self.after_step:
            self.after_step()
        if spec["kind"] == "none":
            return None, spec["result"]
        self.last_execution = {"controller_exit": "NORMAL_TERMINATION"}
        start, end = snapshot.clock_seconds, snapshot.clock_seconds + spec["dur"] + spec["extra"]
        nxt = Snap(snapshot.decision_id + 1, end)
        t = SimpleNamespace(snapshot=snapshot, next_snapshot=nxt, selected_candidate_id=spec["cand"], reward=spec["reward"],
                            duration=spec["dur"], weight=spec["weight"], terminated=spec["term"], truncated=spec["trunc"],
                            reason=spec["reason"])
        res = {"success": spec["success"], "reason": spec["reason"]} if spec["result_dict"] else SimpleNamespace(success=spec["success"], reason=spec["reason"])
        return t, res


class Fake:
    """Minimal stage2a_v11 stand-in for the recorder orchestration (no torch, no environment)."""

    def __init__(self, scripts, bundle=None, gate=None):
        self.scripts = scripts
        self.bundle = bundle or Bundle()
        self.rng = {"calls": []}
        self.s1 = SimpleNamespace(capture_rng=lambda: "RNG", restore_rng=lambda s: self.rng["calls"].append(("restore", s)),
                                  seed_all=lambda n: self.rng["calls"].append(("seed", n)))
        self.ENABLED_SPLITS = {"T_B": "split.json"}
        self.collector = None

    # production-function stand-ins
    def bind_H(self, root):
        return {}

    def make_bundle(self, root, task, split):
        return self.bundle

    def resolve_runtime(self, root):
        return {"task_deadlines": {"T_B": 60.0}}

    def attach_split_cases(self, *a):
        pass

    def make_policy(self, template, method, device):
        return SimpleNamespace(eval=lambda: None)

    def apply_episode_prior(self, method, bundle, snap, sampler=None, eval_original=False):
        return snap, SimpleNamespace(audit_mode="absent"), 0

    def require_case_cache(self, root, rec_):
        return None

    def method_dir_name(self, m):
        return m

    def suite_half_life(self):
        return 23.1

    def collector_cls(self, bundle, policy):
        self.collector = Scripted(bundle, policy, self.scripts)
        return self.collector


def make_model(tmp_path, payload=b"weights-v1"):
    p = tmp_path / "final_n_000010_u1.pt"
    p.write_bytes(payload)
    p.with_suffix(".json").write_text(json.dumps({"sha256": hashlib.sha256(payload).hexdigest(),
                                                   "manifest": {"N": 10, "T": 12.5, "complete_updates": 1, "final": True}}))
    return p


def run_new(tmp_path, scripts, cases, **kw):
    fake = Fake(scripts)
    ckpt = make_model(tmp_path)
    out = tmp_path / "eval.json"
    payload = rec.evaluate_checkpoint_recorded(
        ROOT, "T_B", "B1-K", ckpt, cases, {c: {"case_id": c} for c in cases}, "cpu", {"git_commit": "x"}, out,
        label="unit", v11=fake, collector_cls=fake.collector_cls, load_checkpoint_fn=kw.pop("load", lambda p, m: None), **kw)
    return payload, fake, out, ckpt


FIVE_STEP = [step("PICK_A", 0.6, weight=1.0), step("PLACE_BUFFER_A", 0.6, weight=0.9, extra=0.05), step("PICK_B", 0.6, weight=0.8),
             step("PLACE_A", 0.6, weight=0.7), step("PLACE_BUFFER_B", 0.5, reward=0.5, weight=0.6, success=True, term=True, reason="TASK_SUCCESS")]


# ----------------------------------------------------------------------------- recorder behaviour
def test_multistep_success_actions_clock_and_first_success_once(tmp_path):
    scripts = {"c0": FIVE_STEP[:4] + [step("PLACE_BUFFER_B", 0.5, reward=0.5, success=True, reason="CONTINUE"),
                                       step("EXTRA", 0.2, reward=0.0, success=True, term=True, reason="TASK_SUCCESS")]}
    payload, fake, out, _ = run_new(tmp_path, scripts, ["c0"])
    row = payload["rows"][0]
    assert payload["schema_version"] == rec.SCHEMA_VERSION
    acts = row["actions"]
    assert [a["candidate_id"] for a in acts] == ["PICK_A", "PLACE_BUFFER_A", "PICK_B", "PLACE_A", "PLACE_BUFFER_B", "EXTRA"]
    assert [a["decision_index"] for a in acts] == list(range(6))
    start = row["episode_start_clock_s"]
    assert start == 110.0
    assert acts[0]["clock_start"] == start and acts[0]["clock_end"] == pytest.approx(start + 0.6)
    for prev, nxt in zip(acts, acts[1:]):
        assert nxt["clock_start"] == prev["clock_end"]
    # first independently confirmed success is the 5th action (index 4), recorded once even though the 6th also reports success
    assert row["first_confirmed_success_decision_index"] == 4
    assert row["time_to_first_confirmed_success_s"] == pytest.approx(acts[4]["clock_end"] - start)
    assert row["time_to_first_confirmed_success_label"] == rec.TIME_TO_SUCCESS_LABEL
    assert [a["success_confirmed"] for a in acts] == [False, False, False, False, True, True]
    # clock / duration accounting: the 50 ms gap in step 2 is reported, never silently zero
    assert row["inter_step_clock_gap_s"] == pytest.approx(0.05)
    assert sum(a["duration"] for a in acts[:5]) + row["inter_step_clock_gap_s"] == pytest.approx(row["time_to_first_confirmed_success_s"])
    assert row["episode_end_elapsed_s"] == pytest.approx(acts[-1]["clock_end"] - start)
    assert row["success_seconds"] == pytest.approx(0.2)  # old field: LAST successful skill duration only
    assert row["success_seconds_label"] == rec.SUCCESS_SECONDS_LABEL
    assert all(a["controller_exit"] == "NORMAL_TERMINATION" for a in acts)
    assert set(acts[0]) == set(rec.ACTION_FIELDS)


def test_success_on_single_confirmed_step_time_matches_end(tmp_path):
    payload, _, _, _ = run_new(tmp_path, {"c0": FIVE_STEP}, ["c0"])
    row = payload["rows"][0]
    assert row["success"] is True and row["steps"] == 5 and row["reason"] == "TASK_SUCCESS"
    assert row["time_to_first_confirmed_success_s"] == pytest.approx(row["episode_end_elapsed_s"])
    assert row["success_seconds"] == pytest.approx(0.5)
    assert row["G"] == pytest.approx(0.5 * 0.6)  # reward x weight, as in the old loop


def test_deadline_exit_is_normal_failure(tmp_path):
    ev = SimpleNamespace(success=False, reason="DEADLINE", terminated=True, truncated=False)
    scripts = {"c0": [step("PICK_A", 20.0), step("PLACE_BUFFER_A", 20.0), step("PICK_B", 19.9), none(ev)]}
    row = run_new(tmp_path, scripts, ["c0"])[0]["rows"][0]
    assert row["success"] is False and row["reason"] == "DEADLINE" and row["steps"] == 3
    assert row["time_to_first_confirmed_success_s"] is None and row["first_confirmed_success_decision_index"] is None
    assert row["success_seconds"] is None and row["inter_step_clock_gap_s"] is None
    assert row["no_transition_exit"]["reason"] == "DEADLINE" and row["no_transition_exit"]["result_kind"] == "SimpleNamespace"
    assert row["episode_end_elapsed_s"] == pytest.approx(59.9)
    assert row["outcome_class"] == rec.NORMAL_POLICY_OUTCOME


def test_no_candidate_exit(tmp_path):
    scripts = {"c0": [step("PICK_A", 0.6), none({"terminated": True, "reason": "NO_SAFE_CANDIDATES", "no_transition": True})]}
    row = run_new(tmp_path, scripts, ["c0"])[0]["rows"][0]
    assert row["success"] is False and row["steps"] == 1 and row["reason"] == "NO_SAFE_CANDIDATES"
    assert row["no_transition_exit"]["reason"] == "NO_SAFE_CANDIDATES" and row["no_transition_exit"]["result_kind"] == "dict"
    assert row["outcome_class"] == rec.NORMAL_POLICY_OUTCOME


def test_insufficient_remaining_control_cycle_exit(tmp_path):
    res = {"terminated": True, "truncated": True, "reason": "INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE", "no_transition": True,
           "success": False, "elapsed": 59.97, "remaining": 0.03}
    scripts = {"c0": [step("PICK_A", 30.0), step("PLACE_BUFFER_A", 29.97), none(res)]}
    row = run_new(tmp_path, scripts, ["c0"])[0]["rows"][0]
    assert row["reason"] == "INSUFFICIENT_REMAINING_FOR_CONTROL_CYCLE" and row["success"] is False
    assert row["no_transition_exit"]["elapsed_s"] == pytest.approx(59.97) and row["no_transition_exit"]["remaining_s"] == pytest.approx(0.03)
    assert row["episode_end_elapsed_s"] == pytest.approx(59.97)


def test_case_order_and_rng_isolation_recorded(tmp_path):
    scripts = {c: FIVE_STEP for c in ("c2", "c0", "c1")}
    payload, fake, _, _ = run_new(tmp_path, scripts, ["c2", "c0", "c1"], n_episodes=2)
    assert payload["case_order"] == ["c2", "c0"] and [r["case_id"] for r in payload["rows"]] == ["c2", "c0"]
    assert payload["isolated_rng"] is True and payload["eval_rng_isolation"] == {"rng_captured_before": True, "rng_restored_after": True, "seed_all": 0}
    assert ("seed", 0) in fake.rng["calls"] and ("restore", "RNG") in fake.rng["calls"]
    assert payload["optimizer_steps"] == 0 and payload["eval_action"] == "deterministic_argmax"


def test_payload_identity_blocks(tmp_path):
    payload, _, _, ckpt = run_new(tmp_path, {"c0": FIVE_STEP}, ["c0"])
    assert payload["evaluated_model"] == {"path": str(ckpt), "bytes": ckpt.stat().st_size, "sha256": hashlib.sha256(ckpt.read_bytes()).hexdigest()}
    assert payload["model_sha256_after_last_episode"] == payload["evaluated_model"]["sha256"]
    assert payload["evaluated_generation"]["N"] == 10 and payload["evaluated_generation"]["update"] == 1
    assert payload["evaluated_generation"]["T"] == 12.5 and payload["evaluated_generation"]["final"] is True
    assert payload["hashes"] == {"git_commit": "x"} and payload["status"] == "COMPLETE"


# ----------------------------------------------------------------------------- old-field invariance against the REAL old loop
def _old_rows(tmp_path, scripts, cases):
    import cp_disr.stage2a_v11 as v11
    fake = Fake(scripts)
    ckpt = make_model(tmp_path / "old") if (tmp_path / "old").mkdir() is None else None
    out = tmp_path / "old" / "eval_old.json"
    with mock.patch.multiple(v11, bind_H=fake.bind_H, make_bundle=fake.make_bundle, resolve_runtime=fake.resolve_runtime,
                             attach_split_cases=fake.attach_split_cases, make_policy=fake.make_policy,
                             apply_episode_prior=fake.apply_episode_prior, require_case_cache=fake.require_case_cache,
                             suite_half_life=fake.suite_half_life, ENABLED_SPLITS=fake.ENABLED_SPLITS, s1=fake.s1), \
            mock.patch("cp_disr.collector.Collector", fake.collector_cls), \
            mock.patch("cp_disr.torch_rl.load_checkpoint", lambda p, m: None):
        return v11.eval_episodes(ROOT, "T_B", "B1-K", ckpt, cases, {c: {"case_id": c} for c in cases}, "cpu", {}, out, label="unit")


def test_old_fields_identical_to_real_old_eval_loop(tmp_path):
    ev = SimpleNamespace(success=False, reason="DEADLINE", terminated=True, truncated=False)
    scripts = {
        "ok": FIVE_STEP,
        "late": [step("PICK_A", 20.0), step("PLACE_BUFFER_A", 20.0), step("PICK_B", 19.9), none(ev)],
        "nocand": [step("PICK_A", 0.6), none({"terminated": True, "reason": "NO_SAFE_CANDIDATES", "no_transition": True})],
        "trunc": [step("PICK_A", 0.6, term=True, trunc=True, reason="TIMEOUT_X")],
        "dictres": [step("PICK_A", 0.6, result_dict=True), step("PLACE_BUFFER_B", 0.4, reward=0.4, success=True, term=True, reason="TASK_SUCCESS", result_dict=True)],
    }
    cases = list(scripts)
    old = _old_rows(tmp_path, scripts, cases)
    new, _, _, _ = run_new(tmp_path, scripts, cases)
    assert len(old["rows"]) == len(new["rows"]) == 5
    for o, n in zip(old["rows"], new["rows"]):
        for k in rec.OLD_ROW_FIELDS:
            assert o[k] == n[k], (o["case_id"], k, o[k], n[k])
        assert set(o) == set(rec.OLD_ROW_FIELDS)
    for k in ("success_n", "n", "success_rate", "mean_discounted_return", "isolated_rng", "H", "eval_action", "label", "optimizer_steps", "task", "method"):
        assert old[k] == new[k], k


# ----------------------------------------------------------------------------- fail-closed behaviour
def test_registered_sha_mismatch_fails_before_load(tmp_path):
    loaded = []
    with pytest.raises(rec.ModelIdentityError):
        run_new(tmp_path, {"c0": FIVE_STEP}, ["c0"], expected_sha256="0" * 64, load=lambda p, m: loaded.append(p))
    assert loaded == []


def test_model_hash_change_during_evaluation_fails_closed(tmp_path):
    fake = Fake({"c0": FIVE_STEP, "c1": FIVE_STEP})
    ckpt = make_model(tmp_path)
    original = fake.collector_cls

    def tampering(bundle, policy):
        col = original(bundle, policy)
        col.after_step = lambda: ckpt.write_bytes(b"tampered")
        return col

    out = tmp_path / "eval.json"
    with pytest.raises(rec.ModelIdentityError):
        rec.evaluate_checkpoint_recorded(ROOT, "T_B", "B1-K", ckpt, ["c0", "c1"], {"c0": {"case_id": "c0"}, "c1": {"case_id": "c1"}}, "cpu", {}, out,
                                         label="unit", v11=fake, collector_cls=tampering, load_checkpoint_fn=lambda p, m: None)
    doc = json.loads(out.read_text())
    assert doc["status"] == "ABORTED_MODEL_HASH_CHANGED" and "model_sha256_after_last_episode" not in doc


def test_infrastructure_failure_stops_and_is_not_retried(tmp_path):
    class Boom(Scripted):
        def step(self, snapshot, deterministic=False):
            if self.bundle.starts[-1] == "c1":
                raise RuntimeError("EGL context lost")
            return super().step(snapshot, deterministic)

    fake = Fake({"c0": FIVE_STEP, "c1": FIVE_STEP, "c2": FIVE_STEP})
    ckpt = make_model(tmp_path)
    out = tmp_path / "eval.json"
    with pytest.raises(rec.InfrastructureFailure) as err:
        rec.evaluate_checkpoint_recorded(ROOT, "T_B", "B1-K", ckpt, ["c0", "c1", "c2"], {c: {"case_id": c} for c in ("c0", "c1", "c2")}, "cpu", {}, out,
                                         label="unit", v11=fake, collector_cls=lambda b, p: Boom(b, p, fake.scripts), load_checkpoint_fn=lambda p, m: None)
    assert err.value.partial_path == str(out)
    doc = json.loads(out.read_text())
    assert doc["status"] == "INFRASTRUCTURE_FAILURE_STOPPED" and doc["infrastructure_failure"]["case_id"] == "c1"
    assert [r["case_id"] for r in doc["rows"]] == ["c0"] and fake.bundle.starts == ["c0", "c1"]  # c2 never started, c1 never re-run


def test_environment_closed_and_rng_restored_on_failure(tmp_path):
    closed = []
    fake = Fake({"c0": []})
    fake.bundle.environment = SimpleNamespace(close=lambda: closed.append(1))
    fake.bundle.start_case = lambda case: (_ for _ in ()).throw(RuntimeError("scene build failed"))
    ckpt = make_model(tmp_path)
    with pytest.raises(rec.InfrastructureFailure):
        rec.evaluate_checkpoint_recorded(ROOT, "T_B", "B1-K", ckpt, ["c0"], {"c0": {"case_id": "c0"}}, "cpu", {}, tmp_path / "o.json", label="unit",
                                         v11=fake, collector_cls=fake.collector_cls, load_checkpoint_fn=lambda p, m: None)
    assert closed == [1] and ("restore", "RNG") in fake.rng["calls"]


def test_normal_policy_outcomes_never_become_infrastructure_failures(tmp_path):
    scripts = {"a": [none({"terminated": True, "reason": "NO_SAFE_CANDIDATES", "no_transition": True})],
               "b": [none(SimpleNamespace(success=False, reason="DEADLINE"))]}
    payload, _, _, _ = run_new(tmp_path, scripts, ["a", "b"])
    assert payload["status"] == "COMPLETE" and payload["n"] == 2 and payload["success_n"] == 0


# ----------------------------------------------------------------------------- readers / old schema
def test_old_schema_rows_stay_unverified_and_never_enter_time_column():
    old = {"case_id": "T_B_dev_00", "success": True, "G": 0.5, "steps": 5, "reason": "TASK_SUCCESS", "success_seconds": 2.9, "source_n": 0, "prior_mode": "absent"}
    norm = rec.normalize_old_row(old)
    assert norm["episode_time_status"] == rec.UNVERIFIED and norm["time_to_first_confirmed_success_s"] is None
    assert norm["success_seconds_label"] == rec.SUCCESS_SECONDS_LABEL and norm["success_seconds"] == 2.9
    assert rec.time_to_success_column([old]) == [None]
    new_row = {"case_id": "x", "time_to_first_confirmed_success_s": 3.5, "time_to_first_confirmed_success_label": rec.TIME_TO_SUCCESS_LABEL}
    assert rec.time_to_success_column([old, new_row]) == [None, 3.5]


def test_labels_are_the_frozen_strings():
    assert rec.SUCCESS_SECONDS_LABEL == "FINAL_SKILL_DURATION_ONLY"
    assert rec.TIME_TO_SUCCESS_LABEL == "EPISODE_START_TO_FIRST_INDEPENDENT_CONFIRMED_SUCCESS_SIM_SECONDS"
    assert rec.SCHEMA_VERSION == "eval_row_v2"


# ----------------------------------------------------------------------------- static / identity guards
def test_module_imports_no_optimizer_and_writes_no_training_artifact():
    src = (ROOT / "src/cp_disr/final_tb_eval_record.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(a.name for a in node.names)
        elif isinstance(node, ast.ImportFrom):
            imported.add((node.module or "") + ":" + ",".join(a.name for a in node.names))
    joined = " ".join(sorted(imported))
    for banned in ("torch.optim", "PPO", "save_checkpoint", "Adam", "stage2a_runner", "provider", "vlm"):
        assert banned not in joined, (banned, joined)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)} | {n.attr for n in ast.walk(tree) if isinstance(n, ast.Attribute)}
    for banned in ("PPO", "Adam", "save_checkpoint", "save_ckpt", "zero_grad", "backward", "step_optimizer"):
        assert banned not in names, banned
    assert "load_checkpoint" in joined  # same loader as the old evaluation (hash check + strict state_dict load)
    # the only write is the atomic eval json
    assert src.count("write_text(") == 1 and src.count("os.replace(") == 1


def test_evaluation_writes_only_its_eval_file(tmp_path):
    before = set(p.name for p in tmp_path.iterdir())
    payload, _, out, ckpt = run_new(tmp_path, {"c0": FIVE_STEP}, ["c0"])
    created = set(p.name for p in tmp_path.iterdir()) - before
    assert created == {out.name, ckpt.name, ckpt.with_suffix(".json").name}  # model + sidecar were created by the test helper itself


@pytest.mark.skipif(subprocess.run(["git", "cat-file", "-e", BASELINE + "^{commit}"], cwd=str(ROOT), capture_output=True).returncode != 0,
                    reason="baseline commit not available")
def test_protected_production_files_unchanged_vs_baseline():
    for rel in PROTECTED:
        want = subprocess.check_output(["git", "rev-parse", "%s:%s" % (BASELINE, rel)], cwd=str(ROOT), text=True).strip()
        got = subprocess.check_output(["git", "hash-object", rel], cwd=str(ROOT), text=True).strip()
        assert want == got, rel


def test_recorder_loads_standalone_by_file_path():
    import importlib.util
    spec = importlib.util.spec_from_file_location("standalone_eval_record", ROOT / "src/cp_disr/final_tb_eval_record.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.SCHEMA_VERSION == "eval_row_v2"


def test_no_prior_case_gate_rejects_cache_pointers_and_reads_nothing():
    assert rec.no_prior_case_gate(ROOT, {"case_id": "c0", "source_cache_ref_audit_only": {"cache_dir": "x"}}) is None
    for key in ("cache_dir", "cache_key", "cache_status"):
        with pytest.raises(rec.EvalRecorderError):
            rec.no_prior_case_gate(ROOT, {"case_id": "c0", key: "x"})
