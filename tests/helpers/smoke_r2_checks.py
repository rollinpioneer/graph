"""Behavioural checks for the T_B smoke selector repair (CP-DISR-TB-SMOKE-SELECTOR-REPAIR-R2-1).

The checks import the ACTUAL repaired selector (`cp_disr.final_tb.ScriptedSelector`) and drive the ACTUAL
`cp_disr.collector.Collector.step`.  Nothing of these two is mocked.  Only the physical boundary is replaced by
deterministic test doubles: the clock, the skill executor, observation/perception/verifier/evaluator/safety and the
successor snapshot builder.  The policy is the PRODUCTION `neural.Policy` (B1-K) built on the REAL T_B template and run
on CPU; no MuJoCo/EGL object is constructed and no optimizer step is taken.

Every state below is engineering-only test data (public facts of the T_B_dev_00 initial state copied from the persisted
smoke receipt, later states hand-written).  None of it is a training sample or a performance result.
Plain functions (`check_*(root, tmp)`) so that pytest and the mutation driver can call them.
"""
from __future__ import annotations

import contextlib
import json
import math
from pathlib import Path
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[2]

IDS = ("a:OPEN:container:v1", "a:PICK:second_object:v1", "a:PICK:target:v1", "a:PLACE:second_object:container:v1",
       "a:PLACE:target:container:v1", "a:PLACE_BUFFER:second_object:buffer:v1", "a:PLACE_BUFFER:target:buffer:v1")
SCRIPT = (IDS[1], IDS[5], IDS[0], IDS[2], IDS[4])  # the frozen five-action scripts of T_B_dev_00 and T_B_dev_04
INITIAL = {"p:AtBuffer:second_object:buffer": "FALSE", "p:AtBuffer:target:buffer": "FALSE", "p:GripperEmpty": "TRUE",
           "p:Held:second_object": "FALSE", "p:Held:target": "FALSE", "p:Inside:second_object:container": "FALSE",
           "p:Inside:target:container": "FALSE", "p:OnTable:second_object": "TRUE", "p:OnTable:target": "TRUE",
           "p:Open:container": "FALSE"}


def _facts_chain():
    s0 = dict(INITIAL)
    s1 = {**s0, "p:Held:second_object": "TRUE", "p:GripperEmpty": "FALSE", "p:OnTable:second_object": "FALSE"}
    s2 = {**s1, "p:AtBuffer:second_object:buffer": "TRUE", "p:Held:second_object": "FALSE", "p:GripperEmpty": "TRUE"}
    s3 = {**s2, "p:Open:container": "TRUE"}
    s4 = {**s3, "p:Held:target": "TRUE", "p:GripperEmpty": "FALSE", "p:OnTable:target": "FALSE"}
    s5 = {**s4, "p:Inside:target:container": "TRUE", "p:Held:target": "FALSE", "p:GripperEmpty": "TRUE"}
    return [s0, s1, s2, s3, s4, s5]


T, F = True, False
# Controlled fixtures (NOT reconstructed real states): after each action the action just executed is illegal
# (or not applicable) in the successor while other candidates are legal -- the situation that exposed the old defect.
MASKS = ((T, T, T, F, F, F, F), (F, F, F, T, F, T, F), (T, T, T, F, F, F, F), (F, T, T, F, F, F, F),
         (F, F, F, F, T, F, T), (F, F, F, F, F, F, F))


def make_template(root):
    from tests.helpers.final_tb_checks import make_template as real_template
    return real_template(root)


def build_state(template, decision_id, facts, mask, base_nan=False):
    from cp_disr.common import digest
    from cp_disr.facts import FactRecord, FactStore, Truth
    from cp_disr.rl import Snapshot
    from cp_disr.stage1a_v11 import CAND_DIM, OBS_DIM
    records = tuple(FactRecord(k, Truth(v), 0.0, 0.0, ("engineering_test_only",), Truth.TRUE, 0.0, "ENGINEERING_TEST_ONLY", 0.9)
                    for k, v in sorted(facts.items()))
    base = tuple(0.01 * ((i + decision_id) % 7) for i in range(OBS_DIM))
    if base_nan:
        base = (float("nan"),) + base[1:]
    feats = tuple(tuple(float((i + j) % 3) for j in range(CAND_DIM)) for i in range(len(IDS)))
    return Snapshot("ENG_ENV", "ENG_EP", decision_id, template, FactStore(records), IDS, tuple(bool(m) for m in mask), (),
                    digest(()), base, feats, "engineering_test_only:%d" % decision_id, 0.5 * decision_id,
                    synthetic_unit_fixture=False)


# ------------------------------------------------------------------------------------------ physical-boundary doubles
class FakeClock:
    def __init__(self):
        self.t = 0.0

    def now_seconds(self):
        return self.t

    def duration_seconds(self, start, end):
        return float(end - start)


class FakeExecutor:
    """Counts its own calls independently of the production counting wrapper."""

    def __init__(self, clock, raise_on=None):
        self.clock, self.calls, self.last, self.raise_on = clock, [], None, raise_on

    def execute(self, candidate, timeout):
        self.calls.append(candidate)
        if self.raise_on is not None and len(self.calls) == self.raise_on:
            raise RuntimeError("EXECUTOR_BOUNDARY_FAILURE")
        start = self.clock.t
        self.clock.t += 0.5
        self.last = {"controller_exit": "NORMAL_TERMINATION", "steps": 10, "sim_duration": 0.5, "raw_sim_start": start,
                     "raw_sim_end": self.clock.t, "timeout": False, "interrupted": False, "nan_blocked": False}
        return SimpleNamespace(controller_exit="NORMAL_TERMINATION", evidence_ids=())


class Rig:
    """Real Collector + real selector + production Policy on the real T_B template; doubles only at the physical boundary."""

    def __init__(self, root, tmp, masks=MASKS, terminal_after=5, nan_base_at=(), builder_fault=None, executor_raise_on=None,
                 instrument=True):
        import torch
        from cp_disr import final_tb, rl, stage2a_v11 as v11
        from cp_disr.collector import Collector
        rl.set_suite_half_life(final_tb.H_SECONDS)
        self.torch, self.final_tb = torch, final_tb
        self.template = make_template(root)
        assert {c.id for c in self.template.contracts} >= set(IDS), "real T_B template must contain the smoke candidates"
        chain = _facts_chain()
        self.states = [build_state(self.template, i, chain[min(i, len(chain) - 1)], masks[min(i, len(masks) - 1)],
                                   base_nan=(i in nan_base_at)) for i in range(len(masks))]
        self.clock = FakeClock()
        self.executor = FakeExecutor(self.clock, raise_on=executor_raise_on)
        self.terminal_after = terminal_after
        self.builder_calls = []

        def evaluate(_input):
            done = len(self.executor.calls) >= self.terminal_after
            return SimpleNamespace(success=done, terminated=done, truncated=False, reason="TASK_SUCCESS" if done else "CONTINUE",
                                   reward_events=((0.5, 1),) if done else ())

        def build(previous, facts, observation, execution, end):
            self.builder_calls.append(previous.decision_id)
            if builder_fault == "raise":
                raise RuntimeError("SUCCESSOR_BUILD_FAILED")
            return self.states[min(previous.decision_id + 1, len(self.states) - 1)]

        self.bundle = SimpleNamespace(
            clock=self.clock, task_id="T_B", episode_start_seconds=0.0, executor=self.executor, template=self.template,
            observations=SimpleNamespace(observe=lambda: {}), perception=SimpleNamespace(infer=lambda obs: {}),
            verifier=SimpleNamespace(verify=lambda measured, execution: {}),
            evaluator=SimpleNamespace(deadline=60.0, evaluate=evaluate),
            safety=SimpleNamespace(can_execute=lambda candidate, obs: True, end_no_candidates=lambda *key: "NO_SAFE_CANDIDATES"),
            snapshot_builder=SimpleNamespace(build=build), render_gpu_device_id=0, env_factory=lambda spec: None,
            restore_receipt={"engineering_test_only": True}, snapshot_identity="engineering_test_only",
            environment=SimpleNamespace(close=lambda: None), current_snapshot=None, start_case=lambda case_id: self.states[0])
        self.current = self.states[0]
        self.log_path = Path(tmp) / "events.jsonl"
        if not instrument:  # bare doubles: final_tb.run_smoke installs its own instrumentation
            return
        torch.manual_seed(0)
        self.policy = v11.make_policy(self.template, "B1-K", torch.device("cpu"))
        self.policy.eval()
        self.log = final_tb.SmokeEventLog(self.log_path)
        self.rec = final_tb.SmokeRecorder(self.log)
        self.selector, self.restore = final_tb.install_smoke_instrumentation(self.bundle, self.policy, self.rec)
        self.collector = Collector(self.bundle, self.selector)
        self.collector.reset_episode("ENG_ENV", "ENG_EP")

    def step(self, index, deterministic=True, target=None):
        self.selector.target = SCRIPT[index] if target is None else target
        self.rec.begin_step(index, self.selector.target, self.current)
        transition, result = self.collector.step(self.current, deterministic)
        self.rec.end_step(transition is not None)
        if transition is not None:
            self.current = transition.next_snapshot
        return transition, result

    def plain_forward(self, snapshot, hidden):
        with self.torch.no_grad():
            return self.policy(snapshot, hidden)

    def events(self):
        return [json.loads(line) for line in self.log_path.read_text(encoding="utf-8").splitlines()]


@contextlib.contextmanager
def rig(root, tmp, **kw):
    from cp_disr import rl
    saved = rl._SUITE_H
    try:
        yield Rig(root, tmp, **kw)
    finally:
        rl._SUITE_H = saved


def _expect(exc_type, fn):
    try:
        fn()
    except exc_type as exc:
        return exc
    raise AssertionError("expected %s" % exc_type.__name__)


# ------------------------------------------------------------------------------------------ step-level checks (tests 1-8)
def check_stale_target_next_value(root, tmp):
    """1. Current PICK legal, successor PICK illegal (others legal): real Collector returns a finite next-V and the
    executor is entered exactly once.  The OLD selector forced the stale target in the next-V probe -> NaN."""
    with rig(root, tmp) as r:
        assert r.states[0].mask[IDS.index(SCRIPT[0])] and not r.states[1].mask[IDS.index(SCRIPT[0])] and any(r.states[1].mask)
        t, result = r.step(0)
        assert r.executor.calls == [SCRIPT[0]], r.executor.calls
        assert t.selected_candidate_id == SCRIPT[0]
        assert all(math.isfinite(x) for x in (t.old_logp, t.old_v, t.old_v_next)), (t.old_logp, t.old_v, t.old_v_next)
        out0 = r.plain_forward(r.states[0], r.policy.initial_hidden())
        expected = float(r.plain_forward(r.states[1], out0.hidden).value)
        assert abs(t.old_v_next - expected) < 1e-6, "next-V must be the unmodified policy value"
        assert r.rec.forward_calls == 2 and r.rec.executor_entered == r.rec.executor_returned == r.rec.transitions_returned == 1
        return {"executor_calls": 1, "old_v_next": t.old_v_next, "forward_calls": r.rec.forward_calls}


def check_five_action_script_cursor(root, tmp):
    """2. Five scripted actions: the cursor moves only through the explicit loop; next-V forwards do not advance the
    script nor the committed recurrent history."""
    with rig(root, tmp) as r:
        for i, action in enumerate(SCRIPT):
            t, result = r.step(i)
            assert r.selector.target == action, "a forward moved the script cursor"
            assert t.selected_candidate_id == action and all(math.isfinite(x) for x in (t.old_logp, t.old_v, t.old_v_next))
        assert r.executor.calls == list(SCRIPT) and result.success is True and t.terminated
        assert r.rec.forward_calls == 9, "5 current decisions + 4 next-V probes (terminal skips its probe)"
        prefix = r.collector.prefixes[("ENG_ENV", "ENG_EP")]
        assert [s.decision_id for s in prefix] == [0, 1, 2, 3, 4], "committed history = decision snapshots only"
        events = r.events()
        assert [e["seq"] for e in events] == list(range(1, len(events) + 1)), "event log must be valid JSONL, one record per line"
        begins = [i for i, e in enumerate(events) if e["event"] == "STEP_BEGIN"]
        enters = [i for i, e in enumerate(events) if e["event"] == "EXECUTOR_ENTER"]
        assert len(begins) == len(enters) == 5 and all(b < e for b, e in zip(begins, enters)), "STEP_BEGIN must precede the executor"
        assert [e["event"] for e in events].count("SELECT") == 5
        return {"executor_calls": 5, "forward_calls": r.rec.forward_calls, "events": len(events)}


def check_masked_missing_target_fails_closed(root, tmp):
    """3. A masked/missing/None target (or a non-deterministic call) fails at the real select with zero executor calls."""
    from cp_disr import final_tb
    seen = {}
    for label, target, message in (("masked", IDS[5], "SCRIPT_TARGET_MASKED"),
                                   ("missing", "a:NOT:A:CANDIDATE:v1", "SCRIPT_TARGET_MISSING_OR_AMBIGUOUS"),
                                   ("none", None, "SCRIPT_TARGET_MISSING_OR_AMBIGUOUS")):
        with rig(root, Path(tmp) / label) as r:
            r.selector.target = target
            r.rec.begin_step(0, target, r.current)
            exc = _expect(final_tb.SmokeSelectionError, lambda: r.collector.step(r.current, True))
            assert message in str(exc), (label, str(exc))
            assert r.executor.calls == [] and r.rec.executor_entered == 0 and r.rec.transitions_returned == 0
            seen[label] = str(exc)
    with rig(root, Path(tmp) / "nondeterministic") as r:
        r.selector.target = SCRIPT[0]
        exc = _expect(final_tb.SmokeSelectionError, lambda: r.collector.step(r.current, False))
        assert "SMOKE_SELECTOR_NOT_FOR_TRAINING" in str(exc) and r.executor.calls == []
    dup = SimpleNamespace(candidate_ids=("x", "x"), mask=[True, True], distribution=object())
    exc = _expect(final_tb.SmokeSelectionError, lambda: final_tb._SelectionOnlyOutput(dup, "x").select(True))
    assert "AMBIGUOUS" in str(exc)
    return seen


def check_terminal_and_no_candidate_semantics(root, tmp):
    """4. Genuine terminal skips the next-V forward (0.0); a non-terminal successor without legal candidates keeps the
    original Policy semantics (finite zero value); no legal candidate at the CURRENT state ends without executing."""
    out = {}
    with rig(root, Path(tmp) / "terminal", terminal_after=1) as r:
        t, result = r.step(0)
        assert t.terminated and t.old_v_next == 0.0 and r.rec.forward_calls == 1 and r.executor.calls == [SCRIPT[0]]
        out["terminal"] = {"forward_calls": r.rec.forward_calls}
    masks = (MASKS[0], (F,) * 7)
    with rig(root, Path(tmp) / "nolegal_next", masks=masks) as r:
        t, result = r.step(0)
        assert not t.terminated and r.rec.forward_calls == 2 and r.executor.calls == [SCRIPT[0]]
        probe = r.plain_forward(r.states[1], r.plain_forward(r.states[0], r.policy.initial_hidden()).hidden)
        assert probe.distribution is None and probe.ended_reason == "NO_SAFE_CANDIDATES"
        assert math.isfinite(t.old_v_next) and abs(t.old_v_next - float(probe.value)) < 1e-9
        out["no_legal_next"] = {"old_v_next": t.old_v_next}
    with rig(root, Path(tmp) / "nolegal_now", masks=((F,) * 7, MASKS[0])) as r:
        r.selector.target = SCRIPT[0]
        t, result = r.collector.step(r.current, True)
        assert t is None and result["no_transition"] and r.executor.calls == [] and r.rec.executor_entered == 0
        out["no_legal_now"] = result["reason"]
    return out


def check_stale_action_still_legal(root, tmp):
    """5. If the previous action is still legal in the next state, nothing extra is executed and select() is used once."""
    masks = (MASKS[0], MASKS[0], MASKS[0], MASKS[0], MASKS[0], MASKS[0])
    with rig(root, tmp, masks=masks) as r:
        t, result = r.step(0)
        assert r.executor.calls == [SCRIPT[0]] and r.rec.executor_entered == 1
        events = r.events()
        assert [e["event"] for e in events].count("SELECT") == 1, "the next-V probe must never call select()"
        assert r.rec.forward_calls == 2 and r.states[1].mask[IDS.index(SCRIPT[0])]
        return {"select_events": 1}


def check_nonfinite_policy_not_masked(root, tmp):
    """6. NaN in the real Policy path is never hidden: no nan_to_num, no fallback, no skipped next-V."""
    import torch
    from cp_disr.common import DataIntegrityError
    out = {}
    # (a) NaN hidden/logits at the CURRENT decision: the original Policy raises before any execution
    with rig(root, Path(tmp) / "a", nan_base_at=(0,)) as r:
        _expect(ValueError, lambda: r.step(0))
        assert r.executor.calls == [] and r.rec.executor_entered == 0 and r.rec.stage == "CURRENT_DECISION_FORWARD"
        failed = [e for e in r.events() if e["event"] == "FORWARD_FAILED"]
        assert failed and "Traceback" in failed[0]["traceback"] and failed[0]["executor_counters"]["executor_entered"] == 0
        out["current_forward_nan"] = {"executor_calls": 0}
    # (b) NaN only in the successor (next-V probe): the failure surfaces AFTER one real execution, at the probe
    with rig(root, Path(tmp) / "b", nan_base_at=(1,)) as r:
        _expect(ValueError, lambda: r.step(0))
        assert r.executor.calls == [SCRIPT[0]] and r.rec.executor_entered == r.rec.executor_returned == 1
        assert r.rec.transitions_returned == 0 and r.rec.stage == "NEXT_STATE_VALUE_PROBE"
        out["next_value_probe_nan"] = {"executor_calls": 1, "transitions": 0}
    # (c) NaN value head: the selector must not repair it; the production Transition guard rejects it after execution
    with rig(root, Path(tmp) / "c") as r:
        with torch.no_grad():
            for p in r.policy.v_head.parameters():
                p.fill_(float("nan"))
        _expect(DataIntegrityError, lambda: r.step(0))
        assert r.executor.calls == [SCRIPT[0]] and r.rec.transitions_returned == 0
        ends = [e for e in r.events() if e["event"] == "FORWARD_END"]
        assert ends and ends[0]["output"]["value_finite"] is False, "the NaN value must stay visible in the evidence"
        out["nan_value_head"] = {"executor_calls": 1}
    return out


def check_output_fields_unchanged(root, tmp):
    """7. logits / mask / distribution / value / q / hidden / diagnostics are exactly the plain Policy's."""
    import torch
    with rig(root, tmp) as r:
        hidden = r.policy.initial_hidden()
        ref = r.plain_forward(r.states[0], hidden)
        r.selector.target = SCRIPT[0]
        with torch.no_grad():
            out = r.selector(r.states[0], hidden)
        for name in ("logits", "mask", "value", "q", "hidden"):
            assert torch.equal(getattr(out, name), getattr(ref, name)), name
        assert out.candidate_ids == ref.candidate_ids and type(out.distribution) is type(ref.distribution)
        assert torch.equal(out.distribution.logits, ref.distribution.logits)
        assert out.diagnostics.keys() == ref.diagnostics.keys() and out.ended_reason == ref.ended_reason
        assert torch.isneginf(out.logits[~out.mask]).all() and torch.isfinite(out.logits[out.mask]).all()
        assert out.logits is out._output.logits, "the proxy must expose the wrapped tensors, not copies"
        before = r.selector.target
        assert out.select(True) == (SCRIPT[0], IDS.index(SCRIPT[0])) and r.selector.target == before
        after = r.selector(r.states[1], out.hidden)  # a forward neither consumes nor rewrites the script
        assert r.selector.target == before and torch.equal(after.logits, r.plain_forward(r.states[1], out.hidden).logits)
        return {"n_legal": int(out.mask.sum())}


# ------------------------------------------------------------------------------------------ run_smoke-level (doubles)
def fake_v11(rig_, final_tb):
    from cp_disr import rl, stage2a_v11 as real

    def bind_H(root):
        rl.set_suite_half_life(final_tb.H_SECONDS)
        return {"H": final_tb.H_SECONDS}

    return SimpleNamespace(
        bind_H=bind_H, make_bundle=lambda root, task: rig_.bundle, make_policy=real.make_policy,
        apply_episode_prior=lambda method, bundle, snap, sampler, eval_original=False: (
            snap, SimpleNamespace(audit_mode="absent", edges=()), 0))


def run_smoke_with_doubles(root, out_dir, case_index, prep, tmp, builder_fault=None, executor_raise_on=None, masks=MASKS,
                           terminal_after=5, nan_base_at=()):
    """The REAL `final_tb.run_smoke` (budget charge, counting wrappers, receipts) on a registered R2 directory with the
    physical boundary replaced by the doubles above."""
    import torch
    from cp_disr import final_tb, rl
    saved = rl._SUITE_H
    try:
        r = Rig(root, Path(tmp) / ("rig%d" % case_index), masks=masks, terminal_after=terminal_after,
                builder_fault=builder_fault, executor_raise_on=executor_raise_on, nan_base_at=nan_base_at, instrument=False)
        case = final_tb.select_smoke_cases(root, 2)[case_index - 1]
        assert tuple(case["script"]) == SCRIPT, "the frozen script must equal the engineering fixture script"
        ctx = SimpleNamespace(source_commit=prep, render_gpu_device_id=0)
        res = final_tb.run_smoke(root, out_dir, case_index, 0, ctx, v11_module=fake_v11(r, final_tb),
                                 evidence_fn=lambda pid, gpu: {"on_expected_only": True, "engineering_test_only": True},
                                 device=torch.device("cpu"))
        return res, r
    finally:
        rl._SUITE_H = saved


def register_for_test(root, out_dir, head):
    from cp_disr import final_tb
    git = lambda r, *a: head if a[0] == "rev-parse" else ""
    return final_tb.run_register(root, out_dir, head, "engineering authorization text (test)", stamp="TEST", git=git)


def check_post_execution_failure_is_recorded(root, tmp):
    """8. A failure after execution is recorded with the real executor count (1), zero Transitions, the stage and the
    full traceback; zero Transitions is never read as 'executor not called'."""
    tmp = Path(tmp)
    out_dir = tmp / "launch"
    register_for_test(root, out_dir, "a" * 40)
    res, r = run_smoke_with_doubles(root, out_dir, 1, "a" * 40, tmp / "d", builder_fault="raise")
    assert res["episode_verdict"] != "PASS" and res["failure"]["stage"] == "NEXT_SNAPSHOT_BUILD", res["failure"]["stage"]
    assert "SUCCESSOR_BUILD_FAILED" in res["failure"]["traceback"] and "Traceback" in res["failure"]["traceback"]
    c = res["counters"]
    assert c["executor_calls_entered"] == c["executor_calls_returned"] == 1 and c["transitions_returned"] == 0
    assert r.executor.calls == [SCRIPT[0]] and res["steps"][0]["verdict"] == "ENGINEERING_FAILURE"
    assert c["budget_charged_this_attempt"]["episode_attempts"] == 1 and c["budget_charged_this_attempt"]["skill_calls"] == 1
    persisted = json.loads((out_dir / "smoke_case1.json").read_text())
    assert persisted["failure"]["stage"] == "NEXT_SNAPSHOT_BUILD" and persisted["counters"]["transitions_returned"] == 0
    kinds = [json.loads(line)["event"] for line in (out_dir / "smoke_case1_events.jsonl").read_text().splitlines()]
    assert kinds.index("STEP_BEGIN") < kinds.index("EXECUTOR_ENTER") < kinds.index("EXECUTOR_RETURN") < kinds.index("STEP_FAILED")
    # a NaN successor fails at the next-value probe with its own stage
    out2 = tmp / "other_parent" / "launch2"
    register_for_test(root, out2, "b" * 40)
    res2, r2_ = run_smoke_with_doubles(root, out2, 1, "b" * 40, tmp / "d2", nan_base_at=(1,))
    assert res2["failure"]["stage"] == "NEXT_STATE_VALUE_PROBE" and res2["counters"]["executor_calls_entered"] == 1
    assert res2["counters"]["transitions_returned"] == 0 and "ValueError" in res2["failure"]["type"]
    return {"stage_a": res["failure"]["stage"], "stage_b": res2["failure"]["stage"]}
