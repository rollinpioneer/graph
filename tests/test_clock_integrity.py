"""Limited production clock-integrity regression. No formal RL, no test-ID eval."""
from __future__ import annotations

import math
import time
from types import SimpleNamespace

import numpy as np
import pytest

from cp_disr.common import ClockIntegrityError, DiagnosticAbort
from cp_disr.platforms.libero.clock import DurationProvider
from cp_disr.platforms.libero.skill_executor import SkillExecutor, GRIP_OPEN
from cp_disr.platforms.libero.task_evaluator import TaskEvaluator
from cp_disr.adapters import EvaluationInput
from cp_disr.collector import Collector
from cp_disr.rl import gamma


CONTROL_DT = 0.05
WALL_BUDGET_S = 20.0
CONTROL_STEP_BUDGET = 400


class FakeSim:
    def __init__(self, t=0.0, timestep=0.002):
        self.data = SimpleNamespace(time=float(t))
        self.model = SimpleNamespace(opt=SimpleNamespace(timestep=float(timestep)))


class FakeEnv:
    def __init__(self, perception=None, step_dt=CONTROL_DT, fail_step=False, max_steps=CONTROL_STEP_BUDGET, wall_budget=WALL_BUDGET_S):
        self.sim = FakeSim(0.0)
        self.last_perception = dict(perception or {})
        self.table_top_z = 0.80
        self.container_center = np.array([0.00, 0.00, 0.82], dtype=float)
        self.buffer_center = np.array([0.18, 0.00, 0.82], dtype=float)
        self.eef = np.array([0.00, 0.00, 1.02], dtype=float)
        self.step_dt = float(step_dt)
        self.n_steps = 0
        self.max_steps = int(max_steps)
        self.fail_step = bool(fail_step)
        self.wall_budget = float(wall_budget)
        self._wall0 = time.monotonic()
        self.second_role = "second_object"

    def public_observation(self):
        return {"eef_pos": self.eef.copy()}

    def step_osc(self, dpos, grip, n=1):
        if time.monotonic() - self._wall0 > self.wall_budget:
            raise RuntimeError("DIAGNOSTIC_WALL_BUDGET")
        if self.fail_step:
            raise RuntimeError("step_failed")
        traces = []
        for _ in range(int(n)):
            self.n_steps += 1
            if self.n_steps > self.max_steps:
                raise RuntimeError("DIAGNOSTIC_CONTROL_STEP_BUDGET")
            self.sim.data.time = float(self.sim.data.time) + self.step_dt
            delta = np.asarray(dpos, dtype=float) * 0.01
            self.eef = self.eef + delta
            traces.append({"t": float(self.sim.data.time), "eef": self.eef.copy()})
        return traces

    def hidden_truth(self):
        z = float(self.table_top_z) + 0.20
        return {
            "lid": (1.0, 0.0, 1.0),
            "container": tuple(self.container_center),
            "target": (0.0, 0.0, z),
            "second_object": (0.2, 0.0, z),
            "buffer": tuple(self.buffer_center),
            "table_top_z": self.table_top_z,
        }

    def refresh_perception(self):
        return dict(self.last_perception)


class FakeSafety:
    def __init__(self, allow=True):
        self.allow = allow

    def can_execute(self, candidate_id, obs):
        return bool(self.allow)

    def end_no_candidates(self, env_id, episode_id):
        return "NO_CANDIDATES"


def _executor(env, allow=True):
    clock = DurationProvider(env)
    return SkillExecutor(env, FakeSafety(allow=allow), clock), clock


def _open_perception():
    return {
        "lid": np.array([0.02, 0.00, 0.84], dtype=float),
        "target": np.array([0.05, 0.02, 0.83], dtype=float),
        "second_object": np.array([-0.05, 0.02, 0.83], dtype=float),
    }


@pytest.mark.pure
def test_clock_1_zero_physics_not_positive_ppo_duration():
    env = FakeEnv(perception={})
    clock = DurationProvider(env)
    start = clock.now_seconds()
    d = clock.duration_seconds(start, clock.now_seconds())
    assert d == 0.0
    with pytest.raises(ValueError):
        gamma(d)
    ex, _ = _executor(env)
    out = ex.execute("a:OPEN:container:v1", 8.0)
    assert out["steps"] == 0
    assert float(out["sim_duration"]) <= 0.0
    assert out["controller_exit"].startswith("INTERRUPT_CONTROLLER_EXCEPTION:RuntimeError")
    assert "perception_unavailable" in out["controller_exit"]
    assert abs(float(out["sim_duration"]) - 0.05) > 1e-12


@pytest.mark.pure
def test_clock_2_negative_or_nonfinite_errors():
    env = FakeEnv()
    clock = DurationProvider(env)
    with pytest.raises(ClockIntegrityError):
        clock.duration_seconds(1.0, 0.5)
    with pytest.raises(ClockIntegrityError):
        clock.duration_seconds(float("nan"), 1.0)
    with pytest.raises(ClockIntegrityError):
        clock.duration_seconds(0.0, float("inf"))


@pytest.mark.pure
def test_clock_3_real_control_step_0_05_accepted():
    env = FakeEnv(perception=_open_perception(), step_dt=CONTROL_DT)
    clock = DurationProvider(env)
    start = clock.now_seconds()
    env.step_osc(np.zeros(3), GRIP_OPEN, n=1)
    end = clock.now_seconds()
    d = clock.duration_seconds(start, end)
    assert d == pytest.approx(CONTROL_DT)
    g = gamma(d, H=23.1)
    assert math.isfinite(g) and 0 < g < 1
    assert clock.now_seconds() == pytest.approx(d)


@pytest.mark.pure
def test_clock_4_open_pick_place_place_buffer_execute():
    env = FakeEnv(perception=_open_perception(), max_steps=CONTROL_STEP_BUDGET)
    ex, clock = _executor(env)
    t0 = clock.now_seconds()
    for cid in (
        "a:OPEN:container:v1",
        "a:PICK:target:v1",
        "a:PLACE:target:container:v1",
        "a:PLACE_BUFFER:second_object:buffer:v1",
    ):
        n0 = env.n_steps
        t1 = float(env.sim.data.time)
        out = ex.execute(cid, 12.0)
        assert out["controller_exit"] not in {"INTERRUPT_NAN_ACTION"}
        assert not str(out["controller_exit"]).startswith("INTERRUPT_CONTROLLER_EXCEPTION")
        assert out["steps"] >= 1
        assert float(out["sim_duration"]) == pytest.approx(float(env.sim.data.time) - t1)
        assert float(out["sim_duration"]) == pytest.approx((env.n_steps - n0) * CONTROL_DT)
        assert float(out["sim_duration"]) > 0
    assert clock.now_seconds() - t0 == pytest.approx(env.n_steps * CONTROL_DT)


@pytest.mark.pure
def test_clock_5_repeat_open_does_not_grow_n_without_time():
    env = FakeEnv(perception={}, max_steps=CONTROL_STEP_BUDGET)
    ex, clock = _executor(env)
    durations = []
    for _ in range(8):
        t0 = clock.now_seconds()
        out = ex.execute("a:OPEN:container:v1", 8.0)
        d = clock.duration_seconds(t0, clock.now_seconds())
        durations.append((d, out["sim_duration"], out["controller_exit"], out["steps"]))
        assert d == 0.0
        assert float(out["sim_duration"]) <= 0.0
        assert out["steps"] == 0
    assert clock.now_seconds() == 0.0
    assert env.n_steps == 0


@pytest.mark.pure
def test_clock_6_perception_reject_exception_keep_class():
    env = FakeEnv(perception={})
    ex, _ = _executor(env)
    out = ex.execute("a:OPEN:container:v1", 8.0)
    assert out["interrupted"] is True
    assert "perception_unavailable" in out["controller_exit"]
    env2 = FakeEnv(perception=_open_perception())
    ex2, _ = _executor(env2, allow=False)
    out2 = ex2.execute("a:OPEN:container:v1", 8.0)
    assert out2["rejected"] is True
    assert out2["controller_exit"] == "REJECTED_UNSAFE_OR_INVALID"
    assert out2["steps"] == 1
    assert "CONFIRM_TICK" in out2["states"]
    assert float(out2["sim_duration"]) == pytest.approx(CONTROL_DT)


@pytest.mark.pure
def test_clock_7_tb_train_36_fault_reproduced_as_zero_physics_interrupt():
    env = FakeEnv(perception=_open_perception())
    ex, clock = _executor(env)
    first = ex.execute("a:PICK:target:v1", 8.0)
    assert float(first["sim_duration"]) > 0
    env.last_perception = {}
    loop_n = 0
    t0 = clock.now_seconds()
    for _ in range(5):
        out = ex.execute("a:OPEN:container:v1", 8.0)
        loop_n += 1
        assert out["controller_exit"].startswith("INTERRUPT")
        assert float(out["sim_duration"]) <= 0.0
    assert clock.now_seconds() == pytest.approx(t0)
    assert loop_n == 5


@pytest.mark.pure
def test_clock_8_deadline_and_short_remaining():
    env = FakeEnv(goal=False) if False else FakeEnv()
    ev = TaskEvaluator(env, 1.0, task_id="T_B")
    ev.reset_episode()
    r = ev.evaluate(EvaluationInput("T_B", "e", "ep", (), 1.0, 0.2, 1.0))
    assert r.reason == "DEADLINE" and r.terminated and not r.truncated and r.reward_events == ()
    ev2 = TaskEvaluator(env, 60.0, task_id="T_B")
    ev2.reset_episode()
    r2 = ev2.evaluate(EvaluationInput("T_B", "e", "ep", (), 0.04, 0.0, 0.04))
    assert r2.reason == "CONTINUE" and not r2.terminated


@pytest.mark.pure
def test_clock_9_episode_clock_not_reset_across_fake_update():
    env = FakeEnv()
    clock = DurationProvider(env)
    env.step_osc(np.zeros(3), GRIP_OPEN, n=3)
    t_mid = clock.now_seconds()
    env.step_osc(np.zeros(3), GRIP_OPEN, n=2)
    assert clock.now_seconds() == pytest.approx(t_mid + 2 * CONTROL_DT)
    assert clock._origin == pytest.approx(0.0)


@pytest.mark.pure
def test_clock_10_deadline_no_bootstrap_external_truncation_keeps_value():
    from cp_disr.rl import scalar_targets

    class T:
        def __init__(self, terminated, truncated, nxt):
            self.reward = 0.0
            self.Gamma = 0.99
            self.terminated = terminated
            self.truncated = truncated
            self.old_v = 0.4
            self.old_v_next = nxt
            self.snapshot = SimpleNamespace(env_id="e", episode_id="p", decision_id=0)
            self.next_snapshot = SimpleNamespace(decision_id=1)

    a, v, q = scalar_targets([T(True, False, 9.0)])
    assert q[0] == pytest.approx(0.0)
    a, v, q = scalar_targets([T(False, True, 2.0)])
    assert q[0] == pytest.approx(0.99 * 2.0)


@pytest.mark.pure
def test_clock_11_collector_remaining_uses_episode_relative_clock():
    env = FakeEnv()
    clock = DurationProvider(env)
    env.step_osc(np.zeros(3), GRIP_OPEN, n=10)
    bundle = SimpleNamespace(clock=clock, evaluator=SimpleNamespace(deadline=60.0), episode_start_seconds=0.0)
    col = Collector(bundle, policy=None)
    now = clock.now_seconds()
    elapsed = col._episode_elapsed(now)
    remaining = 60.0 - elapsed
    assert elapsed == pytest.approx(10 * CONTROL_DT)
    assert remaining == pytest.approx(60.0 - 10 * CONTROL_DT)
    col2 = Collector(SimpleNamespace(clock=clock, evaluator=SimpleNamespace(deadline=60.0), episode_start_seconds=now), policy=None)
    assert col2._episode_elapsed(now) == pytest.approx(0.0)


@pytest.mark.pure
def test_clock_11b_collector_rejects_zero_duration_as_diagnostic_abort():
    env = FakeEnv(perception={})
    clock = DurationProvider(env)
    ex, _ = _executor(env)

    class FakeDist:
        def log_prob(self, t):
            return t * 0.0

    class FakeOut:
        distribution = FakeDist()
        value = 0.0
        hidden = None
        logits = None
        def select(self, det):
            return "a:OPEN:container:v1", 0

    class FakePolicy:
        def initial_hidden(self):
            import torch
            return torch.zeros(1)
        def advance_hidden(self, x, hidden):
            return hidden
        def __call__(self, snapshot, hidden=None):
            return FakeOut()

    class ExecWrap:
        def __init__(self, inner):
            self.inner = inner
            self.last = None
        def execute(self, cid, timeout):
            self.last = self.inner.execute(cid, timeout)
            return SimpleNamespace(
                execution_id=self.last["execution_id"],
                controller_exit=self.last["controller_exit"],
                start_seconds=self.last["start_seconds"],
                end_seconds=self.last["end_seconds"],
                evidence_ids=tuple(self.last.get("evidence_ids") or ()),
            )

    class DummySnap:
        synthetic_unit_fixture = False
        env_id = "d0-env-0"
        episode_id = "ep-1"
        template = SimpleNamespace(contracts=[SimpleNamespace(id="a:OPEN:container:v1", timeout_seconds=8.0)])

    bundle = SimpleNamespace(
        clock=clock,
        evaluator=TaskEvaluator(env, 60.0, task_id="T_B"),
        episode_start_seconds=0.0,
        task_id="T_B",
        safety=FakeSafety(True),
        observations=SimpleNamespace(observe=lambda: {"eef_pos": env.eef.copy()}),
        perception=SimpleNamespace(infer=lambda obs: {}),
        verifier=SimpleNamespace(verify=lambda measured, execution: ()),
        executor=ExecWrap(ex),
        snapshot_builder=SimpleNamespace(build=lambda *a, **k: DummySnap()),
    )
    bundle.evaluator.reset_episode()
    col = Collector(bundle, FakePolicy())
    col.reset_episode("d0-env-0", "ep-1")
    with pytest.raises(DiagnosticAbort) as ei:
        col.step(DummySnap(), deterministic=False)
    assert ei.value.payload.get("detail") == "technical_controller_failure"
    assert clock.now_seconds() == 0.0


@pytest.mark.pure
def test_clock_12_empty_patch_and_mask_untouched_by_clock_repair():
    from cp_disr.neural import canonical_method
    assert canonical_method("B1-K") == "B1"
    assert canonical_method("B2") == "B2"
    import inspect
    from cp_disr.platforms.libero import clock as clock_mod
    src = inspect.getsource(clock_mod.DurationProvider.duration_seconds)
    assert "1.0 / 20.0" not in src
    assert "max(0.05" not in src
