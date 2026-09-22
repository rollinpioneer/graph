"""Stage 2A production tests for frozen deadline semantics. No Method change."""
from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from cp_disr.adapters import EvaluationInput
from cp_disr.collector import Collector
from cp_disr.platforms.libero.task_evaluator import TaskEvaluator
from cp_disr.rl import Transition, scalar_targets, gamma
from cp_disr.torch_rl import PPO, save_checkpoint, load_checkpoint


class FakeEnv:
    def __init__(self, goal=False, task_id="T_A"):
        self._goal = bool(goal)
        self.task_id = task_id
        self.second_role = "second_object" if task_id == "T_A" else "interferer"
        self.table_top_z = 0.8
        self.sim = SimpleNamespace(model=SimpleNamespace(opt=SimpleNamespace(timestep=0.002)))

    def hidden_truth(self):
        z = 0.85 if self._goal else 1.6
        return {
            "lid": (1.0, 0.0, 1.0),
            "container": (0.0, 0.0, 0.8),
            "target": (0.0, 0.0, z),
            self.second_role: (0.0, 0.0, z),
            "table_top_z": self.table_top_z,
        }


def _eval(env, deadline, elapsed, start, end, task_id="T_A", ep="ep"):
    ev = TaskEvaluator(env, deadline, task_id=task_id)
    ev.reset_episode()
    return ev, ev.evaluate(EvaluationInput(task_id, "e", ep, (), elapsed, start, end))


@pytest.mark.pure
def test_deadline_1_success_in_time():
    env = FakeEnv(goal=True, task_id="T_A")
    ev, r = _eval(env, 10.0, 4.0, 3.0, 4.0)
    assert r.success and r.terminated and (not r.truncated)
    assert r.reason == "TASK_SUCCESS" and r.reward_events


@pytest.mark.pure
def test_deadline_2_no_success_in_time():
    env = FakeEnv(goal=False)
    ev, r = _eval(env, 10.0, 4.0, 3.0, 4.0)
    assert (not r.success) and (not r.terminated) and (not r.truncated)
    assert r.reason == "CONTINUE" and r.reward_events == ()


@pytest.mark.pure
def test_deadline_3_at_deadline_unsuccessful():
    env = FakeEnv(goal=False)
    ev, r = _eval(env, 5.0, 5.0, 4.0, 5.0)
    assert (not r.success) and r.terminated and (not r.truncated)
    assert r.reason == "DEADLINE" and r.reward_events == ()


@pytest.mark.pure
def test_deadline_4_goal_after_deadline():
    env = FakeEnv(goal=True)
    ev, r = _eval(env, 5.0, 5.2, 4.0, 5.2)
    assert (not r.success) and r.terminated and (not r.truncated)
    assert r.reason == "DEADLINE" and r.reward_events == ()


@pytest.mark.pure
def test_deadline_5_skill_crossing_recorded_as_real_stop():
    env = FakeEnv(goal=False)
    ev = TaskEvaluator(env, 1.0, task_id="T_A")
    ev.reset_episode()
    r = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 1.0, 0.2, 1.0))
    assert r.reason == "DEADLINE" and r.terminated and not r.truncated
    assert ev.time_resolution_seconds == pytest.approx(0.002)


@pytest.mark.pure
def test_deadline_6_external_truncation_keeps_bootstrap():
    # External truncation is a transition flag, not an evaluator DEADLINE.
    class Snap:
        pass
    # scalar target with truncated True, terminated False includes next V
    from dataclasses import dataclass
    from cp_disr.rl import Snapshot
    from cp_disr.facts import FactStore
    from cp_disr.common import digest
    # Use GAE helper only
    class T:
        def __init__(self):
            self.reward = 0.3
            self.Gamma = 0.99
            self.terminated = False
            self.truncated = True
            self.old_v = 0.4
            self.old_v_next = 2.0
            self.snapshot = SimpleNamespace(env_id="e", episode_id="p", decision_id=0)
            self.next_snapshot = SimpleNamespace(decision_id=1)
    a, v, q = scalar_targets([T()])
    assert q[0] == pytest.approx(0.3 + 0.99 * 2.0)


@pytest.mark.pure
def test_deadline_7_buffer_cut_is_not_episode_truncation():
    class T:
        def __init__(self, did=0):
            self.reward = 0.0
            self.Gamma = 0.99
            self.terminated = False
            self.truncated = False
            self.old_v = 1.0
            self.old_v_next = 1.5
            self.snapshot = SimpleNamespace(env_id="e", episode_id="p", decision_id=did)
            self.next_snapshot = SimpleNamespace(decision_id=did + 1)
    a, v, q = scalar_targets([T(0), T(1)])
    assert T(0).truncated is False
    assert q[0] != 0.0


@pytest.mark.pure
def test_deadline_8_clock_not_reset_across_update_is_evaluator_state():
    env = FakeEnv(goal=False)
    ev = TaskEvaluator(env, 10.0, task_id="T_A")
    ev.reset_episode()
    r1 = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 3.0, 2.0, 3.0))
    r2 = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 9.5, 8.0, 9.5))
    assert r1.reason == "CONTINUE" and r2.reason == "CONTINUE"
    r3 = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 10.0, 9.5, 10.0))
    assert r3.reason == "DEADLINE" and r3.terminated


@pytest.mark.pure
def test_deadline_9_deadline_target_no_bootstrap():
    class T:
        def __init__(self):
            self.reward = 0.0
            self.Gamma = 0.99
            self.terminated = True
            self.truncated = False
            self.old_v = 0.4
            self.old_v_next = 9.0
            self.snapshot = SimpleNamespace(env_id="e", episode_id="p", decision_id=0)
            self.next_snapshot = SimpleNamespace(decision_id=1)
    a, v, q = scalar_targets([T()])
    assert q[0] == pytest.approx(0.0)


@pytest.mark.pure
def test_deadline_10_success_reward_not_repeated():
    env = FakeEnv(goal=True)
    ev = TaskEvaluator(env, 10.0, task_id="T_A")
    ev.reset_episode()
    r1 = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 2.0, 1.0, 2.0))
    r2 = ev.evaluate(EvaluationInput("T_A", "e", "ep", (), 3.0, 2.0, 3.0))
    assert r1.success and r1.reward_events
    assert r2.success and r2.reward_events == ()
