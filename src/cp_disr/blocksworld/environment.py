"""Symbolic Blocksworld environment and snapshot builder (runbook 5, 6). Deterministic, fully observable, no UNKNOWN, no simulator, no GPU, no provider.

The environment is driven by the frozen contracts (``state.legal_actions`` / ``state.apply``, tested against the grounded ``SkillContract`` objects). A decision's snapshot carries the
production ``Snapshot`` type so the production PPO / Transition / recurrence code can be reused unchanged.
"""
from __future__ import annotations

from dataclasses import dataclass

from ..common import digest
from ..facts import FactRecord, FactStore, Truth
from ..rl import Snapshot, gamma
from . import contracts as C
from . import state as S

OBS_DIM = 48
_FEATURE_CACHE = {}


@dataclass(frozen=True)
class Case:
    case_id: str
    split: str
    problem: S.Problem
    optimal_length: int
    step_cap: int


def step_cap_for(optimal_length):
    return 2 * optimal_length + 4


def _candidate_features(template):
    key = id(template)
    hit = _FEATURE_CACHE.get(key)
    if hit is not None and hit[0] is template:
        return hit[1]
    from ..c1_blocksworld_policies import candidate_feature
    ids = tuple(c.id for c in sorted(template.contracts, key=lambda c: c.id))
    by_id = {c.id: c for c in template.contracts}
    feats = tuple(candidate_feature(by_id[i].name) for i in ids)
    _FEATURE_CACHE[key] = (template, (ids, feats))
    return ids, feats


def base_input(step_index, step_cap, previous_ok):
    """Fixed 48-vector: step / cap, remaining / cap, previous action executed; no block identity, goal hash, planner distance, optimal action or split information."""
    row = [0.0] * OBS_DIM
    row[0] = step_index / step_cap
    row[1] = max(0, step_cap - step_index) / step_cap
    row[2] = 1.0 if previous_ok else 0.0
    return tuple(row)


class BwEpisode:
    """One episode on one case. ``snapshot()`` is the current decision's production Snapshot; ``step(action_id)`` executes it."""

    def __init__(self, case, env_id="bw-env-0", episode_id="ep-1"):
        self.case = case
        self.env_id = env_id
        self.episode_id = episode_id
        self.problem = case.problem
        self.template = C.template_for(case.problem)
        self.state = case.problem.init
        self.step_index = 0
        self.previous_ok = False
        self.done = False
        self.success = False
        self.reason = "CONTINUE"
        self._ids, self._feats = _candidate_features(self.template)
        self._action_of = {S.action_id(self.problem.names, a): a for a in C.all_actions(self.problem.n)}

    def legal_ids(self):
        return {S.action_id(self.problem.names, a) for a in S.legal_actions(self.state)}

    def snapshot(self):
        facts = S.fact_values(self.problem, self.state)
        records = tuple(FactRecord(k, Truth.TRUE if v else Truth.FALSE, 0.0, 0.0) for k, v in facts.items())
        legal = self.legal_ids()
        mask = tuple(i in legal for i in self._ids)
        return Snapshot(self.env_id, self.episode_id, self.step_index, self.template, FactStore(records), self._ids, mask, (), digest(()),
                        base_input(self.step_index, self.case.step_cap, self.previous_ok), self._feats, "bw:%s:%d" % (self.episode_id, self.step_index), float(self.step_index))

    def step(self, action_id):
        """Execute a legal action: returns (reward, terminated, truncated, reason). Success reward is 1.0 the first time every goal atom is TRUE."""
        if self.done:
            raise RuntimeError("episode finished")
        action = self._action_of[action_id]
        if action not in S.legal_actions(self.state):
            raise RuntimeError("masked action executed: %s" % action_id)
        self.state = S.apply(self.state, action)
        self.step_index += 1
        self.previous_ok = True
        if S.goal_satisfied(self.problem, self.state):
            self.done, self.success, self.reason = True, True, "TASK_SUCCESS"
            return 1.0, True, False, "TASK_SUCCESS"
        if self.step_index >= self.case.step_cap:
            self.done, self.reason = True, "DEADLINE"
            return 0.0, True, False, "DEADLINE"
        return 0.0, False, False, "CONTINUE"


def discount():
    """Per-action discount with the frozen half-life (one symbolic time unit per action)."""
    return gamma(1.0)
