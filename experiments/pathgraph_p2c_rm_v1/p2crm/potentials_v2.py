"""V2 potential wrapper. Does not modify frozen V1 potential bytes."""
from __future__ import annotations

from p2cq_research.potentials import evaluate_all

from .remaining_work_model import RemainingWorkPlanner, UnreachableWorkError, SearchTruncatedError

METHOD_V2 = "PATHGRAPH_REMAINING_WORK_PBRS_V2"


def phi_v2(contract, dyn, planner=None, use_cache=True):
    if planner is None:
        planner = RemainingWorkPlanner(contract)
    phi, cost, expands = planner.potential(dyn, use_cache=use_cache)
    return phi


def evaluate_all_with_v2(contract, dyn, planner=None, clip_hits=None):
    out = evaluate_all(contract, dyn, clip_hits)
    if planner is None:
        planner = RemainingWorkPlanner(contract)
    out[METHOD_V2] = planner.potential(dyn)[0]
    return out


def shaped_training_reward(task_reward, phi_before, phi_after, *, gamma, beta=1.0, terminated=False):
    effective = 0.0 if terminated else phi_after
    bonus = beta * (gamma * effective - phi_before)
    return task_reward + bonus, bonus