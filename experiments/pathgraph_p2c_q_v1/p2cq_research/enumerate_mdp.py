"""Reachable-state export of the task-only finite MDP. No candidate filtering."""
from __future__ import annotations

from collections import deque

from .environment import SkillEnv
from .task_contract import ACTION_NAMES, digest


class BudgetExceeded(RuntimeError):
    pass


def state_id(dyn):
    return "s_" + digest(dyn)[:16]


def enumerate_mdp(contract, max_states=200000):
    env = SkillEnv(contract)
    env.reset()
    init = env.dynamic_public()
    init_id = state_id(init)
    states = {}
    prefixes = {init_id: []}
    q = deque([env.snapshot()])
    expanded = 0
    omitted = 0
    while q:
        snap = q.popleft()
        env.restore(snap)
        dyn = env.dynamic_public()
        sid = state_id(dyn)
        if sid in states:
            continue
        success = contract.goal_satisfied(env.state.valid)
        terminal = bool(success)
        rec = {
            "terminal": terminal,
            "success": bool(success),
            "dynamic_public": dyn,
            "nongraph_public": env.nongraph_public(),
        }
        if terminal:
            rec["outcomes"] = []
            states[sid] = rec
            expanded += 1
            if len(states) > max_states:
                raise BudgetExceeded("STATE_BUDGET_EXCEEDED")
            continue
        # Time is an oracle remainder, not a transition key. Clear horizon latch
        # so enumeration explores task dynamics rather than deadline absorption.
        env.state.t = 0
        env._terminated = False
        mask = env.legal_mask()
        outcomes = []
        rec["valid_actions"] = list(mask)
        rec["outcomes"] = outcomes
        states[sid] = rec
        expanded += 1
        if len(states) > max_states:
            raise BudgetExceeded("STATE_BUDGET_EXCEEDED")
        parent_fp = env.state.fingerprint()
        for a in range(37):
            env.restore(snap)
            env.state.t = 0
            env._terminated = False
            before = env.state.fingerprint()
            if before != parent_fp:
                raise RuntimeError("parent mutated")
            _, reward, term, trunc, _ = env.step(a)
            if trunc:
                omitted += 1
            ndyn = env.dynamic_public()
            nid = state_id(ndyn)
            nsuccess = contract.goal_satisfied(env.state.valid)
            # Sparse task reward: 1 iff successor is a success terminal.
            expected = 1.0 if nsuccess else 0.0
            if abs(reward - expected) > 1e-12:
                raise RuntimeError("reward contract")
            outcomes.append([{"p": 1.0, "next": nid, "task_reward": expected}])
            if nid not in prefixes:
                prefixes[nid] = prefixes[sid] + [{"action": a, "next": nid}]
                q.append(env.snapshot())
            # restore already done next loop
        env.restore(snap)
        if env.state.fingerprint() != parent_fp:
            raise RuntimeError("parent mutated after branch")
    data = {
        "schema": "P2CQ_FINITE_MDP_V1",
        "evidence_tier": "FINITE_ABSTRACT_SKILL_MDP_QUALIFICATION",
        "gamma": contract.gamma,
        "horizon": contract.horizon,
        "initial_state": init_id,
        "task_public": contract.public_dict(),
        "actions": list(ACTION_NAMES),
        "states": states,
        "meta": {
            "expanded": expanded,
            "omitted": omitted,
            "n_states": len(states),
            "complete": omitted == 0 and len(states) <= max_states,
            "structure_hash": contract.structure_hash(),
            "parameter_hash": contract.parameter_hash(),
            "task_hash": contract.task_hash(),
        },
    }
    if omitted != 0:
        data["meta"]["complete"] = False
    return data, prefixes
