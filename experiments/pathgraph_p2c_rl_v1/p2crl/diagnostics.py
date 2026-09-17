"""Prepare-time fairness and leakage tests. Zero gradients."""
from __future__ import annotations
import hashlib
import numpy as np
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import encode_observation
from p2cq_research.potentials import evaluate_all
from p2cq_research.task_contract import canonical
from p2crm.mask_contract import legal_mask_bool
from p2crm.potentials_v2 import METHOD_V2, evaluate_all_with_v2, shaped_training_reward
from p2crm.remaining_work_model import RemainingWorkPlanner
from . import METHODS
from .gym_adapter import RLTaskEnv
from .io_utils import write_new

def observation_has_identity(vec, contract):
    text = ",".join(f"{x:.8f}" for x in vec)
    banned = [str(contract.family_id), contract.split, contract.version, contract.motif]
    banned.extend([x for x in contract.node_ids if x])
    hits = [b for b in banned if b and b in text]
    return hits

def prefix_fairness(contracts, prefixes, methods=None):
    methods = methods or METHODS
    env = SkillEnv(contracts[0])
    rows = []
    for contract, pref in zip(contracts, prefixes):
        states = {}
        for method in methods:
            env.reset(contract)
            for a in pref:
                if env.terminated():
                    break
                env.step(int(a))
            obs = encode_observation(contract, env)
            mask = tuple(bool(x) for x in legal_mask_bool(env))
            dyn = canonical(env.dynamic_public())
            states[method] = (hashlib.sha256(canonical(list(obs)).encode()).hexdigest(), mask, dyn)
        ok = len({v[0] for v in states.values()}) == 1 and len({v[1] for v in states.values()}) == 1 and len({v[2] for v in states.values()}) == 1
        rows.append({"ok": ok, "family_id": contract.family_id})
        if not ok:
            raise RuntimeError("prefix fairness failed")
    return rows

def reward_parity(contracts, n_actions=12):
    if not isinstance(contracts, (list, tuple)):
        raise TypeError("reward_parity requires a motif-balanced contract list")
    if len(contracts) < 8:
        raise RuntimeError("reward_parity needs motif-balanced left/right contracts")
    contract = contracts[0]
    env = SkillEnv(contract)
    env.reset(contract)
    planner = RemainingWorkPlanner(contract)
    errors = []
    adapters = {}
    phi = {}
    dyn = env.dynamic_public()
    table = evaluate_all_with_v2(contract, dyn, planner=planner)
    for method in METHODS:
        adapter = RLTaskEnv(list(contracts), method, 1, 0)
        adapter.reset(options={"contract_index": 0})
        adapters[method] = adapter
        phi[method] = 0.0 if method == "TASK_ONLY_ZERO_V1" else float(table[method])
        if abs(float(adapter._phi) - float(phi[method])) > 1e-10:
            raise RuntimeError(f"initial phi mismatch {method}")
    steps = 0
    while steps < n_actions and not env.terminated():
        mask = legal_mask_bool(env)
        legal = np.flatnonzero(mask)
        if len(legal) == 0:
            break
        action = int(legal[0])
        _o, task_r, term, trunc, _i = env.step(action)
        dyn = env.dynamic_public()
        table = evaluate_all_with_v2(contract, dyn, planner=planner)
        for method in METHODS:
            raw = 0.0 if method == "TASK_ONLY_ZERO_V1" else float(table[method])
            got, _ = shaped_training_reward(task_r, phi[method], raw, gamma=0.99, beta=1.0, terminated=bool(term))
            _ao, ar, at, atr, ainfo = adapters[method].step(action)
            if not (np.isfinite(got) and np.isfinite(ar)):
                errors.append(method)
                raise RuntimeError(f"nonfinite reward {method}")
            if abs(float(ar) - float(got)) > 1e-10 and (not np.isclose(float(ar), float(got), atol=1e-10, rtol=1e-9)):
                raise RuntimeError(f"reward parity failed {method}: adapter={ar} frozen={got}")
            if bool(at) != bool(term):
                raise RuntimeError(f"termination mismatch {method}")
            phi[method] = 0.0 if term else raw
        steps += 1
    return {"steps": steps, "errors": errors}

def cache_consistency(contract):
    env = SkillEnv(contract)
    env.reset()
    dyn = env.dynamic_public()
    a = RemainingWorkPlanner(contract)
    b = RemainingWorkPlanner(contract)
    pa = a.potential(dyn, use_cache=True)[0]
    pb = b.potential(dyn, use_cache=False)[0]
    if abs(pa - pb) > 1e-12:
        raise RuntimeError("cache mismatch")
    return True
