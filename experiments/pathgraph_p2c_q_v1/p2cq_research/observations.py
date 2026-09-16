"""Single public observation encoder. No method argument, no oracle fields."""
from __future__ import annotations

from .task_contract import N, TaskContract, digest
from .environment import SkillEnv, current_xy

LAYOUT_DIM = 224
NODE_FEAT = 12


def encode_observation(contract: TaskContract, env: SkillEnv):
    st = env.state
    remaining = max(0, contract.horizon - st.t)
    vec = [0.0] * LAYOUT_DIM
    p = 0
    for i in range(N):
        present = 1.0 if contract.node_mask[i] else 0.0
        k = contract.transport_steps[i]
        sx, sy = contract.start_xy[i]
        tx, ty = contract.target_xy[i]
        prog = (st.progress[i] / k) if k else 0.0
        feats = [
            float(sx), float(sy), float(tx), float(ty),
            float(prog), float(k / 3.0 if k else 0.0),
            1.0 if st.held[i] else 0.0,
            1.0 if st.valid[i] else 0.0,
            1.0 if st.open_loss[i] else 0.0,
            1.0 if st.recovering[i] else 0.0,
            1.0 if st.invalidated[i] else 0.0,
            present,
        ]
        vec[p:p + NODE_FEAT] = feats
        p += NODE_FEAT
    for i in range(N):
        for c in range(2):
            for j in range(N):
                vec[p] = 1.0 if contract.preconditions_dnf[i][c][j] else 0.0
                p += 1
    for i in range(N):
        for c in range(2):
            vec[p] = 1.0 if contract.precondition_clause_mask[i][c] else 0.0
            p += 1
    for c in range(2):
        for j in range(N):
            vec[p] = 1.0 if contract.goal_dnf[c][j] else 0.0
            p += 1
    for c in range(2):
        vec[p] = 1.0 if contract.goal_clause_mask[c] else 0.0
        p += 1
    for i in range(N):
        for j in range(N):
            vec[p] = 1.0 if contract.invalidation_dependency[i][j] else 0.0
            p += 1
    rule = contract.disturbance_rule
    none = rule["kind"] == "NONE"
    one = rule["kind"] == "ONE_SHOT"
    trig = [0.0] * N
    tgt = [0.0] * N
    thr = 0.0
    if one:
        trig[rule["trigger_node"]] = 1.0
        tgt[rule["target_node"]] = 1.0
        tk = contract.transport_steps[rule["trigger_node"]]
        thr = rule["trigger_progress"] / tk if tk else 0.0
    glob = (
        [1.0 if none else 0.0, 1.0 if one else 0.0]
        + trig + tgt
        + [float(thr), 1.0 if st.disturbance_consumed else 0.0,
           remaining / 128.0, contract.horizon / 128.0]
    )
    if len(glob) != 18:
        raise RuntimeError("global layout")
    vec[p:p + 18] = glob
    p += 18
    if p != LAYOUT_DIM:
        raise RuntimeError(f"layout {p}")
    return tuple(vec)


def observation_bytes(contract, env):
    import struct
    return struct.pack("<224f", *encode_observation(contract, env))


def observation_digest(contract, env):
    import hashlib
    return hashlib.sha256(observation_bytes(contract, env)).hexdigest()


def public_input_digest(contract, env):
    return digest({
        "task": contract.public_dict(),
        "dynamic": env.dynamic_public(),
        "nongraph": env.nongraph_public(),
        "remaining": max(0, contract.horizon - env.state.t),
        "horizon": contract.horizon,
    })
