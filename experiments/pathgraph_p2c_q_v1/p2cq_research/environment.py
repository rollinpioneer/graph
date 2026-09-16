"""Finite abstract skill environment. Imports only the contract and pure math.

Success is goal-DNF on current valid latches. Invalid skill actions are
no-ops that still consume one time unit. PLACE with unsatisfied
preconditions is legal but does not certify the node.
"""
from __future__ import annotations

from copy import deepcopy

from .task_contract import ACTION_NAMES, N, TaskContract, action_index, dnf_satisfied, node_op


class State:
    __slots__ = (
        "progress", "held", "valid", "open_loss", "recovering", "invalidated",
        "held_id", "disturbance_consumed", "t",
    )

    def __init__(self, n=N):
        self.progress = [0] * n
        self.held = [False] * n
        self.valid = [False] * n
        self.open_loss = [False] * n
        self.recovering = [False] * n
        self.invalidated = [False] * n
        self.held_id = -1
        self.disturbance_consumed = False
        self.t = 0

    def clone(self):
        s = State()
        s.progress = list(self.progress)
        s.held = list(self.held)
        s.valid = list(self.valid)
        s.open_loss = list(self.open_loss)
        s.recovering = list(self.recovering)
        s.invalidated = list(self.invalidated)
        s.held_id = self.held_id
        s.disturbance_consumed = self.disturbance_consumed
        s.t = self.t
        return s

    def fingerprint(self):
        # Time is the oracle horizon remainder, not a transition key.
        return (
            tuple(self.progress), tuple(self.held), tuple(self.valid),
            tuple(self.open_loss), tuple(self.recovering), tuple(self.invalidated),
            self.held_id, self.disturbance_consumed,
        )


def current_xy(contract, state, i):
    k = contract.transport_steps[i]
    if k <= 0:
        return list(contract.start_xy[i])
    r = state.progress[i] / k
    sx, sy = contract.start_xy[i]
    tx, ty = contract.target_xy[i]
    return [sx + r * (tx - sx), sy + r * (ty - sy)]


class SkillEnv:
    def __init__(self, contract: TaskContract):
        self.contract = contract
        self.state = State()
        self._terminated = False
        self._success = False
        self._reward_history = []

    def reset(self, contract=None):
        if contract is not None:
            self.contract = contract if isinstance(contract, TaskContract) else TaskContract(contract)
        self.state = State()
        self._terminated = False
        self._success = False
        self._reward_history = []
        return self.public_observation()

    def snapshot(self):
        return {
            "state": self.state.clone(),
            "terminated": self._terminated,
            "success": self._success,
            "reward_history": list(self._reward_history),
        }

    def restore(self, snap):
        self.state = snap["state"].clone()
        self._terminated = bool(snap["terminated"])
        self._success = bool(snap["success"])
        self._reward_history = list(snap["reward_history"])

    def legal_mask(self, st=None):
        st = self.state if st is None else st
        c = self.contract
        mask = [False] * 37
        mask[0] = True
        any_held = st.held_id >= 0
        for i in c.present_nodes():
            mask[action_index(i, "ACQUIRE")] = (not any_held) and (not st.open_loss[i])
            mask[action_index(i, "ADVANCE")] = st.held[i]
            mask[action_index(i, "PLACE")] = st.held[i] and st.progress[i] == c.transport_steps[i]
            mask[action_index(i, "START_RECOVERY")] = st.open_loss[i] and (not any_held)
            mask[action_index(i, "REGRASP")] = st.open_loss[i] and st.recovering[i] and (not any_held)
            mask[action_index(i, "RELEASE")] = st.held[i]
        return mask

    def _apply_invalidation(self, st, seeds):
        stack = list(seeds)
        seen = set()
        while stack:
            i = stack.pop()
            if i in seen:
                continue
            seen.add(i)
            for j in self.contract.present_nodes():
                if not self.contract.invalidation_dependency[i][j]:
                    continue
                if not st.valid[j]:
                    continue
                if self.contract.preconditions_satisfied(j, st.valid):
                    continue
                st.valid[j] = False
                st.invalidated[j] = True
                stack.append(j)

    def _apply_disturbance(self, st):
        rule = self.contract.disturbance_rule
        if rule["kind"] != "ONE_SHOT" or st.disturbance_consumed:
            return
        trig = rule["trigger_node"]
        if st.progress[trig] < rule["trigger_progress"]:
            return
        tgt = rule["target_node"]
        became = []
        if st.valid[tgt]:
            became.append(tgt)
        st.held[tgt] = False
        if st.held_id == tgt:
            st.held_id = -1
        if st.valid[tgt]:
            st.valid[tgt] = False
        st.open_loss[tgt] = True
        st.recovering[tgt] = False
        st.progress[tgt] = st.progress[tgt] // 2
        st.disturbance_consumed = True
        if became:
            self._apply_invalidation(st, became)

    def _apply_skill(self, st, action):
        legal = self.legal_mask(st)
        if action < 0 or action >= 37 or not legal[action]:
            return
        j, op = node_op(action)
        if op == "WAIT":
            return
        c = self.contract
        if op == "ACQUIRE":
            became = []
            if st.valid[j]:
                became.append(j)
            st.held[j] = True
            st.held_id = j
            st.valid[j] = False
            if became:
                self._apply_invalidation(st, became)
        elif op == "ADVANCE":
            st.progress[j] = min(c.transport_steps[j], st.progress[j] + 1)
        elif op == "PLACE":
            st.held[j] = False
            st.held_id = -1
            if c.preconditions_satisfied(j, st.valid):
                st.valid[j] = True
            else:
                st.valid[j] = False
                st.invalidated[j] = True
        elif op == "START_RECOVERY":
            st.recovering[j] = True
        elif op == "REGRASP":
            st.held[j] = True
            st.held_id = j
            st.open_loss[j] = False
            st.recovering[j] = False
        elif op == "RELEASE":
            st.held[j] = False
            st.held_id = -1
            if st.valid[j]:
                st.valid[j] = False
                st.invalidated[j] = True
                self._apply_invalidation(st, [j])
            else:
                st.valid[j] = False

    def step(self, action):
        if self._terminated:
            raise RuntimeError("step after terminal")
        if type(action) is not int or not 0 <= action < 37:
            raise ValueError("action")
        st = self.state
        self._apply_skill(st, action)
        self._apply_disturbance(st)
        st.t += 1
        success = self.contract.goal_satisfied(st.valid)
        reward = 1.0 if success else 0.0
        terminated = bool(success or st.t >= self.contract.horizon)
        truncated = False
        self._success = bool(success)
        self._terminated = terminated
        self._reward_history.append(reward)
        info = {"success": self._success}
        return self.public_observation(), reward, terminated, truncated, info

    def dynamic_public(self, st=None):
        st = self.state if st is None else st
        c = self.contract
        nodes = []
        for i in range(N):
            nodes.append({
                "present": c.node_mask[i],
                "progress": st.progress[i],
                "held": st.held[i],
                "valid": st.valid[i],
                "open_loss": st.open_loss[i],
                "recovering": st.recovering[i],
                "invalidated": st.invalidated[i],
                "xy": current_xy(c, st, i) if c.node_mask[i] else [0.0, 0.0],
            })
        return {
            "nodes": nodes,
            "held_id": st.held_id,
            "disturbance_consumed": st.disturbance_consumed,
        }

    def nongraph_public(self, st=None):
        st = self.state if st is None else st
        c = self.contract
        nodes = []
        for i in range(N):
            nodes.append({
                "present": c.node_mask[i],
                "start_xy": list(c.start_xy[i]),
                "target_xy": list(c.target_xy[i]),
                "K": c.transport_steps[i],
                "progress": st.progress[i],
                "held": st.held[i],
                "valid": st.valid[i],
                "open_loss": st.open_loss[i],
                "recovering": st.recovering[i],
                "invalidated": st.invalidated[i],
            })
        return {
            "nodes": nodes,
            "held_id": st.held_id,
            "disturbance_consumed": st.disturbance_consumed,
        }

    def public_observation(self):
        remaining = max(0, self.contract.horizon - self.state.t)
        return {
            "task_public": self.contract.public_dict(),
            "dynamic_public": self.dynamic_public(),
            "remaining_steps": remaining,
            "horizon": self.contract.horizon,
        }

    def public_reward_state(self):
        return {
            "task_public": self.contract.public_dict(),
            "dynamic_public": self.dynamic_public(),
            "nongraph_public": self.nongraph_public(),
            "remaining_steps": max(0, self.contract.horizon - self.state.t),
            "horizon": self.contract.horizon,
            "terminated": self._terminated,
            "success": self._success,
        }

    def terminated(self):
        return self._terminated

    def success(self):
        return self._success


ACTION_NAMES  # re-export
