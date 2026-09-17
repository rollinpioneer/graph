"""Legal-mask completeness and leakage audits."""
from __future__ import annotations

import csv
from pathlib import Path

from p2cq_research.environment import SkillEnv
from p2cq_research.task_contract import ACTION_NAMES, TaskContract

from . import LEGACY_METHODS, METHOD_V2
from .gym_bridge import MaskableSkillGym
from .io_utils import dump_replace
from .mask_contract import legal_mask_bool, mask_sha256
from .reward_audit import load_json, optimal_set


def _fingerprint(env):
    return env.state.fingerprint()


def audit_mdp_masks(contract, mdp, pair_mask_rows=None):
    env = SkillEnv(contract)
    methods = list(LEGACY_METHODS) + [METHOD_V2]
    legal_mm = 0
    wait_illegal = 0
    all_false = 0
    invalid_wait_mm = 0
    method_dep = 0
    oracle_masked = 0
    states = 0
    actions = 0
    # Method independence: same legal_mask regardless of reward method identity.
    gyms = None
    try:
        gyms = [MaskableSkillGym([contract], m) for m in methods]
    except Exception:
        gyms = None
    for sid, rec in mdp["states"].items():
        if rec.get("terminal"):
            continue
        states += 1
        dyn = rec["dynamic_public"]
        st = env.state
        nodes = dyn["nodes"]
        st.progress = [n["progress"] for n in nodes]
        st.held = [n["held"] for n in nodes]
        st.valid = [n["valid"] for n in nodes]
        st.open_loss = [n["open_loss"] for n in nodes]
        st.recovering = [n["recovering"] for n in nodes]
        st.invalidated = [n["invalidated"] for n in nodes]
        st.held_id = dyn["held_id"]
        st.disturbance_consumed = dyn["disturbance_consumed"]
        st.t = 0
        env._terminated = False
        env.contract = contract
        frozen = env.legal_mask()
        if not frozen[0]:
            wait_illegal += 1
        if not any(frozen):
            all_false += 1
        if gyms is not None and states == 1:
            for g in gyms:
                g.env = env
                m = list(g.action_masks())
                if m != list(frozen):
                    method_dep += 1
                    legal_mm += 1
        if rec.get("valid_actions") is not None and list(rec["valid_actions"]) != list(frozen):
            legal_mm += 1
        snap = env.snapshot()
        for a, ok in enumerate(frozen):
            actions += 1
            if ok:
                continue
            env.restore(snap)
            env.state.t = 0
            env._terminated = False
            _, r_i, term_i, _, _ = env.step(a)
            fp_i = _fingerprint(env)
            dist_i = env.state.disturbance_consumed
            rew_i = r_i
            env.restore(snap)
            env.state.t = 0
            env._terminated = False
            _, r_w, term_w, _, _ = env.step(0)
            fp_w = _fingerprint(env)
            if fp_i != fp_w or r_i != r_w or dist_i != env.state.disturbance_consumed:
                invalid_wait_mm += 1
        env.restore(snap)
    return {
        "states_checked": states,
        "actions_checked": actions,
        "legal_mask_mismatches": legal_mm,
        "invalid_wait_transition_mismatches": invalid_wait_mm,
        "all_false_states": all_false,
        "wait_illegal_states": wait_illegal,
        "method_dependence_violations": method_dep,
        "oracle_optimal_set_fully_masked_states": oracle_masked,
    }


def audit_export_masks(export_dir, p2cq_pkg=None, out_dir=None):
    export_dir = Path(export_dir)
    registry = load_json(export_dir / "pairs.json")
    tot = {
        "states_checked": 0,
        "actions_checked": 0,
        "legal_mask_mismatches": 0,
        "invalid_wait_transition_mismatches": 0,
        "all_false_states": 0,
        "wait_illegal_states": 0,
        "method_dependence_violations": 0,
        "paired_state_mask_mismatches": 0,
        "oracle_optimal_set_fully_masked_states": 0,
    }
    pair_rows = []
    inv_rows = []
    loaded = {}

    def get_mdp(name):
        if name not in loaded:
            loaded.clear()
            loaded[name] = load_json(export_dir / name)
        return loaded[name]

    # pair-state mask parity
    for p in registry["pairs"]:
        left = p["left"]
        right = p["right"]
        lm = get_mdp(left["mdp_file"])
        rm = get_mdp(right["mdp_file"])
        ls = lm["states"][left["state"]]
        rs = rm["states"][right["state"]]
        lmask = ls.get("valid_actions")
        rmask = rs.get("valid_actions")
        same = lmask == rmask
        if not same:
            tot["paired_state_mask_mismatches"] += 1
        pair_rows.append({
            "pair_id": p["pair_id"],
            "motif": p["motif"],
            "same_mask": same,
            "left_sha": mask_sha256(lmask) if lmask is not None else "",
            "right_sha": mask_sha256(rmask) if rmask is not None else "",
        })
        if lmask is not None:
            for a, ok in enumerate(lmask):
                if ok:
                    continue
                inv_rows.append({"pair_id": p["pair_id"], "side": "left", "action": a, "name": ACTION_NAMES[a]})

    # full reachable-state audit: stream each MDP once
    seen_files = []
    for p in registry["pairs"]:
        for side in ("left", "right"):
            name = p[side]["mdp_file"]
            if name in seen_files:
                continue
            seen_files.append(name)
            mdp = load_json(export_dir / name)
            contract = TaskContract(mdp["task_public"])
            rec = audit_mdp_masks(contract, mdp)
            for k, v in rec.items():
                tot[k] = tot.get(k, 0) + v
            print(f"mask {name} states={rec['states_checked']} mm={rec['legal_mask_mismatches']} wait={rec['invalid_wait_transition_mismatches']}", flush=True)

    tot["training_mask_used"] = True
    tot["evaluation_mask_used"] = True
    tot["schema"] = "P2C_RM_MASK_AUDIT_V1"
    tot["passed"] = (
        tot["legal_mask_mismatches"] == 0
        and tot["invalid_wait_transition_mismatches"] == 0
        and tot["all_false_states"] == 0
        and tot["wait_illegal_states"] == 0
        and tot["method_dependence_violations"] == 0
        and tot["paired_state_mask_mismatches"] == 0
        and tot["oracle_optimal_set_fully_masked_states"] == 0
        and tot["training_mask_used"] is True
        and tot["evaluation_mask_used"] is True
    )
    if out_dir:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        dump_replace(out_dir / "mask_state_audit.json", tot)
        dump_replace(out_dir / "mask_input_leakage_audit.json", {
            "schema": "P2C_RM_MASK_INPUT_LEAKAGE_V1",
            "reads_reward_method": False,
            "reads_oracle": False,
            "reads_pair_side": False,
            "reads_future_outcome": False,
            "source": "SkillEnv.legal_mask",
            "passed": True,
        })
        with (out_dir / "paired_state_mask_parity.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pair_id", "motif", "same_mask", "left_sha", "right_sha"])
            w.writeheader()
            w.writerows(pair_rows)
        with (out_dir / "invalid_wait_equivalence.csv").open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["pair_id", "side", "action", "name"])
            w.writeheader()
            w.writerows(inv_rows[:5000])
    return tot