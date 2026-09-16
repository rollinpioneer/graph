"""Export all pre-registered pair slots plus candidate potentials.

Does not read oracle optimal actions to select slots. Failed slots remain
in the registry so the denominator stays 2048.
"""
from __future__ import annotations

import json
from pathlib import Path

from .enumerate_mdp import enumerate_mdp, state_id, BudgetExceeded
from .environment import SkillEnv
from .generator import SLOTS_PER_ROOT, iter_split
from .potentials import REQUIRED, evaluate_all
from .task_contract import canonical


def _dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        raise FileExistsError(path)
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    path.write_text(raw, encoding="ascii")
    return path


def replay_prefix(contract, actions):
    env = SkillEnv(contract)
    env.reset()
    prefix = []
    for a in actions:
        if env.terminated():
            return None, "terminated_before_end"
        before = env.dynamic_public()
        _, _, term, _, _ = env.step(a)
        nxt = state_id(env.dynamic_public())
        prefix.append({"action": int(a), "next": nxt})
        if term and a != actions[-1]:
            return None, "early_terminal"
    remaining = contract.horizon - env.state.t
    return {
        "state": state_id(env.dynamic_public()),
        "remaining_steps": remaining,
        "prefix": prefix,
        "nongraph": env.nongraph_public(),
    }, None


def potentials_table(contract, mdp, clip_hits):
    methods = {k: {} for k in REQUIRED}
    env = SkillEnv(contract)
    for sid, rec in mdp["states"].items():
        env.state.progress = [n["progress"] for n in rec["dynamic_public"]["nodes"]]
        env.state.held = [n["held"] for n in rec["dynamic_public"]["nodes"]]
        env.state.valid = [n["valid"] for n in rec["dynamic_public"]["nodes"]]
        env.state.open_loss = [n["open_loss"] for n in rec["dynamic_public"]["nodes"]]
        env.state.recovering = [n["recovering"] for n in rec["dynamic_public"]["nodes"]]
        env.state.invalidated = [n["invalidated"] for n in rec["dynamic_public"]["nodes"]]
        env.state.held_id = rec["dynamic_public"]["held_id"]
        env.state.disturbance_consumed = rec["dynamic_public"]["disturbance_consumed"]
        vals = evaluate_all(contract, rec["dynamic_public"], clip_hits)
        for k in REQUIRED:
            methods[k][sid] = float(vals[k])
    return {"time_independent": True, "methods": methods}


def export_split(split, out_dir, max_states=200000):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=False)
    pairs = []
    candidate_files = {}
    registry_rows = []
    clip_hits = {"clip": 0, "unreachable": 0, "states": 0}
    n_mdp_states = 0
    omitted = 0
    n_done = 0
    for spec in iter_split(split):
        fam = spec["family"]
        motif = spec["motif"]
        n_done += 1
        print(f"export {split} {n_done}/128 family={fam} motif={motif}", flush=True)
        left_c, right_c = spec["left"], spec["right"]
        l_mdp, _ = enumerate_mdp(left_c, max_states=max_states)
        r_mdp, _ = enumerate_mdp(right_c, max_states=max_states)
        omitted += int(not l_mdp["meta"]["complete"]) + int(not r_mdp["meta"]["complete"])
        n_mdp_states += l_mdp["meta"]["n_states"] + r_mdp["meta"]["n_states"]
        l_name = f"mdp/{fam}_left.json"
        r_name = f"mdp/{fam}_right.json"
        p_l = f"phi/{fam}_left.json"
        p_r = f"phi/{fam}_right.json"
        _dump(out / l_name, l_mdp)
        _dump(out / r_name, r_mdp)
        _dump(out / p_l, potentials_table(left_c, l_mdp, clip_hits))
        _dump(out / p_r, potentials_table(right_c, r_mdp, clip_hits))
        clip_hits["states"] += l_mdp["meta"]["n_states"] + r_mdp["meta"]["n_states"]
        candidate_files[l_name] = p_l
        candidate_files[r_name] = p_r
        slots = spec["slots"]
        if len(slots) != SLOTS_PER_ROOT:
            raise RuntimeError("slot count")
        for si, actions in enumerate(slots):
            pair_id = f"{fam}_slot{si:02d}"
            left_side, lerr = replay_prefix(left_c, actions)
            right_side, rerr = replay_prefix(right_c, actions)
            row = {
                "pair_id": pair_id,
                "family": fam,
                "motif": motif,
                "slot": si,
                "left_error": lerr,
                "right_error": rerr,
                "structure_hash_left": left_c.structure_hash(),
                "structure_hash_right": right_c.structure_hash(),
                "parameter_hash": left_c.parameter_hash(),
            }
            if left_side is None or right_side is None:
                # Keep a syntactically valid witness by using empty prefix at initial if needed.
                # Qualification requires real prefixes; if replay failed, emit initial
                # state with empty prefix so the slot stays in the denominator.
                env_l = SkillEnv(left_c); env_l.reset()
                env_r = SkillEnv(right_c); env_r.reset()
                left_side = {
                    "mdp_file": l_name,
                    "state": state_id(env_l.dynamic_public()),
                    "remaining_steps": left_c.horizon,
                    "prefix": [],
                }
                right_side = {
                    "mdp_file": r_name,
                    "state": state_id(env_r.dynamic_public()),
                    "remaining_steps": right_c.horizon,
                    "prefix": [],
                }
                row["fallback_initial"] = True
            else:
                left_side = {
                    "mdp_file": l_name,
                    "state": left_side["state"],
                    "remaining_steps": left_side["remaining_steps"],
                    "prefix": left_side["prefix"],
                }
                right_side = {
                    "mdp_file": r_name,
                    "state": right_side["state"],
                    "remaining_steps": right_side["remaining_steps"],
                    "prefix": right_side["prefix"],
                }
                row["fallback_initial"] = False
            pairs.append({
                "pair_id": pair_id,
                "family": fam,
                "motif": motif,
                "left": left_side,
                "right": right_side,
            })
            registry_rows.append(row)
    planned = 4 * 32 * 16
    if len(pairs) != planned:
        raise RuntimeError(f"pair count {len(pairs)}")
    payload = {
        "schema": "P2CQ_PAIR_EXPORT_V1",
        "evidence_tier": "FINITE_ABSTRACT_SKILL_MDP_QUALIFICATION",
        "planned_pair_count": planned,
        "pairs": pairs,
        "candidate_files": candidate_files,
    }
    _dump(out / "pairs.json", payload)
    _dump(out / "slot_registry.json", {"rows": registry_rows, "clip_hits": clip_hits,
                                       "n_mdp_states": n_mdp_states, "omitted": omitted})
    return {
        "n_pairs": len(pairs),
        "n_mdps": len(candidate_files),
        "n_mdp_states": n_mdp_states,
        "omitted": omitted,
        "clip_hits": clip_hits,
    }
