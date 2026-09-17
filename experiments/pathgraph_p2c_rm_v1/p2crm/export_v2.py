"""Independent confirmation / smoke pair export. Slot prefixes ignore oracle."""
from __future__ import annotations

from pathlib import Path

from p2cq_research.enumerate_mdp import enumerate_mdp, state_id
from p2cq_research.environment import SkillEnv
from p2cq_research.generator import SLOTS_PER_ROOT
from p2cq_research.task_contract import K_ALLOWED, TaskContract

from .generator_v2 import CONF_START, SMOKE_START, iter_split, planned_pairs
from .io_utils import dump_replace
from .remaining_work_model import RemainingWorkPlanner, UnreachableWorkError, SearchTruncatedError


def replay_prefix(contract, actions):
    env = SkillEnv(contract)
    env.reset()
    prefix = []
    for a in actions:
        if env.terminated():
            return None, "terminated_before_end"
        _, _, term, _, _ = env.step(int(a))
        nxt = state_id(env.dynamic_public())
        prefix.append({"action": int(a), "next": nxt})
        if term and a != actions[-1]:
            return None, "early_terminal"
    remaining = contract.horizon - env.state.t
    return {
        "state": state_id(env.dynamic_public()),
        "remaining_steps": remaining,
        "prefix": prefix,
    }, None


def assert_contract(contract: TaskContract):
    if contract.horizon > 128 or contract.horizon < 1:
        raise RuntimeError("horizon")
    for i in contract.present_nodes():
        if contract.transport_steps[i] not in K_ALLOWED:
            raise RuntimeError(f"K not allowed {contract.transport_steps[i]}")


def export_split(split, out_dir, *, write_mdp=False, progress_cb=None):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    pairs = []
    registry_rows = []
    n_mdp_states = 0
    omitted = 0
    n_done = 0
    mdps = {}
    for spec in iter_split(split):
        n_done += 1
        fam = spec["family"]
        motif = spec["motif"]
        left_c, right_c = spec["left"], spec["right"]
        assert_contract(left_c)
        assert_contract(right_c)
        if progress_cb:
            progress_cb(n_done, fam, motif)
        print(f"export {split} {n_done} family={fam} motif={motif}", flush=True)
        l_mdp, _ = enumerate_mdp(left_c)
        r_mdp, _ = enumerate_mdp(right_c)
        omitted += int(not l_mdp["meta"]["complete"]) + int(not r_mdp["meta"]["complete"])
        n_mdp_states += l_mdp["meta"]["n_states"] + r_mdp["meta"]["n_states"]
        l_name = f"mdp/{fam}_left.json"
        r_name = f"mdp/{fam}_right.json"
        if write_mdp:
            dump_replace(out / l_name, l_mdp)
            dump_replace(out / r_name, r_mdp)
        mdps = {l_name: l_mdp, r_name: r_mdp}
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
        # drop bulky mdps if not writing; caller may use returned mdps for current family only
    planned = planned_pairs(split)
    if len(pairs) != planned:
        raise RuntimeError(f"pair count {len(pairs)} != {planned}")
    payload = {
        "schema": "P2CRM_PAIR_EXPORT_V1",
        "evidence_tier": "FINITE_ABSTRACT_SKILL_MDP_QUALIFICATION",
        "planned_pair_count": planned,
        "pairs": pairs,
        "split": split,
        "family_start": CONF_START if split == "confirmation" else SMOKE_START,
    }
    dump_replace(out / "pairs.json", payload)
    dump_replace(out / "slot_registry.json", {
        "rows": registry_rows,
        "n_mdp_states": n_mdp_states,
        "omitted": omitted,
        "planned_pairs": planned,
        "all_slots_in_denominator": True,
    })
    return {
        "n_pairs": len(pairs),
        "n_mdp_states": n_mdp_states,
        "omitted": omitted,
        "out": str(out),
        "registry": payload,
        "mdps": mdps,
    }


def collision_scan(export_dirs, new_ids):
    seen = set()
    for d in export_dirs:
        p = Path(d) / "pairs.json"
        if not p.exists():
            continue
        data = __import__("json").loads(p.read_text(encoding="utf-8"))
        for pair in data.get("pairs", []):
            seen.add(str(pair.get("family")))
    hits = [i for i in new_ids if str(i) in seen]
    return {"existing": len(seen), "collisions": hits, "passed": not hits}