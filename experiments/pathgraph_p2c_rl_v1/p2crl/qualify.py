"""New-dataset coverage gates. Does not read V1/V2 rewards to select families."""
from __future__ import annotations
from collections import defaultdict
from pathlib import Path
from p2cq_research.environment import SkillEnv
from p2cq_research.observations import LAYOUT_DIM, encode_observation
from p2cq_research.task_contract import TaskContract
from p2crm.mask_contract import legal_mask_bool
from . import MOTIFS, SPLITS
from .contracts import load_protocol, roots_for_split
from .io_utils import load_json, write_new
from .oracle import action_q, conservative_optimal, local_facts, remaining_cost, shortest_from_initial

PRIMARY_SPLITS = ["train_A", "train_B", "train_C", "validation", "test"]


def load_family_contracts(data_dir, fam):
    cdir = Path(data_dir) / "contracts"
    left = TaskContract(load_json(cdir / f"{fam['family_id']}_left.json"))
    right = TaskContract(load_json(cdir / f"{fam['family_id']}_right.json"))
    return left, right


def _run_prefix(env, prefix):
    env.reset()
    for a in prefix:
        if env.terminated():
            return True
        mask = env.legal_mask()
        if not mask[int(a)]:
            env.step(int(a))
            continue
        env.step(int(a))
        if env.terminated():
            return True
    return bool(env.terminated())


def qualify_dataset(protocol, data_dir, out_path, max_states_per=200000, campaign_state_budget=10000000):
    families = load_json(Path(data_dir) / "families.json")["families"]
    q = protocol["data"]["qualification"]
    cache = {}
    expanded_total = 0
    by = {s: {m: {"planned": 0, "separating_roots": 0, "separating_pairs": 0, "solvable": 0,
                  "min_len": [], "slack": [], "dyn": set(), "topo": set(), "full": set()}
              for m in MOTIFS} for s in SPLITS}
    collisions = []
    full_seen = {}
    rejected = []
    n_done = 0
    for fam in families:
        split, motif = fam["split"], fam["motif"]
        slot = by[split][motif]
        slot["planned"] += 1
        try:
            left, right = load_family_contracts(data_dir, fam)
            for side, c in ("left", left), ("right", right):
                env = SkillEnv(c)
                env.reset()
                obs = encode_observation(c, env)
                if len(obs) != LAYOUT_DIM:
                    raise RuntimeError("obs dim")
                mask = legal_mask_bool(env)
                if len(mask) != 37 or not bool(mask[0]):
                    raise RuntimeError("mask")
                cost, exp = shortest_from_initial(c, max_states=max_states_per, cache=cache)
                expanded_total += exp
                if expanded_total > campaign_state_budget:
                    raise RuntimeError("STATE_BUDGET_EXCEEDED")
                if cost == float("inf") or cost > c.horizon:
                    raise RuntimeError("unsolvable")
                slot["min_len"].append(cost)
                slot["slack"].append(c.horizon - cost)
                sig = fam["left_sig" if side == "left" else "right_sig"]
                full = sig["full"]
                if full in full_seen and full_seen[full] != (split, motif, fam["root"], side):
                    collisions.append({"full": full, "a": full_seen[full], "b": (split, motif, fam["root"], side)})
                full_seen[full] = (split, motif, fam["root"], side)
                slot["full"].add(full)
                slot["dyn"].add(sig["dynamics"])
                slot["topo"].add(sig["topology"])
            slot["solvable"] += 1
            n_done += 1
            if n_done % 16 == 0:
                print(f"QUALIFY {n_done}/{len(families)} {split} {motif} sep_roots={slot['separating_roots']} expanded={expanded_total}", flush=True)
            env_l, env_r = SkillEnv(left), SkillEnv(right)
            sep_here = 0
            for pref in fam["prefixes"]:
                env_l.reset(); env_r.reset()
                term_l = _run_prefix(env_l, pref)
                term_r = _run_prefix(env_r, pref)
                if term_l or term_r:
                    continue
                if local_facts(env_l) != local_facts(env_r):
                    continue
                ml = list(legal_mask_bool(env_l).tolist())
                mr = list(legal_mask_bool(env_r).tolist())
                if ml != mr:
                    continue
                ql, _ = action_q(env_l, max_states=max_states_per, cache=cache)
                qr, _ = action_q(env_r, max_states=max_states_per, cache=cache)
                ol = set(conservative_optimal(ql))
                orr = set(conservative_optimal(qr))
                if ol and orr and ol.isdisjoint(orr):
                    sep_here += 1
            slot["separating_pairs"] += sep_here
            if sep_here:
                slot["separating_roots"] += 1
        except Exception as exc:
            rejected.append({"family_id": fam["family_id"], "split": split, "motif": motif, "error": str(exc)})
            write_new(Path(out_path).with_name("qualification_rejected.json"), {"rejected": rejected})
            raise RuntimeError("RL_DATASET_QUALIFICATION_FAILED") from exc

    report = {
        "schema": "P2CRL_DATASET_QUALIFICATION_V1",
        "n_families": len(families),
        "expanded_states": expanded_total,
        "full_contract_collisions": collisions,
        "by_split": {},
        "gates": {},
    }
    n_primary_families = sum(1 for f in families if f["split"] in PRIMARY_SPLITS)
    n_primary_contracts = 2 * n_primary_families
    gates = {
        "constructable_1536": n_primary_contracts == 1536,
        "full_contract_hash_collision_count_max": len(collisions) <= int(q["full_contract_hash_collision_count_max"]),
        "no_rejected": not rejected,
    }
    for split in PRIMARY_SPLITS:
        report["by_split"][split] = {}
        for motif in MOTIFS:
            s = by[split][motif]
            rec = {
                "planned_roots": s["planned"],
                "solvable_roots": s["solvable"],
                "separating_roots": s["separating_roots"],
                "separating_pairs": s["separating_pairs"],
                "unique_dynamics": len(s["dyn"]),
                "unique_topology": len(s["topo"]),
                "unique_full": len(s["full"]),
                "mean_shortest": (sum(s["min_len"]) / len(s["min_len"])) if s["min_len"] else None,
                "mean_slack": (sum(s["slack"]) / len(s["slack"])) if s["slack"] else None,
            }
            report["by_split"][split][motif] = rec
            planned = roots_for_split(protocol, split)
            if s["planned"] != planned or s["solvable"] != planned:
                gates[f"{split}:{motif}:complete"] = False
            else:
                gates[f"{split}:{motif}:complete"] = True
            if split.startswith("train_") or split == "validation":
                gates[f"{split}:{motif}:sep_roots"] = s["separating_roots"] >= int(q["train_separating_roots_per_motif_min"] if split.startswith("train_") else q["validation_separating_roots_per_motif_min"])
            if split == "test":
                gates[f"{split}:{motif}:sep_roots"] = s["separating_roots"] >= int(q["test_separating_roots_per_motif_min"])
                gates[f"{split}:{motif}:sep_pairs"] = s["separating_pairs"] >= int(q["test_separating_pairs_per_motif_min"])
    report["n_primary_families"] = n_primary_families
    report["n_primary_contracts"] = n_primary_contracts
    report["gates"] = gates
    report["passed"] = all(bool(v) is True for v in gates.values())
    report["status"] = "PASS" if report["passed"] else "RL_DATASET_QUALIFICATION_FAILED"
    write_new(out_path, report)
    if not report["passed"]:
        raise RuntimeError("RL_DATASET_QUALIFICATION_FAILED")
    return report
