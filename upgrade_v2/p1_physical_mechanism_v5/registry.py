from __future__ import annotations
import json
from pathlib import Path

def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))

def planned_rollouts(families: dict, cases: dict) -> list[dict]:
    out=[]
    for fi, fam in enumerate(families["families"]):
        for case in cases["case_order"]:
            seed = int(fam["rollout_seed_base"]) + int(case["index"])
            out.append(dict(
                family_id=fam["family_id"], family_seed=int(fam["family_seed"]),
                case_id=case["case_id"], case_index=int(case["index"]),
                task=case["task"], required_semantics=case["required_semantics"],
                rollout_seed=seed, episode_id=f"{fam['family_id']}__{case['case_id']}",
            ))
    if len(out) != 112:
        raise ValueError(len(out))
    ids=[(r["family_id"], r["case_id"], r["rollout_seed"]) for r in out]
    if len(set(ids)) != 112:
        raise ValueError("nonunique")
    return out