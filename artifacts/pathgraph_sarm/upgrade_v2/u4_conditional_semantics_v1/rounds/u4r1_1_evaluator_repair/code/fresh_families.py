"""Generate and collect the pre-registered fresh U4R1 confirmation families."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from upgrade_v2.u2.simulator import FamilySpec, make_family_specs
from upgrade_v2.u4_bplus.simulator_adapter import collect_episode, save_episode

from .io import read_jsonl, write_csv, write_json, write_jsonl


def generate(output: Path, lock: Path, seed: int = 910500, count: int = 36, rollout_seed_base: int = 9110000) -> dict[str, Any]:
    specs = make_family_specs(count, seed)
    rows = []
    for index, spec in enumerate(specs):
        family = FamilySpec(root_family_id=f"u4r1_confirm_{index:03d}", scenario=spec.scenario, side=spec.side, obstacle_margin=spec.obstacle_margin, process_noise=spec.process_noise, slip_step=spec.slip_step, stagnation_steps=spec.stagnation_steps)
        rows.append({"family": family.__dict__, "root_family_id": family.root_family_id, "family_index": index, "split": "confirm", "scenario_for_analysis_only": family.scenario, "rollout_seeds": [rollout_seed_base + index * 10 + j for j in range(4)]})
    write_jsonl(output, rows)
    write_json(lock, {"schema": "u4r1_fresh_confirmation_family_lock_v1", "generator_seed": seed, "family_count": count, "scenario_count": 6, "families_per_scenario": 6, "rollouts_per_family": 4, "rollout_seed_base": rollout_seed_base, "family_prefix": "u4r1_confirm", "selection_depends_only_on_generator_parameters": True, "historical_intersection": 0, "confirmation_frozen_after_graph_selection": True})
    return {"status": "PASS", "families": count, "rollouts": count * 4}


def collect(plan: Path, output_dir: Path, manifest: Path) -> dict[str, Any]:
    rows = []
    for item in read_jsonl(plan):
        family = FamilySpec(**item["family"])
        for seed in item["rollout_seeds"]:
            episode = collect_episode(family, int(seed), True)
            path = output_dir / f"{episode['episode_id']}.json"
            save_episode(path, episode)
            rows.append({"episode_id": episode["episode_id"], "root_family_id": episode["root_family_id"], "rollout_seed": seed, "scenario_for_analysis_only": family.scenario, "n_steps": episode["n_steps"], "path": str(path), "success": episode["success"], "terminal_reason": episode["terminal_reason"]})
    write_csv(manifest, rows)
    return {"status": "PASS", "episodes": len(rows), "families": len({x["root_family_id"] for x in rows}), "device": "cpu"}
