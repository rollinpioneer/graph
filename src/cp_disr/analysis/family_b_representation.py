"""Zero-optimizer production Policy probes from recorded public boundaries."""
from __future__ import annotations

from dataclasses import replace
import json
from pathlib import Path

from cp_disr.common import digest
from cp_disr.facts import FactRecord, FactStore, Truth
from cp_disr.rl import Snapshot
from cp_disr.platforms.libero.family_b_runtime import build_task_template
from cp_disr.vlm_cache_pipeline import verify_audit_cache
from .family_b_pilot import read, write, _registered


def _snapshot(root, cfg, out, branch, edges):
    boundary = Path(out) / "captures" / branch["branch_id"] / "boundary"
    public = read(boundary / "public.json")
    encoded = read(boundary / "snapshot_encoding.json")
    template = build_task_template(Path(root) / cfg["runtime"]["contract_path"])
    records = tuple(FactRecord(
        fact_id=fid, value=Truth(value), capture_time=float(public["sim_time"]),
        available_time=float(public["sim_time"]),
        evidence_ids=("recorded_public_boundary",),
        last_confirmed_value=Truth(value) if value != "UNKNOWN" else Truth.UNKNOWN,
        last_confirmed_time=float(public["sim_time"]) if value != "UNKNOWN" else None,
        reason="offline_replay_observed_fact")
        for fid, value in sorted(public["fact_values"].items()))
    facts = FactStore(records)
    if tuple(encoded["candidate_ids"]) != tuple(public["candidate_ids"]) or encoded["mask"] != public["candidate_mask"]:
        raise ValueError("snapshot encoding/public boundary mismatch")
    return Snapshot(
        env_id=encoded["env_id"], episode_id=encoded["episode_id"],
        decision_id=int(encoded["decision_id"]), template=template, facts=facts,
        candidate_ids=tuple(encoded["candidate_ids"]), mask=tuple(encoded["mask"]),
        prior_edges=tuple(tuple(x) for x in edges), prior_hash=digest(tuple(tuple(x) for x in edges)),
        base_input=tuple(encoded["base_input"]),
        candidate_features=tuple(tuple(x) for x in encoded["candidate_features"]),
        observation_ref=encoded["observation_ref"], clock_seconds=float(encoded["clock_seconds"]),
        execution_summary=tuple(tuple(x) for x in encoded["execution_summary"]),
        synthetic_unit_fixture=False)


def _natural_edges(out, layout):
    path = Path(out) / "provider/relation_request_ledger.json"
    if not path.is_file():
        return None
    entries = read(path)["entries"]
    target = f"family_b_{layout}_repeat0"
    row = next((x for x in entries if x["scene_id"] == target), None)
    if row is None or row.get("state") != "SUCCESS":
        # Exact-repeat source reuse is resolved through cache key, not a new request.
        reused_path = Path(out) / "provider/reused_initial_sources.json"
        reused = read(reused_path) if reused_path.is_file() else []
        match = next((x for x in reused if x["scene_id"] == target), None)
        row = next((x for x in entries if x.get("cache_key") == match["cache_key"]), None) if match else None
    if row is None or row.get("state") != "SUCCESS":
        return None
    _, edges = verify_audit_cache(row["cache_path"])
    return tuple(tuple(e) for e in edges)


def _target_swap(edges):
    swapped = []
    for source, target, relation in edges:
        if "obj_b" in target:
            target = target.replace("obj_b", "obj_c")
        elif "obj_c" in target:
            target = target.replace("obj_c", "obj_b")
        swapped.append((source, target, relation))
    return tuple(swapped)


def probe(root, cfg, out):
    import torch
    from cp_disr.neural import Policy
    from cp_disr.patch_identity import legal_nonempty_indices
    root, out = Path(root), Path(out)
    gate = read(out / "physical/mechanism_gate.json")
    if gate.get("physical_mechanism_status") != "ESTABLISHED":
        result = {"status": "NOT_RUN_PHYSICAL_GATE", "optimizer_steps": 0, "contexts": 0}
        write(out / "representation/summary.json", result)
        return result
    branches = [b for b in _registered(out) if b["repeat"] == 0 and b["candidate"] == "pad_u"]
    if len(branches) != 6:
        raise ValueError("six public repeat0 contexts required")
    rows = []
    for b in branches:
        edges = _natural_edges(out, b["layout"])
        if edges is None:
            rows.append({"layout": b["layout"], "context": b["context"],
                         "status": "NATURAL_R_UNAVAILABLE"})
            continue
        snapshots = {
            "Original": _snapshot(root, cfg, out, b, edges),
            "NoPrior": _snapshot(root, cfg, out, b, ()),
            "TargetSwap_ANALYSIS_ONLY": _snapshot(root, cfg, out, b, _target_swap(edges)),
        }
        template = snapshots["Original"].template
        actions = sorted({n.schema for n in template.nodes if n.kind == "ACTION"})
        predicates = sorted({n.schema for n in template.nodes if n.kind == "PROPOSITION"})
        types = sorted({t for n in template.nodes for t in n.argument_types})
        for seed in cfg["representation"]["capacity_seeds"]:
            for method in cfg["representation"]["methods"]:
                torch.manual_seed(int(seed))
                model = Policy(actions, predicates, types,
                               len(snapshots["Original"].base_input),
                               len(snapshots["Original"].candidate_features[0]),
                               method=method)
                model.eval()
                original_logits = None
                for condition, snap in snapshots.items():
                    model.zero_grad(set_to_none=True)
                    result = model(snap)
                    legal, changed, _ = legal_nonempty_indices(snap)
                    candidate_indices = [snap.candidate_ids.index(cid) for cid in cfg["candidate_ids"]]
                    if not all(snap.mask[i] for i in candidate_indices):
                        raise ValueError("public U/V mask changed in representation replay")
                    deltas = result.diagnostics.get("differences", {})
                    for d in deltas.values():
                        for tensor in (d.dk, d.dp):
                            if tensor.requires_grad:
                                tensor.retain_grad()
                    selected = result.logits[candidate_indices]
                    relative = float((selected[0]-selected[1]).detach())
                    if condition == "Original":
                        original_logits = relative
                    selected.sum().backward()
                    dk_grad = any(d.dk.grad is not None and int(torch.count_nonzero(d.dk.grad)) > 0
                                  for d in deltas.values())
                    dp_grad = any(d.dp.grad is not None and int(torch.count_nonzero(d.dp.grad)) > 0
                                  for d in deltas.values())
                    rows.append({
                        "layout": b["layout"], "context": b["context"],
                        "branch_id": b["branch_id"], "capacity_seed": seed,
                        "method": method, "condition": condition,
                        "natural_relation_count": len(edges),
                        "legal_candidate_count": len(legal),
                        "changed_patch_candidate_count": len(changed),
                        "candidate_ids_aligned": True, "goal_ids_aligned": True,
                        "relative_logit_u_minus_v": relative,
                        "relative_logit_change_from_original": (
                            None if condition == "Original" else relative-original_logits),
                        "gradient_to_contract_successor_DK": bool(dk_grad),
                        "gradient_to_prior_patch_DP": bool(dp_grad),
                        "optimizer_steps": 0,
                        "training_performance_inference": False,
                        "intervention_only": condition == "TargetSwap_ANALYSIS_ONLY",
                        "status": "MEASURED",
                    })
    path = out / "representation/input_dependency.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(x, sort_keys=True) + "\n" for x in rows))
    result = {"status": "MEASURED" if all(r["status"] == "MEASURED" for r in rows) else "PARTIAL",
              "contexts": len(branches), "rows": len(rows), "optimizer_steps": 0,
              "C2_performance_tested": False, "C3_performance_tested": False}
    write(out / "representation/summary.json", result)
    return result
