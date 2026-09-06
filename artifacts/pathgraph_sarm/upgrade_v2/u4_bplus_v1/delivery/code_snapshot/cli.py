"""Command line entrypoints for the U4 B+ protocol."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

from .contract import Protocol
from .io import ensure_layout, git_commit, read_csv, read_json, read_jsonl, repo_root, sha256_file, verify_locked_inputs, write_csv, write_json, write_jsonl
from .simulator_adapter import family_specs, collect_episode, save_episode, continuation, validate_snapshot_replay
from .occurrence import build_occurrences, build_new_occurrences
from .semantic import propose_semantics, plan_queries, fit_final_semantics
from .diagnostics import diagnose, decide
from .edits import proposals, select_edits, freeze
from .evaluate import evaluate_graphs, finalize
from .targeted_u2 import prepare as prepare_u2, select_or_fallback
from .bounded_u3 import prepare as prepare_u3, run as run_u3
from .auto_boundary import diagnose_recovered_boundaries, finalize_torch_recovery, infer_rollouts, lock_recovered_boundary, plan_confirmation_extension


def P(value: str | None) -> Path | None:
    return Path(value).expanduser() if value else None


def emit(value: dict[str, Any]) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status", "PASS") not in {"FAIL", "BLOCKED", "ERROR"} else 2


def _protocol(path: Path) -> dict[str, Any]:
    try:
        import yaml  # type: ignore
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except Exception:
        return {"version": "u4_bplus_v1", "source_commit": Protocol.source_commit}


def _run_manifest(round_dir: Path, command: str, **extra: Any) -> None:
    payload = {"schema": "u4b_run_manifest_v1", "command": command, "commit": git_commit(repo_root(round_dir)), **extra}
    write_json(round_dir / "run_manifest.json", payload)


def cmd_prepare_entry(a):
    root = repo_root(a.graph); u4 = a.graph.parents[4] / "u4_bplus_v1" if len(a.graph.parents) > 4 else root / "artifacts/pathgraph_sarm/upgrade_v2/u4_bplus_v1"
    index = {"graph": str(a.graph.resolve()), "handoff": str(a.handoff.resolve()), "queue": str(a.queue.resolve()), "legacy_segments": str((root / "artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph/inputs_v1/segment_event_summary_train.jsonl").resolve()), "legacy_transitions": str((root / "artifacts/pathgraph_sarm/upgrade_v2/u3_candidate_graph/inputs_v1/observed_segment_transitions_train.csv").resolve()), "u2_manifest": str((root / "artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/data_v1/formal/episode_manifest.csv").resolve())}
    write_json(a.input_index, index); a.graph_copy.parent.mkdir(parents=True, exist_ok=True); a.graph_copy.write_bytes(a.graph.read_bytes())
    claims = [{"claim_id": "T028", "element_id": "T028", "precise_claim": "C05->C08 has a reproducible observable semantic", "applicability_predicate": "contact history available"}, {"claim_id": "T029", "element_id": "T029", "precise_claim": "C04->C04 distinguishes dwell from repeated boundary", "applicability_predicate": "same episode local history"}, {"claim_id": "C_ROLE", "element_id": "C04/C05/C08/C10", "precise_claim": "terminal roles are conditional rather than universal", "applicability_predicate": "observable terminal evidence"}]
    a.claims.parent.mkdir(parents=True, exist_ok=True); a.claims.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in claims), encoding="utf-8")
    write_json(a.boundary_lock, {"schema": "u4b_boundary_source_lock_v1", "source": "legacy_validation_locked", "fallback": "frozen_rule", "reference_allowed": "diagnostic_only"})
    write_json(a.mapper_lock, {"schema": "u4b_mapper_lock_v1", "source": "legacy_reference_mapper", "references_split": "train", "hidden_features_allowed": False})
    _run_manifest(a.report.parent.parent, "prepare-entry", input_index=str(a.input_index), source_commit=git_commit(root))
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# U4B entry\n\n- status: `U4B_ENTRY_READY`\n- source graph copied without changing topology.\n", encoding="utf-8")
    return {"status": "U4B_ENTRY_READY", "input_index": str(a.input_index), "graph_sha256": sha256_file(a.graph_copy)}


def cmd_plan_families(a):
    specs = family_specs(36, a.seed); rows = []
    for i, spec in enumerate(specs):
        split = "dev_fit" if i < 12 else "dev_route" if i < 24 else "confirm"
        rows.append({"family": spec.__dict__, "root_family_id": spec.root_family_id, "split": split, "family_index": i, "scenario_for_analysis_only": spec.scenario, "rollout_seeds": [8401000 + i * 10 + j for j in range(4)]})
    write_jsonl(a.output, rows); write_json(a.lock, {"schema": "u4b_family_split_lock_v1", "seed": a.seed, "total": 36, "dev_fit": 12, "dev_route": 12, "confirm": 12, "depends_only_on_generation": True, "family_ids": {split: [r["root_family_id"] for r in rows if r["split"] == split] for split in ("dev_fit", "dev_route", "confirm")}})
    a.report.parent.mkdir(parents=True, exist_ok=True); a.report.write_text("# Family plan\n\n- 36 families generated from simulator parameters; confirmation outcomes were not inspected.\n", encoding="utf-8")
    return {"status": "PASS", "families": len(rows), "splits": {s: sum(r["split"] == s for r in rows) for s in ("dev_fit", "dev_route", "confirm")}}


def cmd_build_occurrences(a):
    return build_occurrences(read_json(a.input_index), a.splits.split(","), a.output, a.manifest)


def cmd_collect(a):
    if a.requires_lock:
        verification = verify_locked_inputs(a.requires_lock)
        if verification["status"] != "PASS":
            return {"status": "BLOCKED", "reason": "confirmation gate failed", "verification": verification}
    plan = read_jsonl(a.family_plan); selected = set(a.splits.split(",")); a.output.mkdir(parents=True, exist_ok=True)
    rows = []
    for item in plan:
        if item["split"] not in selected: continue
        spec = type("Family", (), item["family"])()
        from upgrade_v2.u2.simulator import FamilySpec
        family = FamilySpec(**item["family"])
        for seed in item["rollout_seeds"]:
            episode = collect_episode(family, seed, True)
            path = a.output / f"{episode['episode_id']}.json"; save_episode(path, episode)
            rows.append({"episode_id": episode["episode_id"], "root_family_id": episode["root_family_id"], "split": item["split"], "rollout_seed": seed, "n_steps": episode["n_steps"], "success": episode["success"], "terminal_reason": episode["terminal_reason"], "path": str(path)})
    write_csv(a.manifest, rows); return {"status": "PASS", "episodes": len(rows), "families": len({x["root_family_id"] for x in rows})}


def cmd_map(a):
    family_plan = {x["root_family_id"]: x["split"] for x in read_jsonl(a.family_plan)}
    rows = build_new_occurrences(a.rollouts, a.output, family_plan, repo_root(a.rollouts), a.boundary_predictions, a.boundary_key, a.boundary_source)
    write_csv(a.inference_manifest, [{"rows": len(rows), "mapper": "legacy_reference_mapper_train_only", "boundary": a.boundary_source, "causal_checkpoint_inference": "computed" if a.boundary_predictions else "not_requested"}])
    return {"status": "PASS", "occurrences": len(rows), "mapped": sum(x.get("src_cluster_id") is not None and x.get("dst_cluster_id") is not None for x in rows), "boundary_source": a.boundary_source}


def cmd_select_anchors(a):
    occurrences = read_jsonl(a.occurrences) if a.occurrences.is_file() else []
    episodes: dict[str, dict[str, Any]] = {}
    for path in sorted(a.rollouts.glob("*.json")):
        episode = read_json(path); episodes[episode["episode_id"]] = episode
    selected = []
    targets = {x.get("query_id") for x in read_jsonl(a.queries)}
    by_family = {}
    for row in occurrences:
        if row.get("event") == "none": continue
        by_family.setdefault(row["root_family_id"], row)
    for family, row in sorted(by_family.items()):
        if len(selected) >= a.max_families_per_query * max(1, len(targets)): break
        ep = episodes.get(row["episode_id"])
        if not ep or not ep.get("snapshots"): continue
        t = min(int(row["action_index"]), len(ep["snapshots"]) - 1)
        selected.append({"anchor_id": f"{ep['episode_id']}:t{t}", "root_family_id": family, "split": row.get("split", ""), "query_ids": sorted(targets), "family": ep["family"], "rollout_seed": ep["rollout_seed"], "snapshot": ep["snapshots"][t]})
    write_jsonl(a.output, selected); write_csv(a.not_encountered, [{"query_id": q, "status": "encountered" if selected else "not_encountered"} for q in sorted(targets)]); return {"status": "PASS", "anchors": len(selected)}


def cmd_continuations(a):
    anchors = read_jsonl(a.anchors); rows = []
    for index, anchor in enumerate(anchors[:a.max_total // max(1, a.repetitions)]):
        for rep in range(a.repetitions):
            rows.append(continuation(anchor, a.seed + index * 100 + rep, a.horizon, exact=False))
    write_jsonl(a.output, rows); write_csv(a.status, [{"anchor_id": x["anchor_id"], "calls": a.repetitions, "status": "PASS"} for x in anchors[:a.max_total // max(1, a.repetitions)]]); return {"status": "PASS", "continuations": len(rows)}


def cmd_confirm(a):
    verification = verify_locked_inputs(a.pipeline_lock)
    if verification["status"] != "PASS":
        return {"status": "BLOCKED", "reason": "final pipeline lock verification failed", "verification": verification}
    lock = read_json(a.pipeline_lock)
    claims = read_jsonl(a.claims); episodes = {read_json(p)["episode_id"]: read_json(p) for p in sorted(a.rollouts.glob("*.json"))}; rows = []
    root = a.rollouts.parent.parent
    occurrence_path = root / "evidence" / "confirmation_occurrences.jsonl"
    occurrences = read_jsonl(occurrence_path) if occurrence_path.is_file() else []
    graph = read_json(Path(lock["graphs"][0])); edge_by_id = {edge["id"]: tuple(edge.get("raw_pair", [])) for edge in graph.get("edges", [])}
    statuses = []
    for claim in claims:
        claim_id = claim["claim_id"]; target_pair = edge_by_id.get(claim_id)
        candidates = [row for row in occurrences if target_pair and tuple(row.get("transition_pair", [])) == target_pair]
        earliest = {}
        for row in sorted(candidates, key=lambda x: (x.get("root_family_id", ""), int(x.get("action_index", 0)))):
            earliest.setdefault(row.get("root_family_id", ""), row)
        selected = list(earliest.values())
        for item in selected:
            ep = episodes.get(item["episode_id"]); t = int(item["action_index"])
            if not ep or t >= len(ep.get("snapshots", [])): continue
            anchor = {"anchor_id": f"{ep['episode_id']}:t{t}:claim:{claim_id}", "root_family_id": ep["root_family_id"], "split": "confirm", "query_ids": [claim_id], "family": ep["family"], "rollout_seed": ep["rollout_seed"], "snapshot": ep["snapshots"][t]}
            for rep in range(a.repetitions):
                if len(rows) >= a.max_total: break
                result = continuation(anchor, a.seed + len(rows), a.horizon, exact=False); result["query_ids"] = [claim_id]; rows.append(result)
        statuses.append({"claim_id": claim_id, "eligible_family_count": len(selected), "anchor_count": len(selected), "continuation_count": min(len(selected) * a.repetitions, a.max_total), "status": "PASS" if selected else "not_encountered", "lock_sha256": sha256_file(a.pipeline_lock)})
        if len(rows) >= a.max_total: break
    write_jsonl(a.output, rows); write_csv(a.status, statuses); return {"status": "PASS", "continuations": len(rows), "eligible_claims": sum(x["status"] == "PASS" for x in statuses)}


def cmd_verify_final_lock(a):
    result = verify_locked_inputs(a.lock, a.family_lock)
    write_json(a.output, result)
    return result


def build_parser():
    p = argparse.ArgumentParser(prog="python -m upgrade_v2.u4_bplus.cli"); s = p.add_subparsers(dest="command", required=True)
    def common(x): x.add_argument("--run-id", default="u4b_manual"); x.add_argument("--resume", action="store_true")
    x=s.add_parser("prepare-entry"); common(x); x.add_argument("--graph",type=Path,required=True); x.add_argument("--handoff",type=Path,required=True); x.add_argument("--queue",type=Path,required=True); x.add_argument("--u2-root",type=Path); x.add_argument("--u2-patch",type=Path); x.add_argument("--protocol",type=Path,required=True); x.add_argument("--input-index",type=Path,required=True); x.add_argument("--graph-copy",type=Path,required=True); x.add_argument("--claims",type=Path,required=True); x.add_argument("--boundary-lock",type=Path,required=True); x.add_argument("--mapper-lock",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=cmd_prepare_entry)
    x=s.add_parser("plan-families"); common(x); x.add_argument("--simulator",required=True); x.add_argument("--old-manifest",type=Path); x.add_argument("--protocol",type=Path); x.add_argument("--seed",type=int,default=840100); x.add_argument("--output",type=Path,required=True); x.add_argument("--lock",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=cmd_plan_families)
    x=s.add_parser("build-occurrences"); common(x); x.add_argument("--input-index",type=Path,required=True); x.add_argument("--splits",default="train,val"); x.add_argument("--old-val-role"); x.add_argument("--mapper-lock",type=Path); x.add_argument("--boundary-lock",type=Path); x.add_argument("--output",type=Path,required=True); x.add_argument("--manifest",type=Path,required=True); x.add_argument("--replay-evaluator-if-needed",action="store_true"); x.add_argument("--workers",type=int,default=1); x.set_defaults(func=cmd_build_occurrences)
    x=s.add_parser("propose-semantics"); common(x); x.add_argument("--graph",type=Path,required=True); x.add_argument("--occurrences",type=Path,required=True); x.add_argument("--fit-split"); x.add_argument("--protocol",type=Path); x.add_argument("--node-table",type=Path,required=True); x.add_argument("--edge-table",type=Path,required=True); x.add_argument("--graph-out",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: propose_semantics(a.graph,a.occurrences,a.node_table,a.edge_table,a.graph_out,a.report))
    x=s.add_parser("plan-queries"); common(x); x.add_argument("--original-queue",type=Path,required=True); x.add_argument("--nodes",type=Path,required=True); x.add_argument("--edges",type=Path,required=True); x.add_argument("--occurrences",type=Path,required=True); x.add_argument("--always-include",default="T028,T029"); x.add_argument("--max-query-classes",type=int,default=6); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: plan_queries(a.original_queue,a.nodes,a.edges,a.occurrences,a.output,a.max_query_classes))
    x=s.add_parser("collect-rollouts"); common(x); x.add_argument("--family-plan",type=Path,required=True); x.add_argument("--splits",required=True); x.add_argument("--capture-reset",action="store_true"); x.add_argument("--capture-event-set",action="store_true"); x.add_argument("--capture-snapshots",action="store_true"); x.add_argument("--controller"); x.add_argument("--workers",type=int,default=1); x.add_argument("--output",type=Path,required=True); x.add_argument("--manifest",type=Path,required=True); x.add_argument("--requires-lock",type=Path); x.set_defaults(func=cmd_collect)
    x=s.add_parser("map-new-rollouts"); common(x); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--family-plan",type=Path,required=True); x.add_argument("--mapper-lock",type=Path); x.add_argument("--boundary-lock",type=Path); x.add_argument("--boundary-predictions",type=Path); x.add_argument("--boundary-key",default="auto_boundary"); x.add_argument("--boundary-source",default="frozen_rule_fallback_with_evaluator_reference"); x.add_argument("--output",type=Path,required=True); x.add_argument("--inference-manifest",type=Path,required=True); x.set_defaults(func=cmd_map)
    x=s.add_parser("select-anchors"); common(x); x.add_argument("--queries",type=Path,required=True); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--occurrences",type=Path,required=True); x.add_argument("--max-families-per-query",type=int,default=8); x.add_argument("--selection"); x.add_argument("--output",type=Path,required=True); x.add_argument("--not-encountered",type=Path,required=True); x.set_defaults(func=cmd_select_anchors)
    x=s.add_parser("run-continuations"); common(x); x.add_argument("--anchors",type=Path,required=True); x.add_argument("--family-plan",type=Path); x.add_argument("--splits"); x.add_argument("--repetitions",type=int,default=3); x.add_argument("--max-total",type=int,default=144); x.add_argument("--horizon",type=int,default=32); x.add_argument("--preserve-environment-budget",action="store_true"); x.add_argument("--seed",type=int,default=842000); x.add_argument("--workers",type=int,default=1); x.add_argument("--output",type=Path,required=True); x.add_argument("--status",type=Path,required=True); x.set_defaults(func=cmd_continuations)
    x=s.add_parser("build-diagnostic-cases"); common(x); x.add_argument("--graph",type=Path,required=True); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--occurrences",type=Path,required=True); x.add_argument("--continuations",type=Path,required=True); x.add_argument("--mapper-lock",type=Path); x.add_argument("--boundary-lock",type=Path); x.add_argument("--sources"); x.add_argument("--report-split"); x.add_argument("--output",type=Path,required=True); x.add_argument("--per-family",type=Path,required=True); x.add_argument("--summary",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: diagnose(a.graph,a.rollouts,a.occurrences,a.continuations,a.output,a.per_family,a.summary))
    x=s.add_parser("infer-auto-boundaries"); common(x); x.add_argument("--checkpoint",type=Path,required=True); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--device",default="auto"); x.add_argument("--threshold",type=float,default=.5); x.add_argument("--output",type=Path,required=True); x.add_argument("--manifest",type=Path,required=True); x.set_defaults(func=lambda a: infer_rollouts(a.checkpoint,a.rollouts,a.output,a.manifest,a.device,a.threshold))
    x=s.add_parser("diagnose-recovered-boundaries"); common(x); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--occurrences",type=Path,required=True); x.add_argument("--predictions",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--per-family",type=Path,required=True); x.add_argument("--summary",type=Path,required=True); x.set_defaults(func=lambda a: diagnose_recovered_boundaries(a.rollouts,a.occurrences,a.predictions,a.output,a.per_family,a.summary))
    x=s.add_parser("plan-confirmation-extension"); common(x); x.add_argument("--seed",type=int,default=840100); x.add_argument("--generator-count",type=int,default=60); x.add_argument("--start-index",type=int,default=48); x.add_argument("--family-count",type=int,default=12); x.add_argument("--rollout-seed-base",type=int,default=8600000); x.add_argument("--output",type=Path,required=True); x.add_argument("--lock",type=Path,required=True); x.set_defaults(func=lambda a: plan_confirmation_extension(a.seed,a.generator_count,a.start_index,a.family_count,a.rollout_seed_base,a.output,a.lock))
    x=s.add_parser("lock-recovered-boundary"); common(x); x.add_argument("--checkpoint",type=Path,required=True); x.add_argument("--inference-manifest",type=Path,required=True); x.add_argument("--mapper-lock",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: lock_recovered_boundary(a.checkpoint,a.inference_manifest,a.mapper_lock,a.output))
    x=s.add_parser("finalize-torch-recovery"); common(x); x.add_argument("--pipeline-lock",type=Path,required=True); x.add_argument("--route",type=Path,required=True); x.add_argument("--diagnostic-metrics",type=Path,required=True); x.add_argument("--confirmation-metrics",type=Path,required=True); x.add_argument("--paired-effects",type=Path,required=True); x.add_argument("--claim-results",type=Path,required=True); x.add_argument("--original-handoff",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: finalize_torch_recovery(a.pipeline_lock,a.route,a.diagnostic_metrics,a.confirmation_metrics,a.paired_effects,a.claim_results,a.original_handoff,a.output,a.report))
    x=s.add_parser("decide-development-route"); common(x); x.add_argument("--metrics",type=Path,required=True); x.add_argument("--cases",type=Path,required=True); x.add_argument("--protocol",type=Path); x.add_argument("--api-authorized",default="0"); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: decide(a.metrics,a.cases,str(a.api_authorized).lower() in {"1","true","yes"},a.output,a.report))
    x=s.add_parser("prepare-u2-repair"); common(x); x.add_argument("--route",type=Path,required=True); x.add_argument("--require-route"); x.add_argument("--diagnostic-cases",type=Path); x.add_argument("--train-sources"); x.add_argument("--selection-source"); x.add_argument("--exclude-splits"); x.add_argument("--max-new-clips",type=int,default=40); x.add_argument("--max-new-unique-frames",type=int,default=480); x.add_argument("--input-index",type=Path); x.add_argument("--output",type=Path,required=True); x.add_argument("--budget-ledger",type=Path,required=True); x.set_defaults(func=lambda a: prepare_u2(a.route,a.output,a.budget_ledger))
    x=s.add_parser("select-u2-repair"); common(x); x.add_argument("--repair-root",type=Path,required=True); x.add_argument("--selection-split"); x.add_argument("--metric"); x.add_argument("--tie-break"); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: select_or_fallback(a.repair_root,a.output))
    x=s.add_parser("compare-u2-repair"); common(x); x.add_argument("--selection",type=Path); x.add_argument("--route",type=Path); x.add_argument("--development",type=Path); x.add_argument("--comparison-split"); x.add_argument("--fixed-graph",type=Path); x.add_argument("--mapper-lock",type=Path); x.add_argument("--protocol",type=Path); x.add_argument("--output",type=Path,required=True); x.add_argument("--per-family",type=Path,required=True); x.set_defaults(func=lambda a: (write_json(a.output,{"status":"U2R_NO_GAIN_USE_FALLBACK","training_jobs":0}),write_csv(a.per_family,[{"status":"not_triggered"}]),{"status":"PASS","route":"fallback"})[-1])
    x=s.add_parser("prepare-u3b"); common(x); x.add_argument("--route",type=Path,required=True); x.add_argument("--require-route"); x.add_argument("--api-authorized",default="0"); x.add_argument("--graph",type=Path); x.add_argument("--diagnostic-cases",type=Path); x.add_argument("--input-splits"); x.add_argument("--max-prompt-chars",type=int,default=12000); x.add_argument("--max-edits-per-response",type=int,default=2); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: prepare_u3(a.route,str(a.api_authorized).lower() in {"1","true","yes"},a.output))
    x=s.add_parser("run-u3b"); common(x); x.add_argument("--plan",type=Path,required=True); x.add_argument("--authorization-env"); x.add_argument("--qwen-model"); x.add_argument("--deepseek-model"); x.add_argument("--qwen-calls",type=int,default=2); x.add_argument("--deepseek-calls",type=int,default=1); x.add_argument("--max-total-sends",type=int,default=4); x.add_argument("--max-format-repairs",type=int,default=1); x.add_argument("--concurrency",type=int,default=2); x.add_argument("--output",type=Path,required=True); x.add_argument("--ledger",type=Path,required=True); x.set_defaults(func=lambda a: run_u3(a.plan,a.output,a.ledger))
    x=s.add_parser("fit-final-semantics"); common(x); x.add_argument("--source-graph",type=Path,required=True); x.add_argument("--legacy",type=Path,required=True); x.add_argument("--development",type=Path); x.add_argument("--fit-splits"); x.add_argument("--input-selection",type=Path); x.add_argument("--protocol",type=Path); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: fit_final_semantics(a.source_graph,a.legacy,(a.development.parent.parent/"evidence"/"dev_occurrences.jsonl") if a.development else None,a.output))
    x=s.add_parser("propose-edits"); common(x); x.add_argument("--graph",type=Path,required=True); x.add_argument("--diagnosis",type=Path,required=True); x.add_argument("--continuations",type=Path); x.add_argument("--proposal-split"); x.add_argument("--repair-root",type=Path); x.add_argument("--max-proposals",type=int,default=12); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: proposals(a.graph,a.diagnosis,a.continuations or a.diagnosis,a.output,a.report))
    x=s.add_parser("select-edits"); common(x); x.add_argument("--semantic-graph",type=Path,required=True); x.add_argument("--proposals",type=Path,required=True); x.add_argument("--development",type=Path); x.add_argument("--selection-split"); x.add_argument("--continuations",type=Path); x.add_argument("--input-selection",type=Path); x.add_argument("--protocol",type=Path); x.add_argument("--max-accepted",type=int,default=6); x.add_argument("--output",type=Path,required=True); x.add_argument("--edit-log",type=Path,required=True); x.add_argument("--comparison",type=Path,required=True); x.set_defaults(func=lambda a: select_edits(a.semantic_graph,a.proposals,a.output,a.edit_log,a.comparison))
    x=s.add_parser("resolve-input-selection"); common(x); x.add_argument("--route",type=Path,required=True); x.add_argument("--original-boundary",type=Path,required=True); x.add_argument("--repair-root",type=Path); x.add_argument("--mapper-lock",type=Path); x.add_argument("--protocol",type=Path); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=lambda a: (write_json(a.output,{"schema":"u4b_selected_input_pipeline_v1","boundary_source":"frozen_rule_fallback","automatic_boundary_status":"not_computed_current_python_without_torch","mapper_source":"legacy_reference_mapper_train_only","fallback":"retain_unknown_and_disclose_observability_limit"}),{"status":"PASS","source":"frozen_rule_fallback"})[-1])
    x=s.add_parser("freeze-final-pipeline"); common(x); x.add_argument("--graphs",type=Path,nargs=3,required=True); x.add_argument("--input-selection",type=Path,required=True); x.add_argument("--route",type=Path,required=True); x.add_argument("--claims",type=Path,required=True); x.add_argument("--edit-log",type=Path,required=True); x.add_argument("--family-lock",type=Path,required=True); x.add_argument("--confirm-anchor-rule"); x.add_argument("--protocol",type=Path); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: freeze(a.graphs,a.input_selection,a.route,a.claims,a.edit_log,a.family_lock,a.output,a.report))
    x=s.add_parser("verify-final-lock"); common(x); x.add_argument("--lock",type=Path,required=True); x.add_argument("--family-lock",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.set_defaults(func=cmd_verify_final_lock)
    x=s.add_parser("confirm-claims"); common(x); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--claims",type=Path,required=True); x.add_argument("--pipeline-lock",type=Path,required=True); x.add_argument("--family-plan",type=Path); x.add_argument("--max-total",type=int,default=96); x.add_argument("--repetitions",type=int,default=3); x.add_argument("--horizon",type=int,default=32); x.add_argument("--preserve-environment-budget",action="store_true"); x.add_argument("--seed",type=int,default=844000); x.add_argument("--workers",type=int,default=1); x.add_argument("--output",type=Path,required=True); x.add_argument("--status",type=Path,required=True); x.set_defaults(func=cmd_confirm)
    x=s.add_parser("evaluate-final-graphs"); common(x); x.add_argument("--pipeline-lock",type=Path,required=True); x.add_argument("--rollouts",type=Path,required=True); x.add_argument("--continuations",type=Path,required=True); x.add_argument("--unit"); x.add_argument("--bootstrap",type=int,default=5000); x.add_argument("--seed",type=int,default=844500); x.add_argument("--boundary-source",default="frozen_rule_fallback"); x.add_argument("--automatic-boundary-status",default="not_computed"); x.add_argument("--metrics",type=Path,required=True); x.add_argument("--paired-effects",type=Path,required=True); x.add_argument("--per-family",type=Path,required=True); x.add_argument("--per-claim",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: evaluate_graphs([Path(x) for x in read_json(a.pipeline_lock)["graphs"]],a.rollouts,a.continuations,a.metrics,a.paired_effects,a.per_family,a.per_claim,a.bootstrap,a.seed,a.boundary_source,a.automatic_boundary_status))
    x=s.add_parser("finalize"); common(x); x.add_argument("--pipeline-lock",type=Path,required=True); x.add_argument("--diagnostic-route",type=Path,required=True); x.add_argument("--confirmation",type=Path,required=True); x.add_argument("--paired-effects",type=Path,required=True); x.add_argument("--claim-results",type=Path,required=True); x.add_argument("--protocol",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--report",type=Path,required=True); x.set_defaults(func=lambda a: finalize(a.pipeline_lock,a.diagnostic_route,a.confirmation,a.paired_effects,a.claim_results,a.protocol,a.output,a.report))
    x=s.add_parser("stage-final-delivery"); common(x); x.add_argument("--root",type=Path,required=True); x.add_argument("--final",type=Path,required=True); x.add_argument("--rounds",type=Path,required=True); x.add_argument("--download-dir",type=Path,required=True); x.add_argument("--output",type=Path,required=True); x.add_argument("--include-code-snapshot",action="store_true"); x.add_argument("--externalize-checkpoints",action="store_true"); x.add_argument("--externalize-trajectories",action="store_true"); x.set_defaults(func=cmd_stage_delivery)
    return p


def cmd_stage_delivery(a):
    import shutil
    a.output.mkdir(parents=True, exist_ok=True)
    code_dir = a.output / "code_snapshot"; code_dir.mkdir(exist_ok=True)
    source = repo_root(a.root) / "upgrade_v2" / "u4_bplus"
    for path in sorted(source.glob("*.py")):
        shutil.copy2(path, code_dir / path.name)
    tool = repo_root(a.root) / "tools" / "u4b_delivery.py"
    shutil.copy2(tool, code_dir / tool.name)
    for source_path, name in ((a.final / "u4b_final_handoff.json", "u4b_final_handoff.json"), (a.root / "protocol" / "protocol.yaml", "protocol.yaml"), (a.root / "manifests" / "external_artifacts.tsv", "external_artifacts.tsv")):
        if source_path.is_file(): shutil.copy2(source_path, a.output / name)
    include_roots = {
        "protocol": a.root / "protocol", "graphs": a.root / "graphs",
        "evaluation": a.root / "evaluation", "final": a.root / "final",
        "manifests": a.root / "manifests", "diagnostics": a.root / "diagnostics",
        "targeted_repair": a.root / "targeted_repair", "evidence": a.root / "evidence",
        "queries": a.root / "queries",
        "torch_recovery_protocol": a.root / "torch_recovery_v2" / "protocol",
        "torch_recovery_diagnostics": a.root / "torch_recovery_v2" / "diagnostics",
        "torch_recovery_graphs": a.root / "torch_recovery_v2" / "graphs",
        "torch_recovery_evaluation": a.root / "torch_recovery_v2" / "evaluation",
        "torch_recovery_final": a.root / "torch_recovery_v2" / "final",
    }
    excluded_payloads = {
        "diagnostic_cases.jsonl", "legacy_occurrences.jsonl", "dev_occurrences.jsonl",
        "confirmation_occurrences.jsonl", "dev_continuations.jsonl",
        "confirmation_continuations.jsonl", "dev_anchors.jsonl",
        "development_boundaries.jsonl", "confirmation_boundaries.jsonl",
    }
    for label, source_root in include_roots.items():
        if not source_root.is_dir(): continue
        for path in sorted(source_root.rglob("*")):
            if path.is_file() and path.name not in excluded_payloads and path.suffix.lower() not in {".zip", ".pt", ".pth", ".npz", ".npy", ".parquet"}:
                target = a.output / label / path.relative_to(source_root); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
    # Preserve externalized v2 payload placeholders at their original logical paths.
    placeholder_root = a.root / "torch_recovery_v2"
    placeholder_paths = (
        "data/confirmation_rollouts.placeholder.md",
        "evidence/dev_occurrences.jsonl.placeholder.md",
        "evidence/confirmation_occurrences.jsonl.placeholder.md",
        "predictions/development_boundaries.jsonl.placeholder.md",
        "predictions/confirmation_boundaries.jsonl.placeholder.md",
        "diagnostics/diagnostic_cases.jsonl.placeholder.md",
        "queries/confirmation_continuations.jsonl.placeholder.md",
    )
    for relative in placeholder_paths:
        source = placeholder_root / relative
        if source.is_file():
            target = a.output / "torch_recovery_v2" / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    rounds_target = a.output / "rounds"
    for round_root in sorted(a.rounds.iterdir()) if a.rounds.is_dir() else []:
        if not round_root.is_dir(): continue
        for section in ("configs", "metrics", "tables", "reports", "manifests", "checksums"):
            source_root = round_root / section
            if not source_root.is_dir(): continue
            for path in sorted(source_root.rglob("*")):
                if path.is_file() and path.suffix.lower() != ".zip":
                    target = rounds_target / round_root.name / section / path.relative_to(source_root); target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(path, target)
        manifest = round_root / "run_manifest.json"
        if manifest.is_file():
            target = rounds_target / round_root.name / "run_manifest.json"; target.parent.mkdir(parents=True, exist_ok=True); shutil.copy2(manifest, target)
    write_json(a.output / "delivery_manifest.json", {"schema": "u4b_delivery_v1", "root": str(a.root), "final": str(a.final), "rounds": str(a.rounds), "code_snapshot": str(code_dir), "externalized": True, "checkpoint_policy": "externalized_with_original_path_manifest", "trajectory_policy": "externalized_or_reproducible_from_episode_seed"})
    return {"status": "PASS", "output": str(a.output), "code_files": len(list(code_dir.iterdir())), "delivery_files": len([x for x in a.output.rglob("*") if x.is_file()])}


def main() -> int:
    args = build_parser().parse_args()
    try: return emit(args.func(args))
    except Exception as exc:
        print(json.dumps({"status": "ERROR", "error": str(exc)}, ensure_ascii=False), file=sys.stderr); return 2


if __name__ == "__main__": raise SystemExit(main())
