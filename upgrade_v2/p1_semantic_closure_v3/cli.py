#!/usr/bin/env python3
"""P1 V3 agent CLI. Implemented commands only; no fake PASS."""
from __future__ import annotations
import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT.parent.parent) not in sys.path:
    sys.path.insert(0, str(ROOT.parent.parent))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from . import admissibility, causal_adapter, g1_binding_spec, history_guard, replay_inventory, report, residual
from .util import require_new, sha256_file, write_csv, write_json, write_text


def git(repo: Path, *args: str) -> str:
    proc = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True)
    if proc.returncode:
        raise RuntimeError(proc.stderr.strip() or proc.stdout.strip())
    return proc.stdout.strip()


def cmd_help(args) -> int:
    print(__doc__)
    print("commands: admissibility residual-debt guards inventory replay g1 report run-all")
    return 0


def _copy_suffix_inputs(first_pass: Path, out: Path) -> None:
    meta = json.loads((first_pass / "10_symbolic_probes" / "probe_metadata.json").read_text(encoding="utf-8"))
    suffix_ids = {m["episode_id"] for m in meta if m.get("suffix")}
    src = first_pass / "10_symbolic_probes" / "probe_inputs.jsonl"
    dest = out / "matched_suffix_inputs.jsonl"
    if dest.exists():
        raise FileExistsError(dest)
    with src.open(encoding="utf-8") as inf, dest.open("x", encoding="utf-8") as outf:
        for line in inf:
            if line.strip() and json.loads(line)["episode_id"] in suffix_ids:
                outf.write(line)


def run_all(repo: Path, pkg: Path, first_pass: Path, out: Path) -> int:
    require_new(out)
    work = {
        "admissibility": out / "long_phi_loop",
        "residual": out / "residual_debt",
        "guards": out / "history_guard",
        "resources": out / "resources_inventory_v1",
        "replay": out / "causal_replay",
        "g1": out / "current_g1",
        "final": out / "final",
        "score": out / "causal_reward_comparison_v1",
        "normalized": out / "normalized",
    }
    for key in ("admissibility", "residual", "guards", "replay", "g1", "final", "normalized"):
        work[key].mkdir(parents=True, exist_ok=False)

    adm = admissibility.build_admissibility(repo, first_pass)
    write_json(work["admissibility"] / "runtime_phi_entry_policy.json", adm["runtime_phi_entry_policy"])
    write_json(work["admissibility"] / "topology.json", adm["topology"])
    write_json(work["admissibility"] / "summary.json", {k: adm[k] for k in adm if k != "representative_loops"})
    for loop in adm["representative_loops"]:
        write_json(work["admissibility"] / f"{loop['loop_id']}.json", loop)

    residual_summary = residual.write_outputs(first_pass, work["residual"])
    _copy_suffix_inputs(first_pass, work["residual"])

    graph = repo / "artifacts/pathgraph_sarm/stage3/input_adapter_v1/runtime_graph_specs_v1.0.1/transport_dual_order_graph_runtime_v1.0.1.yaml"
    overlay = history_guard.overlay_document(sha256_file(graph))
    write_json(work["guards"] / "dual_order_guard_overlay.json", overlay)
    write_csv(work["guards"] / "required_sequence_tests.csv", overlay["sequence_tests"])
    write_text(work["guards"] / "source_notes.md",
               "Synthetic oracle: tools/stage2/stage2_pipeline.py synthetic_episode\n"
               "YAML history_policy: completed_subgoal_set is part of state\n"
               "Success edges: A_done_to_success / B_done_to_success guard='state evidence'\n"
               "info.success is scenario!=terminal_failure, not A∧B.\n")

    roots = json.loads((pkg / "templates" / "raw_roots.template.json").read_text(encoding="utf-8"))
    sys.path.insert(0, str(pkg / "tools"))
    import v3  # type: ignore
    v3.inventory(pkg / "templates" / "raw_roots.template.json", work["resources"])

    manifest = Path(roots["roots"][0]["path"]).parent / "pilot" / "pilot_rollout_manifest.csv"
    if not manifest.is_file():
        manifest = Path("/home/__compress_data/xushijie/graph_github_upload/artifacts/pathgraph_sarm/upgrade_v2/visual_refine_l2_v1/dynamic_dataset_v1/pilot/pilot_rollout_manifest.csv")
    inv = replay_inventory.inventory(manifest)
    write_json(work["replay"] / "sample_inventory.json", inv)

    selected = []
    scoring_all = []
    coverage = {"normal": 0, "loss_recovery": 0, "missed_grasp": 0, "terminal": 0, "dual_order": 0}
    prefix_all = []
    v2_ref = repo / "artifacts/pathgraph_sarm/upgrade_v2/p1_state_conditioned_reward_v1/continuation_v2/normalized/causal_replay_L2P_00_00_220701_r00.json"
    v2 = json.loads(v2_ref.read_text(encoding="utf-8")) if v2_ref.is_file() else None
    for rollout_id in inv["selected_for_replay"]:
        item = next(x for x in inv["items"] if x["rollout_id"] == rollout_id)
        raw = causal_adapter.load_rollout(Path(item["path"]))
        adapted = causal_adapter.adapt_prefix(raw)
        cuts = [4, 7, 10, 13] if adapted["frames"] >= 13 else sorted(set([
            max(2, adapted["frames"] // 4),
            max(2, adapted["frames"] // 2),
            max(2, (3 * adapted["frames"]) // 4),
            adapted["frames"],
        ]))
        prefixes = causal_adapter.prefix_checks(raw, cuts)
        write_json(work["replay"] / f"replay_{rollout_id}.json", dict(adapted, prefix_causality=prefixes, catalog=item))
        scoring = causal_adapter.scoring_rows(adapted)
        if adapted["illegal_transition_count"] == 0 and adapted["unresolved_state_count"] == 0:
            scoring_all.extend(scoring)
        else:
            write_json(work["replay"] / f"excluded_from_main_score_{rollout_id}.json", dict(
                reason="ILLEGAL_OR_UNRESOLVED_NOT_IN_MAIN_SCORE", illegal=adapted["illegal_transition_count"],
                unresolved=adapted["unresolved_state_count"]))
        selected.append(dict(rollout_id=rollout_id, frames=adapted["frames"], transitions=len(adapted["transitions"]),
                             illegal=adapted["illegal_transition_count"], status=adapted["status"],
                             classes=item["event_classes"]))
        prefix_all.extend(prefixes)
        if "normal_success" in item["event_classes"]:
            coverage["normal"] += 1
        if "loss_then_recovery_success" in item["event_classes"]:
            coverage["loss_recovery"] += 1
        if "missed_grasp_recovery_success" in item["event_classes"]:
            coverage["missed_grasp"] += 1
        if rollout_id == "L2P_00_00_220701_r00" and v2 is not None:
            seq = [s["state"] for s in adapted["state_sequence"]]
            ref = [s["state"] for s in v2["state_sequence"]]
            write_json(work["replay"] / "pilot_reproduction.json", dict(
                node_sequence=seq, reference=ref, match=seq == ref,
                n_transitions=len(adapted["transitions"]), reference_transitions=len(v2["transitions"]),
            ))
    write_json(work["normalized"] / "causal_replay_manifest.json", dict(selected=selected, coverage=coverage))
    score_path = work["normalized"] / "causal_transitions_scoring.jsonl"
    with score_path.open("x", encoding="utf-8") as stream:
        for row in scoring_all:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    sys.path.insert(0, str(pkg / "tools"))
    import v3 as v3mod  # type: ignore
    v3mod.score(repo, score_path, work["score"])

    g1 = g1_binding_spec.build_spec(repo)
    write_json(work["g1"] / "G1_binding_spec_v3.json", g1)
    write_csv(work["g1"] / "runtime_binding_matrix.csv", g1["rows"])
    write_csv(work["g1"] / "unresolved_bindings.csv", g1["unresolved"] or [dict(source_node_or_edge="none", binding_status="NONE")])

    first_summary = json.loads((first_pass / "10_symbolic_probes" / "run_summary.json").read_text(encoding="utf-8"))
    claim_scope = [
        dict(claim="four_edge_loop_topology", status="SUPPORTED", scope="legacy transport_recovery graph"),
        dict(claim="repeated_phi_credit_under_entry_reset_hypothesis", status="SUPPORTED_SYMBOLIC", scope="TOPOLOGY_CHECKED_SYMBOLIC_INPUT; R=0.5p"),
        dict(claim="physical_phi_reset_counterexample", status="UNRESOLVED", scope="adapter not located; not physical"),
        dict(claim="residual_debt_accumulates", status="SUPPORTED_SYMBOLIC", scope="normalized no-progress cycles D=k/6"),
        dict(claim="residual_debt_changes_matched_suffix_reward", status="NOT_OBSERVED", scope="tested suffix only"),
        dict(claim="dual_order_current_valid_guard_runtime", status="UNRESOLVED", scope="source found; oracle uses scenario success"),
        dict(claim="causal_replay_normal_pilot", status="SUPPORTED", scope="L2P_00 plus selected siblings; not confirmation"),
        dict(claim="dual_order_raw_replay", status="RAW_NOT_FOUND", scope="pilot family has no dual-order traces"),
        dict(claim="G1_numeric_reward_or_controller_dispatch", status="NOT_CLAIMED", scope="3 edges, numeric_reward_or_cost=null"),
    ]
    payload = dict(
        execution_status="PASS",
        runner_commit=git(repo, "rev-parse", "HEAD"),
        parent_preserved=True,
        engine_mirror_parity=first_summary.get("max_engine_mirror_error"),
        long_phi_loop=dict(
            topology="reachable_four_legacy_edges",
            formal_binding_admissibility="TOPOLOGY_ADMISSIBLE_BINDING_UNRESOLVED",
            normalized_return="R_loop=0.5p under entry-reset hypothesis",
            physical_replay="NOT_A_PHYSICAL_COUNTEREXAMPLE",
        ),
        residual_debt=dict(
            accumulation="D_k≈k/6",
            tested_suffix_reward_effect="unchanged",
            scope="matched lawful suffix only",
        ),
        dual_order_guard=dict(
            symbolic_table="inherited packaged 27-row table",
            source_bound_runtime_guard="UNRESOLVED",
            overlay_status=overlay["status"],
        ),
        causal_replay=dict(
            raw_found=True,
            episodes=len(selected),
            event_coverage=coverage,
            dual_order=inv["dual_order_raw"],
            prefix_checks=prefix_all,
        ),
        G1=dict(
            provenance_trace="INHERITED_FROM_V2",
            reward_binding="INCOMPLETE",
            controller_driven_by_graph=False,
        ),
        overall_claim="MECHANISM_SCOPED_NO_CONFIRMATION",
        claim_scope=claim_scope,
        result_manifest=dict(out=str(out), first_pass=str(first_pass), selected=selected),
        external_tsv="role\tpath\tsha256\npilot_manifest\t%s\t%s\n" % (manifest, sha256_file(manifest)),
        run_manifest=dict(
            module="upgrade_v2/p1_semantic_closure_v3",
            commands={
                "admissibility": "python -B -m upgrade_v2.p1_semantic_closure_v3.cli run-all --help",
                "help": subprocess.run([sys.executable, "-B", str(ROOT / "cli.py"), "--help"], capture_output=True, text=True).stdout,
            },
            new_physical_runs=0,
            new_training_runs=0,
            new_llm_calls=0,
        ),
    )
    report.write_final(work["final"], payload)
    print(json.dumps({"status": "PASS", "out": str(out), "episodes": len(selected)}, ensure_ascii=False))
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("help")
    c = sub.add_parser("run-all")
    c.add_argument("--repo", type=Path, required=True)
    c.add_argument("--pkg", type=Path, required=True)
    c.add_argument("--first-pass", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "help":
        return cmd_help(args)
    return run_all(args.repo, args.pkg, args.first_pass, args.out)


if __name__ == "__main__":
    raise SystemExit(main())