"""U4R1 command line: each subcommand has explicit inputs and outputs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .confirm import build_lock, build_occurrences_confirm, evaluate as evaluate_confirm, verify_lock
from .evaluator_v2 import compare_old_new, evaluate_occurrence, rescore
from .fresh_families import collect, generate
from .historical_lock import freeze_history, verify_history
from .io import environment_audit, read_json, read_jsonl, sha256_file, write_csv, write_json, write_jsonl
from .multigraph import build_graphs
from .occurrence_table import build_occurrences, write_occurrence_summary
from .package import package_complete, package_round
from .select_graph import evaluate_graphs, select
from .separability import fit_baselines, summarize_mixed


def _status(value: dict) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if value.get("status", "PASS") not in {"FAIL", "BLOCKED", "ERROR"} else 2


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upgrade_v2.u4_conditional.cli")
    s = p.add_subparsers(dest="command", required=True)
    x = s.add_parser("freeze-history"); x.add_argument("--repo", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--paths", type=Path, nargs="+", required=True)
    x = s.add_parser("validate-inputs"); x.add_argument("--lock", type=Path, required=True)
    x = s.add_parser("run-evaluator-tests")
    x = s.add_parser("compare-evaluator-contracts"); x.add_argument("--old", type=Path, required=True); x.add_argument("--new", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("rescore-frozen-confirmation"); x.add_argument("--graph", type=Path, required=True); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("compare-old-new-metrics"); x.add_argument("--old", type=Path, required=True); x.add_argument("--new", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("build-semantic-occurrence-table"); x.add_argument("--rollouts", type=Path, required=True); x.add_argument("--repo", type=Path, required=True); x.add_argument("--split", default="development"); x.add_argument("--output", type=Path, required=True); x.add_argument("--summary", type=Path)
    x = s.add_parser("summarize-mixed-pairs"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("fit-separability-baselines"); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--table", type=Path)
    x = s.add_parser("build-conditional-graphs"); x.add_argument("--raw-graph", type=Path, required=True); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output-dir", type=Path, required=True); x.add_argument("--edit-log", type=Path)
    x = s.add_parser("evaluate-conditional-graphs"); x.add_argument("--graphs", type=Path, nargs="+", required=True); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--table", type=Path, required=True)
    x = s.add_parser("select-conditional-graph"); x.add_argument("--metrics", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--report", type=Path)
    x = s.add_parser("generate-fresh-families"); x.add_argument("--output", type=Path, required=True); x.add_argument("--lock", type=Path, required=True); x.add_argument("--seed", type=int, default=910500); x.add_argument("--count", type=int, default=36); x.add_argument("--rollout-seed-base", type=int, default=9110000)
    x = s.add_parser("infer-fresh-boundaries"); x.add_argument("--checkpoint", type=Path, required=True); x.add_argument("--rollouts", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--manifest", type=Path, required=True); x.add_argument("--device", default="auto")
    x = s.add_parser("build-confirmation-occurrences"); x.add_argument("--rollouts", type=Path, required=True); x.add_argument("--repo", type=Path, required=True); x.add_argument("--predictions", type=Path); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("evaluate-fresh-confirmation"); x.add_argument("--graphs", type=Path, nargs="+", required=True); x.add_argument("--occurrences", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--paired", type=Path, required=True); x.add_argument("--family-table", type=Path, required=True); x.add_argument("--pipeline-lock", type=Path, required=True); x.add_argument("--family-lock", type=Path, required=True)
    x = s.add_parser("build-confirmation-lock"); x.add_argument("--graphs", type=Path, nargs="+", required=True); x.add_argument("--selection", type=Path, required=True); x.add_argument("--protocol", type=Path, required=True); x.add_argument("--family-lock", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("decide-final"); x.add_argument("--selection", type=Path, required=True); x.add_argument("--metrics", type=Path, required=True); x.add_argument("--paired", type=Path, required=True); x.add_argument("--separability", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--report", type=Path, required=True)
    x = s.add_parser("package-round"); x.add_argument("--root", type=Path, required=True); x.add_argument("--output", type=Path, required=True)
    x = s.add_parser("package-complete"); x.add_argument("--root", type=Path, required=True); x.add_argument("--output", type=Path, required=True); x.add_argument("--rounds", type=Path, nargs="*")
    return p


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    c = a.command
    if c == "freeze-history": return _status(freeze_history(a.repo, a.output, a.paths))
    if c == "validate-inputs": return _status(verify_history(a.lock))
    if c == "run-evaluator-tests":
        import unittest
        suite = unittest.defaultTestLoader.loadTestsFromName("upgrade_v2.u4_conditional.test_evaluator_v2")
        result = unittest.TextTestRunner(verbosity=1).run(suite)
        return 0 if result.wasSuccessful() else 2
    if c == "compare-evaluator-contracts":
        result = compare_old_new(read_jsonl(a.old), read_jsonl(a.new)); write_json(a.output, result); return _status(result)
    if c == "rescore-frozen-confirmation":
        graph = read_json(a.graph); rows = rescore(graph, read_jsonl(a.occurrences)); write_jsonl(a.output, rows); return _status({"status": "PASS", "rows": len(rows), "evaluator_semantics": True})
    if c == "compare-old-new-metrics":
        result = compare_old_new(read_jsonl(a.old), read_jsonl(a.new)); write_json(a.output, result); return _status(result)
    if c == "build-semantic-occurrence-table":
        result = build_occurrences(a.rollouts, a.output, a.repo, a.split)
        if a.summary: write_occurrence_summary(read_jsonl(a.output), a.summary)
        return _status(result)
    if c == "summarize-mixed-pairs": return _status(summarize_mixed(a.occurrences, a.output))
    if c == "fit-separability-baselines": return _status(fit_baselines(a.occurrences, a.output, a.table))
    if c == "build-conditional-graphs": return _status(build_graphs(a.raw_graph, a.occurrences, a.output_dir, a.edit_log))
    if c == "evaluate-conditional-graphs": return _status(evaluate_graphs(a.graphs, a.occurrences, a.output, a.table))
    if c == "select-conditional-graph": return _status(select(a.metrics, a.output, a.report))
    if c == "generate-fresh-families": return _status(generate(a.output, a.lock, a.seed, a.count, a.rollout_seed_base))
    if c == "infer-fresh-boundaries":
        from upgrade_v2.u4_bplus.auto_boundary import infer_rollouts
        result = infer_rollouts(a.checkpoint, a.rollouts, a.output, a.manifest, a.device); return _status(result)
    if c == "build-confirmation-occurrences": return _status(build_occurrences_confirm(a.rollouts, a.output, a.repo, a.predictions))
    if c == "evaluate-fresh-confirmation": return _status(evaluate_confirm(a.graphs, a.occurrences, a.output, a.paired, a.family_table, a.pipeline_lock, a.family_lock))
    if c == "build-confirmation-lock": return _status(build_lock(a.output, a.graphs, a.selection, a.protocol, a.family_lock))
    if c == "decide-final": return _status(__import__("upgrade_v2.u4_conditional.handoff", fromlist=["decide"]).decide(a.selection, a.metrics, a.paired, a.separability, a.output, a.report))
    if c == "package-round": return _status(package_round(a.root, a.output))
    if c == "package-complete": return _status(package_complete(a.root, a.output, a.rounds or []))
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
