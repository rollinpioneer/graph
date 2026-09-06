"""CLI for the deterministic U3 evidence-grounding bridge."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .common import read_json, sha256_file, secret_scan, write_json
from .data_graph import build_data_only_graph
from .evaluate_graphs import evaluate_graphs
from .evidence_catalog import build_evidence_handles
from .freeze import freeze_u3_result
from .ground_edges import ground_edges
from .ground_nodes import ground_nodes
from .handoff import build_u4_handoff
from .package import package_complete, package_round
from .predicate_profiles import build_cluster_profiles
from .select_graphs import select_graphs
from .semantic_graph import normalize_semantic_candidates
from .validate_grounding import assemble_grounded_graphs, validate_grounding


def _path(value: str) -> Path: return Path(value).expanduser()
def _list(value: str) -> list[float]: return [float(x) for x in value.split(",") if x]
def _emit(value: dict) -> int:
    print(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)); return 0 if value.get("status") not in {"FAIL", "REPAIR_GROUNDING_LOGIC", "REPAIR_EVIDENCE_SOURCE"} else 2


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="python -m upgrade_v2.u3_grounding.cli"); s = p.add_subparsers(dest="command", required=True)
    x=s.add_parser("freeze-u3-result"); x.add_argument("--execution-summary",type=_path,required=True); x.add_argument("--handoff",type=_path,required=True); x.add_argument("--hard-checks",type=_path,required=True); x.add_argument("--scores",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--lock",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("build-evidence-handles"); x.add_argument("--clusters",type=_path,required=True); x.add_argument("--transitions",type=_path,required=True); x.add_argument("--fallback",type=_path,required=True); x.add_argument("--registry",type=_path,required=True); x.add_argument("--output-dir",type=_path,required=True); x.add_argument("--manifest",type=_path,required=True)
    x=s.add_parser("build-cluster-profiles"); x.add_argument("--cluster-catalog",type=_path,required=True); x.add_argument("--predicate-vocabulary",type=_path,required=True); x.add_argument("--train-segments",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--table",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("build-data-only-graph"); x.add_argument("--cluster-profiles",type=_path,required=True); x.add_argument("--transition-catalog",type=_path,required=True); x.add_argument("--minimum-transition-families",type=int,required=True); x.add_argument("--minimum-cluster-families",type=int,required=True); x.add_argument("--merge-predicate-jaccard",type=float,required=True); x.add_argument("--merge-transition-jaccard",type=float,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--edit-log",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("normalize-semantic-candidates"); x.add_argument("--hard-checks",type=_path,required=True); x.add_argument("--candidate-root",type=_path,required=True); x.add_argument("--candidate-ids",nargs="+",required=True); x.add_argument("--output-dir",type=_path,required=True); x.add_argument("--manifest",type=_path,required=True)
    x=s.add_parser("ground-nodes"); x.add_argument("--candidate-dir",type=_path,required=True); x.add_argument("--cluster-profiles",type=_path,required=True); x.add_argument("--thresholds",required=True); x.add_argument("--top-k",type=int,required=True); x.add_argument("--output-dir",type=_path,required=True); x.add_argument("--table",type=_path,required=True)
    x=s.add_parser("ground-edges"); x.add_argument("--candidate-dir",type=_path,required=True); x.add_argument("--node-grounding",type=_path,required=True); x.add_argument("--transition-catalog",type=_path,required=True); x.add_argument("--fallback-catalog",type=_path,required=True); x.add_argument("--top-k",type=int,required=True); x.add_argument("--output-dir",type=_path,required=True); x.add_argument("--table",type=_path,required=True)
    x=s.add_parser("assemble-grounded-graphs"); x.add_argument("--semantic-candidates",type=_path,required=True); x.add_argument("--node-grounding",type=_path,required=True); x.add_argument("--edge-grounding",type=_path,required=True); x.add_argument("--thresholds",required=True); x.add_argument("--output-dir",type=_path,required=True); x.add_argument("--manifest",type=_path,required=True)
    x=s.add_parser("validate-grounding"); x.add_argument("--manifest",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("evaluate-graphs"); x.add_argument("--grounded-graph-root",type=_path); x.add_argument("--data-only-graph",type=_path,required=True); x.add_argument("--selected",type=_path); x.add_argument("--dataset",type=_path,required=True); x.add_argument("--boundary-cache",type=_path); x.add_argument("--reward-cache",type=_path); x.add_argument("--split",required=True); x.add_argument("--statistics-unit",required=True); x.add_argument("--protocol",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--details",type=_path,required=True); x.add_argument("--bootstrap-root-families",type=int,default=5000); x.add_argument("--bootstrap-seed",type=int,default=20261011); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("select-graphs"); x.add_argument("--val-metrics",type=_path,required=True); x.add_argument("--require-start-success-reachable",action="store_true"); x.add_argument("--max-contradicted-edge-rate",type=float,required=True); x.add_argument("--max-unresolved-edge-rate",type=float,required=True); x.add_argument("--one-per-source",action="store_true"); x.add_argument("--output",type=_path,required=True); x.add_argument("--lock",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("build-u4-handoff"); x.add_argument("--decision",type=_path,required=True); x.add_argument("--selected",type=_path,required=True); x.add_argument("--val-metrics",type=_path,required=True); x.add_argument("--test-metrics",type=_path,required=True); x.add_argument("--edge-details",type=_path,required=True); x.add_argument("--fallback-policy",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--element-table",type=_path,required=True); x.add_argument("--query-queue",type=_path,required=True); x.add_argument("--copy-graphs",type=_path,required=True); x.add_argument("--report",type=_path,required=True)
    x=s.add_parser("package-round"); x.add_argument("--round-dir",type=_path,required=True); x.add_argument("--output",type=_path,required=True); x.add_argument("--max-file-mb",type=int,default=200)
    x=s.add_parser("package-complete"); x.add_argument("--root",type=_path,required=True); x.add_argument("--final-root",type=_path,required=True); x.add_argument("--round-zip-dir",type=_path); x.add_argument("--output",type=_path,required=True); x.add_argument("--max-file-mb",type=int,default=200)
    x=s.add_parser("secret-scan"); x.add_argument("--paths",nargs="+",type=_path,required=True); x.add_argument("--output",type=_path,required=True)
    return p


def main() -> int:
    a=build_parser().parse_args(); c=a.command
    if c == "freeze-u3-result": return _emit(freeze_u3_result(execution_summary=a.execution_summary,handoff=a.handoff,hard_checks=a.hard_checks,scores=a.scores,output=a.output,lock=a.lock,report=a.report))
    if c == "build-evidence-handles": return _emit(build_evidence_handles(clusters=a.clusters,transitions=a.transitions,fallback=a.fallback,registry=a.registry,output_dir=a.output_dir,manifest=a.manifest))
    if c == "build-cluster-profiles": return _emit(build_cluster_profiles(cluster_catalog=a.cluster_catalog,predicate_vocabulary=a.predicate_vocabulary,train_segments=a.train_segments,output=a.output,table=a.table,report=a.report))
    if c == "build-data-only-graph": return _emit(build_data_only_graph(cluster_profiles=a.cluster_profiles,transition_catalog=a.transition_catalog,minimum_transition_families=a.minimum_transition_families,minimum_cluster_families=a.minimum_cluster_families,merge_predicate_jaccard=a.merge_predicate_jaccard,merge_transition_jaccard=a.merge_transition_jaccard,output=a.output,edit_log=a.edit_log,report=a.report))
    if c == "normalize-semantic-candidates": return _emit(normalize_semantic_candidates(hard_checks=a.hard_checks,candidate_root=a.candidate_root,candidate_ids=a.candidate_ids,output_dir=a.output_dir,manifest=a.manifest))
    if c == "ground-nodes": return _emit(ground_nodes(candidate_dir=a.candidate_dir,cluster_profiles=a.cluster_profiles,thresholds=_list(a.thresholds),top_k=a.top_k,output_dir=a.output_dir,table=a.table))
    if c == "ground-edges": return _emit(ground_edges(candidate_dir=a.candidate_dir,node_grounding=a.node_grounding,transition_catalog=a.transition_catalog,fallback_catalog=a.fallback_catalog,top_k=a.top_k,output_dir=a.output_dir,table=a.table))
    if c == "assemble-grounded-graphs": return _emit(assemble_grounded_graphs(semantic_candidates=a.semantic_candidates,node_grounding=a.node_grounding,edge_grounding=a.edge_grounding,thresholds=_list(a.thresholds),output_dir=a.output_dir,manifest=a.manifest))
    if c == "validate-grounding": return _emit(validate_grounding(manifest=a.manifest,output=a.output,report=a.report))
    if c == "evaluate-graphs": return _emit(evaluate_graphs(grounded_graph_root=a.grounded_graph_root,data_only_graph=a.data_only_graph,selected=a.selected,dataset=a.dataset,split=a.split,statistics_unit=a.statistics_unit,protocol=a.protocol,output=a.output,details=a.details,report=a.report,bootstrap=a.bootstrap_root_families,seed=a.bootstrap_seed))
    if c == "select-graphs": return _emit(select_graphs(val_metrics=a.val_metrics,require_start_success_reachable=a.require_start_success_reachable,max_contradicted_edge_rate=a.max_contradicted_edge_rate,max_unresolved_edge_rate=a.max_unresolved_edge_rate,one_per_source=a.one_per_source,output=a.output,lock=a.lock,report=a.report))
    if c == "build-u4-handoff": return _emit(build_u4_handoff(decision=a.decision,selected=a.selected,val_metrics=a.val_metrics,test_metrics=a.test_metrics,edge_details=a.edge_details,fallback_policy=a.fallback_policy,output=a.output,element_table=a.element_table,query_queue=a.query_queue,copy_graphs=a.copy_graphs,report=a.report))
    if c == "package-round": return _emit(package_round(round_dir=a.round_dir,output=a.output,max_file_mb=a.max_file_mb))
    if c == "package-complete": return _emit(package_complete(root=a.root,final_root=a.final_root,round_zip_dir=a.round_zip_dir,output=a.output,max_file_mb=a.max_file_mb))
    if c == "secret-scan":
        result = secret_scan(a.paths); write_json(a.output, result); return _emit(result)
    raise AssertionError(c)


if __name__ == "__main__": raise SystemExit(main())
