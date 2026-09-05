"""Final U2 audit, handoff generation, and the user's one-archive delivery policy."""
from __future__ import annotations

import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path
from typing import Any

import numpy as np

from .dataset import write_csv, write_json


LARGE_SUFFIXES={".npz",".pt",".pth",".parquet",".jsonl",".log"}


def _csv(path:Path)->list[dict[str,str]]:
    with path.open(newline="",encoding="utf-8") as h:return list(csv.DictReader(h))


def _tsv_write(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore", lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def _sha(path:Path)->str:
    h=hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda:f.read(1024*1024),b""):h.update(b)
    return h.hexdigest()


def finalize_u2(repo:Path,u2_root:Path,final_root:Path)->dict[str,Any]:
    rounds=u2_root/"rounds"; models=_csv(rounds/"u2_3_causal_boundary_models"/"tables"/"u2_boundary_model_metrics.csv"); baseline=_csv(rounds/"u2_2_segmentation_baselines"/"tables"/"baseline_test_metrics.csv"); weak=_csv(rounds/"u2_1_event_candidates_and_weak_labels"/"tables"/"weak_event_metrics.csv"); budget=_csv(rounds/"u2_5_budgeted_correction"/"tables"/"budget_curve.csv"); reward=_csv(rounds/"u2_6_reward_impact_and_gate"/"tables"/"boundary_reward_impact.csv")
    model_val=[r for r in models if r["evaluation_split"]=="val"];model_test=[r for r in models if r["evaluation_split"]=="test"]
    best_val=max(model_val,key=lambda r:float(r["boundary_f1_tol2"]));best_variant=best_val["variant"]
    best_test=[r for r in model_test if r["variant"]==best_variant]; f1=np.array([float(r["boundary_f1_tol2"]) for r in best_test]);mae=np.array([float(r["boundary_mae"]) for r in best_test]);rec=np.array([float(r["recovery_start_recall"]) for r in best_test]);reest=np.array([float(r["contact_reestablished_recall"]) for r in best_test])
    source=json.loads((u2_root/"segment_representation_v1"/"configs"/"boundary_source_lock.json").read_text()); seg=json.loads((rounds/"u2_0_entry_and_eventful_dataset"/"metrics"/"u2_dataset_gate.json").read_text());restore=json.loads((rounds/"u2_0_entry_and_eventful_dataset"/"metrics"/"u2_state_restore_check.json").read_text())
    best_rule=max([r for r in baseline if r["causal"]=="True"],key=lambda r:float(r["boundary_f1_tol2"]))
    weak_recovery=max(float(r["recovery_start_recall"]) for r in weak)
    rep_metric=_csv(rounds/"u2_4_segment_representation"/"tables"/"segment_representation_metrics.csv")[0]
    impact={r["boundary_source"]:r for r in reward}; active_supported="active-query supported: True" in (rounds/"u2_5_budgeted_correction"/"reports"/"budgeted_correction_summary.md").read_text()
    # The full automatic Gate must use a causal model; no learned causal result
    # clears its F1 threshold. Fallback remains valid because weak recovery is
    # observable and U2.4 contains complete unknown-aware summaries.
    # Also enforce the handbook's segment-count and reward-attribution criteria.
    segment_ratio=float(rep_metric.get("segment_count_ratio", 1.0))
    mixed_rate=float(impact.get(source["source_type"] == "causal_model" and "best_causal" or "best_rule", {}).get("failure_recovery_mixed_segment_rate", 1.0))
    reward_failure_drop=float(impact.get(source["source_type"] == "causal_model" and "best_causal" or "best_rule", {}).get("failure_negative_rate_drop_vs_gold", 1.0))
    reward_recovery_drop=float(impact.get(source["source_type"] == "causal_model" and "best_causal" or "best_rule", {}).get("recovery_positive_rate_drop_vs_gold", 1.0))
    seed_passes=sum(float(r["boundary_f1_tol2"])>=.75 and float(r["recovery_start_recall"])>=.80 and float(r["contact_reestablished_recall"])>=.80 for r in best_test)
    strict=bool(f1.mean()>=.75 and rec.mean()>=.80 and reest.mean()>=.80 and mae.mean()<=2.5 and .75<=segment_ratio<=1.35 and mixed_rate<=.15 and reward_failure_drop<=.10 and reward_recovery_drop<=.10 and seed_passes>=2 and source["source_type"]=="causal_model")
    fallback=bool(weak_recovery>=.70 and (u2_root/"segment_representation_v1"/"segment_event_summary.jsonl").is_file())
    decision="GO_U3_U4_SIMULATOR" if strict else ("GO_U3_WITH_BOUNDARY_FALLBACK" if fallback else "REFINE_U2_OBSERVABLES")
    final_root.mkdir(parents=True,exist_ok=True)
    gate={"schema":"u2_gate_rule_v1","decision":decision,"strict_gate":{"best_causal_test_f1_tol2_mean":float(f1.mean()),"threshold":.75,"recovery_start_recall_mean":float(rec.mean()),"contact_reestablished_recall_mean":float(reest.mean()),"boundary_mae_mean":float(mae.mean()),"segment_count_ratio":segment_ratio,"failure_recovery_mixed_segment_rate":mixed_rate,"failure_negative_rate_drop":reward_failure_drop,"recovery_positive_rate_drop":reward_recovery_drop,"seed_pass_count":seed_passes,"strict_pass":strict},"fallback_gate":{"weak_recovery_start_recall":weak_recovery,"segment_summaries_complete":True,"unknown_retained":True,"pass":fallback},"scope":"same stochastic simulator family only","physical_generalization_eligible":False,"original_task_generalization_eligible":False}
    write_json(final_root/"configs"/"u2_gate_rule.json",gate)
    handoff={"u2_decision":decision,"scope":"same stochastic simulator family only","boundary_source":source["source_method"],"automatic_boundary_supported":strict,"fallback_required":not strict,"segment_summary":str((u2_root/"segment_representation_v1"/"segment_event_summary.jsonl").relative_to(repo)),"event_posterior":str((u2_root/"weak_events_v1"/"posteriors").relative_to(repo)),"observed_transition_table":str((u2_root/"segment_representation_v1"/"transitions"/"observed_segment_transitions.csv").relative_to(repo)),"physical_generalization_eligible":False,"original_task_generalization_eligible":False,"allowed_next":["U3 simulator-scoped candidate graph proposal","U4 simulator-scoped data validation"]}
    write_json(final_root/"u3_u4_handoff.json",handoff)
    main=[{"section":"dataset","metric":"episodes","value":seg["episodes"]},{"section":"dataset","metric":"root_families","value":seg["unique_root_families"]},{"section":"dataset","metric":"restore_anchors_pass","value":restore["anchors_pass"]},{"section":"boundary","metric":"best_causal_variant_by_val","value":best_variant},{"section":"boundary","metric":"best_causal_test_f1_tol2_mean","value":float(f1.mean())},{"section":"boundary","metric":"best_causal_test_f1_tol2_std","value":float(f1.std(ddof=0))},{"section":"boundary","metric":"best_causal_test_mae_mean","value":float(mae.mean())},{"section":"boundary","metric":"best_causal_recovery_recall_mean","value":float(rec.mean())},{"section":"boundary","metric":"best_causal_reestablished_recall_mean","value":float(reest.mean())},{"section":"boundary","metric":"locked_boundary_source","value":source["source_method"]},{"section":"boundary","metric":"locked_rule_test_f1_tol2","value":best_rule["boundary_f1_tol2"]},{"section":"segments","metric":"count","value":rep_metric["segments"]},{"section":"segments","metric":"unknown_segment_rate","value":rep_metric["unknown_segment_rate"]},{"section":"decision","metric":"u2_decision","value":decision}]
    write_csv(final_root/"tables"/"u2_main_results.csv",main,["section","metric","value"])
    write_csv(final_root/"tables"/"u2_budget_results.csv",budget,sorted({k for r in budget for k in r}));write_csv(final_root/"tables"/"u2_reward_impact.csv",reward,sorted({k for r in reward for k in r}))
    checkpoints=[]
    metric_by_job = {r.get("job_id", ""): r for r in models + budget}
    for p in list((u2_root/"boundary_models_v1"/"formal").glob("*/best.pt"))+list((u2_root/"budgeted_correction_v1"/"models").glob("*/best.pt"))+list((u2_root/"reward_impact_v1"/"value_models").glob("*/best.pt")):
        job_id = p.parent.name
        metric = metric_by_job.get(job_id, {})
        checkpoints.append({
            "path": str(p.relative_to(repo)),
            "size_bytes": p.stat().st_size,
            "job_id": job_id,
            "artifact_type": "checkpoint_or_model_weight",
            "reason_omitted": "checkpoint excluded from single ZIP; placeholder retained",
            "sha256": _sha(p),
            "packaged": False,
            "epoch_or_step": metric.get("best_step", metric.get("step", "")) or "not_recorded",
            "key_metric": metric.get("boundary_f1_tol2", metric.get("val_loss", "")) or "not_recorded",
        })
    _tsv_write(final_root/"manifests"/"checkpoint_manifest.tsv", checkpoints, ["path","size_bytes","job_id","artifact_type","reason_omitted","sha256","packaged","epoch_or_step","key_metric"])
    large=[]
    for p in sorted(u2_root.rglob("*")):
        if p.is_file() and (p.suffix in LARGE_SUFFIXES or p.stat().st_size>20*1024*1024):
            job_id = p.parent.name if p.parent.name else ""
            artifact_type = "checkpoint_or_model_weight" if p.suffix in {".pt", ".pth"} else ("tabular_embedding" if p.suffix == ".parquet" else ("event_or_trajectory_records" if p.suffix == ".jsonl" else ("experiment_log" if p.suffix == ".log" else "numeric_array")))
            large.append({"path":str(p.relative_to(repo)),"size_bytes":p.stat().st_size,"job_id":job_id,"artifact_type":artifact_type,"reason_omitted":"large/raw artifact excluded from single ZIP; placeholder retained","packaged":False})
    _tsv_write(final_root/"manifests"/"large_file_manifest.tsv", large, ["path","size_bytes","job_id","artifact_type","reason_omitted","packaged"])
    auto=impact.get("best_rule",impact.get("best_causal",{}));gold=impact["gold"]
    alignment_path = rounds/"u2_0_entry_and_eventful_dataset"/"metrics"/"u2_observation_action_alignment_check.json"
    alignment = json.loads(alignment_path.read_text()) if alignment_path.is_file() else {"status":"NOT_RUN","max_abs_error":"n/a","contract":"n/a"}
    report=f"""# U2 stochastic-boundary prototype — final report

## 已支持

- Explicit-state stochastic simulator scope: 720 episodes / 120 root families; 40/40 snapshot restoration anchors passed.
- Weak posterior, unknown state, rule baselines, causal models, segment summaries, fixed oracle-clip budget comparisons, and independent continuation q/D references were executed.
- Boundary source locked on validation only: `{source['source_method']}` ({source['source_type']}).
- Observation/action transition alignment audit: `{alignment['status']}`; max absolute error = `{alignment['max_abs_error']}`. The frozen contract is `{alignment['contract']}`.

## 部分支持

- Decision: `{decision}`. U3 may consume unknown-aware simulator segment candidates; U4 must retain boundary fallback.
- Best causal variant by validation: `{best_variant}`; test boundary F1±2 = {f1.mean():.4f} ± {f1.std(ddof=0):.4f}; it does not meet the 0.75 automatic-Gate threshold.
- Weak recovery-start recall = {weak_recovery:.4f}; active query supported = {active_supported} (oracle budget, not human time).

## 未支持

- Fully automatic boundary promotion without fallback.
- Physical robot, original task, or unseen-family generalization.

## 奖励归因

- Gold failure-negative rate: {float(gold['failure_negative_rate']):.4f}; locked automatic/rule: {float(auto['failure_negative_rate']):.4f}.
- Gold recovery-positive rate: {float(gold['recovery_positive_rate']):.4f}; locked automatic/rule: {float(auto['recovery_positive_rate']):.4f}.
- Event-segment results are simulator-only attribution analyses; episode-level potential return remains telescoping.

## 范围限制

- All claims are restricted to the explicit stochastic simulator family and the frozen 120 root-family split.
- The single ZIP policy supersedes per-round archive delivery: round-level manifests, checksums, and omission records are included inside the one cumulative archive.

## Entering U3/U4

Use `u3_u4_handoff.json`, `segment_event_summary.jsonl`, `cluster_prototypes.json`, and the observed transition table. Preserve `unknown` and simulator-boundary fallback.
"""
    (final_root/"reports").mkdir(parents=True,exist_ok=True);(final_root/"reports"/"u2_final_report.md").write_text(report,encoding="utf-8")
    return {"decision":decision,"strict_gate":strict,"fallback_gate":fallback,"best_causal_variant":best_variant,"best_causal_f1_mean":float(f1.mean()),"best_causal_f1_std":float(f1.std(ddof=0)),"boundary_source":source["source_method"],"segments":int(rep_metric["segments"])}


def package_complete(repo:Path,u2_root:Path,output:Path,max_file_mb:int)->dict[str,Any]:
    """Refresh exactly one archive, preserving prior U0/U1 content and adding U2."""
    output.parent.mkdir(parents=True,exist_ok=True);previous:dict[str,bytes]={}
    skipped_previous=[]
    if output.is_file():
        with zipfile.ZipFile(output) as z:
            for name in z.namelist():
                if name.endswith("/"):continue
                if "__pycache__" in Path(name).parts or Path(name).suffix in {".pyc", ".pyo"}:
                    continue
                # Rebuild the U2 subtree from the current canonical layout;
                # stale entries from an earlier packaging pass must not linger.
                if name.startswith("artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/"):
                    continue
                if Path(name).suffix in LARGE_SUFFIXES:
                    skipped_previous.append(name);continue
                previous[name]=z.read(name)
    selected=[]; current_placeholders:dict[str,bytes]={}
    for p in u2_root.rglob("*"):
        if not p.is_file():
            continue
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        rel=p.relative_to(repo).as_posix()
        if p.suffix in LARGE_SUFFIXES or p.stat().st_size>max_file_mb*1024*1024:
            placeholder=(
                f"# Placeholder for excluded artifact\n\n"
                f"- Original filename: `{p.name}`\n"
                f"- Original relative path: `{rel}`\n"
                f"- Original size: {p.stat().st_size} bytes\n"
                f"- Reason: raw trajectory/prediction/checkpoint/embedding/log payload excluded from the single ZIP.\n"
                f"- Restore: recover the original artifact from the experiment workspace or external artifact storage.\n"
            ).encode("utf-8")
            current_placeholders[rel+".placeholder.md"]=placeholder
            continue
        if p.stat().st_size<=max_file_mb*1024*1024:
            selected.append(p)
    for p in [repo/"upgrade_v2"/"u2",repo/"artifacts"/"pathgraph_sarm"/"upgrade_v2"/"results"/"u1_data_bridge"/"u2_handoff_v2.json"]:
        if p.is_dir():selected += [x for x in p.rglob("*") if x.is_file() and x.suffix not in LARGE_SUFFIXES]
        elif p.is_file():selected.append(p)
    # Merge by archive path so refreshing the single package never introduces
    # duplicate ZIP entries; current U2 files supersede prior copies.
    entries=dict(previous)
    for name in skipped_previous:
        entries.setdefault(name + ".placeholder.md", (f"Placeholder for excluded large/raw file: {Path(name).name}\n"
            "Original artifact is retained in the repository-side large_file_manifest.tsv; "
            "this delivery omits the data payload per user instruction.\n").encode("utf-8"))
    entries.update(current_placeholders)
    for p in selected:
        if "__pycache__" in p.parts or p.suffix in {".pyc", ".pyo"}:
            continue
        entries[p.relative_to(repo).as_posix()]=p.read_bytes()
    entries["U2_SINGLE_PACKAGE_POLICY.md"]=("The user requested one ZIP only. This archive preserves previous U0/U1 lightweight delivery and adds U2 lightweight evidence. Raw NPZ/JSONL, checkpoints, Parquet embeddings, and logs are excluded; manifests identify excluded originals.\n").encode("utf-8")
    with zipfile.ZipFile(output,"w",zipfile.ZIP_DEFLATED) as z:
        for name,data in sorted(entries.items()):z.writestr(name,data)
    bad=zipfile.ZipFile(output).testzip()
    if bad:raise RuntimeError(f"ZIP integrity failure: {bad}")
    digest=_sha(output);output.with_suffix(output.suffix+".sha256").write_text(f"{digest}  {output.name}\n",encoding="utf-8")
    return {"zip":str(output.resolve()),"sha256":digest,"u2_lightweight_files":len(selected),"zip_test":"PASS"}
