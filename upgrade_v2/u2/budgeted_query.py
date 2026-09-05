"""Train-only oracle-clip query queues and fixed-budget correction experiments."""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from .dataset import load_episode, read_csv, write_csv, write_json
from .evaluate import evaluate_predictions
from .jobs import (_read_tsv, _write_tsv, build_inference_jobs, launch_inference_jobs,
                   launch_training_jobs, select_checkpoints)


PROTOCOL: dict[str, Any] = {
    "schema": "u2_budget_protocol_v1",
    "budget_unit": "simulator_gold_oracle_clip_budget_not_human_minutes",
    "budgets": [0, 15, 30, 60],
    "strategies": ["random_stratified", "entropy_only", "entropy_x_event_importance", "expected_boundary_disagreement"],
    "importance": {"recovery_start": 2.0, "contact_off_failure": 2.0, "contact_reestablished": 2.0, "terminal_failure": 1.5, "goal_enter": 1.2, "other": 1.0},
    "selection_split": "train",
    "test_used_for_queue_selection": False,
}


def freeze_protocol(path: Path) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(PROTOCOL, sort_keys=False), encoding="utf-8")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".sha256").write_text(f"{digest}  {path.name}\n", encoding="utf-8")
    return {"protocol": str(path.resolve()), "sha256": digest}


def _importance(event: int) -> float:
    return {3: 2.0, 4: 2.0, 5: 2.0, 9: 1.5, 7: 1.2}.get(event, 1.0)


def _candidate_rows(dataset: Path, weak: Path, boundary_predictions: Path) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    # Use only train episodes.  Predicted uncertainty and observable weak
    # events form a queue; simulator gold is not consulted here.
    for row in read_csv(dataset / "episode_manifest.csv"):
        if row["split"] != "train": continue
        with np.load(weak / "weak_plus_small_gold_calibration" / f"{row['episode_id']}.npz") as d:
            entropy = d["posterior_entropy"]; argmax = d["event_argmax"]; boundary = d["boundary_probability"]
        # The current frozen boundary source can be a rule baseline.  If a
        # causal prediction exists, its probability contributes disagreement;
        # otherwise a neutral observable-only disagreement is retained.
        causal_values=[]
        possible = list(boundary_predictions.glob(f"*/train/{row['episode_id']}.npz"))
        for path in possible:
            with np.load(path) as d: causal_values.append(d["boundary_probability"].astype(float))
        disagreement=np.std(np.stack(causal_values),axis=0) if len(causal_values)>=2 else np.abs(boundary-(causal_values[0] if causal_values else boundary))
        for t in range(len(boundary)):
            e = int(argmax[t]); imp = _importance(e)
            candidates.append({"clip_id": f"{row['episode_id']}:t{t:03d}", "episode_id": row["episode_id"], "start_t": max(0, t - 2), "end_t": min(len(boundary) - 1, t + 2), "root_family_id": row["root_family_id"], "uncertainty": float(entropy[t]), "event_importance": imp, "model_disagreement": float(disagreement[t]), "estimated_steps": min(len(boundary) - 1, t + 2) - max(0, t - 2) + 1, "predicted_event_id": e})
    return candidates


def build_queues(dataset: Path, weak: Path, boundary_predictions: Path, budgets: list[int], strategies: list[str], seeds: list[int], output_root: Path, manifest: Path) -> list[dict[str, Any]]:
    candidates = _candidate_rows(dataset, weak, boundary_predictions); output_root.mkdir(parents=True, exist_ok=True); out: list[dict[str, Any]] = []
    for budget in budgets:
        for strategy in strategies:
            for seed in seeds:
                rng = np.random.default_rng(seed); order = list(range(len(candidates)))
                if strategy == "random_stratified":
                    # Shuffle after stratifying rough event-importance tiers.
                    order = []
                    for imp in sorted({x["event_importance"] for x in candidates}, reverse=True):
                        indices = [i for i,x in enumerate(candidates) if x["event_importance"] == imp]; rng.shuffle(indices); order.extend(indices)
                else:
                    def score(x: dict[str, Any]) -> float:
                        if strategy == "entropy_only": return x["uncertainty"]
                        if strategy == "entropy_x_event_importance": return x["uncertainty"] * x["event_importance"]
                        return x["uncertainty"] * x["event_importance"] * (0.25 + x["model_disagreement"]) / x["estimated_steps"]
                    order.sort(key=lambda i: (score(candidates[i]), rng.random()), reverse=True)
                selected = [dict(candidates[i], strategy=strategy, seed=seed, rank=rank + 1, budget=budget) for rank, i in enumerate(order[:budget])]
                name = f"budget{budget}_{strategy}_s{seed}.csv"; queue = output_root / name
                fields = ["clip_id", "episode_id", "start_t", "end_t", "root_family_id", "strategy", "seed", "rank", "budget", "uncertainty", "event_importance", "model_disagreement", "estimated_steps", "predicted_event_id"]
                write_csv(queue, selected, fields)
                out.append({"queue": str(queue.resolve()), "name": name, "budget": budget, "strategy": strategy, "seed": seed, "clips": len(selected), "split": "train", "test_used": False})
    write_csv(manifest, out, list(out[0])); return out


def reveal_oracle_queues(queue_root: Path, dataset: Path, output_root: Path, provenance: str) -> list[dict[str, Any]]:
    allowed = {r["episode_id"]: r for r in read_csv(dataset / "episode_manifest.csv") if r["split"] == "train"}; output_root.mkdir(parents=True, exist_ok=True); manifest=[]
    for queue in sorted(queue_root.glob("budget*.csv")):
        rows = list(csv.DictReader(queue.open()))
        revealed=[]
        for row in rows:
            if row["episode_id"] not in allowed: raise RuntimeError("query queue contains a non-train episode")
            gold=load_episode(allowed[row["episode_id"]]); left=int(row["start_t"]); right=int(row["end_t"])+1
            row.update({"gold_event_ids": json.dumps(gold["gold_event_id"][left:right].tolist()), "gold_boundaries": json.dumps(gold["gold_boundary"][left:right].tolist()), "provenance": provenance, "not_human_annotation": True})
            revealed.append(row)
        target=output_root / queue.name; write_csv(target,revealed,list(revealed[0]) if revealed else list(rows[0]) if rows else ["clip_id"])
        manifest.append({"queue":str(target.resolve()),"clips":len(revealed),"provenance":provenance,"split":"train"})
    write_csv(output_root / "revealed_manifest.csv",manifest,list(manifest[0])); return manifest


def build_budget_jobs(dataset: Path, weak: Path, revealed: Path, budgets: list[int], strategies: list[str], seeds: list[int], steps: int, output_root: Path, table: Path) -> list[dict[str, Any]]:
    rows=[]
    for seed in seeds:
        rows.append({"job_id":f"budget0_s{seed}","budget":0,"strategy":"none","seed":seed,"variant":"causal_weak_only","dataset":str(dataset.resolve()),"weak_posteriors":str(weak.resolve()),"oracle_clips":"","steps":steps,"output_dir":str((output_root/f"budget0_s{seed}").resolve()),"test_used_for_selection":False,"cuda_required":True})
    for budget in budgets:
        if budget == 0: continue
        for strategy in strategies:
            for seed in seeds:
                q=revealed / f"budget{budget}_{strategy}_s{seed}.csv"
                if not q.is_file(): raise FileNotFoundError(q)
                rows.append({"job_id":f"budget{budget}_{strategy}_s{seed}","budget":budget,"strategy":strategy,"seed":seed,"variant":"causal_weak_plus_oracle_budget","dataset":str(dataset.resolve()),"weak_posteriors":str(weak.resolve()),"oracle_clips":str(q.resolve()),"steps":steps,"output_dir":str((output_root/f"budget{budget}_{strategy}_s{seed}").resolve()),"test_used_for_selection":False,"cuda_required":True})
    _write_tsv(table,rows); return rows


def evaluate_budget(job_root: Path, dataset: Path, selection_split: str, test_split: str, output: Path, per_event: Path, report: Path) -> tuple[list[dict[str, Any]], bool]:
    rows=[]; event_rows=[]
    for result_path in sorted(job_root.glob("*/train_result.json")):
        result=json.loads(result_path.read_text()); name=result["job_id"]; pred_root=result_path.parent
        # Validation comes from the train job itself; frozen test is in the
        # dedicated prediction directory one level up.
        for split, root in [(selection_split, pred_root / "val_predictions"), (test_split, job_root.parent / "predictions" / name / test_split)]:
            summary, events=evaluate_predictions(dataset,root,name,split,2)
            parts=name.split("_"); budget=0 if name.startswith("budget0") else int(name.split("_")[0].replace("budget","")); strategy="none" if budget==0 else "_".join(parts[1:-1])
            summary.update({"job_id":name,"budget":budget,"strategy":strategy,"seed":result["seed"],"evaluation_split":split,"reviewed_clips":budget,"oracle_budget_not_human_time":True,"test_used_for_selection":False}); rows.append(summary); event_rows.extend(events)
    write_csv(output,rows,sorted({k for r in rows for k in r})); write_csv(per_event,event_rows,list(event_rows[0]))
    test=[r for r in rows if r["evaluation_split"]==test_split]
    def mean(budget:int,strategy:str,key:str)->float:
        vals=[float(r[key]) for r in test if int(r["budget"])==budget and r["strategy"]==strategy]; return float(np.mean(vals)) if vals else float("nan")
    active=mean(30,"entropy_x_event_importance","boundary_f1_tol2"); random=mean(30,"random_stratified","boundary_f1_tol2")
    active_rec=mean(30,"entropy_x_event_importance","recovery_start_recall"); random_rec=mean(30,"random_stratified","recovery_start_recall")
    supported=(active-random>=.03) or (active_rec-random_rec>=.08)
    report.parent.mkdir(parents=True,exist_ok=True); report.write_text(f"# U2 budgeted correction\n\n- oracle budget, not human time: true\n- zero F1±2: {mean(0,'none','boundary_f1_tol2'):.4f}\n- random 30 F1±2: {random:.4f}\n- active 30 F1±2: {active:.4f}\n- active-query supported: {supported}\n",encoding="utf-8")
    return rows,supported
