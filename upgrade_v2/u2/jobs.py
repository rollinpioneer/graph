"""Frozen U2.3 training/inference job manifests and GPU-safe launcher."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from .dataset import load_episode, read_csv, write_csv, write_json
from .evaluate import evaluate_predictions
from .infer_boundary import infer
from .train_boundary import train_job


def _read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as h: return list(csv.DictReader(h, delimiter="\t"))
def _write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as h:
        w = csv.DictWriter(h, fieldnames=sorted({k for r in rows for k in r}), delimiter="\t", lineterminator="\n", extrasaction="ignore"); w.writeheader(); w.writerows(rows)


def select_gold_clips(dataset: Path, split: str, budget: int, seed: int, output: Path) -> None:
    import numpy as np
    rng = np.random.default_rng(seed); choices: list[dict[str, Any]] = []
    for row in read_csv(dataset / "episode_manifest.csv"):
        if row["split"] != split: continue
        gold = load_episode(row)["gold_event_id"]
        for t in np.where(gold != 0)[0]:
            choices.append({"clip_id": f"{row['episode_id']}:t{int(t):03d}", "episode_id": row["episode_id"], "root_family_id": row["root_family_id"], "split": split, "start_t": max(0, int(t) - 2), "end_t": min(len(gold) - 1, int(t) + 2), "center_t": int(t), "event_id": int(gold[t]), "label_source": "simulator_gold_oracle_budget", "not_human_minutes": True})
    rng.shuffle(choices); selected = choices[:budget]; write_csv(output, selected, list(selected[0]))


def build_jobs(mode: str, dataset: Path, weak: Path, variants: list[str], seeds: list[int], oracle: Path, steps: int, output_root: Path, table: Path, commands: Path | None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for variant in variants:
        for seed in seeds:
            job_id = f"{variant}_s{seed}"; rows.append({"job_id": job_id, "mode": mode, "variant": variant, "seed": seed, "dataset": str(dataset.resolve()), "weak_posteriors": str(weak.resolve()), "oracle_clips": str(oracle.resolve()), "steps": steps, "output_dir": str((output_root / job_id).resolve()), "test_used_for_selection": False, "cuda_required": True})
    _write_tsv(table, rows)
    if commands:
        commands.mkdir(parents=True, exist_ok=True)
        for row in rows:
            (commands / f"{row['job_id']}.sh").write_text(f"{sys.executable} -m upgrade_v2.u2.cli run-boundary-job --job-table {table.resolve()} --job-id {row['job_id']}\n", encoding="utf-8")
    return rows


def run_training_job(table: Path, job_id: str) -> dict[str, Any]:
    row = next((row for row in _read_tsv(table) if row["job_id"] == job_id), None)
    if row is None: raise ValueError(f"unknown job id: {job_id}")
    result = train_job(row); return result


def launch_training_jobs(table: Path, gpu_ids: list[str], status_output: Path) -> None:
    jobs = _read_tsv(table); pending = list(jobs); running: list[tuple[subprocess.Popen[str], dict[str, str], str]] = []; status: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < len(gpu_ids):
            row = pending.pop(0); gpu = gpu_ids[len(running) % len(gpu_ids)]; env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen([sys.executable, "-m", "upgrade_v2.u2.cli", "run-boundary-job", "--job-table", str(table), "--job-id", row["job_id"]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            running.append((proc, row, gpu))
        proc, row, gpu = running.pop(0); stdout, _ = proc.communicate()
        result_file = Path(row["output_dir"]) / "train_result.json"; result = json.loads(result_file.read_text()) if proc.returncode == 0 and result_file.is_file() else {"status": "FAIL", "error": stdout[-4000:]}
        status.append({"job_id": row["job_id"], "variant": row["variant"], "seed": row["seed"], "gpu": gpu, "return_code": proc.returncode, "status": result.get("status", "FAIL"), "cuda_used": result.get("cuda_used", False), "detail": result.get("error", "")})
    _write_tsv(status_output, status)
    if any(row["status"] != "PASS" for row in status): raise RuntimeError("one or more U2.3 training jobs failed")


def select_checkpoints(job_root: Path, table: Path, output: Path, lock: Path, checkpoint_manifest: Path) -> list[dict[str, Any]]:
    selection: list[dict[str, Any]] = []
    for job in _read_tsv(table):
        result = json.loads((Path(job["output_dir"]) / "train_result.json").read_text())
        selection.append({"job_id": job["job_id"], "variant": job["variant"], "seed": job["seed"], "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"], "best_step": result["best_step"], "boundary_f1_tol2": result["boundary_f1_tol2"], "recovery_start_recall": result["recovery_start_recall"], "boundary_mae": result["boundary_mae"], "cuda_used": result["cuda_used"], "selection_split": "val", "test_used_for_selection": False, "teacher_checkpoint": result.get("offline_teacher_checkpoint", "")})
    write_csv(output, selection, list(selection[0])); _write_tsv(checkpoint_manifest, selection)
    write_json(lock, {"selection_split": "val", "metric": "boundary_f1_tol2", "tiebreaker": ["recovery_start_recall", "boundary_mae"], "checkpoints": len(selection), "test_used_for_selection": False, "locked": True})
    return selection


def build_inference_jobs(selection: Path, dataset: Path, splits: list[str], output_root: Path, table: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for item in read_csv(selection):
        for split in splits:
            rows.append({"inference_id": f"{item['job_id']}_{split}", "job_id": item["job_id"], "variant": item["variant"], "seed": item["seed"], "checkpoint": item["checkpoint"], "dataset": str(dataset.resolve()), "split": split, "output_dir": str((output_root / item["job_id"] / split).resolve()), "cuda_required": True})
    _write_tsv(table, rows); return rows


def run_inference_job(table: Path, inference_id: str) -> None:
    item = next((x for x in _read_tsv(table) if x["inference_id"] == inference_id), None)
    if item is None: raise ValueError(f"unknown inference id: {inference_id}")
    # The weak path is always adjacent to the formal data tree in the U2 env;
    # retain it as an explicit job field if the layout changes later.
    weak = Path(os.environ.get("U2_WEAK", "artifacts/pathgraph_sarm/upgrade_v2/u2_stochastic_boundary/weak_events_v1")) / "posteriors"
    infer(Path(item["checkpoint"]), Path(item["dataset"]), weak, item["split"], Path(item["output_dir"]))
    write_json(Path(item["output_dir"]).parent / f"{item['split']}_inference_result.json", {"inference_id": inference_id, "status": "PASS", "cuda_used": True})


def launch_inference_jobs(table: Path, gpu_ids: list[str], status_output: Path) -> None:
    pending = _read_tsv(table); running: list[tuple[subprocess.Popen[str], dict[str, str], str]] = []; status: list[dict[str, Any]] = []
    while pending or running:
        while pending and len(running) < len(gpu_ids):
            item = pending.pop(0); gpu = gpu_ids[len(running) % len(gpu_ids)]; env = os.environ.copy(); env["CUDA_VISIBLE_DEVICES"] = gpu
            proc = subprocess.Popen([sys.executable, "-m", "upgrade_v2.u2.cli", "run-boundary-inference-job", "--job-table", str(table), "--inference-id", item["inference_id"]], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
            running.append((proc, item, gpu))
        proc, item, gpu = running.pop(0); stdout, _ = proc.communicate()
        marker = Path(item["output_dir"]).parent / f"{item['split']}_inference_result.json"
        status.append({"inference_id": item["inference_id"], "job_id": item["job_id"], "split": item["split"], "gpu": gpu, "return_code": proc.returncode, "status": "PASS" if proc.returncode == 0 and marker.is_file() else "FAIL", "detail": "" if proc.returncode == 0 else stdout[-4000:]})
    _write_tsv(status_output, status)
    if any(x["status"] != "PASS" for x in status): raise RuntimeError("one or more U2.3 inference jobs failed")


def evaluate_models(dataset: Path, prediction_root: Path, selection: Path, output: Path, per_event: Path, per_seed: Path, report: Path) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []; events: list[dict[str, Any]] = []
    for item in read_csv(selection):
        for split in ("val", "test"):
            summary, detail = evaluate_predictions(dataset, prediction_root / item["job_id"] / split, item["variant"], split, 2)
            summary.update({"job_id": item["job_id"], "variant": item["variant"], "seed": item["seed"], "evaluation_split": split, "causal": True, "test_used_for_selection": False}); summaries.append(summary); events.extend(detail)
    write_csv(output, summaries, sorted({k for r in summaries for k in r})); write_csv(per_event, events, list(events[0])); write_csv(per_seed, summaries, sorted({k for r in summaries for k in r}))
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text("# U2 boundary-model summary\n\n" + "\n".join(f"- {x['job_id']} ({x['evaluation_split']}): F1±2={x['boundary_f1_tol2']:.4f}" for x in summaries) + "\n", encoding="utf-8")
    return summaries


def evaluate_offline_teachers(dataset: Path, prediction_root: Path, formal_root: Path, output: Path, per_event: Path, report: Path) -> list[dict[str, Any]]:
    """Report noncausal teacher upper bounds separately from deployable students."""
    rows=[]; details=[]
    for teacher in sorted(formal_root.glob("offline_teacher_to_causal_s*/teacher.pt")):
        seed=teacher.parent.name.rsplit("s",1)[1];job_id=f"offline_teacher_s{seed}"
        for split in ("val","test"):
            summary,event=evaluate_predictions(dataset,prediction_root/job_id/split,job_id,split,2)
            summary.update({"job_id":job_id,"seed":seed,"evaluation_split":split,"causal":False,"offline_noncausal":True,"role":"training_teacher_upper_bound","test_used_for_selection":False});rows.append(summary);details.extend(event)
    write_csv(output,rows,sorted({k for r in rows for k in r}));write_csv(per_event,details,list(details[0]));report.parent.mkdir(parents=True,exist_ok=True);report.write_text("# U2 offline teacher upper bound\n\n"+"\n".join(f"- {r['job_id']} {r['evaluation_split']}: F1±2={r['boundary_f1_tol2']:.4f}" for r in rows)+"\n",encoding="utf-8");return rows
