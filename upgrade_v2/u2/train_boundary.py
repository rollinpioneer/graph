"""GPU training for the three frozen U2 boundary-model variants."""
from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .boundary_model import CausalBoundaryGRU, OfflineBoundaryTeacher
from .dataset import load_episode, read_csv, write_json
from .evaluate import evaluate_predictions, prf
from .weak_labels import DEFAULT_RULES, candidate_scores


def _load_sequences(dataset: Path, weak_root: Path, split: str) -> list[dict[str, Any]]:
    sequences: list[dict[str, Any]] = []
    for row in read_csv(dataset / "episode_manifest.csv"):
        if row["split"] != split: continue
        episode = load_episode(row); p = weak_root / "weak_plus_small_gold_calibration" / f"{row['episode_id']}.npz"
        if p.is_file():
            with np.load(p) as d: posterior = {"boundary": d["boundary_probability"], "events": d["event_probability"], "unknown": d["unknown"]}
        else:
            score, _ = candidate_scores(episode["observations"], episode["actions"], DEFAULT_RULES)
            logits = score * 3.0; logits[:, 0] += 3.0; probs = np.exp(logits - logits.max(1, keepdims=True)); probs /= probs.sum(1, keepdims=True)
            posterior = {"boundary": 1 - probs[:, 0], "events": probs, "unknown": np.zeros(len(probs), dtype=np.int8)}
        sequences.append({"row": row, "obs": episode["observations"], "gold_event": episode["gold_event_id"], "gold_boundary": episode["gold_boundary"], **posterior})
    return sequences


def _batch(samples: list[dict[str, Any]], indices: np.ndarray, device: torch.device) -> dict[str, torch.Tensor]:
    chosen = [samples[int(i)] for i in indices]; length = max(len(x["obs"]) for x in chosen); B = len(chosen)
    obs = np.zeros((B, length, 17), np.float32); boundary = np.zeros((B, length), np.float32); events = np.zeros((B, length, 11), np.float32); unknown = np.zeros((B, length), np.float32); gold_event = np.zeros((B, length), np.int64); gold_boundary = np.zeros((B, length), np.float32); mask = np.zeros((B, length), np.float32)
    for i, item in enumerate(chosen):
        t = len(item["obs"]); obs[i,:t] = item["obs"]; boundary[i,:t] = item["boundary"]; events[i,:t] = item["events"]; unknown[i,:t] = item["unknown"]; gold_event[i,:t] = item["gold_event"]; gold_boundary[i,:t] = item["gold_boundary"]; mask[i,:t] = 1
    return {"obs": torch.from_numpy(obs).to(device), "boundary": torch.from_numpy(boundary).to(device), "events": torch.from_numpy(events).to(device), "unknown": torch.from_numpy(unknown).to(device), "gold_event": torch.from_numpy(gold_event).to(device), "gold_boundary": torch.from_numpy(gold_boundary).to(device), "mask": torch.from_numpy(mask).to(device), "items": chosen}


def _loss(out: dict[str, torch.Tensor], batch: dict[str, torch.Tensor], oracle_clips: dict[str, list[tuple[int, int]]], variant: str) -> torch.Tensor:
    mask = batch["mask"]; boundary_target = batch["boundary"].clone(); event_target = batch["events"].clone()
    if variant.startswith("causal_weak_plus_oracle"):
        for index, item in enumerate(batch["items"]):
            for start, end in oracle_clips.get(item["row"]["episode_id"], []):
                left = max(0, start); right = min(int(mask[index].sum().item()), end + 1)
                if left < right:
                    boundary_target[index, left:right] = batch["gold_boundary"][index, left:right]
                    event_target[index, left:right] = F.one_hot(batch["gold_event"][index, left:right], 11).float()
    bce = F.binary_cross_entropy_with_logits(out["boundary_logit"], boundary_target, reduction="none")
    kl = F.kl_div(F.log_softmax(out["event_logits"], -1), event_target, reduction="none").sum(-1)
    unk = F.binary_cross_entropy_with_logits(out["unknown_logit"], batch["unknown"], reduction="none")
    smooth = (out["embedding"][:,1:] - out["embedding"][:,:-1]).pow(2).mean(-1)
    smooth_mask = (torch.sigmoid(out["boundary_logit"][:,:-1]) < .1).float() * mask[:,1:]
    return ((bce + kl + .15 * unk) * mask).sum() / mask.sum() + .01 * (smooth * smooth_mask).sum() / smooth_mask.sum().clamp_min(1)


@torch.no_grad()
def _validation_metrics(model: torch.nn.Module, sequences: list[dict[str, Any]], device: torch.device) -> dict[str, float]:
    """Small in-memory validation-only selector; never reads test labels."""
    model.eval(); predicted=[]; gold=[]; recovery_pred=[]; recovery_gold=[]
    for item in sequences:
        out=model(torch.from_numpy(item["obs"]).unsqueeze(0).to(device)); bp=torch.sigmoid(out["boundary_logit"])[0].cpu().numpy()>=.5; event=out["event_logits"][0].argmax(-1).cpu().numpy()
        predicted.append(bp);gold.append(item["gold_boundary"].astype(bool));recovery_pred.append(event==4);recovery_gold.append(item["gold_event"]==4)
    boundary=prf(np.concatenate(predicted),np.concatenate(gold),2);recovery=prf(np.concatenate(recovery_pred),np.concatenate(recovery_gold),2)
    model.train();return {"boundary_f1_tol2":boundary["f1"],"boundary_mae":boundary["mae"],"recovery_start_recall":recovery["recall"]}


def train_job(job: dict[str, str]) -> dict[str, Any]:
    if not torch.cuda.is_available(): raise RuntimeError("CUDA is required for U2.3 training")
    device = torch.device("cuda"); seed = int(job["seed"]); torch.manual_seed(seed); np.random.seed(seed)
    dataset = Path(job["dataset"]); weak = Path(job["weak_posteriors"]); outdir = Path(job["output_dir"]); outdir.mkdir(parents=True, exist_ok=True)
    train = _load_sequences(dataset, weak, "train"); val = _load_sequences(dataset, weak, "val"); variant = job["variant"]
    model: torch.nn.Module = CausalBoundaryGRU()
    model.to(device); model.train(); optimizer = torch.optim.AdamW(model.parameters(), lr=.001, weight_decay=.0001)
    oracle_clips: dict[str, list[tuple[int, int]]] = {}
    oracle_file = Path(job["oracle_clips"])
    if oracle_file.is_file():
        for row in csv.DictReader(oracle_file.open()):
            center = int(row.get("center_t", 0)); start = int(row.get("start_t", center - 2)); end = int(row.get("end_t", center + 2))
            oracle_clips.setdefault(row["episode_id"], []).append((start, end))
    rng = np.random.default_rng(seed); losses: list[float] = []; steps = int(job["steps"]); teacher: torch.nn.Module | None = None
    if variant == "offline_teacher_to_causal":
        # Train the explicitly noncausal teacher first, then distil it into the
        # saved causal GRU.  The deployable checkpoint is always the student.
        teacher = OfflineBoundaryTeacher().to(device).train(); teacher_opt = torch.optim.AdamW(teacher.parameters(), lr=.001, weight_decay=.0001)
        for _ in range(max(100, steps // 3)):
            batch = _batch(train, rng.integers(0, len(train), size=min(64, len(train))), device)
            teacher_opt.zero_grad(set_to_none=True); loss = _loss(teacher(batch["obs"]), batch, oracle_clips, "causal_weak_only"); loss.backward(); teacher_opt.step()
        torch.save({"state_dict": teacher.state_dict(), "variant": "offline_teacher", "seed": seed, "architecture": "OfflineBoundaryTeacher", "cuda_used": True}, outdir / "teacher.pt")
        teacher.eval()
    best_score=-float("inf");best_step=0;best_state=None;validation_history=[]
    for step in range(steps):
        batch = _batch(train, rng.integers(0, len(train), size=min(64, len(train))), device)
        optimizer.zero_grad(set_to_none=True); out = model(batch["obs"]); loss = _loss(out, batch, oracle_clips, variant)
        if teacher is not None:
            with torch.no_grad(): target = teacher(batch["obs"])
            loss = loss + .35 * F.kl_div(F.log_softmax(out["event_logits"], -1), F.softmax(target["event_logits"], -1), reduction="batchmean") + .15 * F.mse_loss(torch.sigmoid(out["boundary_logit"]), torch.sigmoid(target["boundary_logit"]))
        if not torch.isfinite(loss): raise RuntimeError("non-finite U2.3 loss")
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0); optimizer.step(); losses.append(float(loss.item()))
        if (step + 1) % 100 == 0 or step + 1 == steps:
            metric=_validation_metrics(model,val,device);metric["step"]=step+1;validation_history.append(metric)
            score=metric["boundary_f1_tol2"]
            if score>best_score:
                best_score=score;best_step=step+1;best_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}
    if best_state is None: raise RuntimeError("no validation checkpoint was produced")
    model.load_state_dict(best_state);checkpoint = outdir / "best.pt"; torch.save({"state_dict": model.state_dict(), "variant": variant, "seed": seed, "steps": steps, "best_step":best_step, "validation_every":100, "architecture": type(model).__name__, "cuda_used": True}, checkpoint)
    # Validation prediction uses the checkpoint-producing model only; no test labels are read.
    predict_model(model, dataset, weak, "val", outdir / "val_predictions")
    metric, _ = evaluate_predictions(dataset, outdir / "val_predictions", variant, "val", 2)
    digest = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    result = {"job_id": job["job_id"], "variant": variant, "seed": seed, "status": "PASS", "cuda_used": True, "device": torch.cuda.get_device_name(device), "steps": steps, "validation_every":100,"validation_history":validation_history,"loss_final": losses[-1], "loss_finite": bool(np.isfinite(losses).all()), "checkpoint": str(checkpoint.resolve()), "checkpoint_sha256": digest, "best_step": best_step, "test_used_for_selection": False, "offline_teacher_checkpoint": str((outdir / "teacher.pt").resolve()) if teacher is not None else "", **metric}
    write_json(outdir / "train_result.json", result); return result


@torch.no_grad()
def predict_model(model: torch.nn.Module, dataset: Path, weak: Path, split: str, output: Path) -> None:
    device = next(model.parameters()).device; output.mkdir(parents=True, exist_ok=True); model.eval()
    for item in _load_sequences(dataset, weak, split):
        x = torch.from_numpy(item["obs"]).unsqueeze(0).to(device); out = model(x); probs = F.softmax(out["event_logits"], -1)[0].cpu().numpy().astype(np.float32); bp = torch.sigmoid(out["boundary_logit"])[0].cpu().numpy().astype(np.float32); unknown = (torch.sigmoid(out["unknown_logit"])[0].cpu().numpy() >= .5).astype(np.int8)
        event = probs.argmax(1).astype(np.int8); event[bp < .5] = 0
        np.savez_compressed(output / f"{item['row']['episode_id']}.npz", boundary_prediction=(bp >= .5).astype(np.int8), boundary_probability=bp, event_prediction=event, event_probability=probs, unknown=unknown, embedding=out["embedding"][0].cpu().numpy().astype(np.float32))
    model.train()


def load_and_predict(checkpoint: Path, dataset: Path, weak: Path, split: str, output: Path) -> None:
    data = torch.load(checkpoint, map_location="cuda")
    model: torch.nn.Module = OfflineBoundaryTeacher() if data.get("architecture") == "OfflineBoundaryTeacher" else CausalBoundaryGRU(); model.load_state_dict(data["state_dict"]); model.cuda(); predict_model(model, dataset, weak, split, output)
