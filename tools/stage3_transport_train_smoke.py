#!/usr/bin/env python3
"""Run one formal-size Transport-MH optimizer step without creating a train run."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    started = time.monotonic()
    config_path = args.config.resolve()
    output_path = args.output.resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cfg = OmegaConf.load(config_path)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one CUDA-visible device, got {torch.cuda.device_count()}"
        )

    torch.manual_seed(int(cfg.training.seed))
    torch.cuda.manual_seed_all(int(cfg.training.seed))
    device = torch.device(str(cfg.training.device))
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)

    dataset_started = time.monotonic()
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    dataloader = DataLoader(dataset, **cfg.dataloader)
    batch = next(iter(dataloader))
    dataset_seconds = time.monotonic() - dataset_started

    model = hydra.utils.instantiate(cfg.policy)
    model.set_normalizer(dataset.get_normalizer())
    optimizer = hydra.utils.instantiate(cfg.optimizer, params=model.parameters())
    model.to(device)
    batch = dict_apply(batch, lambda value: value.to(device, non_blocking=True))

    step_started = time.monotonic()
    optimizer.zero_grad(set_to_none=True)
    loss = model.compute_loss(batch)
    if not torch.isfinite(loss):
        raise RuntimeError(f"non-finite loss before backward: {loss.item()}")
    loss.backward()

    grad_tensors = 0
    grad_values = 0
    grad_norm_sq = 0.0
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        grad_tensors += 1
        grad_values += parameter.grad.numel()
        if not torch.isfinite(parameter.grad).all():
            raise RuntimeError("non-finite gradient detected")
        grad_norm_sq += float(parameter.grad.detach().float().norm().item()) ** 2
    if grad_tensors == 0:
        raise RuntimeError("no gradients were produced")

    optimizer.step()
    torch.cuda.synchronize(device)
    step_seconds = time.monotonic() - step_started
    parameters_finite = all(
        bool(torch.isfinite(parameter).all()) for parameter in model.parameters()
    )
    if not parameters_finite:
        raise RuntimeError("optimizer step produced non-finite parameters")

    dataset_path = Path(str(cfg.task.dataset.dataset_path)).resolve()
    result = {
        "status": "PASS",
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "visible_cuda_device": 0,
        "gpu_name": torch.cuda.get_device_name(device),
        "gpu_total_memory_mib": round(
            torch.cuda.get_device_properties(device).total_memory / 1024**2
        ),
        "peak_allocated_memory_mib": round(torch.cuda.max_memory_allocated(device) / 1024**2, 2),
        "peak_reserved_memory_mib": round(torch.cuda.max_memory_reserved(device) / 1024**2, 2),
        "batch_size": int(cfg.dataloader.batch_size),
        "batch_obs_shape": list(batch["obs"].shape),
        "batch_action_shape": list(batch["action"].shape),
        "train_samples": len(dataset),
        "train_batches_per_epoch": len(dataloader),
        "model_parameters_after_normalizer": sum(
            parameter.numel() for parameter in model.parameters()
        ),
        "trainable_model_parameters": sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        ),
        "loss": float(loss.detach().cpu()),
        "gradient_tensor_count": grad_tensors,
        "gradient_value_count": grad_values,
        "gradient_l2_norm": math.sqrt(grad_norm_sq),
        "parameters_finite_after_step": parameters_finite,
        "dataset_load_seconds": round(dataset_seconds, 3),
        "optimizer_step_seconds": round(step_seconds, 3),
        "wall_time_seconds": round(time.monotonic() - started, 3),
        "config_path": str(config_path),
        "config_sha256": sha256(config_path),
        "dataset_path": str(dataset_path),
        "dataset_sha256": sha256(dataset_path),
    }
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
