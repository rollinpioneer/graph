#!/usr/bin/env python3
"""Exercise the formal Transport-MH TRAK stack on one source/target batch."""

from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path

import hydra
import numpy as np
import torch
import trak
from torch.utils.data import DataLoader

from diffusion_policy.common.pytorch_util import dict_apply
from diffusion_policy.common.trak_util import get_parameter_names, get_policy_from_checkpoint


MODELOUT = "diffusion_policy.data_attribution.modelout_functions.DiffusionLowdimFunctionalModelOutput"
GRADIENT_COMPUTER = "diffusion_policy.data_attribution.gradient_computers.DiffusionLowdimFunctionalGradientComputer"


def directory_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def timed(result: dict, name: str, function) -> None:
    started = time.monotonic()
    function()
    torch.cuda.synchronize()
    result[name + "_seconds"] = round(time.monotonic() - started, 3)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--source-samples", type=int, default=4096)
    parser.add_argument("--proj-dim", type=int, default=4000)
    parser.add_argument("--proj-max-batch-size", type=int, default=32)
    parser.add_argument("--num-timesteps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise FileExistsError(args.output_dir)
    args.output_dir.mkdir(parents=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("smoke requires exactly one CUDA-visible device")
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    wall_started = time.monotonic()
    result = {
        "status": "RUNNING",
        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "unset"),
        "gpu_name": torch.cuda.get_device_name(device),
        "checkpoint": str(args.checkpoint.resolve()),
        "batch_size": args.batch_size,
        "requested_source_samples": args.source_samples,
        "proj_dim": args.proj_dim,
        "proj_max_batch_size": args.proj_max_batch_size,
        "num_timesteps": args.num_timesteps,
        "seed": args.seed,
    }

    load_started = time.monotonic()
    policy, cfg = get_policy_from_checkpoint(args.checkpoint, device=device)
    dataset = hydra.utils.instantiate(cfg.task.dataset)
    loader = DataLoader(dataset, batch_size=args.batch_size, num_workers=0, shuffle=False)
    iterator = iter(loader)
    result["load_seconds"] = round(time.monotonic() - load_started, 3)

    if args.source_samples < args.proj_dim:
        raise ValueError("source_samples must be at least proj_dim when lambda_reg=0")
    if args.source_samples % args.batch_size:
        raise ValueError("source_samples must be divisible by batch_size")
    source_count = args.source_samples
    grad_wrt = get_parameter_names(policy, ["model."])
    result["gradient_parameter_count"] = sum(
        dict(policy.named_parameters())[name].numel() for name in grad_wrt
    )
    task = hydra.utils.get_class(MODELOUT)(loss_fn="square")
    gradient_computer = hydra.utils.get_class(GRADIENT_COMPUTER)
    save_dir = args.output_dir / "trak"
    traker = trak.TRAKer(
        model=policy,
        task=task,
        train_set_size=source_count,
        gradient_computer=gradient_computer,
        device=device,
        grad_wrt=grad_wrt,
        proj_dim=args.proj_dim,
        proj_max_batch_size=args.proj_max_batch_size,
        lambda_reg=0.0,
        save_dir=str(save_dir),
        use_half_precision=False,
    )
    traker.load_checkpoint(policy.state_dict(), model_id=0)

    def add_timesteps(batch: dict) -> dict:
        batch = dict(batch)
        batch["timesteps"] = torch.randint(
            cfg.policy.noise_scheduler.num_train_timesteps,
            (batch["action"].shape[0], args.num_timesteps),
        ).long()
        return dict_apply(batch, lambda value: value.to(device))

    def featurize() -> None:
        consumed = 0
        while consumed < source_count:
            batch = next(iterator)
            count = int(batch["action"].shape[0])
            if count != args.batch_size:
                raise RuntimeError("source smoke received a short batch")
            traker.featurize(add_timesteps(batch), num_samples=count)
            consumed += count

    timed(result, "featurize_source", featurize)
    result["featurize_one_batch_seconds"] = round(
        result["featurize_source_seconds"] / (source_count / args.batch_size), 3
    )
    timed(result, "finalize_features", lambda: traker.finalize_features(
        model_ids=[0], hessian_lim=source_count
    ))
    target_batch = next(iterator)
    target_count = int(target_batch["action"].shape[0])
    if target_count != args.batch_size:
        raise RuntimeError("target smoke received a short batch")
    traker.start_scoring_checkpoint(
        exp_name="smoke_targets",
        checkpoint=policy.state_dict(),
        model_id=0,
        num_targets=target_count,
    )
    timed(result, "score_one_batch", lambda: traker.score(
        batch=add_timesteps(target_batch), num_samples=target_count
    ))
    scores_holder: dict[str, np.ndarray] = {}

    def finalize_scores() -> None:
        scores_holder["scores"] = np.asarray(traker.finalize_scores(
            exp_name="smoke_targets", model_ids=[0], allow_skip=False
        ))

    timed(result, "finalize_scores", finalize_scores)
    scores = scores_holder["scores"]
    if scores.shape != (source_count, target_count) or not np.all(np.isfinite(scores)):
        raise RuntimeError(f"invalid smoke score matrix: shape={scores.shape}")

    result.update({
        "status": "PASS",
        "source_samples": source_count,
        "target_samples": target_count,
        "score_shape": list(scores.shape),
        "score_min": float(scores.min()),
        "score_max": float(scores.max()),
        "peak_allocated_memory_mib": round(torch.cuda.max_memory_allocated() / 1024**2, 2),
        "peak_reserved_memory_mib": round(torch.cuda.max_memory_reserved() / 1024**2, 2),
        "output_bytes": directory_bytes(args.output_dir),
        "wall_time_seconds": round(time.monotonic() - wall_started, 3),
    })
    (args.output_dir / "result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(result, indent=2), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
