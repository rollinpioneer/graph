"""Single-job execution. Release checks run before any model construction."""
from __future__ import annotations
import os
from pathlib import Path
from .backends import close_model, get_backend
from .checkpointing import save_milestone, state_dict_digest
from .constants_b import FORMAL_STEPS, GRADIENT_UPDATES_SEMANTICS, MILESTONES, SMOKE_STEPS
from .errors import ProtocolViolation
from .io_utils import sha256_file, write_new
from .release import ReleaseRejected, assert_no_test_payload, validate_release
from .training_shape import shape_for_job
from . import rng_state

def _job_dir(campaign_root, job):
    kind = "smoke" if job.get("kind") == "smoke" else "training"
    return Path(campaign_root) / kind / job["job_id"]

def _write_identity(job_dir, job, release, backend_name, shape):
    write_new(job_dir / "run_identity.json", {
        "schema": "P2CRL_RUN_IDENTITY_V1",
        "job_id": job["job_id"],
        "kind": job["kind"],
        "method": job.get("method"),
        "draw": job.get("draw"),
        "policy_seed": job.get("policy_seed"),
        "run_seed": job.get("run_seed"),
        "backend": backend_name,
        "training_shape": shape.as_dict(),
    })
    write_new(job_dir / "release_binding.json", {
        "schema": "P2CRL_RELEASE_BINDING_V1",
        "campaign_id": release.get("campaign_id"),
        "attempt_id": release.get("attempt_id"),
        "status": release.get("status"),
        "runner_commit": release.get("runner_commit") or release.get("runner_v2_commit"),
        "protocol_sha256": release.get("protocol_sha256"),
        "plan_sha256": release.get("plan_sha256") or release.get("job_plan_sha256"),
        "dataset_manifest_sha256": release.get("dataset_manifest_sha256"),
        "source_lock_sha256": release.get("source_lock_sha256"),
    })

def execute_job(
    *,
    repo=None,
    data_root=None,
    release,
    plan_job,
    protocol_path=None,
    source_lock_path=None,
    dataset_manifest_path=None,
    amendment_path=None,
    campaign_root,
    ledger=None,
    backend="fake",
    register_only=False,
    pid=None,
    tombstone_registry_path=None,
):
    """Run one smoke/formal job. Fake/register-only never calls learn()."""
    validate_release(
        release,
        repo=repo,
        protocol_path=protocol_path,
        plan_path=None,
        source_lock_path=source_lock_path,
        dataset_manifest_path=dataset_manifest_path,
        amendment_path=amendment_path,
        require_active=True,
        require_clean=repo is not None,
        require_detached=False,
        tombstone_registry_path=tombstone_registry_path,
    )
    if backend != "fake" and register_only:
        raise ReleaseRejected("FAKE_BACKEND_REQUIRED", "zero-gradient drill")
    if backend != "fake" and backend != "real":
        raise ValueError(backend)
    job = dict(plan_job)
    shape = shape_for_job(job)
    job_dir = _job_dir(campaign_root, job)
    job_dir.mkdir(parents=True, exist_ok=True)
    if ledger is not None:
        ledger.claim(job["job_id"], directory=job_dir, pid=pid or os.getpid())
        ledger.mark_running(job["job_id"], directory=str(job_dir), pid=pid or os.getpid())
    _write_identity(job_dir, job, release, backend, shape)
    be = get_backend(backend)
    if register_only or backend == "fake":
        model = be.construct(
            release, [], job.get("method"), int(job.get("run_seed") or 0),
            shape.n_envs, shape.n_steps, shape.batch_size, shape.n_epochs, shape=shape,
        )
        write_new(job_dir / "initialization.json", {
            "schema": "P2CRL_INIT_V1",
            "num_timesteps": 0,
            "learn_called": False,
            "gradient_updates": 0,
            "optimizer_step_count": 0,
            "backend": backend,
            "n_envs": shape.n_envs,
            "n_steps": shape.n_steps,
            "rollout_quantum": shape.rollout_quantum,
            "policy_sha256": state_dict_digest(model),
        })
        rec = {
            "schema": "P2CRL_JOB_COMPLETE_V1",
            "status": "REGISTERED_ZERO_GRADIENT",
            "job_id": job["job_id"],
            "environment_steps": 0,
            "learn_called": bool(getattr(model, "learn_called", False) or be.learn_called),
            "gradient_updates": 0,
            "gradient_updates_semantics": GRADIENT_UPDATES_SEMANTICS,
            "optimizer_step_count": 0,
            "invalid_selected": 0,
            "nonfinite": 0,
            "backend": backend,
            "training_shape": shape.as_dict(),
        }
        write_new(job_dir / "registered.json", rec)
        write_new(job_dir / "complete.json", rec)
        write_new(job_dir / "manifest.json", {
            "schema": "P2CRL_JOB_MANIFEST_V1",
            "job_id": job["job_id"],
            "files": sorted(p.name for p in job_dir.iterdir() if p.is_file()),
        })
        if ledger is not None:
            ledger.mark_complete(job["job_id"], environment_steps=0, gradient_updates=0, directory=str(job_dir))
        return job_dir

    if data_root is None:
        raise ProtocolViolation("data_root required for real training")
    from .data_access import load_job_contracts
    contracts = load_job_contracts(data_root, job)
    for c in contracts:
        assert_no_test_payload(c)
    from .learner import seed_all
    seed_all(int(job.get("run_seed") or 0))
    limit = SMOKE_STEPS if job.get("kind") == "smoke" else FORMAL_STEPS
    shape.validate_target(limit)
    model = None
    try:
        model = be.construct(
            release, contracts, job["method"], int(job.get("run_seed") or 0),
            shape.n_envs, shape.n_steps, shape.batch_size, shape.n_epochs, shape=shape,
        )
        write_new(job_dir / "initialization.json", {
            "schema": "P2CRL_INIT_V1",
            "num_timesteps": 0,
            "learn_called": False,
            "gradient_updates": 0,
            "optimizer_step_count": 0,
            "backend": backend,
            "n_envs": int(model.n_envs),
            "n_steps": int(model.n_steps),
            "rollout_quantum": shape.rollout_quantum,
            "policy_sha256": state_dict_digest(model),
        })
        ckpt_dir = job_dir / "checkpoints"
        index = []
        milestones = [0, limit] if job.get("kind") == "smoke" else list(MILESTONES)
        index.append(save_milestone(model, ckpt_dir, 0, post_update=False))
        from .validation_worker import run_isolated_validation
        for target in milestones[1:]:
            delta = int(target) - int(model.num_timesteps)
            if delta > 0:
                be.train_segment(model, delta, expected_after=int(target), shape=shape)
            index.append(save_milestone(model, ckpt_dir, int(target), post_update=True))
            before = rng_state.capture(model, extra={"sampler": "parent"})
            run_isolated_validation(
                checkpoint=ckpt_dir / f"policy_{int(target)}.zip",
                data_root=data_root,
                out_dir=Path(campaign_root) / "validation" / job["job_id"] / str(int(target)),
                method=job["method"],
                backend=backend,
            )
            after = rng_state.capture(model, extra={"sampler": "parent"})
            if before != after:
                raise ProtocolViolation("validation mutated trainer state")
        if int(model.num_timesteps) != int(limit):
            raise ProtocolViolation("job timestep mismatch")
        opt = int(getattr(model, "_p2crl_optimizer_step_count", 0) or 0)
        expected_opt = shape.expected_optimizer_steps(limit)
        if opt != expected_opt:
            raise ProtocolViolation(f"optimizer_step_count={opt} expected={expected_opt}")
        rec = {
            "schema": "P2CRL_SMOKE_COMPLETE_V2" if job.get("kind") == "smoke" else "P2CRL_JOB_COMPLETE_V1",
            "status": "TRAINING_COMPLETE",
            "job_id": job["job_id"],
            "environment_steps": int(model.num_timesteps),
            "learn_called": True,
            "rollout_quantum": shape.rollout_quantum,
            "rollout_iterations": int(getattr(model, "_p2crl_rollout_iteration_count", 0) or 0),
            "ppo_epoch_count": int(getattr(model, "_p2crl_ppo_epoch_count", 0) or 0),
            "optimizer_step_count": opt,
            "gradient_updates": opt,
            "gradient_updates_semantics": GRADIENT_UPDATES_SEMANTICS,
            "invalid_selected": 0,
            "nonfinite": 0,
            "backend": backend,
            "final_checkpoint_actual_step": int(model.num_timesteps),
            "training_shape": shape.as_dict(),
            "checkpoints": index,
        }
        write_new(job_dir / "complete.json", rec)
        write_new(job_dir / "checkpoint_index.json", {"schema": "P2CRL_CKPT_INDEX_V1", "items": index})
        write_new(job_dir / "manifest.json", {
            "schema": "P2CRL_JOB_MANIFEST_V1",
            "job_id": job["job_id"],
            "complete_sha256": sha256_file(job_dir / "complete.json"),
        })
        if ledger is not None:
            ledger.mark_complete(
                job["job_id"],
                environment_steps=int(model.num_timesteps),
                gradient_updates=opt,
                directory=str(job_dir),
            )
        return job_dir
    finally:
        if model is not None:
            close_model(model)