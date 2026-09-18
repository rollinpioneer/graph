"""Release validation. Must run before model or vecenv construction."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .constants_b import (
    ATTEMPT_02_ID, ATTEMPT_03_ID, CAMPAIGN_ID, DATASET_MANIFEST_SHA256, FORMAL_JOB_LIMIT, FORMAL_STEP_LIMIT,
    KNOWN_DEVIATIONS, PARALLEL_JOBS_MAX, PLAN_SHA256, PREREGISTRATION_COMMIT,
    PROTOCOL_SEMANTIC_SHA256, PROTOCOL_SHA256, SMOKE_JOB_LIMIT, SMOKE_STEPS,
)
from .contracts import load_protocol
from .io_utils import hash_json, load_json, sha256_file
from .training_shape import FORMAL_SHAPE, SMOKE_SHAPE

ALLOWED_RELEASE_SCHEMAS = (
    "P2CRL_CAMPAIGN_RELEASE_V1",
    "P2CRL_CAMPAIGN_RELEASE_V2",
    "P2CRL_CAMPAIGN_RELEASE_V3",
)

FROZEN_RUNTIME_VERSIONS = {
    "stable-baselines3": "2.7.1",
    "sb3-contrib": "2.7.1",
    "gymnasium": "1.2.2",
}


class ReleaseRejected(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _runner_commit(release):
    if release.get("schema") == "P2CRL_CAMPAIGN_RELEASE_V3":
        return release.get("runner_v3_commit")
    if release.get("schema") == "P2CRL_CAMPAIGN_RELEASE_V2":
        return release.get("runner_v2_commit") or release.get("runner_commit")
    return release.get("runner_commit")


def _plan_sha(release):
    return release.get("plan_sha256") or release.get("job_plan_sha256")


def validate_release(
    release,
    *,
    repo=None,
    protocol_path=None,
    plan_path=None,
    source_lock_path=None,
    dataset_manifest_path=None,
    amendment_path=None,
    runtime_lock_path=None,
    require_active=True,
    require_clean=True,
    require_detached=False,
    tombstone_registry_path=None,
    release_path=None,
    terminal_receipt_path=None,
):
    if not release or release.get("schema") not in ALLOWED_RELEASE_SCHEMAS:
        raise ReleaseRejected("RELEASE_SCHEMA")
    schema = release.get("schema")
    if release.get("campaign_id") != CAMPAIGN_ID:
        raise ReleaseRejected("CAMPAIGN_ID")
    if require_active:
        if release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE")
    if release.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ReleaseRejected("PREREGISTRATION_COMMIT")
    runner = _runner_commit(release)
    if not runner:
        raise ReleaseRejected("RUNNER_COMMIT")
    if protocol_path is not None:
        if sha256_file(protocol_path) != PROTOCOL_SHA256 or release.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ReleaseRejected("PROTOCOL_HASH")
        proto = load_protocol(protocol_path)
        if hash_json(proto) != PROTOCOL_SEMANTIC_SHA256 or release.get("protocol_semantic_sha256") != PROTOCOL_SEMANTIC_SHA256:
            raise ReleaseRejected("PROTOCOL_SEMANTIC_HASH")
    if plan_path is not None:
        if sha256_file(plan_path) != PLAN_SHA256 or _plan_sha(release) != PLAN_SHA256:
            raise ReleaseRejected("PLAN_HASH")
    if dataset_manifest_path is not None:
        if sha256_file(dataset_manifest_path) != DATASET_MANIFEST_SHA256 or release.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256:
            raise ReleaseRejected("DATASET_MANIFEST_HASH")
    if source_lock_path is not None:
        if release.get("source_lock_sha256") != sha256_file(source_lock_path):
            raise ReleaseRejected("SOURCE_LOCK_HASH")
    if release.get("known_deviations") is not None and release.get("known_deviations") != KNOWN_DEVIATIONS:
        raise ReleaseRejected("KNOWN_DEVIATIONS")
    if schema == "P2CRL_CAMPAIGN_RELEASE_V1" and release.get("known_deviations") != KNOWN_DEVIATIONS:
        raise ReleaseRejected("KNOWN_DEVIATIONS")
    if int(release.get("formal_job_limit", -1)) != FORMAL_JOB_LIMIT:
        raise ReleaseRejected("FORMAL_JOB_LIMIT")
    if int(release.get("formal_step_limit", -1)) != FORMAL_STEP_LIMIT:
        raise ReleaseRejected("FORMAL_STEP_LIMIT")
    if int(release.get("smoke_job_limit", -1)) != SMOKE_JOB_LIMIT:
        raise ReleaseRejected("SMOKE_JOB_LIMIT")
    if int(release.get("smoke_steps_each", -1)) != SMOKE_STEPS:
        raise ReleaseRejected("SMOKE_STEPS")
    if int(release.get("parallel_jobs_max", -1)) not in (1, PARALLEL_JOBS_MAX):
        raise ReleaseRejected("PARALLEL_JOBS")
    if not release.get("explicit_user_execution_instruction_ref"):
        raise ReleaseRejected("EXPLICIT_INSTRUCTION")
    if schema in ("P2CRL_CAMPAIGN_RELEASE_V2", "P2CRL_CAMPAIGN_RELEASE_V3"):
        expected_attempt = ATTEMPT_02_ID if schema == "P2CRL_CAMPAIGN_RELEASE_V2" else ATTEMPT_03_ID
        if release.get("attempt_id") != expected_attempt:
            raise ReleaseRejected("ATTEMPT_ID")
        smoke = release.get("smoke_shape") or {}
        formal = release.get("formal_shape") or {}
        if (int(smoke.get("n_envs", -1)), int(smoke.get("n_steps", -1)), int(smoke.get("batch_size", -1)), int(smoke.get("n_epochs", -1))) != (
            SMOKE_SHAPE.n_envs, SMOKE_SHAPE.n_steps, SMOKE_SHAPE.batch_size, SMOKE_SHAPE.n_epochs
        ):
            raise ReleaseRejected("SMOKE_SHAPE")
        if (int(formal.get("n_envs", -1)), int(formal.get("n_steps", -1)), int(formal.get("batch_size", -1)), int(formal.get("n_epochs", -1))) != (
            FORMAL_SHAPE.n_envs, FORMAL_SHAPE.n_steps, FORMAL_SHAPE.batch_size, FORMAL_SHAPE.n_epochs
        ):
            raise ReleaseRejected("FORMAL_SHAPE")
        root = str(release.get("attempt_root") or "")
        if root.endswith("/campaign_v1") or root.endswith("\\campaign_v1"):
            raise ReleaseRejected("OLD_ATTEMPT_ROOT")
        if _plan_sha(release) not in (None, PLAN_SHA256) and _plan_sha(release) != PLAN_SHA256:
            raise ReleaseRejected("PLAN_HASH")
    if schema == "P2CRL_CAMPAIGN_RELEASE_V3":
        if runtime_lock_path is None:
            raise ReleaseRejected("RUNTIME_LOCK_REQUIRED")
        if release.get("runtime_lock_sha256") != sha256_file(runtime_lock_path):
            raise ReleaseRejected("RUNTIME_LOCK_HASH")
        runtime = load_json(runtime_lock_path)
        if runtime.get("status") != "FROZEN_PREFLIGHT_PASSED":
            raise ReleaseRejected("RUNTIME_PREFLIGHT")
        if runtime.get("learn_called") is not False or int(runtime.get("optimizer_steps", -1)) != 0:
            raise ReleaseRejected("RUNTIME_PREFLIGHT_GRADIENT")
        versions = {item.get("name"): item.get("version") for item in runtime.get("packages", [])}
        if any(versions.get(name) != version for name, version in FROZEN_RUNTIME_VERSIONS.items()):
            raise ReleaseRejected("RUNTIME_VERSION")
    if amendment_path is not None:
        amd = load_json(amendment_path)
        if schema == "P2CRL_CAMPAIGN_RELEASE_V3":
            if amd.get("category") != "NON_SCIENTIFIC_RUNTIME_ENVIRONMENT_AND_ATTEMPT_03_ENABLEMENT":
                raise ReleaseRejected("AMENDMENT_CATEGORY")
            for key in (
                "scientific_protocol_changed", "job_plan_changed", "dataset_changed", "methods_changed",
                "reward_changed", "mask_changed", "formal_ppo_changed", "statistics_changed",
                "training_logic_changed",
            ):
                if amd.get(key) is not False:
                    raise ReleaseRejected("AMENDMENT_SCIENCE", key)
            if amd.get("runner_v3_commit") != runner:
                raise ReleaseRejected("AMENDMENT_RUNNER_COMMIT")
            if amd.get("runtime_lock_sha256") != release.get("runtime_lock_sha256"):
                raise ReleaseRejected("AMENDMENT_RUNTIME_LOCK")
            if release.get("runtime_amendment_sha256") != sha256_file(amendment_path):
                raise ReleaseRejected("AMENDMENT_HASH")
        elif schema == "P2CRL_CAMPAIGN_RELEASE_V2":
            if amd.get("category") != "NON_SCIENTIFIC_SMOKE_ROLLOUT_QUANTUM_AND_ACCOUNTING_FIX":
                raise ReleaseRejected("AMENDMENT_CATEGORY")
            for key in (
                "scientific_protocol_changed", "job_plan_changed", "dataset_changed", "methods_changed",
                "reward_changed", "mask_changed", "formal_ppo_changed", "statistics_changed",
            ):
                if amd.get(key) is not False:
                    raise ReleaseRejected("AMENDMENT_SCIENCE", key)
            if amd.get("runner_v2_commit") not in (None, runner) and amd.get("runner_v2_commit") != runner:
                raise ReleaseRejected("AMENDMENT_RUNNER_COMMIT")
            expected_amd = release.get("runner_v2_amendment_sha256") or release.get("amendment_sha256")
            if expected_amd != sha256_file(amendment_path):
                raise ReleaseRejected("AMENDMENT_HASH")
        else:
            if amd.get("category") != "NON_SCIENTIFIC_EXECUTION_RUNNER_COMPLETION":
                raise ReleaseRejected("AMENDMENT_CATEGORY")
            for key in ("scientific_protocol_changed", "dataset_changed", "methods_changed",
                        "reward_changed", "mask_changed", "statistics_changed"):
                if amd.get(key) is not False:
                    raise ReleaseRejected("AMENDMENT_SCIENCE", key)
            if amd.get("only_execution_runner_completed") is not True:
                raise ReleaseRejected("AMENDMENT_SCOPE")
            if amd.get("runner_commit") != release.get("runner_commit"):
                raise ReleaseRejected("AMENDMENT_RUNNER_COMMIT")
            if release.get("amendment_sha256") != sha256_file(amendment_path):
                raise ReleaseRejected("AMENDMENT_HASH")
    if tombstone_registry_path is not None:
        from .release_terminal_registry import assert_release_not_tombstoned, assert_sha_not_tombstoned
        if release_path is not None:
            assert_release_not_tombstoned(release_path, tombstone_registry_path)
        if terminal_receipt_path is not None:
            rec = load_json(terminal_receipt_path)
            if rec.get("status") != "TERMINAL" or rec.get("reusable") is not False:
                raise ReleaseRejected("TERMINAL_RECEIPT")
            if release.get("prior_terminal_receipt_sha256") not in (None, sha256_file(terminal_receipt_path)):
                if release.get("prior_terminal_receipt_sha256") != sha256_file(terminal_receipt_path):
                    raise ReleaseRejected("TERMINAL_RECEIPT_HASH")
    if repo is not None:
        repo = Path(repo)
        head = git(repo, "rev-parse", "HEAD")
        if head != runner:
            raise ReleaseRejected("RUNNER_HEAD_MISMATCH", head)
        if require_clean and git(repo, "status", "--porcelain"):
            raise ReleaseRejected("WORKTREE_DIRTY")
        if require_detached:
            try:
                git(repo, "symbolic-ref", "-q", "HEAD")
                raise ReleaseRejected("NOT_DETACHED")
            except subprocess.CalledProcessError:
                pass
    return True


def assert_no_test_payload(family_or_contract):
    fid = None
    split = None
    if isinstance(family_or_contract, dict):
        fid = family_or_contract.get("family_id")
        split = family_or_contract.get("split")
    else:
        fid = getattr(family_or_contract, "family_id", None)
        split = getattr(family_or_contract, "split", None)
    if split == "test" or (fid is not None and 1325000 <= int(fid) < 1326000):
        raise ReleaseRejected("TEST_CONTAMINATION_PROTOCOL_VIOLATION", str(fid))
