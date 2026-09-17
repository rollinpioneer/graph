"""Release validation. Must run before model or vecenv construction."""
from __future__ import annotations
import subprocess
from pathlib import Path
from .constants_b import (
    CAMPAIGN_ID, DATASET_MANIFEST_SHA256, FORMAL_JOB_LIMIT, FORMAL_STEP_LIMIT,
    KNOWN_DEVIATIONS, PARALLEL_JOBS_MAX, PLAN_SHA256, PREREGISTRATION_COMMIT,
    PROTOCOL_SEMANTIC_SHA256, PROTOCOL_SHA256, SMOKE_JOB_LIMIT, SMOKE_STEPS,
)
from .contracts import load_protocol
from .io_utils import hash_json, load_json, sha256_file


class ReleaseRejected(RuntimeError):
    def __init__(self, code, detail=""):
        super().__init__(f"{code}: {detail}" if detail else code)
        self.code = code
        self.detail = detail


def git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def validate_release(
    release,
    *,
    repo=None,
    protocol_path=None,
    plan_path=None,
    source_lock_path=None,
    dataset_manifest_path=None,
    amendment_path=None,
    require_active=True,
    require_clean=True,
    require_detached=False,
):
    if not release or release.get("schema") != "P2CRL_CAMPAIGN_RELEASE_V1":
        raise ReleaseRejected("RELEASE_SCHEMA")
    if release.get("campaign_id") != CAMPAIGN_ID:
        raise ReleaseRejected("CAMPAIGN_ID")
    if require_active:
        if release.get("status") != "ACTIVE" or release.get("training_release") is not True:
            raise ReleaseRejected("RELEASE_NOT_ACTIVE")
    if release.get("preregistration_commit") != PREREGISTRATION_COMMIT:
        raise ReleaseRejected("PREREGISTRATION_COMMIT")
    if not release.get("runner_commit"):
        raise ReleaseRejected("RUNNER_COMMIT")
    if protocol_path is not None:
        if sha256_file(protocol_path) != PROTOCOL_SHA256 or release.get("protocol_sha256") != PROTOCOL_SHA256:
            raise ReleaseRejected("PROTOCOL_HASH")
        proto = load_protocol(protocol_path)
        if hash_json(proto) != PROTOCOL_SEMANTIC_SHA256 or release.get("protocol_semantic_sha256") != PROTOCOL_SEMANTIC_SHA256:
            raise ReleaseRejected("PROTOCOL_SEMANTIC_HASH")
    if plan_path is not None:
        if sha256_file(plan_path) != PLAN_SHA256 or release.get("plan_sha256") != PLAN_SHA256:
            raise ReleaseRejected("PLAN_HASH")
    if dataset_manifest_path is not None:
        if sha256_file(dataset_manifest_path) != DATASET_MANIFEST_SHA256 or release.get("dataset_manifest_sha256") != DATASET_MANIFEST_SHA256:
            raise ReleaseRejected("DATASET_MANIFEST_HASH")
    if source_lock_path is not None:
        if release.get("source_lock_sha256") != sha256_file(source_lock_path):
            raise ReleaseRejected("SOURCE_LOCK_HASH")
    if release.get("known_deviations") != KNOWN_DEVIATIONS:
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
    if amendment_path is not None:
        amd = load_json(amendment_path)
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
    if repo is not None:
        repo = Path(repo)
        head = git(repo, "rev-parse", "HEAD")
        if head != release.get("runner_commit"):
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
