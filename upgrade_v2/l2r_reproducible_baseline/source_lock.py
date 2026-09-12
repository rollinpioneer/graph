from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from .protocol import canonical_hash, sha256

RUNNER_GLOBS = (
    "upgrade_v2/l2r_reproducible_baseline/*.py",
)
DEPENDENCY_FILES = (
    "upgrade_v2/l2r_task_context/collector.py",
    "upgrade_v2/l2r_task_context/io.py",
    "upgrade_v2/l2r_task_context/repair_collection.py",
    "upgrade_v2/l2r_hold_evidence/probe_adapter.py",
    "upgrade_v2/visual_refine_l2/dynamic_simulator.py",
    "upgrade_v2/visual_refine_l2/repaired_simulator.py",
    "upgrade_v2/visual_refine_l2/renderer.py",
    "upgrade_v2/visual_refine_l2/vision.py",
)


def current_commit(repo: Path) -> str:
    return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], check=True, text=True, stdout=subprocess.PIPE).stdout.strip()


def runner_file_hashes(repo: Path) -> dict[str, str]:
    paths: set[Path] = set()
    for pattern in RUNNER_GLOBS:
        paths.update(repo.glob(pattern))
    paths.update(repo / relative for relative in DEPENDENCY_FILES)
    result = {}
    for path in sorted(paths):
        if not path.is_file():
            raise FileNotFoundError(path)
        result[str(path.relative_to(repo))] = sha256(path)
    return result


def working_tree_clean(repo: Path) -> bool:
    return not bool(
        subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            text=True,
            stdout=subprocess.PIPE,
            check=True,
        ).stdout.strip()
    )


def pre_execution_model_xml(protocol: dict[str, Any]) -> str:
    """Render the frozen XML template without importing MuJoCo or building a model."""
    from upgrade_v2.visual_refine_l2.dynamic_simulator import _xml, family_spec

    spec = family_spec(
        protocol["root_family_id"],
        protocol["scenario"],
        int(protocol["family_seed"]),
        int(protocol["rollout_seed"]),
        probe_variant="default",
    )
    physical = {
        key: getattr(spec, key)
        for key in ("object_radius", "friction", "camera_jitter", "object_x", "object_y", "target_x", "target_y")
    }
    if canonical_hash(physical) != protocol["physical_spec_sha256"]:
        raise ValueError("physical specification changed")
    return _xml(spec)


def make_pre_execution_source_lock(
    repo: Path,
    protocol: dict[str, Any],
    protocol_sha256: str,
    environment_contract: Path,
) -> dict[str, Any]:
    from .environment import ENVIRONMENT_CONTRACT_SCHEMA, REQUIRED_ENVIRONMENT

    environment_value = json.loads(environment_contract.read_text(encoding="utf-8"))
    if (
        environment_value.get("schema") != ENVIRONMENT_CONTRACT_SCHEMA
        or environment_value.get("status") != "FROZEN_ZERO_PHYSICS"
        or environment_value.get("static_contract_version") != protocol["static_contract_version"]
        or environment_value.get("required") != REQUIRED_ENVIRONMENT
    ):
        raise ValueError("environment contract is not the frozen v2 contract")
    xml = pre_execution_model_xml(protocol)
    lock: dict[str, Any] = {
        "schema": "l2rar2_r14b_pre_execution_source_lock_v3",
        "static_contract_version": protocol["static_contract_version"],
        "runner_commit": current_commit(repo),
        "generation_runner_file_hashes": runner_file_hashes(repo),
        "protocol_sha256": protocol_sha256,
        "program_sha256": protocol["program_sha256"],
        "physical_spec_sha256": protocol["physical_spec_sha256"],
        "generated_model_xml_sha256": sha256_bytes(xml.encode("utf-8")),
        "environment_contract_sha256": sha256(environment_contract),
        "family_seed": int(protocol["family_seed"]),
        "git_status_clean": working_tree_clean(repo),
        "source_lock_sha256": None,
    }
    return finalize_source_lock(lock)


def sha256_bytes(value: bytes) -> str:
    import hashlib

    return hashlib.sha256(value).hexdigest()


def finalize_source_lock(lock: dict[str, Any]) -> dict[str, Any]:
    result = dict(lock)
    result["source_lock_sha256"] = canonical_hash({key: result[key] for key in result if key != "source_lock_sha256"})
    return result
