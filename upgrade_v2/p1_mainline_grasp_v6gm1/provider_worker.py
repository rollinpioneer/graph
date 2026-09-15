"""Probe third-party pretrained grasp providers. Never fakes weights or outputs."""
from __future__ import annotations
import json, os, shutil, subprocess, sys
from pathlib import Path

def probe() -> dict:
    status = {
        "provider_status": "NOT_ATTEMPTED",
        "model_id": None,
        "code_commit": None,
        "checkpoint_sha256": None,
        "license_status": None,
        "official_demo_evidence": None,
        "python": sys.executable,
        "notes": [],
    }
    any_paths = list(Path("/home").glob("**/anygrasp_sdk"))[:3] if False else []
    # bounded search
    candidates = []
    for p in [
        Path("/home/xushijie/anygrasp_sdk"),
        Path("/home/xushijie/contact_graspnet"),
        Path("/home/__compress_data/xushijie/contact_graspnet"),
        Path("/home/__compress_data/xushijie/anygrasp_sdk"),
        Path("/opt/contact_graspnet"),
    ]:
        if p.exists():
            candidates.append(str(p))
    status["found_paths"] = candidates
    try:
        import importlib
        importlib.import_module("contact_graspnet")
        status["notes"].append("contact_graspnet importable in current interpreter")
    except Exception as e:
        status["notes"].append(f"contact_graspnet_import:{type(e).__name__}")
    try:
        import importlib
        importlib.import_module("gsnet")
        status["notes"].append("gsnet/anygrasp importable")
        status["license_status"] = "UNKNOWN_NEEDS_MACHINE_LICENSE"
    except Exception as e:
        status["notes"].append(f"anygrasp_import:{type(e).__name__}")
        if status["license_status"] is None:
            status["license_status"] = "NOT_FOUND_NO_LICENSE_CHECK_PERFORMED"
    if not candidates:
        status["provider_status"] = "BLOCKED_DEPENDENCY"
        status["model_id"] = "NVlabs/contact_graspnet"
        status["official_demo_evidence"] = "NOT_RUN_NO_CHECKPOINTS_OR_REPO"
    else:
        status["provider_status"] = "REPO_PRESENT_DEMO_NOT_RUN"
        status["found_paths"] = candidates
    return status
