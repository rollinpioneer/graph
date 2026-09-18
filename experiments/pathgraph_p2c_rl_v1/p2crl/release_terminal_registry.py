"""Attempt isolation: tombstoned releases cannot be reused."""
from __future__ import annotations
from pathlib import Path
from .io_utils import load_json, sha256_file
from .release import ReleaseRejected


def load_registry(path):
    rec = load_json(path)
    if rec.get("schema") != "P2CRL_RELEASE_TOMBSTONE_REGISTRY_V1":
        raise ReleaseRejected("TOMBSTONE_SCHEMA")
    return rec


def release_file_sha256(path):
    return sha256_file(path)


def assert_release_not_tombstoned(release_path, registry_path):
    if registry_path is None:
        return True
    registry = load_registry(registry_path)
    sha = release_file_sha256(release_path)
    for entry in registry.get("entries") or []:
        if entry.get("release_sha256") == sha:
            raise ReleaseRejected("RELEASE_TOMBSTONED", sha)
    return True


def assert_sha_not_tombstoned(release_sha256, registry_path):
    if registry_path is None:
        return True
    registry = load_registry(registry_path)
    for entry in registry.get("entries") or []:
        if entry.get("release_sha256") == release_sha256:
            raise ReleaseRejected("RELEASE_TOMBSTONED", release_sha256)
    return True