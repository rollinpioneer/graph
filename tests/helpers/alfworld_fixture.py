"""Shared fixtures for ALFWorld tests (skip cleanly when the dataset/catalog is not present)."""
import hashlib
import json
import os

import pytest

RUN_DIR = os.environ.get("ALFWORLD_RUN_DIR", os.path.join(os.path.dirname(__file__), "..", "..", "runs", "alfworld_prior_reliance"))
DATA_ROOT = os.environ.get("ALFWORLD_DATA_ROOT", "/home/xushijie2/xsj2_alf/data")


def require_data():
    pytest.importorskip("textworld")
    pytest.importorskip("alfworld")
    if not (os.path.exists(os.path.join(RUN_DIR, "catalog.json")) and os.path.isdir(DATA_ROOT)):
        pytest.skip("ALFWorld data/catalog not available")


def load_all():
    from cp_disr.platforms.alfworld import data

    rows = data.load_catalog(os.path.join(RUN_DIR, "catalog.json"))
    splits = json.load(open(os.path.join(RUN_DIR, "splits.json")))
    tables = data.Tables(json.load(open(os.path.join(RUN_DIR, "tables.json"))))
    return rows, splits, tables


def gamepath(rel):
    return os.path.join(DATA_ROOT, rel)


class FakePrior:
    """Deterministic synthetic prior (no API): class score from a hash of the class name."""

    def scores(self, pub, tables):
        from cp_disr.platforms.alfworld.prior_provider import instance_scores

        cs = {c: 0.05 + (int(hashlib.sha256((pub.goal_otype + c).encode()).hexdigest(), 16) % 1000) / 1000.0 for c in {r.rsplit(" ", 1)[0] for r in pub.receptacles}}
        return instance_scores(cs, pub.receptacles, pub.feasible, tables.can_contain, pub.goal_otype)
