"""Small handoff helpers kept separate from evaluation for audit consumers."""
from __future__ import annotations
from .io import read_json, write_json


def build(root, status, route, output):
    result = {"schema": "u4b_handoff_manifest_v1", "scientific_status": status, "development_route": route, "root": str(root), "u3_history": "U3_INCONCLUSIVE", "scope": "same explicit stochastic simulator family only", "physical_generalization_eligible": False}
    write_json(output, result); return result
