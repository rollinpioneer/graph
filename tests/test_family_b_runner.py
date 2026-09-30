"""Frozen branch/receipt planning tests; no physical environment."""
from pathlib import Path
from types import SimpleNamespace

import pytest

from cp_disr.analysis import family_b_pilot as p
from cp_disr.facts import Truth

ROOT = Path(__file__).resolve().parents[1]
CFG = p.config(ROOT, "configs/final_master/s4_family_b_staging.yaml")


def test_freeze_has_24_unique_branches_four_technical_and_paired_seeds(tmp_path):
    result = p.freeze(ROOT, CFG, tmp_path)
    assert result["status"] == "FROZEN"
    branches = p._registered(tmp_path)
    assert len(branches) == len({b["branch_id"] for b in branches}) == 24
    assert len(p._registered(tmp_path, "technical")) == 4
    assert len(p._registered(tmp_path, "remaining")) == 20
    by = {}
    for b in branches:
        by.setdefault((b["layout"], b["repeat"]), set()).add(b["restore_seed"])
    assert len(by) == 4 and all(len(v) == 1 for v in by.values())
    assert p.read(tmp_path / "physical/budget_ledger.json")["physical_witness_episodes"]["used"] == 0
    assert p.read(tmp_path / "geometry/candidate_selection.json")["analytic_candidates_examined"] == 2


def test_unreleased_remaining_wave_cannot_dispatch(tmp_path):
    p.freeze(ROOT, CFG, tmp_path)
    with pytest.raises(FileNotFoundError):
        p.run_wave(ROOT, CFG, tmp_path, "remaining")


def test_mask_rejection_happens_before_executor():
    snapshot = SimpleNamespace(candidate_ids=("a:PICK:carrier:v1",), mask=(False,))
    fake = SimpleNamespace(executor=SimpleNamespace(execute=lambda *a: pytest.fail("executed")))
    with pytest.raises(p.BranchStop, match="SCRIPT_DIVERGENCE_MASK"):
        p._action(fake, snapshot, "a:PICK:carrier:v1", "setup", [], 60.0)


def test_geometry_proxy_reverses_and_rejects_alias():
    layouts = p.read(ROOT / CFG["runtime"]["layouts_path"])
    rows = p.geometry_precheck(layouts)
    assert len(rows) == 8
    broken = p.read(ROOT / CFG["runtime"]["layouts_path"])
    broken["layouts"][0]["pad_v_xy"] = broken["layouts"][0]["pad_u_xy"]
    with pytest.raises(ValueError, match="pad alias"):
        p.geometry_precheck(broken)
