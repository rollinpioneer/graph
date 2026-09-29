"""G01-G09: the three frozen configurations are safe, evidence-backed and rank the two orders the right way."""
from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from cp_disr.analysis import s4_family_a_soft_ordering_mvp as m

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/final_master/s4_family_a_soft_ordering_mvp.yaml"
CFG = m.load_config(CONFIG)


def _reference_evidence_present(root):
    """True only where the frozen S1 and V2 pilot evidence trees are checked out (the real worktree)."""
    s1 = root / CFG["sources"]["s1_recovery"]
    pilot = root / CFG["sources"]["pilot_output"] / "captures"
    split = root / CFG["sources"]["s1_split"]
    return s1.is_dir() and pilot.is_dir() and split.is_file()


def _locate(rel):
    """Locate a repo-relative source file; tolerates the offline harness, which mirrors the package without `src/`."""
    bare = rel[4:] if rel.startswith("src/") else rel
    roots = [ROOT, *(p for p in Path(m.__file__).resolve().parents)]
    for r in roots:
        for cand in (r / rel, r / bare, r / "src" / bare):
            if cand.is_file():
                return cand
    return None


HAVE_EVIDENCE = _reference_evidence_present(ROOT)
needs_evidence = pytest.mark.skipif(not HAVE_EVIDENCE, reason="frozen S1/V2 reference evidence is not checked out")


@pytest.fixture(scope="module")
def out():
    d = Path(tempfile.mkdtemp())
    (d / "geometry").mkdir(parents=True, exist_ok=True)
    (d / "stage_manifest.json").write_text('{"phases_done": []}', encoding="utf-8")
    return d


@pytest.fixture(scope="module")
def bank(out):
    if not HAVE_EVIDENCE:
        pytest.skip("frozen S1/V2 reference evidence is not checked out")
    m.derive_anchor_bank(ROOT, CONFIG, out)
    return m.rd(out / "geometry/safe_anchor_bank.json")


@pytest.fixture(scope="module")
def frozen(out, bank):
    m.freeze_configs(ROOT, CONFIG, out)
    return m.rd(out / "geometry/frozen_configs.json")


@needs_evidence
def test_G01_anchor_bank_only_uses_saved_successful_evidence(bank):
    assert bank["status"] == "SAFE"
    assert bank["environment_constructions"] == 0
    assert bank["reference_geometry"]["derivation"] == "saved_evidence_only"
    assert bank["reference_geometry"]["environment_constructions"] == 0
    assert bank["anchors"]
    for a in bank["anchors"]:
        assert a["pick_success_observed"] is True
        assert a["place_success_observed"] is True or a["same_geometry_clause_used"] is True
        assert a["engineering_exception"] is False
        assert a["source_hash"]
        assert a["source_case"] and a["source_branch"]
    src = bank["reference_geometry"]["s1_source"]
    assert src["task_success"] is True
    assert bank["reference_geometry"]["v2_source"]["pick_place_observed"] is True


@needs_evidence
def test_G02_target_first_delta_proxy_at_least_threshold(bank):
    d = bank["reference_geometry"]["delta_proxy_target_first_layout_m"]
    assert d >= float(CFG["geometry"]["target_first_min_delta_proxy_m"])
    assert d == pytest.approx(float(CFG["geometry"]["expected_delta_proxy_m"]), abs=1e-3)


@needs_evidence
def test_G03_second_first_delta_proxy_at_most_negative_threshold(frozen):
    d = frozen["configs"]["SO_SECOND_FIRST"]["delta_proxy_m"]
    assert d <= float(CFG["geometry"]["second_first_max_delta_proxy_m"])
    assert frozen["checks"]["second_first_ok"] is True


@needs_evidence
def test_G04_neutral_abs_delta_proxy_small(bank, frozen):
    d = frozen["configs"]["SO_NEUTRAL"]["delta_proxy_m"]
    assert abs(d) <= float(CFG["geometry"]["neutral_abs_delta_proxy_m"])
    assert frozen["checks"]["neutral_ok"] is True
    assert abs(bank["reference_geometry"]["delta_proxy_neutral_layout_m"] - d) < 1e-12


@needs_evidence
def test_G05_target_and_second_configs_are_exact_identity_swap(frozen):
    cfg = frozen["configs"]
    tf, sf = cfg["SO_TARGET_FIRST"], cfg["SO_SECOND_FIRST"]
    assert tf["target_xy"] == sf["second_xy"]
    assert tf["second_xy"] == sf["target_xy"]
    assert tf["container_xy"] == sf["container_xy"]
    assert tf["buffer_xy"] == sf["buffer_xy"]
    assert sf["identity_swap_of"] == "SO_TARGET_FIRST"
    assert abs(tf["delta_proxy_m"] + sf["delta_proxy_m"]) <= 1e-12
    assert frozen["checks"]["exact_identity_swap"] is True


@needs_evidence
def test_G06_no_initial_overlap(frozen):
    sep = float(CFG["geometry"]["min_object_separation_m"])
    for cid, c in frozen["configs"].items():
        assert m.dist2(c["target_xy"], c["second_xy"]) >= sep, cid


@needs_evidence
def test_G07_all_configs_inside_validated_workspace(frozen, bank):
    half = float(CFG["geometry"]["workspace_half_extent_m"])
    for cid, c in frozen["configs"].items():
        for key in ("target_xy", "second_xy"):
            assert abs(c[key][0]) <= half and abs(c[key][1]) <= half, (cid, key)
        assert m.dist2(c["target_xy"], c["container_xy"]) >= float(CFG["geometry"]["min_object_separation_m"])
        assert m.dist2(c["second_xy"], c["container_xy"]) >= float(CFG["geometry"]["min_object_separation_m"])
    assert all(a["workspace_valid"] for a in bank["anchors"])
    assert all(a["container_clearance"] for a in bank["anchors"])


@needs_evidence
def test_G08_frozen_configs_are_read_only(out, frozen):
    p = out / "geometry/frozen_configs.json"
    assert p.is_file()
    assert (p.stat().st_mode & 0o777) == 0o444
    assert frozen["frozen"] is True
    assert frozen["environment_constructions"] == 0
    assert frozen["derivation"] == "saved_evidence_only"


def test_G09_proxy_not_imported_by_runtime_planner_or_policy():
    rels = ["src/cp_disr/platforms/libero/tp_so_mvp_runtime.py",
            "src/cp_disr/baselines/b_plan.py",
            "src/cp_disr/platforms/libero/skill_executor.py"]
    checked = []
    for rel in rels:
        p = _locate(rel)
        if p is None:
            continue
        checked.append(rel)
        text = p.read_text(encoding="utf-8")
        for token in ("delta_proxy", "path_proxy"):
            assert token not in text, f"{rel} references {token}"
    # the runtime is our own new file; it must always be locatable
    assert "src/cp_disr/platforms/libero/tp_so_mvp_runtime.py" in checked
    # the proxy exists only in the offline analysis module (this one)
    assert "delta_proxy" in Path(m.__file__).read_text(encoding="utf-8")