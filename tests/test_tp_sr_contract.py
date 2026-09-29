from pathlib import Path

import pytest

from cp_disr.platforms.libero import tp_sr_runtime as rt
from cp_disr.analysis import s4_tp_sr_design as d

ROOT = Path(__file__).resolve().parents[1]
TIMEOUTS = {"PICK": 9.0, "PLACE": 8.0, "PLACE_BUFFER": 8.0}


def contracts():
    return rt.ground_task_contracts(ROOT / "configs/runtime/tp_sr_contract_registry.yaml", TIMEOUTS)


def test_C01_exactly_four_allowed_actions():
    assert sorted(c.id for c in contracts()) == sorted(rt.ALLOWED_ACTION_IDS)


def test_C02_open_not_registered():
    assert not any(":OPEN:" in c.id for c in contracts())
    text = (ROOT / "configs/runtime/tp_sr_contract_registry.yaml").read_text()
    assert "name: OPEN" not in text


def test_C03_both_nominal_routes_reachable():
    r = d.contract_reachability(ROOT)
    assert r["direct_reachable"] and r["relocation_reachable"]
    assert list(d.DIRECT_SEQ) in r["routes"] and list(d.RELOC_SEQ) in r["routes"]


def test_C04_excluded_groundings_absent():
    ids = {c.id for c in contracts()}
    for x in rt.EXCLUDED_GROUNDINGS:
        assert x not in ids


def test_C05_template_propositions_equal_verifier_facts():
    t = rt.build_task_template(contracts())
    assert {n.id for n in t.nodes if n.kind == "PROPOSITION"} == set(rt.VERIFIER_FACT_IDS)


def test_C06_no_hard_geometry_facts_and_pick_place_buffer_is_contract_redundant():
    t = rt.build_task_template(contracts())
    props = {n.id for n in t.nodes if n.kind == "PROPOSITION"}
    assert not any(h in p for p in props for h in rt.FORBIDDEN_HARD_FACTS)
    pb = next(c for c in contracts() if c.id == "a:PLACE_BUFFER:interferer:buffer:v1")
    assert "p:Held:interferer" in {a.id for a in pb.pre_pos}   # PICK(i)->PLACE_BUFFER(i) is already a contract edge


def test_C07_nearest_palette_mask_removes_interferer_crosstalk():
    import numpy as np
    from cp_disr.platforms.libero.d0_env import COLORS
    from cp_disr.platforms.libero import perception as P
    img = np.zeros((4, 4, 3), dtype=np.float32) + 0.5
    img[0, 0] = 0.75 * np.asarray(COLORS["interferer"][:3])  # shaded cyan face
    tol = P.THRESHOLDS["color_tol"]
    # production mask lets the cyan pixel leak into the container/buffer masks (this is the defect)
    assert P._mask(img, COLORS["container"], tol)[0, 0] or P._mask(img, COLORS["buffer"], tol)[0, 0]
    for name in ("container", "buffer", "target", "lid"):
        assert not rt._nearest_palette_mask(img, COLORS[name], tol)[0, 0]
    assert rt._nearest_palette_mask(img, COLORS["interferer"], tol)[0, 0]
