"""CP-DISR-TB-STRUCT-GEN-V1: frozen outcome rules."""
from cp_disr import struct_gen_analysis as A

CELLS = A.TEST_CELLS


def mk(T, cells=None, seeds=None):
    cell_rate = {f: {c: (cells or {}).get(f, {}).get(c, T[f]) for c in CELLS} for f in A.FAMILIES}
    per_seed = {f: {s: (seeds or {}).get(f, {}).get(s, T[f]) for s in A.SEEDS} for f in A.FAMILIES}
    return A.decide(T, cell_rate, per_seed)


def test_outcome_a_strongly_strengthened():
    r = mk({"B2": 0.9, "ABS": 0.8, "+E": 0.3, "NC": 0.2})
    assert r["outcome"] == "OUTCOME_A" and r["effect_on_paper"] == "STRONGLY_STRENGTHENED"


def test_outcome_a_strengthened_when_delta_is_moderate():
    r = mk({"B2": 0.7, "ABS": 0.6, "+E": 0.4, "NC": 0.3})
    assert r["outcome"] == "OUTCOME_A" and r["effect_on_paper"] == "STRENGTHENED"


def test_outcome_b_when_everything_is_low_or_equal():
    assert mk({"B2": 0.2, "ABS": 0.15, "+E": 0.1, "NC": 0.1})["outcome"] == "OUTCOME_B"
    assert mk({"B2": 0.55, "ABS": 0.52, "+E": 0.5, "NC": 0.47})["outcome"] in ("OUTCOME_B", "OUTCOME_D")


def test_outcome_c_only_b2():
    assert mk({"B2": 0.8, "ABS": 0.3, "+E": 0.3, "NC": 0.3})["outcome"] == "OUTCOME_C"


def test_outcome_d_when_nc_generalises_like_b2():
    assert mk({"B2": 0.7, "ABS": 0.7, "+E": 0.2, "NC": 0.65})["outcome"] == "OUTCOME_D"


def test_outcome_e_when_seed_ordering_is_unstable():
    seeds = {"B2": {0: 0.9, 1: 0.1, 2: 0.2}, "ABS": {0: 0.9, 1: 0.1, 2: 0.2}, "+E": {0: 0.2, 1: 0.8, 2: 0.2}, "NC": {0: 0.1, 1: 0.1, 2: 0.2}}
    r = mk({"B2": 0.4, "ABS": 0.4, "+E": 0.4, "NC": 0.13}, seeds=seeds)
    assert r["outcome"] == "OUTCOME_E" and r["effect_on_paper"] == "INCONCLUSIVE"


def test_rules_are_evaluated_in_the_frozen_order():
    # NC high and B2 high: OUTCOME_D takes precedence over OUTCOME_A
    assert mk({"B2": 0.9, "ABS": 0.9, "+E": 0.1, "NC": 0.85})["outcome"] == "OUTCOME_D"
