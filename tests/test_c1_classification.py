"""CP-DISR-C1-MECH-CONFIRM-V1: frozen classification rules on synthetic outcomes (pure python)."""
from cp_disr import c1_classification as K

CELLS = K.CELLS


def rates(values):
    return dict(zip(CELLS, values))


def totals(C):
    return {m: sum(v.values()) / len(v) for m, v in C.items()}


def make(b2, ab, q):
    C = {"B2": rates(b2), "ABS": rates(ab), "QMARK": rates(q)}
    return totals(C), C


def test_M1_shared_advantage():
    T, C = make([1.0, 0.9, 0.9, 0.9], [0.9, 0.9, 0.8, 0.8], [0.5, 0.5, 0.5, 0.5])
    assert K.classify_main(T, C)["class"] == "M1"


def test_M1_needs_three_cells_for_both():
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.9, 0.9, 0.9, 0.5], [0.7, 0.7, 0.7, 0.7])     # ABS leads by >= 0.10 only in 3 cells but T gap = 0.075
    assert K.classify_main(T, C)["class"] != "M1"


def test_M2_b2_specific_when_abs_is_close_to_qmark():
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.5, 0.5, 0.5, 0.5], [0.5, 0.5, 0.5, 0.5])
    out = K.classify_main(T, C)
    assert out["class"] == "M2" and out["facts"]["m2_clauses"]["abs_close_to_qmark"]


def test_M2_via_b2_over_abs_gap_and_via_dual_goal_concentration():
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.6, 0.6, 0.6, 0.6], [0.7, 0.7, 0.7, 0.7])      # T(B2)-T(Q)=0.3, ABS-Q = -0.1 -> close too; use a wider gap
    assert K.classify_main(T, C)["class"] == "M2"
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.8, 0.9, 0.9, 0.9], [0.9, 0.95, 0.5, 0.5])
    out = K.classify_main(T, C)
    assert out["class"] == "M2" and out["facts"]["m2_clauses"]["b2_advantage_concentrated_in_dual_goal_cells"]


def test_M3_relational_query_sufficient():
    T, C = make([1.0, 1.0, 0.9, 0.9], [0.9, 0.9, 0.8, 0.8], [0.95, 0.95, 0.9, 0.9])
    assert K.classify_main(T, C)["class"] == "M3"
    T, C = make([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 0.7])       # one cell trails by 0.3 -> still allowed (at most 1)
    assert K.classify_main(T, C)["class"] == "M3"
    T, C = make([1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 0.7, 0.7])       # two cells trail by more than 0.2
    assert K.classify_main(T, C)["class"] == "M4"


def test_M4_is_the_residual_class():
    T, C = make([0.8, 0.8, 0.8, 0.8], [0.7, 0.7, 0.7, 0.7], [0.65, 0.65, 0.65, 0.65])
    assert K.classify_main(T, C)["class"] == "M4"


def test_classes_are_exclusive_over_a_grid():
    import itertools
    for b2, ab, q in itertools.product((0.2, 0.6, 1.0), repeat=3):
        T, C = make([b2] * 4, [ab] * 4, [q] * 4)
        m3 = T["QMARK"] >= T["B2"] - 0.1 - 1e-9 and T["QMARK"] >= T["ABS"] - 0.1 - 1e-9
        m12 = T["B2"] - T["QMARK"] >= 0.2 - 1e-9
        assert not (m3 and m12)


def test_b2_vs_abs_labels():
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.6, 0.6, 1.0, 1.0], [0.5] * 4)
    assert K.classify_b2_vs_abs(T, C)["label"] == "DIFFERENCE_STABILITY_SIGNAL"
    T, C = make([1.0, 1.0, 0.9, 0.9], [0.95, 0.95, 0.9, 0.9], [0.5] * 4)
    assert K.classify_b2_vs_abs(T, C)["label"] == "ABSOLUTE_SUCCESSOR_SUFFICIENT"
    T, C = make([1.0, 1.0, 1.0, 1.0], [0.5, 0.9, 0.9, 0.9], [0.5] * 4)       # T gap 0.2 but only one cell leads by >= 0.2; ABS trails by 0.2: neither rule holds
    assert K.classify_b2_vs_abs(T, C)["label"] == "UNRESOLVED"


def test_pooling_and_id_qualification_and_recommendation():
    per = {0: {"T": {"B2": 1.0}, "C": {"B2": rates([1.0] * 4)}}, 1: {"T": {"B2": 0.5}, "C": {"B2": rates([0.5] * 4)}}}
    T, C = K.pool(per)
    assert T["B2"] == 0.75 and C["B2"]["IN_S"] == 0.75
    assert K.id_qualified(12, 0, None) and not K.id_qualified(11, 0, None) and not K.id_qualified(12, 1, None) and not K.id_qualified(12, 0, "boom")
    assert K.route_recommendation("M1", 0.9, 0.5, True, False)["recommendation"] == "ENTER_MAINLINE"
    assert K.route_recommendation("M3", 0.9, 0.9, True, False)["recommendation"] == "NARROW_AND_CONTINUE"
    assert K.route_recommendation("M4", 0.9, 0.9, True, True)["recommendation"] == "PAUSE"
    assert K.route_recommendation("M2", 0.4, 0.3, True, False)["recommendation"] == "PAUSE"
    assert K.route_recommendation("M1", 0.9, 0.9, False, False)["recommendation"] == "PAUSE"
