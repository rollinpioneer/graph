from upgrade_v2.visual_refine_l2.predicate_dsl import FALSE, TRUE, UNKNOWN, evaluate, satisfied_edges, tri_and, tri_not, tri_or


def test_three_valued_logic_and_windows() -> None:
    assert tri_and([TRUE, UNKNOWN]) == UNKNOWN
    assert tri_and([FALSE, UNKNOWN]) == FALSE
    assert tri_or([FALSE, UNKNOWN]) == UNKNOWN
    assert tri_not(UNKNOWN) == UNKNOWN
    history = [{"p": TRUE}, {"p": TRUE}, {"p": FALSE}]
    assert evaluate({"op": "CONSECUTIVE", "arg": "p", "n": 2}, history, 1) == TRUE
    assert evaluate({"op": "EVER", "arg": "p", "window": 2}, history, 2) == TRUE


def test_multiple_satisfied_edges_are_ambiguous() -> None:
    result = satisfied_edges([{"id": "a", "condition": "p"}, {"id": "b", "condition": "p"}], [{"p": TRUE}])
    assert result["status"] == "ambiguous"
