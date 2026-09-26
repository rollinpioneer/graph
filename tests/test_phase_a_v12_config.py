import pytest

from cp_disr import phase_a_v12 as p


pytestmark = pytest.mark.pure


def test_only_authorized_pair_and_e16_caps():
    assert p.AUTHORIZED == {
        "Full": "v12_E16_T_C_Full_s0",
        "B2": "v12_E16_T_C_B2_s0",
    }
    assert p.N_CAP == 16384
    assert p.ROLLOUT_N == 1024
    assert p.MAX_UPDATES == 16
    assert p.EVAL_POINTS == (0, 4096, 8192, 16384)


def test_v11_adapter_is_narrow():
    p._configure_v11()
    assert p.v11.TASKS == ("T_C",)
    assert p.v11.METHODS == ("B2", "Full")
    assert p.v11.N_CAP == 16384
    assert p.v11.DEV_EPISODES == 10
    assert set(p.v11.PLANNED.values()) == set(p.AUTHORIZED.values())


def test_checkpoint_selection_prefers_better_worst_then_earlier_n():
    rows = [
        {"skill_transitions": 0, "mean_discounted_return": 0.0, "checkpoint": "a"},
        {"skill_transitions": 4096, "mean_discounted_return": 1.0, "checkpoint": "b"},
        {"skill_transitions": 8192, "mean_discounted_return": 0.5, "checkpoint": "c"},
        {"skill_transitions": 16384, "mean_discounted_return": 0.0, "checkpoint": "d"},
    ]
    selected, doc = p._select_checkpoint(rows)
    assert selected == "b"
    assert doc["selection_rule"] == "max_mean3_then_max_worst_then_earlier_N"
