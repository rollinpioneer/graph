import pytest

from cp_disr import phase_a_v13_r1 as r1


pytestmark = pytest.mark.pure


def test_r1_allowlist_and_budget():
    assert r1.TASK == "T_B"
    assert r1.METHODS == ("B1-K", "B2")
    assert r1.AUTHORIZED == {
        "B1-K": "v13_R1_T_B_B1K_s0_E16",
        "B2": "v13_R1_T_B_B2_s0_E16",
    }
    assert r1.N_CAP == 16384
    assert r1.MAX_UPDATES == 16
    assert r1.ROLLOUT_N == 1024
    assert r1.EVAL_POINTS == (0, 4096, 8192, 16384)


def test_configure_routes_only_r1():
    r1.configure()
    assert r1.v11.TASKS == ("T_B",)
    assert r1.v11.METHODS == ("B1-K", "B2")
    assert r1.v11.N_CAP == 16384
    assert r1.v11.DEV_EPISODES == 10
    assert r1.v11.ENABLED_SPLITS["T_B"] == r1.DEV10_REL
    assert set(r1.v11.PLANNED.values()) == set(r1.AUTHORIZED.values())
