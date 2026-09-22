import math
import pytest
from cp_disr.rl import gamma, interval_reward, cumulative_weights

@pytest.mark.pure
def test_T19_duration():
    assert gamma(1) == pytest.approx(0.5, abs=1e-10, rel=1e-9)
    assert gamma(2) == pytest.approx(0.25, abs=1e-10, rel=1e-9)
    assert gamma(1, H=2) == pytest.approx(2 ** (-0.5), abs=1e-10, rel=1e-9)
    assert interval_reward(1, [(0.5, 1)]) == pytest.approx(2 ** (-0.5), abs=1e-10, rel=1e-9)
    assert cumulative_weights([1, 2, 1]) == pytest.approx((1, 0.5, 0.125), abs=1e-10, rel=1e-9)
    for d in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            gamma(d)
