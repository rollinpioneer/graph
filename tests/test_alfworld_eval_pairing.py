"""Evaluation RNG is per (game, repeat, common seed): earlier episodes cannot change later ones."""
import pytest

from .helpers.alfworld_fixture import gamepath, load_all, require_data

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")


def test_per_episode_results_are_independent_of_which_other_games_were_evaluated():
    require_data()
    from cp_disr.platforms.alfworld.prior_provider import SyntheticPrior
    from cp_disr.platforms.alfworld.trainer import evaluate, make_policy

    rows, splits, tables = load_all()
    g = splits["dev"][:3]
    pol = make_policy("B2", "cpu", 0).eval()  # stochastic policy: consumes RNG every decision
    a, b = [], []
    evaluate(pol, g, tables, SyntheticPrior(), "/home/xushijie2/xsj2_alf/data", repeats=2, seed=7, record=False, per_episode=a)
    evaluate(pol, [g[2]], tables, SyntheticPrior(), "/home/xushijie2/xsj2_alf/data", repeats=2, seed=7, record=False, per_episode=b)
    key = lambda rows_: {(r["game"], r["repeat"]): (r["J"], r["raw_steps"]) for r in rows_ if r["game"] == g[2]}
    assert key(a) == key(b) and len(key(a)) == 2
