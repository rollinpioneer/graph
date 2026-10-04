"""E0: macros execute real admissible commands, record raw durations, and respect the 50-step cap."""
import random

from .helpers.alfworld_fixture import gamepath, load_all, require_data


def _drive(ep, rng):
    results = []
    while not ep.done:
        pub = ep.public()
        results.append(ep.execute(*rng.choice(sorted(pub.legal))))
    return results


def test_durations_equal_raw_command_counts_and_games_are_solvable():
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode

    rows, splits, tables = load_all()
    rng = random.Random(7)
    kinds = set()
    for rel in splits["dev"][:40]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            res = _drive(ep, rng)
            assert sum(r.duration for r in res) == ep.raw_steps == len(ep.commands)
            assert all(r.duration == len(r.commands) for r in res)
            assert ep.won != ep.truncated  # random search may hit the 50-step cap, never both/neither
            assert (res[-1].status == "WON") == ep.won and (not ep.won or ep.success_step == ep.raw_steps)
            kinds |= {r.kind for r in res}
            assert all(r.duration >= 1 for r in res)
        finally:
            ep.close()
    assert kinds == {"CHECK", "TAKE", "PUT"}


def test_games_are_solvable_through_the_macro_interface():
    require_data()
    from cp_disr.baselines import alfworld_scripts as S

    rows, splits, tables = load_all()
    for rel in splits["dev"][:40]:
        r = S.run_episode(gamepath(rel), tables, S.belief_optimal)
        assert r["won"] and r["raw_steps"] <= 50


def test_every_raw_command_is_admissible_and_co_located_receptacles_work():
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode

    rows, splits, tables = load_all()
    saw_examine = False
    for rel in splits["dev"]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            for r in sorted(ep.feasible):
                if ep.done or ep.target_found:
                    break
                res = ep.check(r)  # ProtocolError would be raised on any inadmissible command
                saw_examine |= any(c.startswith("examine ") for c in res.commands)
        finally:
            ep.close()
    assert saw_examine  # the shared-location path was exercised


def test_step_cap_truncates_without_reward():
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode

    rows, splits, tables = load_all()
    ep = AlfEpisode(gamepath(splits["dev"][0]), tables, max_steps=2)
    try:
        for r in sorted(ep.feasible)[:3]:
            if ep.done:
                break
            res = ep.check(r)
        assert ep.done and ep.raw_steps <= 2 and not ep.won
        assert res.status in ("TRUNCATED", "WON")
    finally:
        ep.close()


def test_mask_equals_contract_preconditions_along_episodes():
    require_data()
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.snapshot import build_snapshot, mask_matches_preconditions

    rows, splits, tables = load_all()
    rng = random.Random(3)
    for rel in splits["dev"][:30]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            step = 0
            while not ep.done:
                pub = ep.public()
                assert mask_matches_preconditions(build_snapshot(pub, (), "e", "x", step))
                ep.execute(*rng.choice(sorted(pub.legal)))
                step += 1
        finally:
            ep.close()
