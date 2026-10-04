"""E0: nominal overlays are read-only; building them neither steps the env nor advances time."""
from .helpers.alfworld_fixture import FakePrior, gamepath, load_all, require_data


def test_nominal_overlay_is_read_only_and_does_not_touch_env():
    require_data()
    from cp_disr.contracts import nominal_overlay
    from cp_disr.graph import four_views
    from cp_disr.platforms.alfworld.adapter import AlfEpisode
    from cp_disr.platforms.alfworld.snapshot import build_snapshot, prior_edges_for

    rows, splits, tables = load_all()
    prior = FakePrior()
    for rel in splits["dev"][:12]:
        ep = AlfEpisode(gamepath(rel), tables)
        try:
            ep.check(sorted(ep.feasible)[0])
            pub = ep.public()
            probe = build_snapshot(pub, (), "e", "x", 0)
            edges = prior_edges_for(probe.template, prior.scores(pub, tables))
            snap = build_snapshot(pub, edges, "e", "x", 0)
            before_facts, before_steps, before_cmds = dict(snap.facts.values), ep.raw_steps, list(ep.commands)
            contracts = {c.id: c for c in snap.template.contracts}
            for cid, ok in zip(snap.candidate_ids, snap.mask):
                if not ok:
                    continue
                overlay = nominal_overlay(contracts[cid], snap.facts.values)
                assert dict(snap.facts.values) == before_facts  # base untouched
                assert set(overlay.patch) <= set(before_facts)
                four_views(snap.template, snap.facts.values, edges, contracts[cid])
            assert (ep.raw_steps, list(ep.commands)) == (before_steps, before_cmds)
            assert dict(snap.facts.values) == before_facts
        finally:
            ep.close()


def test_nominal_check_never_prewrites_target_location():
    """CHECK's nominal effect leaves target_in(r) UNKNOWN and never asserts target presence/absence."""
    from cp_disr.facts import Truth
    from cp_disr.platforms.alfworld.pddl_contracts import check_schema

    c = check_schema().ground({"r": "drawer 1"})
    assert [a.id for a in c.effects.unknown] == ["p:target_in:drawer 1"]
    assert not any("target_in" in a.id for a in c.effects.add + c.effects.delete)
    assert c.effects.assignments()["p:target_in:drawer 1"] == Truth.UNKNOWN
