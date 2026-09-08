from upgrade_v2.visual_refine_l2.dataset import SCENARIOS, make_family_specs


def test_development_and_fresh_family_ids_are_disjoint() -> None:
    development = {item.root_family_id for item in make_family_specs(48, list(SCENARIOS), 220710, 2210000, "L2D")}
    fresh = {item.root_family_id for item in make_family_specs(24, list(SCENARIOS), 220799, 2290000, "L2C")}
    assert len(development) == 48
    assert len(fresh) == 24
    assert not development & fresh
