import pytest
from cp_disr.graph import four_views

@pytest.mark.pure
def test_T04_node_alignment(fixture):
    for c in fixture['contracts'][:2]:
        k,h,ki,hi=four_views(fixture['template'],fixture['facts'].values,fixture['edges'],c)
        assert k.template.nodes==h.template.nodes==ki.template.nodes==hi.template.nodes
        assert k.template.goal_refs==h.template.goal_refs==ki.template.goal_refs==hi.template.goal_refs
        assert ki.values==hi.values and k.values==h.values
        assert k.edges!=h.edges
