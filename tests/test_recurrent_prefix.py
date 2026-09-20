import pytest
from dataclasses import replace

@pytest.mark.torch_runtime
def test_T22_recurrent_prefix(snap,policy):
    import torch
    import json
    from pathlib import Path
    from cp_disr.torch_rl import RecurrentState,prefix_hidden
    history=[replace(snap,decision_id=i,base_input=tuple(float(j==i%4) for j in range(4))) for i in range(3)]
    state=RecurrentState();state.reset('e0');state.reset('e1')
    fixture=json.loads((Path(__file__).parent/'fixtures/snapshots_two_envs.json').read_text())
    assert fixture['synthetic_unit_fixture'] is True
    for i,base in enumerate(fixture['envs'][1]['history']):state.commit('e1',replace(snap,env_id='e1',episode_id='ep1',decision_id=i,base_input=tuple(base)))
    other=prefix_hidden(policy,state.histories['e1'])
    for s in history:state.commit('e0',s)
    next_s=replace(snap,decision_id=3)
    a=state.probe(policy,'e0',next_s);b=state.probe(policy,'e0',next_s)
    assert torch.equal(a.hidden,b.hidden) and len(state.histories['e0'])==3
    full=policy.initial_hidden()
    for s in history:full=policy.advance_hidden(s.base_input,full)
    assert not torch.allclose(full,other)
    assert torch.allclose(prefix_hidden(policy,history),full,atol=1e-6,rtol=1e-5)
    state.reset('e1');assert len(state.histories['e0'])==3
    h=prefix_hidden(policy,history[:1]);h=policy.advance_hidden(history[1].base_input,h);h=policy.advance_hidden(history[2].base_input,h)
    h.sum().backward();assert policy.gru.weight_hh.grad is not None and torch.count_nonzero(policy.gru.weight_hh.grad)>0
    with torch.no_grad():policy.gru.weight_hh.add_(.03)
    updated=prefix_hidden(policy,history);full=policy.initial_hidden()
    for s in history:full=policy.advance_hidden(s.base_input,full)
    assert torch.allclose(updated,full,atol=1e-6,rtol=1e-5)
