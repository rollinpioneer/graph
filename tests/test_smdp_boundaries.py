import pytest,json,math
from dataclasses import replace
from pathlib import Path
from cp_disr.rl import Transition,scalar_targets

@pytest.mark.pure
def test_T14_boundaries(snap):
    f=json.loads((Path(__file__).parent/'fixtures/smdp_transitions.json').read_text());n=replace(snap,decision_id=1,clock_seconds=1.)
    d=math.log(f['Gamma'])/math.log(.99)
    def transition(term,trunc):return Transition(snap,n,snap.candidate_ids[0],-.5,.4,2.,.3,d,1.,term,trunc,'synthetic',())
    for name,term,trunc in [('terminated',True,False),('truncated',False,True),('buffer_end',False,False)]:
        a,v,q=scalar_targets([transition(term,trunc)]);assert q[0]==pytest.approx(f['expected'][name],abs=1e-10)
        assert v[0]==pytest.approx(q[0],abs=1e-10)
    with pytest.raises(ValueError):replace(transition(False,True),next_snapshot=replace(n,episode_id='reset'))
    t=transition(False,True);second=replace(t,snapshot=replace(snap,env_id='other'),next_snapshot=replace(n,env_id='other'),reward=1.)
    assert scalar_targets([t,second])[0][0]==pytest.approx(.3+.9*2-.4,abs=1e-10)
