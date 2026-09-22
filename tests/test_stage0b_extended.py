"""Additional production regressions; all inputs are synthetic unit fixtures."""
from dataclasses import replace
import pytest
from cp_disr.common import DataIntegrityError,digest
from cp_disr.rl import remap_stored,Transition,Rollout,scalar_targets

@pytest.mark.pure
def test_T10_recompute_mask_shape_rejected(snap):
    with pytest.raises(DataIntegrityError):
        remap_stored(snap,snap.candidate_ids,snap.mask+(True,))

@pytest.mark.torch_runtime
def test_T22_recurrent_commit_identity(snap,policy):
    from cp_disr.torch_rl import RecurrentState
    state=RecurrentState();state.reset('e0')
    with pytest.raises(DataIntegrityError):state.commit('e0',replace(snap,env_id='e1'))
    state.commit('e0',snap)
    with pytest.raises(DataIntegrityError):state.commit('e0',replace(snap,episode_id='other',decision_id=1))
    with pytest.raises(DataIntegrityError):state.probe(policy,'e0',replace(snap,env_id='other',decision_id=1))
    assert state.histories['e0']==[snap]

@pytest.mark.torch_runtime
def test_T12_fp64_hand_targets():
    import torch
    from cp_disr.torch_rl import q_targets,value_targets,q_loss
    old=torch.tensor([2.,2.,2.],dtype=torch.float64,requires_grad=True)
    target=q_targets(torch.tensor([.3]*3,dtype=torch.float64),torch.tensor([.9]*3,dtype=torch.float64),torch.tensor([True,False,False]),old)
    torch.testing.assert_close(target,torch.tensor([.3,2.1,2.1],dtype=torch.float64),atol=1e-10,rtol=1e-9)
    v=value_targets(old,torch.tensor([-.5,.25,1.],dtype=torch.float64,requires_grad=True))
    torch.testing.assert_close(v,torch.tensor([1.5,2.25,3.],dtype=torch.float64),atol=1e-10,rtol=1e-9)
    assert not target.requires_grad and target.grad_fn is None and v.grad_fn is None
    q=torch.zeros(3,3,dtype=torch.float64,requires_grad=True)
    q_loss(q,torch.tensor([0,1,2]),target).backward()
    assert old.grad is None and torch.count_nonzero(q.grad[~torch.eye(3,dtype=torch.bool)])==0

@pytest.mark.pure
def test_T14_gae_continuity_and_reset(snap):
    import math
    n=replace(snap,decision_id=1)
    t=Transition(snap,n,snap.candidate_ids[0],-.5,.4,2.,.3,-math.log2(.9),1.,False,False,'synthetic',())
    t2=Transition(n,replace(n,decision_id=2),snap.candidate_ids[0],-.5,2.,0.,1.,1.,.9,True,False,'synthetic',(snap,))
    a,v,q=scalar_targets([t,t2])
    assert a==pytest.approx((1.7+.9*.95*(-1),-1),abs=1e-10,rel=1e-9)
    assert v==pytest.approx((.4+a[0],1.),abs=1e-10,rel=1e-9)
    assert scalar_targets([replace(t,truncated=True),t2])[0][0]==pytest.approx(1.7,abs=1e-10,rel=1e-9)

@pytest.mark.torch_runtime
def test_T13_each_of_four_encodings_keeps_gradient(fixture,policy):
    import torch
    from cp_disr.neural import differences
    from cp_disr.graph import four_views
    c=next(c for c in fixture['contracts'] if c.name=='MOVE')
    d=differences(policy.encoder,four_views(fixture['template'],fixture['facts'].values,fixture['edges'],c))
    assert len({id(x) for x in d.encodings})==4
    for x in d.encodings:x.retain_grad()
    d.dp.square().sum().backward()
    for x in d.encodings:assert x.grad is not None and torch.isfinite(x.grad).all() and torch.count_nonzero(x.grad)>0

@pytest.mark.torch_runtime
def test_T18_production_ppo_recomputes_four_epochs(snap,policy,monkeypatch):
    import torch,socket
    from cp_disr.torch_rl import PPO
    from cp_disr.prior import PriorSampler
    def forbidden(*args,**kwargs):raise AssertionError('PPO attempted API/network or prior resampling')
    monkeypatch.setattr(PriorSampler,'start',forbidden)
    monkeypatch.setattr(socket.socket,'connect',forbidden)
    with torch.no_grad():old=policy(snap)
    t=Transition(snap,replace(snap,decision_id=1),snap.candidate_ids[0],float(old.distribution.log_prob(torch.tensor(0))),float(old.value),.2,.3,1.,1.,False,True,'synthetic_unit_fixture',())
    buffer=Rollout();buffer.append(t);before=policy.encoder.forward_calls;state=digest(snap)
    observed=[]
    hook=policy.encoder.register_forward_hook(lambda m,a,out:observed.append((m,tuple(p._version for p in m.parameters()),out.requires_grad)))
    logs=PPO(policy).update(buffer,epochs=4,minibatch=1)
    hook.remove()
    assert [x['epoch'] for x in logs]==[0,1,2,3] and len(buffer.transitions)==0
    assert policy.encoder.forward_calls>before+4 and all(x[0] is policy.encoder and x[2] for x in observed)
    assert len({x[1] for x in observed})>=4 and digest(snap)==state
    assert all(torch.isfinite(torch.tensor(x['total'])) for x in logs)

@pytest.mark.torch_runtime
def test_T20_prior_cannot_enter_base_context(snap,policy):
    import torch
    empty=replace(snap,prior_edges=(),prior_hash=digest(()))
    a=policy(snap);b=policy(empty)
    assert torch.equal(a.hidden,b.hidden) and a.candidate_ids==b.candidate_ids and torch.equal(a.mask,b.mask)
    for i,cid in enumerate(a.candidate_ids):
        torch.testing.assert_close(a.logits[i]-a.diagnostics['delta'][cid],b.logits[i],atol=1e-6,rtol=1e-5)
    assert snap.facts is empty.facts and snap.base_input==empty.base_input

@pytest.mark.torch_runtime
def test_T24_registered_wait_and_no_candidates(snap,fixture):
    import torch
    from cp_disr.contracts import SkillContract,Effects
    from cp_disr.graph import build_template
    from cp_disr.neural import Policy
    wait=SkillContract('WAIT',(),(),(),Effects(),'synthetic-v1','synthetic unit test; not runtime authorization',timeout_seconds=1.)
    cs=fixture['contracts']+(wait,)
    template=build_template(cs,snap.template.goals,fixture['registry'].predicate_types,fixture['objects'])
    model=Policy({n.schema for n in template.nodes if n.kind=='ACTION'},{n.schema for n in template.nodes if n.kind=='PROPOSITION'},{x for n in template.nodes for x in n.argument_types},4,3)
    waiting=replace(snap,template=template,candidate_ids=(wait.id,),candidate_features=((0.,0.,0.),),mask=(True,))
    out=model(waiting)
    assert out.select()==(wait.id,0) and out.distribution.probs.item()==1
    assert all(torch.count_nonzero(x)==0 for x in (out.diagnostics['differences'][wait.id].dk,out.diagnostics['differences'][wait.id].dh,out.diagnostics['differences'][wait.id].dp))
    none=model(replace(waiting,candidate_ids=(),candidate_features=(),mask=()))
    assert none.distribution is None and none.ended_reason=='NO_SAFE_CANDIDATES'

@pytest.mark.torch_runtime
def test_T25_ablation_parameter_and_loss_isolation(snap,policy):
    import torch
    from cp_disr.torch_rl import ppo_losses
    before={k:v.detach().clone() for k,v in policy.state_dict().items()}
    outputs={}
    for method in ['Full','A_DD','A_Q','A_B']:
        policy.method=method;outputs[method]=policy(snap)
        assert all(torch.equal(before[k],v) for k,v in policy.state_dict().items())
    assert torch.equal(outputs['Full'].logits,outputs['A_Q'].logits)
    for cid in snap.candidate_ids:
        i=snap.candidate_ids.index(cid)
        base=outputs['Full'].logits[i]-outputs['Full'].diagnostics['delta'][cid]
        for method in ['A_DD','A_B']:
            torch.testing.assert_close(outputs[method].logits[i]-outputs[method].diagnostics['delta'][cid],base,atol=1e-6,rtol=1e-5)
    x=torch.tensor([.1,.2]);args=(x,x,torch.tensor([1.,-1.]),torch.ones(2),x,torch.zeros(2),x,torch.ones(2),x)
    full=ppo_losses(*args,lambda_q=.1);aq=ppo_losses(*args,lambda_q=0.)
    torch.testing.assert_close(full['total']-aq['total'],.1*full['q'],atol=1e-6,rtol=1e-5)
    assert all(torch.equal(full[k],aq[k]) for k in ['actor','v','q'])

@pytest.mark.torch_runtime
def test_T15_canonical_tie_break(snap,policy):
    import torch
    out=policy(snap);out.logits=torch.zeros_like(out.logits)
    assert out.select(deterministic=True)[0]==min(snap.candidate_ids)
    out.candidate_ids=tuple(reversed(out.candidate_ids))
    assert out.select(deterministic=True)[0]==min(snap.candidate_ids)
