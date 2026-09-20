import pytest
from dataclasses import replace

@pytest.mark.torch_runtime
def test_T15_permutations(snap,policy):
    import torch
    from cp_disr.facts import FactStore,FactRecord
    from cp_disr.common import digest
    a=policy(snap);perm=tuple(reversed(range(len(snap.candidate_ids))))
    b=policy(replace(snap,candidate_ids=tuple(snap.candidate_ids[i] for i in perm),candidate_features=tuple(snap.candidate_features[i] for i in perm),mask=tuple(snap.mask[i] for i in perm)))
    assert torch.allclose(a.logits[list(perm)],b.logits,atol=1e-6,rtol=1e-5)
    def rename(s):return s.replace(':red',':renamed')
    t=snap.template
    nodes=tuple(replace(n,id=rename(n.id),arguments=tuple('renamed' if x=='red' else x for x in n.arguments)) for n in t.nodes)
    def atom(x):return replace(x,arguments=tuple('renamed' if y=='red' else y for y in x.arguments))
    cs=tuple(replace(c,bound_arguments=tuple('renamed' if x=='red' else x for x in c.bound_arguments),pre_pos=tuple(map(atom,c.pre_pos)),pre_neg=tuple(map(atom,c.pre_neg)),effects=replace(c.effects,add=tuple(map(atom,c.effects.add)),delete=tuple(map(atom,c.effects.delete)),unknown=tuple(map(atom,c.effects.unknown)))) for c in t.contracts)
    nt=replace(t,nodes=nodes,edges=tuple((rename(x),rename(y),r) for x,y,r in t.edges),goals=tuple(replace(g,fact_id=rename(g.fact_id)) for g in reversed(t.goals)),contracts=cs)
    edges=tuple((rename(x),rename(y),r) for x,y,r in snap.prior_edges)
    ns=replace(snap,template=nt,facts=FactStore(tuple(replace(f,fact_id=rename(f.fact_id)) for f in snap.facts.records)),candidate_ids=tuple(map(rename,snap.candidate_ids)),prior_edges=edges,prior_hash=digest(edges))
    c=policy(ns);assert torch.allclose(a.logits,c.logits,atol=1e-6,rtol=1e-5)
