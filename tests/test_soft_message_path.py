import pytest

@pytest.mark.torch_runtime
def test_T23_soft_path_and_linear_witness(fixture,policy):
    import torch
    from cp_disr.graph import four_views
    from cp_disr.neural import differences
    # Preserved hand-computable linear witnesses are additional checks, not the production test.
    def propagation(n,edges,delta,target,layers=4):
        m=torch.eye(n,dtype=torch.float64)
        for a,b in edges:m[b,a]+=1;m[a,b]+=1
        return (torch.linalg.matrix_power(m,layers)@torch.tensor(delta,dtype=torch.float64))[target].item()
    k=[(0,1),(1,2),(2,3)];h=k+[(0,2)]
    assert propagation(4,k,[0,1,0,0],3)==9
    assert propagation(4,h,[0,1,0,0],3)==14
    # Historical ordering relation with the same numeric adjacency has the same witness,
    # but the production validator rejects that type (T09).
    assert propagation(4,h,[0,1,0,0],3)-propagation(4,k,[0,1,0,0],3)==5
    assert propagation(3,[(0,1),(0,2)],[0,1,0],2)-propagation(3,[(0,1)],[0,1,0],2)==8
    assert propagation(3,[(0,1),(0,2)],[0,0,0],2)==0
    # Explicit deterministic multidimensional weights, no reliance on random initialization.
    with torch.no_grad():
        for i,p in enumerate(policy.encoder.parameters()):
            p.copy_(.07*torch.sin(torch.arange(p.numel(),dtype=p.dtype).reshape(p.shape)*.13+i*.31))
        for norm in policy.encoder.norms:norm.weight.fill_(1.);norm.bias.zero_()
    c=next(c for c in fixture['contracts'] if c.name=='MOVE')
    d=differences(policy.encoder,four_views(fixture['template'],fixture['facts'].values,fixture['edges'],c))
    assert torch.max(torch.abs(d.dp))>1e-6
    d.dp.square().sum().backward();grads=[p.grad for p in policy.encoder.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)
    empty=differences(policy.encoder,four_views(fixture['template'],fixture['facts'].values,(),c));assert torch.count_nonzero(empty.dp)==0
