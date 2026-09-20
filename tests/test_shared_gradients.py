import pytest

@pytest.mark.torch_runtime
def test_T13_shared_gradients(snap,policy):
    import torch
    ids={id(p) for p in policy.encoder.parameters()};assert len(ids)==len(list(policy.encoder.parameters()))
    for kind in ['actor','value','q']:
        policy.zero_grad();out=policy(snap)
        for d in out.diagnostics['differences'].values():
            assert all(x.grad_fn is not None for x in d.encodings)
        loss=-out.distribution.log_prob(torch.tensor(0)) if kind=='actor' else out.value.square()+out.value if kind=='value' else out.q[0].square()+out.q[0]
        loss.backward();grads=[p.grad for p in policy.encoder.parameters() if p.grad is not None]
        assert grads and all(torch.isfinite(g).all() for g in grads) and any(torch.count_nonzero(g)>0 for g in grads)
