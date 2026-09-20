"""CPU/GPU FP32 agreement, with absolute/relative errors for every output."""
import copy,json,os
from pathlib import Path
import pytest
from cp_disr.common import seed32

@pytest.mark.torch_runtime
def test_T13_T23_cpu_gpu_fp32_outputs_and_gradients(fixture,snap,tmp_path):
    import torch
    from cp_disr.neural import Policy
    if not torch.cuda.is_available():pytest.skip("GPU-only parity check: CUDA unavailable")
    torch.manual_seed(seed32('unit_fixture','device_parity',0,0))
    t=fixture['template'];model=Policy({n.schema for n in t.nodes if n.kind=='ACTION'},{n.schema for n in t.nodes if n.kind=='PROPOSITION'},{x for n in t.nodes for x in n.argument_types},4,3).cpu()
    gpu=copy.deepcopy(model).cuda();results=[]
    def compare(name,a,b):
        a=a.detach().cpu().double();b=b.detach().cpu().double();diff=(a-b).abs();rel=diff/a.abs().clamp_min(1e-12)
        bad=diff>(1e-6+1e-5*a.abs())
        results.append({'name':name,'max_absolute_error':float(diff.max()) if diff.numel() else 0.,'max_relative_error':float(rel.max()) if rel.numel() else 0.,'failure_locations':bad.nonzero().tolist(),'status':'FAIL' if bad.any() else 'PASS','atol':1e-6,'rtol':1e-5})
    for loss_name in ['actor','value','q']:
        model.zero_grad();gpu.zero_grad();a=model(snap);b=gpu(snap)
        for field in ['logits','value','q','hidden']:compare(loss_name+'/'+field,getattr(a,field),getattr(b,field))
        compare(loss_name+'/probabilities',a.distribution.probs,b.distribution.probs)
        for cid in a.diagnostics['differences']:
            for field in ['dk','dh','dp']:compare(loss_name+'/'+cid+'/'+field,getattr(a.diagnostics['differences'][cid],field),getattr(b.diagnostics['differences'][cid],field))
            compare(loss_name+'/'+cid+'/delta',a.diagnostics['delta'][cid],b.diagnostics['delta'][cid])
        for out in [a,b]:
            loss=-out.distribution.log_prob(torch.tensor(0,device=out.logits.device)) if loss_name=='actor' else out.value.square()+out.value if loss_name=='value' else out.q[0].square()+out.q[0]
            loss.backward()
        for (name,p),(other,q) in zip(model.named_parameters(),gpu.named_parameters()):
            assert name==other and (p.grad is None)==(q.grad is None)
            if p.grad is not None:compare(loss_name+'/gradient/'+name,p.grad,q.grad)
    path=Path(os.environ.get('CP_DISR_PARITY_REPORT',str(tmp_path/'numeric_comparison.json')));path.write_text(json.dumps(results,indent=2)+'\n')
    assert all(x['status']=='PASS' for x in results),[x for x in results if x['status']=='FAIL']
