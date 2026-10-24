"""Zero-weight-update expected-risk check on cached class logits and GT I1.
Contact risk is a detached table. No differentiation through contact solver.
"""
from pathlib import Path
import json
import numpy as np
import torch
ROOT=Path(__file__).resolve().parent
torch.set_num_threads(1)
rows=[];probes=[]
for file in sorted((ROOT/'evaluation').glob('*.npz')):
 a=np.load(file);s,f=map(int,file.stem.split('_'));r=a['I1'];mask=np.isfinite(r);risk=torch.tensor(np.nan_to_num(r,nan=0),dtype=torch.float64)
 valid=torch.tensor(mask);z=torch.tensor(a['angle_logits'].T,dtype=torch.float64,requires_grad=True)
 assert z.shape[1]==13
 for temperature in [1.,.5,2.]:
  p=torch.softmax((z[:,:12]/temperature).masked_fill(~valid,-torch.inf),dim=1);loss=(p*risk).sum(1)
  g=torch.autograd.grad(loss.sum(),z)[0];analytic=p*(risk-loss[:,None])/temperature
  err=float((g[:,:12]-analytic).abs().max());assert err<1e-12 and torch.isfinite(g).all() and (g[:,12]==0).all()
  # Finite difference in a deterministic direction over the full 13 logits.
  v=torch.cos(torch.arange(z.numel(),dtype=torch.float64)).reshape(z.shape);eps=1e-4
  def value(zp):return (torch.softmax((zp[:,:12]/temperature).masked_fill(~valid,-torch.inf),dim=1)*risk).sum(1)
  fd=(value(z.detach()+eps*v)-value(z.detach()-eps*v))/(2*eps);dot=(g*v).sum(1)
  assert (fd-dot).abs().max()<1e-7
  if temperature==1:
   for j,k in enumerate(a['candidate_id']):rows.append(dict(scene=s,frame=f,candidate_id=int(k),contact_classes=int(mask[j].sum()),gradient_max=float(g[j].abs().max()),expected_I1=float(loss[j]),analytic_error=err,finite_difference_error=float(abs(fd[j]-dot[j])),invalid_class_gradient=float(g[j,12])))
  for bound in [.25,.5,1.]:
   # Ephemeral intervention; original logits and every network weight untouched.
   delta=-bound*g/g.abs().amax(1,keepdim=True).clamp_min(1e-12);after=z.detach()+delta
   new=after[:,:12].argmax(1).numpy();afterloss=value(after).detach().numpy()
   for j,k in enumerate(a['candidate_id']):probes.append(dict(scene=s,frame=f,candidate_id=int(k),temperature=temperature,bound=bound,old_class=int(a['original_angle'][j]),new_class=int(new[j]),expected_risk_before=float(loss[j]),expected_risk_after=float(afterloss[j]),max_logit_change=float(delta[j].abs().max())))
(ROOT/'gradient_checks.json').write_text(json.dumps(dict(rows=rows,probes=probes,model_updates=0,gt_assisted=True,actual_model_outputs_changed=False)))
print(json.dumps(dict(candidates=len(rows),nonzero=sum(r['gradient_max']>1e-9 for r in rows),max_fd_error=max(r['finite_difference_error'] for r in rows),probes=len(probes))))
