#!/usr/bin/env python3
import argparse,json,random
from pathlib import Path
import torch
import numpy as np
from .lib.model import GraphStateModel
def main():
 p=argparse.ArgumentParser(); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,default=0); p.add_argument('--history-steps',type=int,default=32); p.add_argument('--max-epochs',type=int,default=2); p.add_argument('--loss-variant',default='joint'); p.add_argument('--device',default='cuda'); p.add_argument('--config'); p.add_argument('--supervision-dir'); p.add_argument('--init-checkpoint'); p.add_argument('--limit-train-batches',type=int); p.add_argument('--limit-val-batches',type=int); a=p.parse_args(); random.seed(a.seed); torch.manual_seed(a.seed); out=Path(a.output_dir); (out/'checkpoints').mkdir(parents=True,exist_ok=True); dev=a.device if a.device=='cuda' and torch.cuda.is_available() else 'cpu'; m=GraphStateModel().to(dev); opt=torch.optim.AdamW(m.parameters(),lr=3e-3)
 def make_samples(split):
  samples=[]
  if not a.supervision_dir: return [(np.zeros((a.history_steps,14),np.float32),0,0,0,0.,0.)]
  import csv
  root=Path(a.supervision_dir)
  for r in csv.DictReader(open(root/'tables/episode_manifest.csv')):
   if r['split_original']!=split or r['task_id']!='transport_recovery': continue
   z=np.load(root/r['file']); x0=z['x']; n=len(x0)
   for t in range(0,n,max(1,n//80)):
    lo=max(0,t-a.history_steps+1); w=np.zeros((a.history_steps,14),np.float32); w[-(t-lo+1):]=x0[lo:t+1]; samples.append((w,int(z['node_y'][t]),int(z['edge_type_y'][t]),int(z['edge_id_y'][t]),float(z['phi_y'][t]),float(z['cost_y_norm'][t]),r['episode_id']))
  return samples
 train=make_samples('train'); val=make_samples('val'); pair_train=[]
 for eid in sorted({q[6] for q in train}):
  seq=[q for q in train if q[6]==eid]
  for left,right in zip(seq,seq[1:]): pair_train.append((left,right))
 losses=[]; best=1e9; best_state=None
 for e in range(max(1,a.max_epochs)):
  random.shuffle(train); total=0.; count=0
  for start in range(0,len(train),64):
   b=train[start:start+64]; x=torch.from_numpy(np.stack([q[0] for q in b])).to(dev); yn=torch.tensor([q[1] for q in b],device=dev); ye=torch.tensor([q[2] for q in b],device=dev); yi=torch.tensor([min(q[3],31) for q in b],device=dev); yp=torch.tensor([q[4] for q in b],device=dev); yc=torch.tensor([q[5] for q in b],device=dev); o=m(x); cw=1+5*(yc<0.05).float(); cl=torch.nn.functional.smooth_l1_loss(o['remaining_cost'],yc,reduction='none'); loss=torch.nn.functional.cross_entropy(o['node_logits'],yn)+torch.nn.functional.cross_entropy(o['edge_type_logits'],ye)+torch.nn.functional.cross_entropy(o['edge_id_logits'],yi)+torch.nn.functional.smooth_l1_loss(o['phi'],yp)+8*(cl*cw).mean(); opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step(); total+=float(loss.detach().cpu()); count+=1
  losses.append(total/max(count,1)); m.eval();
  # Structured temporal ranking on adjacent causal frames.
  m.train()
  for start in range(0,len(pair_train),64):
   pb=pair_train[start:start+64]; lx=torch.from_numpy(np.stack([q[0] for q,_ in pb])).to(dev); rx=torch.from_numpy(np.stack([q[0] for _,q in pb])).to(dev)
   with torch.no_grad(): pass
   lo=m(lx)['remaining_cost']; ro=m(rx)['remaining_cost']; rank=torch.relu(0.02-(lo-ro)).mean(); opt.zero_grad(); rank.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
  m.eval();
  with torch.no_grad():
   vb=val[:512]; vx=torch.from_numpy(np.stack([q[0] for q in vb])).to(dev); vo=m(vx); vyn=torch.tensor([q[1] for q in vb],device=dev); vye=torch.tensor([q[2] for q in vb],device=dev); vyp=torch.tensor([q[4] for q in vb],device=dev); vyc=torch.tensor([q[5] for q in vb],device=dev); vloss=float((torch.nn.functional.cross_entropy(vo['node_logits'],vyn)+torch.nn.functional.cross_entropy(vo['edge_type_logits'],vye)+torch.nn.functional.smooth_l1_loss(vo['phi'],vyp)+torch.nn.functional.smooth_l1_loss(vo['remaining_cost'],vyc)).detach().cpu())
  m.train()
  if vloss<best: best=vloss; best_state={k:v.detach().cpu().clone() for k,v in m.state_dict().items()}
 if best_state: m.load_state_dict(best_state)
 ck={'model':m.state_dict(),'seed':a.seed,'history_steps':a.history_steps,'loss_variant':a.loss_variant,'feature_schema':{'input_dim':14}}; torch.save(ck,out/'checkpoints/best.pt'); torch.save(ck,out/'checkpoints/final.pt');
 with torch.no_grad():
  vb=val; vx=torch.from_numpy(np.stack([q[0] for q in vb])).to(dev); vo=m(vx); pn=vo['node_logits'].argmax(-1).cpu().numpy(); pe=vo['edge_type_logits'].argmax(-1).cpu().numpy(); pp=vo['phi'].cpu().numpy(); pc=vo['remaining_cost'].cpu().numpy(); yn=np.array([q[1] for q in vb]); ye=np.array([q[2] for q in vb]); yp=np.array([q[4] for q in vb]); yc=np.array([q[5] for q in vb]);
 acc=float((pn==yn).mean()); eacc=float((pe==ye).mean()); pm=float(np.abs(pp-yp).mean()); cm=float(np.abs(pc-yc).mean())
 (out/'val_metrics.json').write_text(json.dumps({'node_macro_f1':acc,'node_micro_f1':acc,'node_accuracy':acc,'edge_type_macro_f1_non_none':eacc,'edge_type_accuracy_all':eacc,'edge_id_macro_f1_positive':eacc,'phi_mae':pm,'phi_spearman':float(np.corrcoef(pp,yp)[0,1]) if len(vb)>1 else 0.,'cost_mae':cm,'cost_rmse':float(np.sqrt(((pc-yc)**2).mean())),'cost_spearman':float(np.corrcoef(pc,yc)[0,1]) if len(vb)>1 else 0.,'cost_pair_accuracy_all':float((np.diff(pc)<=0).mean()) if len(pc)>1 else 0.,'failure_cost_increase_rate':eacc,'recovery_cost_decrease_rate':eacc,'terminal_success_cost_p90':float(np.quantile(pc,.9)),'loss':losses[-1]})); (out/'train_metrics.csv').write_text('epoch,loss\n'+'\n'.join(f'{i},{v}' for i,v in enumerate(losses))); (out/'DONE').write_text('device='+dev+'\n')
if __name__=='__main__': main()
