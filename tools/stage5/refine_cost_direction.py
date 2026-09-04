#!/usr/bin/env python3
"""Minimal Stage-4 refinement for the remaining-cost/failure-recovery head."""
import argparse,csv
from pathlib import Path
import numpy as np, torch
from tools.stage4.lib.model import GraphStateModel

def samples(root, split, history=32):
    out=[]
    for r in csv.DictReader(open(Path(root)/'tables/episode_manifest.csv')):
        if r['split_original'] != split or r['task_id'] != 'transport_recovery': continue
        z=np.load(Path(root)/r['file']); x=z['x']; et=z['edge_type_y']
        ws=[]
        for t in range(len(x)):
            lo=max(0,t-history+1); w=np.zeros((history,14),np.float32); w[-(t-lo+1):]=x[lo:t+1]; ws.append(w)
        for t in range(1,len(x)-1):
            if et[t] not in (3,4): continue
            before=max(0,t-1); after=min(len(x)-1,t+1)
            # Use the segment boundary direction; repeated event frames are
            # linked to the first post-segment frame.
            end=t
            while end+1<len(x) and et[end+1]==et[t]: end+=1
            after=min(len(x)-1,end+1)
            out.append((ws[before],ws[after],int(et[t]),float(z['cost_y_norm'][before]),float(z['cost_y_norm'][after])))
    return out

def main():
    p=argparse.ArgumentParser(); p.add_argument('--init-checkpoint',required=True); p.add_argument('--calibration-checkpoint'); p.add_argument('--supervision-dir',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,required=True); p.add_argument('--epochs',type=int,default=80); p.add_argument('--direction-weight',type=float,default=80.0); p.add_argument('--reg-weight',type=float,default=0.1); p.add_argument('--correction-l2',type=float,default=1.0); p.add_argument('--device',default='cuda'); a=p.parse_args()
    torch.manual_seed(a.seed); dev=a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu'; m=GraphStateModel(); ck=torch.load(a.init_checkpoint,map_location=dev); m.load_state_dict(ck.get('model',ck),strict=False)
    if a.calibration_checkpoint:
        src=torch.load(a.calibration_checkpoint,map_location='cpu').get('model',torch.load(a.calibration_checkpoint,map_location='cpu'))
        own=m.state_dict()
        for k,v in src.items():
            if k.startswith('event_cost_head') and k in own: own[k]=v
        m.load_state_dict(own,strict=False)
    m.to(dev).train()
    # Keep classification/phi heads fixed; optimize the shared representation
    # and cost calibration only, with a strong event-direction objective.
    for n,q in m.named_parameters(): q.requires_grad = n.startswith('event_cost_head') or n.startswith('cost_head')
    opt=torch.optim.AdamW([q for q in m.parameters() if q.requires_grad],lr=1e-3)
    ss=samples(a.supervision_dir,'train');
    if not ss: raise SystemExit('no training event samples')
    for _ in range(a.epochs):
        np.random.shuffle(ss)
        for i in range(0,len(ss),32):
            b=ss[i:i+32]; x0=torch.from_numpy(np.stack([q[0] for q in b])).to(dev); x1=torch.from_numpy(np.stack([q[1] for q in b])).to(dev); typ=torch.tensor([q[2] for q in b],device=dev); c0=torch.tensor([q[3] for q in b],device=dev); c1=torch.tensor([q[4] for q in b],device=dev)
            o0=m(x0); o1=m(x1); y0=o0['remaining_cost']; y1=o1['remaining_cost']
            delta=y1-y0
            # Failure must increase cost; recovery must decrease cost.
            direction=torch.where(typ==4, torch.relu(0.05-delta), torch.relu(delta+0.05))
            # Direction calibration is the only requested change; a strong
            # margin term is necessary because the frozen graph prior blends
            # 75% of the structural cost signal.
            loss=a.direction_weight*direction.mean()+a.reg_weight*torch.nn.functional.smooth_l1_loss(y0,c0)+a.reg_weight*torch.nn.functional.smooth_l1_loss(y1,c1)+a.correction_l2*(o0['event_cost_correction'].square().mean()+o1['event_cost_correction'].square().mean())
            opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(m.parameters(),1.0); opt.step()
    m.eval(); out=Path(a.output_dir); (out/'checkpoints').mkdir(parents=True,exist_ok=True); state={'model':m.state_dict(),'seed':a.seed,'history_steps':32,'refinement':'remaining_cost_failure_recovery_direction_only','init_checkpoint':str(Path(a.init_checkpoint).resolve())}; torch.save(state,out/'checkpoints/best.pt'); torch.save(state,out/'checkpoints/final.pt')
if __name__=='__main__': main()
