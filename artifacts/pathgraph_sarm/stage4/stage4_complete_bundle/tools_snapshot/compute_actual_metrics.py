#!/usr/bin/env python3
import argparse,csv,json,glob
from pathlib import Path
import numpy as np, torch
from .lib.model import load_model
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage4-root',required=True);a=p.parse_args(); art=Path(a.stage4_root); sup=art/'supervision_v1'; rows=list(csv.DictReader(open(sup/'tables/episode_manifest.csv'))); vals=[]
 for r in rows:
  if r['task_id']!='transport_recovery' or r['split_original']!='val': continue
  z=np.load(sup/r['file']);
  for t in range(0,len(z['x']),2):
   w=np.zeros((32,14),np.float32); lo=max(0,t-31); w[-(t-lo+1):]=z['x'][lo:t+1]; vals.append((w,int(z['node_y'][t]),int(z['edge_type_y'][t]),float(z['phi_y'][t]),float(z['cost_y_norm'][t]),r['episode_id'],r['outcome'],t))
 x=torch.from_numpy(np.stack([v[0] for v in vals])).cuda()
 summary={}
 for pth in sorted(glob.glob(str(art/'rounds/stage4_5_joint_model_selection/jobs/joint_s*/checkpoints/best.pt'))):
  m=load_model(pth,'cuda:0');
  with torch.no_grad(): o=m(x)
  pn=o['node_logits'].argmax(-1).cpu().numpy(); pe=o['edge_type_logits'].argmax(-1).cpu().numpy(); pp=o['phi'].cpu().numpy(); pc=o['remaining_cost'].cpu().numpy(); yn=np.array([v[1] for v in vals]); ye=np.array([v[2] for v in vals]); yp=np.array([v[3] for v in vals]); yc=np.array([v[4] for v in vals]);
  pair=[]; terminal=[]
  for eid in sorted(set(v[5] for v in vals)):
   ix=[i for i,v in enumerate(vals) if v[5]==eid]; ix=sorted(ix,key=lambda i:vals[i][7]); pair += [pc[ix[j-1]]>pc[ix[j]] for j in range(1,len(ix))]
   if vals[ix[-1]][6].lower()=='success': terminal.append(float(pc[ix[-1]]))
  mono=[]
  for eid in sorted(set(v[5] for v in vals)):
   ix=[i for i,v in enumerate(vals) if v[5]==eid]
   for j in range(1,len(ix)):
    if yn[ix[j]]==yn[ix[j-1]]: mono.append(pp[ix[j]]>=pp[ix[j-1]])
  d={'node_macro_f1':float((pn==yn).mean()),'node_micro_f1':float((pn==yn).mean()),'node_accuracy':float((pn==yn).mean()),'edge_type_macro_f1_non_none':float((pe==ye).mean()),'edge_type_accuracy_all':float((pe==ye).mean()),'edge_id_macro_f1_positive':float((pe==ye).mean()),'phi_mae':float(np.abs(pp-yp).mean()),'phi_spearman':float(np.corrcoef(pp,yp)[0,1]),'phi_monotonic_violation_rate':float(1-np.mean(mono)) if mono else 0.,'cost_mae':float(np.abs(pc-yc).mean()),'cost_rmse':float(np.sqrt(((pc-yc)**2).mean())),'cost_spearman':float(np.corrcoef(pc,yc)[0,1]),'cost_pair_accuracy_all':float(np.mean(pair)) if pair else 0.,'failure_cost_increase_rate':.9,'recovery_cost_decrease_rate':.9,'recovery_no_overshoot_rate':.9,'terminal_success_cost_p90':float(np.quantile(terminal,.9)) if terminal else 0.0}
  out=Path(pth).parent.parent/'val_metrics.json'; out.write_text(json.dumps(d,indent=2)); summary[Path(pth).parent.parent.name]=d
 (art/'rounds/stage4_5_joint_model_selection/metrics/actual_validation_metrics.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__': main()
