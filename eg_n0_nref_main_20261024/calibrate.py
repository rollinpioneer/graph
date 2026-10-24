"""Freeze thresholds using only the 48 predetermined training frames."""
import json,time,hashlib
from pathlib import Path
import numpy as np
R=Path(__file__).resolve().parent;P=json.loads((R/'protocol.json').read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def load():
 data=[];manifest=[]
 for s in P['training_scenes']:
  parts=[]
  for f in P['training_frames']:
   cp=R/f'cpu/train_{s:04d}_{f:04d}.npz';npth=R/f'nv/train_{s:04d}_{f:04d}.npz';a=np.load(cp);v=np.load(npth)
   row={k:a[k].ravel() for k in ['collision','empty','EG_cd','full_EG','R1','R2','R3','R4','ref_occ','ref_unk']};row.update({k:v[k] for k in ['nv_occ','nv_unk']});parts.append(row)
   manifest.extend([dict(path=str(x.relative_to(R)),sha256=sha(x)) for x in [cp,npth]])
  one={k:np.concatenate([r[k] for r in parts]) for k in parts[0]};one['scene']=s;data.append(one)
 return data,manifest
def rates(data,masks):
 vals=[]
 for a,m in zip(data,masks):
  pos=a['collision']&~a['empty'];neg=~a['collision']&~a['empty'];assert pos.any() and neg.any()
  vals.append([float(m[pos].mean()),float(m[neg].mean())])
 return np.mean(vals,axis=0)
def fit(data,key,budget):
 vals=np.concatenate([a[key] for a in data]);pw=[];nw=[]
 for a in data:
  pos=a['collision']&~a['empty'];neg=~a['collision']&~a['empty'];pw.extend(pos/pos.sum()/len(data));nw.extend(neg/neg.sum()/len(data))
 order=np.argsort(vals,kind='stable');x=vals[order];pos=np.array(pw)[order];neg=np.array(nw)[order]
 ends=np.r_[np.flatnonzero(x[1:]!=x[:-1]),len(x)-1];thresholds=x[ends]
 recall=np.maximum(0,1-np.cumsum(pos)[ends]);fpr=np.maximum(0,1-np.cumsum(neg)[ends])
 feasible=np.flatnonzero(fpr<=budget+1e-12);best=feasible[np.lexsort((thresholds[feasible],fpr[feasible],-recall[feasible]))[0]]
 threshold=float(thresholds[best]);actual=rates(data,[a[key]>threshold for a in data]);assert actual[1]<=budget+1e-10
 curve_indices=np.unique(np.linspace(0,len(thresholds)-1,min(400,len(thresholds))).astype(int))
 return dict(threshold=threshold,recall=float(actual[0]),false_filter=float(actual[1]),target_false_filter=budget),dict(threshold=thresholds[curve_indices].tolist(),recall=recall[curve_indices].tolist(),false_filter=fpr[curve_indices].tolist())
if __name__=='__main__':
 assert not (R/'frozen_config.json').exists(),'Do not refit frozen calibration'
 data,manifest=load();base=rates(data,[a['EG_cd'] for a in data]);budgets=dict(OP1=float(base[1]),OP2=min(1.,float(base[1]+.05)))
 frozen=dict(primary='N_ref',created_unix=time.time(),protocol_sha256=sha(R/'protocol.json'),calibration_input_manifest=manifest,baseline=dict(recall=float(base[0]),false_filter=float(base[1])),workpoints={},kappa={},scope='training only; frozen before C3/P2 labels and scores')
 curves={}
 for family in ['ref','nv']:
  choices=[]
  for k in P['kappa_grid']:
   key=f'{family}_{k}'
   for a in data:a[key]=a[family+'_occ']+k*a[family+'_unk']
   res,curve=fit(data,key,budgets['OP1']);choices.append((res['recall'],-res['false_filter'],-k,k));curves[key]=curve
  frozen['kappa'][family]=max(choices)[3]
 for op,budget in budgets.items():
  configs={}
  for name,key in [('R1','R1'),('R2','R2'),('R3','R3'),('R4','R4'),('N_ref',f'ref_{frozen["kappa"]["ref"]}'),('N_ref_free','ref_0'),('nvblox',f'nv_{frozen["kappa"]["nv"]}'),('nvblox_free','nv_0')]:
   configs[name],curves[op+'_'+name]=fit(data,key,budget);configs[name]['key']=key
  best=max(['R1','R2','R3','R4'],key=lambda n:(configs[n]['recall'],-configs[n]['false_filter'],-['R1','R2','R3','R4'].index(n)))
  frozen['workpoints'][op]=dict(methods=configs,best_simple=best)
 frozen['baseline_full_depth']=dict(zip(['recall','false_filter'],map(float,rates(data,[a['full_EG'] for a in data]))))
 (R/'calibration_curves.json').write_text(json.dumps(curves))
 (R/'frozen_config.json').write_text(json.dumps(frozen,indent=2))
 (R/'frozen_config.sha256').write_text(sha(R/'frozen_config.json')+'\n')
 print(json.dumps({k:v for k,v in frozen.items() if k!='calibration_input_manifest'},indent=2),flush=True)
