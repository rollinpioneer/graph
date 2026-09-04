#!/usr/bin/env python3
import argparse,glob,gzip,json,hashlib
from pathlib import Path
import numpy as np
def entropy(v):
    mu=v.mean(0); h=-(mu*np.log(np.clip(mu,1e-8,1))).sum(); mi=h-np.mean([-(x*np.log(np.clip(x,1e-8,1))).sum() for x in v]); return mu.tolist(),float(h),float(mi)
def main():
    p=argparse.ArgumentParser();p.add_argument('--inputs',nargs='+',required=True);p.add_argument('--output',required=True);p.add_argument('--summary',required=True);a=p.parse_args(); paths=sorted(set(x for pat in a.inputs for x in glob.glob(pat))); assert len(paths)==3,paths; maps=[]
    for path in paths:
        d={}
        with gzip.open(path,'rt') as f:
            for line in f:
                q=json.loads(line); key=(q['task_id'],q['episode_id'],q['content_group_id'],q['split'],q['step']);
                if key in d: raise SystemExit('duplicate prediction key')
                d[key]=q
        maps.append(d)
    keys=set(maps[0]); assert all(set(d)==keys for d in maps[1:]),'seed key mismatch'; out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True); h=hashlib.sha256(); count=0
    with gzip.open(out,'wt') as f:
        for key in sorted(keys):
            qs=[d[key] for d in maps];
            if any(len({q[k] for q in qs})!=1 for k in ('node_gt','edge_type_gt','edge_id_gt')): raise SystemExit('GT mismatch')
            nm,neh,nmi=entropy(np.asarray([q['node_probs'] for q in qs],float)); em,eeh,emi=entropy(np.asarray([q['edge_type_probs'] for q in qs],float)); im,_,_=entropy(np.asarray([q['edge_id_probs'] for q in qs],float)); ph=np.asarray([q['phi_pred'] for q in qs]); co=np.asarray([q['remaining_cost_pred'] for q in qs]); q=qs[0]
            d={'episode_id':q['episode_id'],'content_group_id':q['content_group_id'],'task_id':q['task_id'],'scenario':q['scenario'],'split':q['split'],'step':q['step'],'node_gt':q['node_gt'],'edge_type_gt':q['edge_type_gt'],'edge_id_gt':q['edge_id_gt'],'phi_gt':q['phi_gt'],'remaining_cost_gt':q['remaining_cost_gt'],'node_probs_mean':nm,'edge_type_probs_mean':em,'edge_id_probs_mean':im,'node_pred':int(np.argmax(nm)),'edge_type_pred':int(np.argmax(em)),'edge_id_pred':int(np.argmax(im)),'node_predictive_entropy':neh,'node_mutual_information':nmi,'edge_predictive_entropy':eeh,'edge_mutual_information':emi,'phi_mean':float(np.clip(ph.mean(),0,1)),'phi_std':float(ph.std(ddof=1)),'remaining_cost_mean':float(max(0,co.mean())),'remaining_cost_std':float(co.std(ddof=1)),'per_seed_phi':ph.tolist(),'per_seed_remaining_cost':co.tolist()}; line=json.dumps(d,separators=(',',':'))+'\n'; f.write(line); h.update(line.encode()); count+=1
    Path(a.summary).write_text(json.dumps({'rows':count,'seed_inputs':paths,'prediction_sha256':h.hexdigest(),'statistics_unit':'content_group_id'},indent=2))
if __name__=='__main__': main()
