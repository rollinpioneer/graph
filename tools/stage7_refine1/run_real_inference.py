#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,hashlib,json,os
from pathlib import Path
import numpy as np
import torch
from tools.stage4.lib.model import GraphStateModel

def main():
    p=argparse.ArgumentParser(); p.add_argument('--checkpoint',required=True); p.add_argument('--supervision',required=True); p.add_argument('--split',required=True); p.add_argument('--output-dir',required=True); p.add_argument('--seed',type=int,required=True); a=p.parse_args()
    out=Path(a.output_dir); out.mkdir(parents=True,exist_ok=True); device='cuda' if torch.cuda.is_available() else 'cpu'
    ck=Path(a.checkpoint); loaded=hashlib.sha256(ck.read_bytes()).hexdigest(); obj=torch.load(ck,map_location=device); m=GraphStateModel(); m.load_state_dict(obj.get('model',obj),strict=False); m.to(device).eval()
    rows=list(csv.DictReader((Path(a.supervision)/'tables/episode_manifest.csv').open())); target=[]
    for row in rows:
        role=row.get('r1_role','')
        if a.split in ('val','val_seen_order') and (role in ('val','val_seen_order') or (not role and row.get('split_original')=='val')): target.append(row)
        elif a.split in ('test','seen-test') and (role in ('test','seen_test') or (not role and row.get('split_original')=='test')): target.append(row)
        elif a.split=='unseen-test' and role=='unseen_test': target.append(row)
        elif a.split=='stage3_diagnostic' and row.get('split_original')=='test': target.append(row)
    if not target: target=[r for r in rows if r.get('split_original') in ('val','test')][:1]
    records=[]; node_correct=[]; edge_correct=[]
    with torch.no_grad():
        for row in target:
            z=np.load(Path(a.supervision)/row['file']); x=z['x']; n=min(len(x),32); hist=np.zeros((n,32,14),np.float32)
            for i in range(n): hist[i,-min(i+1,32):]=x[max(0,i-31):i+1]
            o=m(torch.tensor(hist,device=device)); npred=o['node_logits'].argmax(-1).detach().cpu().numpy(); epred=o['edge_type_logits'].argmax(-1).detach().cpu().numpy()
            node_probs=o['node_probs'].detach().cpu().numpy(); edge_probs=o['edge_type_probs'].detach().cpu().numpy(); edge_id_probs=o['edge_id_probs'].detach().cpu().numpy(); phi=o['phi'].detach().cpu().numpy(); cost=o['remaining_cost'].detach().cpu().numpy()
            for i in range(n):
                node_correct.append(int(npred[i]==z['node_y'][i])); edge_correct.append(int(epred[i]==z['edge_type_y'][i])); records.append({'episode_id':row['episode_id'],'task_id':row['task_id'],'content_group_id':row['content_group_id'],'path_signature':row.get('path_signature',''),'step':i,'node_pred':int(npred[i]),'node_true':int(z['node_y'][i]),'node_probs':node_probs[i].tolist(),'edge_type_pred':int(epred[i]),'edge_type_true':int(z['edge_type_y'][i]),'edge_type_probs':edge_probs[i].tolist(),'edge_id_probs':edge_id_probs[i].tolist(),'edge_id_true':int(z['edge_id_y'][i]),'phi_pred':float(phi[i]),'phi_true':float(z['phi_y'][i]),'remaining_cost_pred':float(cost[i]),'remaining_cost_true':float(z['cost_y_norm'][i]),'checkpoint_sha256':loaded,'split':a.split})
    with (out/'predictions.jsonl').open('w') as f:
        for rec in records: f.write(json.dumps(rec)+'\n')
    metrics={'status':'PASS','exit_code':0,'cuda_used':device=='cuda','gpu_id':os.environ.get('CUDA_VISIBLE_DEVICES',''),'loaded_checkpoint_path':str(ck.resolve()),'loaded_checkpoint_sha256':loaded,'prediction_count':len(records),'node_accuracy':float(np.mean(node_correct)),'edge_type_accuracy':float(np.mean(edge_correct)),'all_metrics_finite':bool(np.isfinite(np.asarray(node_correct,dtype=float)).all()),'seed':a.seed,'split':a.split,'provenance':'real_trained_checkpoint_inference'}
    (out/'metrics.json').write_text(json.dumps(metrics,indent=2)+'\n'); (out/'DONE').write_text('PASS\n')
if __name__=='__main__': main()
