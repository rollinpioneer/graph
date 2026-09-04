#!/usr/bin/env python3
"""Run frozen three-seed PathGraph inference over every policy-data episode.

It emits only model-derived transition/chunk fields.  The weight normalisation is
fit separately per task using train chunks and then applied unchanged to validation.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

def read_jsonl(path): return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
def adapt(p):
    q=dict(p); q['per_seed_remaining_cost']=q.get('per_model_remaining_cost',[q['remaining_cost_mean']]); q['per_seed_phi']=q.get('per_model_phi',[q['phi_mean']]); return q

def main():
 p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--data',type=Path,required=True);p.add_argument('--inputs',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args()
 cfg=yaml.safe_load(a.config.read_text()); data=a.data.resolve(); inputs=a.inputs.resolve(); out=a.out.resolve(); out.mkdir(parents=True,exist_ok=True)
 sys.path.insert(0,str(inputs)); from code.reward_engine import PathGraphRewardEngine
 from tools.stage4.lib.ensemble import PathGraphEnsemble
 bundle=inputs/'configs/model_bundle_persistent.json'; model=PathGraphEnsemble.from_bundle(bundle,a.device)
 reward_cfg=yaml.safe_load((inputs/'configs/reward_config_v1.yaml').read_text()); engine=PathGraphRewardEngine(reward_cfg)
 rows=[]
 for manifest_name in ('train_manifest.jsonl','val_manifest.jsonl'):
  split=manifest_name.split('_')[0]
  for ep in read_jsonl(data/manifest_name):
   d=np.load(ep['episode_path']); obs=d['observations']; acts=d['action_applied']; stream=model.new_stream(ep['task_id']); state=engine.new_episode(ep['task_id'],ep['episode_id'],3); prev=None; per=[]
   for t,x in enumerate(obs):
    cur=adapt(stream.step(x))
    if prev is None:
     vals={'reward_mu':0.,'reward_std':0.,'reward_lcb':0.,'weight_positive':0.,'cost_component':0.,'phi_component':0.,'loop_penalty':0.,'recovery_cap_delta':0.,'edge_type_pred':-1,'edge_id_pred':-1,'node_confidence':0.,'edge_confidence':0.}
    else:
     r=engine.step(prev,cur,state); vals={'reward_mu':r.reward_mu,'reward_std':r.reward_std,'reward_lcb':r.reward_lcb,'weight_positive':r.weight_positive,'cost_component':r.cost_delta_mu,'phi_component':r.phi_delta_mu,'loop_penalty':r.loop_penalty,'recovery_cap_delta':r.failure_debt_after-r.failure_debt_before,'edge_type_pred':r.edge_type_pred,'edge_id_pred':r.edge_id_pred,'node_confidence':r.node_confidence,'edge_confidence':r.edge_confidence}
    per.append(vals); prev=cur
   for start in range(0,len(obs),int(cfg['policy']['action_chunk_size'])):
    end=min(len(obs),start+int(cfg['policy']['action_chunk_size'])); seg=per[start:end]
    row={'episode_id':ep['episode_id'],'content_group_id':ep['content_group_id'],'task_id':ep['task_id'],'scenario':ep['scenario'],'order':ep['order'],'split':split,'episode_path':ep['episode_path'],'t_start':start,'t_end':end,'action_count':end-start}
    for k in ('reward_mu','reward_std','reward_lcb','weight_positive','cost_component','phi_component','loop_penalty','recovery_cap_delta','node_confidence','edge_confidence'):row[k]=float(np.sum([z[k] for z in seg]))
    for k in ('edge_type_pred','edge_id_pred'):row[k]=int(seg[-1][k])
    rows.append(row)
 table=pd.DataFrame(rows)
 normalizers={}
 for task in sorted(table.task_id.unique()):
  train=table[(table.task_id==task)&(table.split=='train')]['weight_positive'].to_numpy(); q=float(np.quantile(train,.99)); normalizers[task]={'positive_q99':q,'train_nonzero_ratio':float(np.mean(train>0))}
  denom=max(q,1e-8); mask=table.task_id==task; table.loc[mask,'positive_norm']=np.clip(table.loc[mask,'weight_positive']/denom,0,1)
 gamma=0.75
 # Compatible non-graph transition rewards are derived from raw physical progress.
 # The canonical sequential baseline is frozen A->B, never selected using test outcomes.
 linear=[]; sequential=[]
 for row in table.itertuples(index=False):
  z=np.load(row.episode_path)['observations']; s,e=int(row.t_start),int(row.t_end)
  if row.task_id=='transport_recovery': progress=z[:,0]
  else: progress=z[:,0]+z[:,1]
  delta=float(np.maximum(0,np.diff(progress[max(0,s):e])).sum())
  linear.append(delta)
  sequential.append(delta if row.task_id=='transport_recovery' or row.order=='A_first' else 0.0)
 table['linear_transition_return']=linear; table['sequential_transition_return']=sequential
 sources={'bc_all':np.ones(len(table),dtype=float),'linear_sarm_equiv':np.asarray(linear),'sequential_transition':np.asarray(sequential),'pathgraph_reward_v1_locked':table['weight_positive'].to_numpy(float)}
 tables=[]; selection={}
 for method,raw in sources.items():
  mt=table.copy(); mt['method']=method; mt['transition_return']=raw; mt['raw_positive_weight']=np.maximum(0,raw)
  if method=='bc_all': mt['gamma']=0.; mt['train_q99']=1.; mt['normalized_weight']=1.
  else:
   chosen=None
   for candidate in (1.0,.75,.5):
    candidate_w=np.zeros(len(mt),dtype=float); per_task={}
    for task in sorted(mt.task_id.unique()):
     mask=(mt.task_id==task).to_numpy(); tr=mask&(mt.split.to_numpy()=='train'); q=float(np.quantile(mt.loc[tr,'raw_positive_weight'],.99)); u=np.clip(mt.loc[mask,'raw_positive_weight'].to_numpy()/max(q,1e-8),0,1); v=u**candidate; mean=float(v[tr[mask]].mean()) if np.any(tr[mask]) else 1.; candidate_w[mask]=np.clip(v/max(mean,1e-8),0,5); per_task[task]={'q99':q,'mean_norm':mean}
    trainmask=mt.split.to_numpy()=='train'; w=candidate_w[trainmask]; ess=(w.sum()**2)/(len(w)*(w*w).sum()) if len(w) and (w*w).sum() else 0.; rec=(mt.task_id.to_numpy()=='transport_recovery')&trainmask; coverage=float(np.mean(mt.raw_positive_weight.to_numpy()[rec]>0)) if np.any(rec) else 0.
    if ess>=.25 and float(np.mean(w==0))<=.85 and coverage>=.5 and np.isfinite(candidate_w).all(): chosen=(candidate,candidate_w,per_task,ess,coverage);break
   if chosen is None: raise RuntimeError(f'no train-only gamma satisfied weight gates for {method}')
   mt['gamma']=chosen[0]; mt['normalized_weight']=chosen[1]; mt['train_q99']=mt.task_id.map({k:v['q99'] for k,v in chosen[2].items()}); selection[method]={'gamma':chosen[0],'per_task':chosen[2],'train_ess_ratio':chosen[3],'recovery_positive_coverage':chosen[4]}
  tables.append(mt)
 combined=pd.concat(tables,ignore_index=True)
 (out/'chunk_weights').mkdir(exist_ok=True)
 for method,mt in combined.groupby('method'):
  pq.write_table(pa.Table.from_pandas(mt,preserve_index=False),out/'chunk_weights'/f'{method}.parquet')
 pq.write_table(pa.Table.from_pandas(combined, preserve_index=False),out/'chunk_weights.parquet')
 combined.to_csv(out/'chunk_weights.csv',index=False)
 lock={'locked':True,'selection_source':'train_weight_distribution_only','policy_test_used':False,'chunk_horizon':int(cfg['policy']['action_chunk_size']),'methods':selection,'bc_all':{'constant_weight':1.0}}
 (out/'weight_selection_lock.json').write_text(json.dumps(lock,indent=2)+'\n')
 (out/'linear_baseline_source.json').write_text(json.dumps({'source':'canonical_linear_progress_adapter','stage3_checkpoint_used':False,'reason':'new real-action environment observations have no Stage3 learned linear checkpoint adapter'},indent=2)+'\n')
 (out/'sequential_baseline_source.json').write_text(json.dumps({'source':'stage3 canonical sequential stage semantics','canonical_order':'A_first','policy_test_used':False},indent=2)+'\n')
 (out/'weight_normalizers.json').write_text(json.dumps({'selection':selection,'source':'frozen Stage5 PathGraph ensemble'},indent=2)+'\n')
 summary={'rows':len(combined),'base_chunks':len(table),'methods':combined.method.value_counts().to_dict(),'positive_weight_ratio':float(np.mean(table.weight_positive>0)),'finite':bool(np.isfinite(combined.select_dtypes(include=[np.number]).to_numpy()).all()),'normalizers':normalizers,'selection':selection}
 (out/'weight_inference_summary.json').write_text(json.dumps(summary,indent=2)+'\n'); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
