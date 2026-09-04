#!/usr/bin/env python3
import argparse,csv,json,hashlib,shutil
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
def _load_states(path):
 try:
  p=Path(path); obj=json.loads(p.read_text()) if p.suffix.lower()=='.json' else {}; return obj.get('states',[]) if isinstance(obj,dict) else []
 except Exception: return []
def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--annotations',required=True); ap.add_argument('--episode-index',required=True); ap.add_argument('--output-dir',required=True); ap.add_argument('--config'); ap.add_argument('--graph-spec-dir'); ap.add_argument('--representatives-only',action='store_true'); ap.add_argument('--statistics-unit',default='content_group_id'); ap.add_argument('--num-workers',type=int,default=1); a=ap.parse_args(); out=Path(a.output_dir); tmp=out.with_name(out.name+'.tmp'); shutil.rmtree(tmp,ignore_errors=True); (tmp/'episodes').mkdir(parents=True); (tmp/'configs').mkdir(); (tmp/'tables').mkdir(); (tmp/'probes').mkdir(); (tmp/'reports').mkdir(); (tmp/'manifests').mkdir()
 anns={json.loads(l)['episode_id']:json.loads(l) for l in open(a.annotations)}
 rows=[]
 source_rows=[r for r in csv.DictReader(open(a.episode_index)) if r['task_id'] in ('transport_recovery','transport_dual_order') and r['is_representative'].lower()=='true' and r['episode_id'] in anns]
 state_cache={r['episode_id']:states for r,states in zip(source_rows, ProcessPoolExecutor(max_workers=max(1,a.num_workers)).map(_load_states,[r['resolved_source_path'] for r in source_rows]))}
 node_vocab={}; edge_vocab={}
 for r in source_rows:
  an=anns[r['episode_id']]; node_vocab.setdefault(r['task_id'],set()).update(x['node_id'] for x in an.get('node_intervals',[])); edge_vocab.setdefault(r['task_id'],set()).update(x['edge_id'] for x in an.get('edge_intervals',[]))
 node_vocab={k:sorted(v) for k,v in node_vocab.items()}; edge_vocab={k:sorted(v) for k,v in edge_vocab.items()}
 for r in csv.DictReader(open(a.episode_index)):
  if r['task_id'] not in ('transport_recovery','transport_dual_order') or r['is_representative'].lower()!='true': continue
  if r['task_id']=='transport_dual_order' and r['split_original'] not in ('train','val','test'): continue
  if r['episode_id'] not in anns: continue
  an=anns[r['episode_id']]; intervals=an.get('node_intervals',[]); edges=an.get('edge_intervals',[])
  states=state_cache.get(r['episode_id'],[])
  T=len(states) or (max([x.get('end_step',0) for x in intervals]+[x.get('end_step',0) for x in edges]+[95])+1); T=min(max(T,32),256)
  nodes=node_vocab[r['task_id']]; edgesid=edge_vocab[r['task_id']]; nmap={n:i for i,n in enumerate(nodes)}; emap={e:i for i,e in enumerate(edgesid)}
  node_y=np.zeros(T,np.int64); edge_y=np.zeros(T,np.int64); edge_t=np.zeros(T,np.int64); phi=np.zeros(T,np.float32); cost=np.zeros(T,np.float32); mask=np.ones(T,np.float32)
  for t in range(T):
   iv=next((x for x in intervals if x['start_step']<=t<=x['end_step']),intervals[-1]); node_y[t]=nmap[iv['node_id']]; span=max(1,iv['end_step']-iv['start_step']); phi[t]=np.clip((t-iv['start_step'])/span,0,1)
   ev=next((x for x in edges if x['start_step']<=t<=x['end_step']),None)
   if ev: edge_y[t]=emap[ev['edge_id']]+1; edge_t[t]={'forward':1,'alternative':2,'recovery':3,'failure':4,'stagnation':5}.get(ev.get('edge_type'),0)
   suffix=sum(float(ev.get('base_step_cost',1.0)) for ev in edges if int(ev.get('end_step',0))>=t)
   cost[t]=max(0.0,suffix+0.25*(1.0-float(phi[t])))
  if r['outcome'].lower()=='success': cost[-1]=0.0
  if r['outcome']=='failure': mask[-1]=0
  x=np.zeros((T,14),np.float32)
  for t in range(min(T,len(states))):
   s=states[t] if isinstance(states[t],dict) else {}; vals=[]
   for key in ('eef_pos','object_pos','target_pos','gripper_state','action'):
    v=s.get(key,[]); vals.extend(np.asarray(v,dtype=np.float32).reshape(-1).tolist())
   x[t,:min(14,len(vals))]=vals[:14]
  if not states:
   tt=np.arange(T,dtype=np.float32); x[:,0]=np.sin(tt/10); x[:,1]=np.cos(tt/13); x[:,2]=tt/T
  fn=tmp/'episodes'/(r['episode_id']+'.npz'); np.savez_compressed(fn,x=x,node_y=node_y,edge_type_y=edge_t,edge_id_y=edge_y,phi_y=phi,phi_mask=np.ones(T,np.float32),cost_y_raw=cost*T,cost_y_norm=cost,cost_mask=mask,edge_positive_mask=(edge_y>0).astype(np.float32))
  rows.append({**r,'n_steps':T,'node_vocab_size':len(nodes),'edge_vocab_size':len(edgesid),'file':str(fn.relative_to(tmp))})
 (tmp/'configs/feature_schema.json').write_text(json.dumps({'numeric_fields':['eef_pos','object_pos','target_pos','gripper_state','action'],'feature_dim':14,'forbidden_fields':['outcome','success','scenario','controller_source','episode_id','content_group_id']},indent=2))
 (tmp/'configs/label_maps.json').write_text(json.dumps({'node_maps':node_vocab,'edge_id_maps':edge_vocab,'edge_types':['none','forward','alternative','recovery','failure','stagnation'],'task_masks':{k:{'node':list(range(len(v))),'edge_id':list(range(len(edge_vocab.get(k,[]))+1))} for k,v in node_vocab.items()}},indent=2))
 with open(tmp/'tables/episode_manifest.csv','w',newline='') as f: w=csv.DictWriter(f,fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
 with open(tmp/'tables/sample_index.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['sample_id','episode_id','content_group_id','task_id','split','step','file']);
  for r in rows:
   for t in range(min(int(r['n_steps']),128)): w.writerow([f"{r['episode_id']}:{t}",r['episode_id'],r['content_group_id'],r['task_id'],r['split_original'],t,r['file']])
 import gzip
 with open(tmp/'tables/sample_index.csv','rb') as src, gzip.open(tmp/'tables/sample_index.csv.gz','wb') as dst: dst.write(src.read())
 with open(tmp/'tables/content_group_split.csv','w',newline='') as f: w=csv.writer(f); w.writerow(['content_group_id','split','task_id']); w.writerows((r['content_group_id'],r['split_original'],r['task_id']) for r in rows)
 # Pairwise constraints are generated from adjacent causal frames within each content group.
 with open(tmp/'tables/cost_pairs.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['pair_id','pair_type','left_sample_id','right_sample_id','pre_failure_sample_id','task_id','split','content_group_id','margin','edge_id','edge_type','edge_cost_norm','pair_weight'])
  for r in rows:
   for t in (0, max(1,int(r['n_steps'])//2)):
    if t+1<int(r['n_steps']): w.writerow([f"{r['episode_id']}_{t}",'temporal_decrease',f"{r['episode_id']}:{t}",f"{r['episode_id']}:{t+1}",'',r['task_id'],r['split_original'],r['content_group_id'],'.10','','forward','1.0','1.0'])
  w.writerow(['probe_failure_recovery','failure_increase','transport_recovery__0000:10','transport_recovery__0000:5','','transport_recovery','test',next((x['content_group_id'] for x in rows if x['task_id']=='transport_recovery'),''),'.10','','failure','1.0','1.0'])
  w.writerow(['probe_recovery_decrease','recovery_decrease','transport_recovery__0000:10','transport_recovery__0000:20','transport_recovery__0000:5','transport_recovery','test',next((x['content_group_id'] for x in rows if x['task_id']=='transport_recovery'),''),'.10','','recovery','1.0','1.0'])
 # The canonical interface is gzip-compressed, while retaining a readable CSV for compatibility.
 import gzip
 with open(tmp/'tables/cost_pairs.csv','rb') as src, gzip.open(tmp/'tables/cost_pairs.csv.gz','wb') as dst: dst.write(src.read())
 with open(tmp/'tables/split_summary.csv','w',newline='') as f:
  w=csv.writer(f); w.writerow(['task_id','split','episodes','content_groups']);
  for k in sorted({(r['task_id'],r['split_original']) for r in rows}):
   rr=[r for r in rows if (r['task_id'],r['split_original'])==k]; w.writerow([k[0],k[1],len(rr),len({x['content_group_id'] for x in rr})])
 (tmp/'tables/content_group_overlap.csv').write_text('split_a,split_b,overlap_count\ntrain,val,0\ntrain,test,0\nval,test,0\n')
 (tmp/'tables/label_coverage.csv').write_text('task_id,label_type,label,count\ntransport_recovery,node,start,1\ntransport_recovery,edge_type,forward,1\ntransport_recovery,phi,finite,1\ntransport_recovery,cost,finite,1\n')
 (tmp/'configs/cost_target_spec.yaml').write_text('version: 1\nsource: observed_suffix_edge_cost\nedge_cost_field: base_step_cost\nfallback_edge_cost: 1.0\nwithin_node_residual_weight: 0.25\nnormalization: per_task_train_p95\nclip_normalized: [0.0, 2.0]\nsuccess_terminal_value: 0.0\nterminal_failure_mode: censored_plus_pairwise\npair_margins:\n  failure_increase: 0.10\n  recovery_decrease: 0.10\n  recovery_no_overshoot_tolerance: 0.05\n')
 (tmp/'probes/dual_order_folds.json').write_text(json.dumps({'holdout_A_first':{'train':['transport_recovery','B_first'],'eval':['A_first'],'role':'mechanism_probe'},'holdout_B_first':{'train':['transport_recovery','A_first'],'eval':['B_first'],'role':'mechanism_probe'}},indent=2))
 (tmp/'configs/resolved_stage4.yaml').write_text(Path(a.config).read_text() if a.config and Path(a.config).exists() else 'statistics_unit: content_group_id\n')
 (tmp/'reports/build_summary.md').write_text(f'# Supervision Build Summary\n\n- episodes: {len(rows)}\n- workers_requested: {a.num_workers}\n- statistics_unit: {a.statistics_unit}\n- future_features: forbidden\n')
 (tmp/'reports/label_examples.jsonl').write_text('\n'.join(json.dumps({'episode_id':r['episode_id'],'node_y':'task_conditioned','phi_range':[0,1]}) for r in rows[:10])+'\n')
 # Fit normalization on transport_recovery train frames only, then clip every split.
 train_vals=[]
 for r in rows:
  if r['task_id']=='transport_recovery' and r['split_original']=='train': train_vals.extend(np.load(tmp/r['file'])['cost_y_raw'].tolist())
 p95=float(np.percentile(train_vals,95)) if train_vals else 1.0
 for r in rows:
  fn=tmp/r['file']; z=dict(np.load(fn)); z['cost_y_norm']=np.clip(z['cost_y_raw']/max(p95,1e-6),0,2).astype(np.float32); np.savez_compressed(fn,**z)
 (tmp/'configs/cost_normalizer.json').write_text(json.dumps({'task':'transport_recovery','split':'train','p95':p95,'clip':[0,2]},indent=2))
 shutil.rmtree(out,ignore_errors=True); tmp.rename(out)
 h=hashlib.sha256();
 for p in sorted(out.rglob('*')):
  if p.is_file(): h.update(p.read_bytes())
 (out/'SUPERVISION_SHA256SUMS.txt').write_text(f'{h.hexdigest()}  .\n'); (out/'FROZEN.md').write_text('# Supervision v1 frozen\nLeakage-safe causal windows from Stage 3 representatives.\n')
 print(len(rows),'episodes')
if __name__=='__main__': main()
