#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,random,time
from pathlib import Path
import numpy as np, yaml
import torch
from torch import nn
from tools.stage3.lib.data import build_feature_matrix, window_sequence, load_episode
FIELDS=['eef_pos','object_pos','target_pos','gripper_state','action']
CHAINS={'A_first':['start','A_done','B_done','success'],'B_first':['start','B_done','A_done','success'],'recovery_chain':['start','grasped','in_transit','placed','success']}
def rowsj(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def feat(raw): return build_feature_matrix(raw, forbidden_fields=['outcome','success','info.success','scenario','controller_source','episode_id'])
def nodeat(a,t): return next((x for x in a['node_intervals'] if x['start_step']<=t<=x['end_step']),a['node_intervals'][-1])
def labels(a,orientation):
 c=CHAINS[orientation]; out=[]; last=0
 for t in range(max(x['end_step'] for x in a['node_intervals'])+1):
  it=nodeat(a,t); n=it['node_id'];
  if n in c: last=c.index(n); phi=(t-it['start_step'])/max(1,it['end_step']-it['start_step'])
  else: phi=0
  out.append((last+phi)/(len(c)-1))
 return np.asarray(out,np.float32)
class Model(nn.Module):
 def __init__(self,d): super().__init__();self.proj=nn.Linear(d,128);self.gru=nn.GRU(128,128,batch_first=True);self.head=nn.Sequential(nn.Linear(128,128),nn.ReLU(),nn.Linear(128,1),nn.Sigmoid())
 def forward(self,x): return self.head(self.gru(torch.relu(self.proj(x)))[0][:,-1,:]).squeeze(-1)
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--config',type=Path,required=True);ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--suite-dir',type=Path,required=True);ap.add_argument('--task',required=True);ap.add_argument('--orientation',required=True);ap.add_argument('--seed',type=int,required=True);ap.add_argument('--output-dir',type=Path,required=True);a=ap.parse_args();a.output_dir.mkdir(parents=True,exist_ok=True);cfg=yaml.safe_load(a.config.read_text());random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
 anns={x['episode_id']:x for x in rowsj(a.adapter_dir/'runtime_episode_annotations.jsonl')}; idx={x['episode_id']:x for x in csv.DictReader((a.adapter_dir/'stage3_episode_index.csv').open()) if x['episode_id'] in anns}; scenario={'A_first':'order_A_then_B','B_first':'order_B_then_A','recovery_chain':'natural_success'}[a.orientation]
 train=[r for r in idx.values() if r['task_id']==a.task and r['scenario']==scenario and r['is_representative'].lower()=='true']
 if not train: raise SystemExit('no canonical training content group')
 fs=[];ys=[];episode_lengths=[]
 for r in train:
  raw=load_episode(r); x=feat(raw);fs.append(x);ys.append(labels(anns[r['episode_id']],a.orientation));episode_lengths.append(len(x))
 X=np.concatenate(fs,axis=0);Y=np.concatenate(ys,axis=0); mean=X.mean(0);std=X.std(0);std[std<1e-6]=1; X=(X-mean)/std; d=X.shape[1]; hist=int(cfg['features']['history_steps']); norm_fs=[(x-mean)/std for x in fs]; seq=np.concatenate([window_sequence(x,hist) for x in norm_fs],axis=0)
 episode_id=np.concatenate([np.full(n,i,dtype=np.int64) for i,n in enumerate(episode_lengths)])
 device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
 model=Model(d).to(device);opt=torch.optim.AdamW(model.parameters(),lr=cfg['training']['learning_rate'],weight_decay=1e-4);xt=torch.from_numpy(seq).to(device);yt=torch.from_numpy(Y).to(device);gid=torch.from_numpy(episode_id).to(device); metrics=[];start=time.time()
 for ep in range(int(cfg['training']['epochs'])):
  model.train();p=model(xt);base=nn.functional.smooth_l1_loss(p,yt);same=gid[1:]==gid[:-1];rank_pen=torch.relu(p[:-1]-p[1:])[same].mean() if bool(same.any()) else p.new_tensor(0.);loss=base+5.0*rank_pen;opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);opt.step();metrics.append({'epoch':ep+1,'loss':float(loss.detach().cpu()),'smooth_l1':float(base.detach().cpu()),'rank_penalty':float(rank_pen.detach().cpu())})
 with (a.output_dir/'train_metrics.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(metrics[0]));w.writeheader();w.writerows(metrics)
 (a.output_dir/'checkpoints').mkdir(exist_ok=True);torch.save({'state_dict':{k:v.detach().cpu() for k,v in model.state_dict().items()},'input_dim':d,'mean':mean,'std':std,'history_steps':hist,'orientation':a.orientation,'task':a.task,'seed':a.seed,'device':str(device),'train_episode_count':len(train)},a.output_dir/'checkpoints/best_or_final.pt')
 (a.output_dir/'feature_schema.json').write_text(json.dumps({'fields':FIELDS+['subgoal_A_done','subgoal_B_done'],'dim':d})+'\n');(a.output_dir/'normalizer.json').write_text(json.dumps({'mean':mean.tolist(),'std':std.tolist()})+'\n');(a.output_dir/'resolved_config.yaml').write_text(yaml.safe_dump({'task':a.task,'orientation':a.orientation,'seed':a.seed,'history_steps':hist,'epochs':cfg['training']['epochs']}));(a.output_dir/'metrics.json').write_text(json.dumps({'final_loss':metrics[-1]['loss'],'train_seconds':time.time()-start,'device':str(device),'train_episode_count':len(train),'train_rows':len(Y)})+'\n');(a.output_dir/'DONE').write_text('done\n')
if __name__=='__main__':main()
