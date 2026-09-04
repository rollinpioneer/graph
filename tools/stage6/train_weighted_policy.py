#!/usr/bin/env python3
"""The sole Stage-6 policy trainer: identical batch sequence, weighted loss only."""
from __future__ import annotations
import argparse,hashlib,json,os,random,traceback
from pathlib import Path
import numpy as np,pyarrow.parquet as pq,torch,yaml
from tools.stage6.policy_model import ChunkPolicy

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def build_samples(data_root, weights, task, method, horizon):
 w=pq.read_table(weights).to_pandas();w=w[(w.task_id==task)&(w.method==method)].copy();assert not w.duplicated(['episode_id','t_start']).any()
 out=[]
 for r in w.itertuples(index=False):
  z=np.load(r.episode_path); o=z['observations']; a=z['action_applied'];s,e=int(r.t_start),int(r.t_end); target=np.zeros((horizon,a.shape[1]),np.float32);mask=np.zeros(horizon,np.float32);n=e-s;target[:n]=a[s:e];mask[:n]=1;out.append((r.split,o[s],target,mask,float(r.normalized_weight),r.episode_id,s))
 return out,sha(weights)
def as_arrays(samples,split):
 q=[x for x in samples if x[0]==split];return (np.stack([x[1] for x in q]),np.stack([x[2] for x in q]),np.stack([x[3] for x in q]),np.asarray([x[4] for x in q],np.float32),q)
def main():
 p=argparse.ArgumentParser();p.add_argument('--task',required=True);p.add_argument('--method',required=True);p.add_argument('--policy-seed',type=int,required=True);p.add_argument('--config',type=Path,required=True);p.add_argument('--dataset-root',type=Path,required=True);p.add_argument('--weight-file',type=Path,required=True);p.add_argument('--init-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-optimizer-steps',type=int,required=True);p.add_argument('--device',default='cuda:0');a=p.parse_args(); out=a.output_dir.resolve();(out/'checkpoints').mkdir(parents=True,exist_ok=True);(out/'metrics').mkdir(exist_ok=True)
 try:
  c=yaml.safe_load(a.config.read_text());dev=torch.device(a.device if torch.cuda.is_available() else 'cpu');torch.manual_seed(a.policy_seed);np.random.seed(a.policy_seed);random.seed(a.policy_seed)
  samples,weight_sha=build_samples(a.dataset_root,a.weight_file,a.task,a.method,int(c['policy']['action_chunk_size']));tx,ty,tm,tw,_=as_arrays(samples,'train');vx,vy,vm,vw,_=as_arrays(samples,'val'); mean=tx.mean(0);std=tx.std(0);std=np.maximum(std,1e-4);tx=(tx-mean)/std;vx=(vx-mean)/std
  m=ChunkPolicy(c['policy']['obs_dim'],c['policy']['action_dim'],c['policy']['action_chunk_size'],c['policy']['hidden_dim']).to(dev); init=torch.load(a.init_checkpoint,map_location='cpu');m.load_state_dict(init['model']);opt=torch.optim.AdamW(m.parameters(),lr=float(c['policy']['learning_rate']))
  g=torch.Generator().manual_seed(a.policy_seed+101);order=torch.randperm(len(tx),generator=g).tolist();cursor=0;bs=int(c['policy']['batch_size']);best=float('inf');records=[]
  with (out/'train_metrics.jsonl').open('w') as trainlog, (out/'val_metrics.jsonl').open('w') as vallog:
   for step in range(1,a.max_optimizer_steps+1):
    if cursor+bs>len(order): order=torch.randperm(len(tx),generator=g).tolist();cursor=0
    ix=np.asarray(order[cursor:cursor+bs]);cursor+=bs; x=torch.tensor(tx[ix],device=dev);y=torch.tensor(ty[ix],device=dev);mask=torch.tensor(tm[ix],device=dev);weights=torch.tensor(tw[ix],device=dev)
    pred=m(x);per=((pred-y).square().mean(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1);loss=(per*weights).sum()/weights.sum().clamp_min(1e-8);opt.zero_grad(set_to_none=True);loss.backward();torch.nn.utils.clip_grad_norm_(m.parameters(),1.0);opt.step()
    if step==1 or step%100==0 or step==a.max_optimizer_steps:
     rec={'optimizer_step':step,'train_weighted_action_loss':float(loss.detach().cpu()),'weight_mean':float(weights.mean().cpu()),'weight_std':float(weights.std(unbiased=False).cpu()),'cuda_used':dev.type=='cuda'};trainlog.write(json.dumps(rec)+'\n');trainlog.flush()
    if step%200==0 or step==a.max_optimizer_steps:
     m.eval();vals=[]
     with torch.no_grad():
      for s in range(0,len(vx),bs):
       x=torch.tensor(vx[s:s+bs],device=dev);y=torch.tensor(vy[s:s+bs],device=dev);mask=torch.tensor(vm[s:s+bs],device=dev);per=((m(x)-y).square().mean(-1)*mask).sum(-1)/mask.sum(-1).clamp_min(1);vals.extend(per.cpu().tolist())
     val=float(np.mean(vals));vr={'optimizer_step':step,'val_action_loss':val};vallog.write(json.dumps(vr)+'\n');vallog.flush();records.append(vr);payload={'model':m.state_dict(),'optimizer_step':step,'normalizer_mean':mean,'normalizer_std':std,'task_id':a.task,'method':a.method,'policy_seed':a.policy_seed,'init_sha256':sha(a.init_checkpoint),'weight_sha256':weight_sha}
     torch.save(payload,out/'checkpoints/latest.pt')
     if val<best:best=val;torch.save(payload,out/'checkpoints/best_val.pt')
     m.train()
  status={'status':'PASS','task_id':a.task,'method':a.method,'policy_seed':a.policy_seed,'optimizer_steps':a.max_optimizer_steps,'best_val_action_loss':best,'cuda_used':dev.type=='cuda','cuda_visible_devices':os.getenv('CUDA_VISIBLE_DEVICES',''),'init_sha256':sha(a.init_checkpoint),'weight_sha256':weight_sha,'dataset_samples_train':len(tx),'dataset_samples_val':len(vx),'weight_join_coverage':1.0,'weighted_sampler_used':False,'checkpoint_selection':'val_action_loss_min'}
  (out/'resolved_config.yaml').write_text(yaml.safe_dump(c,sort_keys=True));(out/'status.json').write_text(json.dumps(status,indent=2)+'\n');print(json.dumps(status,indent=2))
 except Exception as e:
  (out/'status.json').write_text(json.dumps({'status':'FAILED','error':str(e),'traceback':traceback.format_exc()},indent=2)+'\n');raise
if __name__=='__main__':main()
