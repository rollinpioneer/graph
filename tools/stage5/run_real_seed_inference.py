#!/usr/bin/env python3
import argparse,csv,gzip,json,hashlib
from pathlib import Path
import numpy as np,torch
from tools.stage4.lib.model import load_model
def main():
 p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--supervision-dir',required=True);p.add_argument('--episode-manifest',required=True);p.add_argument('--split',required=True);p.add_argument('--task-id',default='all');p.add_argument('--history-steps',type=int,default=32);p.add_argument('--stride',type=int,default=1);p.add_argument('--batch-size',type=int,default=512);p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);a=p.parse_args();dev=a.device if a.device.startswith('cuda') and torch.cuda.is_available() else 'cpu';m=load_model(a.checkpoint,dev);root=Path(a.supervision_dir); rows=[]
 for r in csv.DictReader(open(a.episode_manifest)):
  if r['split_original']!=a.split or (a.task_id!='all' and r['task_id']!=a.task_id):continue
  z=np.load(root/r['file']);x0=z['x'];
  for t in range(0,len(x0),max(1,a.stride)):
   w=np.zeros((a.history_steps,14),np.float32);lo=max(0,t-a.history_steps+1);w[-(t-lo+1):]=x0[lo:t+1];rows.append((r,z,t,w))
 out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);h=hashlib.sha256()
 with gzip.open(out,'wt',encoding='utf-8') as f:
  for i in range(0,len(rows),a.batch_size):
   batch=rows[i:i+a.batch_size]; xx=torch.from_numpy(np.stack([q[3] for q in batch])).to(dev)
   with torch.no_grad():o=m(xx)
   for j,(r,z,t,w) in enumerate(batch):
    d={'episode_id':r['episode_id'],'content_group_id':r['content_group_id'],'task_id':r['task_id'],'scenario':r['scenario'],'split':a.split,'controller_source':'real_checkpoint','step':t,'seed':a.seed,'node_gt':int(z['node_y'][t]),'edge_type_gt':int(z['edge_type_y'][t]),'edge_id_gt':int(z['edge_id_y'][t]),'phi_gt':float(z['phi_y'][t]),'remaining_cost_gt':float(z['cost_y_norm'][t]),'node_probs':o['node_probs'][j].detach().cpu().tolist(),'edge_type_probs':o['edge_type_probs'][j].detach().cpu().tolist(),'edge_id_probs':o['edge_id_probs'][j].detach().cpu().tolist(),'node_pred':int(o['node_probs'][j].argmax()),'edge_type_pred':int(o['edge_type_probs'][j].argmax()),'edge_id_pred':int(o['edge_id_probs'][j].argmax()),'phi_pred':float(o['phi'][j].cpu()),'remaining_cost_pred':float(o['remaining_cost'][j].cpu())}
    line=(json.dumps(d,separators=(',',':'))+'\n').encode();h.update(line);f.write(line.decode())
 (out.with_suffix(out.suffix+'.sha256')).write_text(h.hexdigest()+'  '+out.name+'\n');(out.parent/'DONE').write_text('checkpoint_forward_complete\n')
if __name__=='__main__':main()
