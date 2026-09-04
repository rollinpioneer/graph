#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json
from pathlib import Path
import numpy as np, torch
from train_linear_sarm import Model,feat,nodeat,CHAINS
from tools.stage3.lib.linearization import compute_within_node_phi
def rowsj(p): return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--resolved-config',type=Path,required=True);ap.add_argument('--adapter-dir',type=Path,required=True);ap.add_argument('--suite-dir',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);m=Model(ck['input_dim']);m.load_state_dict(ck['state_dict']);m.eval();anns={x['episode_id']:x for x in rowsj(a.adapter_dir/'runtime_episode_annotations.jsonl')};ds=list(csv.DictReader((a.suite_dir/'tables/diagnostic_episodes.csv').open()));lines=[]
 for d in ds:
  if d['task_id']!=ck['task']:continue
  raw=json.loads(Path(d['source_path']).read_text());x=(feat(raw)-ck['mean'])/ck['std'];h=ck['history_steps'];seq=np.stack([x[max(0,i-h+1):i+1] if i>=h-1 else np.pad(x[:i+1],((h-i-1,0),(0,0))) for i in range(len(x))]);p=m(torch.from_numpy(seq)).detach().numpy();an=anns[d['episode_id']]
  for t,v in enumerate(p):
   it=nodeat(an,t);e=next((z for z in an['edge_intervals'] if z['end_step']==t),None); chain=CHAINS[ck['orientation']]; rank=chain.index(it['node_id']) if it['node_id'] in chain else None; phi=compute_within_node_phi(t,it,an.get('progress_anchors'));lines.append({'episode_id':d['episode_id'],'task_id':d['task_id'],'scenario':d['scenario'],'split':d['split_original'],'content_group_id':d['content_group_id'],'method':'learned_linear_sarm','orientation':ck['orientation'],'seed':ck['seed'],'step':t,'node_id_runtime':it['node_id'],'edge_id_runtime':e['edge_id'] if e else None,'semantic_edge_type':e['edge_type'] if e else None,'progress':float(v),'reward_delta':0.0 if t==0 else float(v-p[t-1]),'stage_rank':rank,'within_node_phi':phi,'uncertainty':None,'controller_source':d['controller_source']})
 a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(''.join(json.dumps(x)+'\n' for x in lines))
if __name__=='__main__':main()
