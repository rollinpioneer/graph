#!/usr/bin/env python3
import argparse,csv,hashlib,json,os
from pathlib import Path
def h(p):
 x=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''): x.update(b)
 return x.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--model-bundle',required=True);p.add_argument('--output',required=True);a=p.parse_args(); d=json.load(open(a.model_bundle)); rows=[]
 for i in d['checkpoints']:
  pth=Path(i['path']); assert pth.is_file(); got=h(pth); assert got==i['sha256'],(pth,got,i['sha256']); rows.append({'seed':i['seed'],'path':str(pth),'size_bytes':pth.stat().st_size,'sha256':got,'history_steps':i.get('history_steps',32),'refinement':i.get('refinement','')})
 with open(a.output,'w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]),delimiter='\t');w.writeheader();w.writerows(rows)
 print('PERSISTENT_REWARD_CHECKPOINTS_OK'); [print(*r.values(),sep='\t') for r in rows]
if __name__=='__main__':main()
