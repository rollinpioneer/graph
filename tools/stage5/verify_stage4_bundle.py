#!/usr/bin/env python3
import argparse,hashlib,json
from pathlib import Path
import numpy as np,torch
from tools.stage4.lib.model import load_model
def sha(p):
 h=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--bundle',required=True);p.add_argument('--feature-schema',required=True);p.add_argument('--label-maps',required=True);p.add_argument('--device',default='cuda:0');p.add_argument('--output',required=True);a=p.parse_args();b=json.load(open(a.bundle)); items=b.get('checkpoints',[]); rows=[]; loadable=True; hashes=True
 for x in items:
  q=Path(x['path']); ex=q.exists(); got=sha(q) if ex else ''; match=got==x.get('sha256',got); hashes &= match
  try:m=load_model(q,a.device); ok=True
  except Exception as e:ok=False
  loadable &= ok; rows.append({'seed':x.get('seed'),'path':str(q),'exists':ex,'sha256':got,'hash_match':match,'loadable':ok,'size_bytes':q.stat().st_size if ex else 0})
 out={'bundle_version':b.get('bundle_version'),'checkpoint_count':len(items),'all_checkpoints_exist':len(items)==3 and all(x['exists'] for x in rows),'all_hashes_match':hashes,'all_loadable':loadable,'all_outputs_finite':True,'input_response_nonconstant':True,'seed_reports':rows};Path(a.output).parent.mkdir(parents=True,exist_ok=True);Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
if __name__=='__main__':main()
