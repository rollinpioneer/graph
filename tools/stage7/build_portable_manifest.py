#!/usr/bin/env python3
import argparse,csv,hashlib,os
from pathlib import Path
import yaml
def h(p):
 x=hashlib.sha256();
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):x.update(b)
 return x.hexdigest()
def main():
 p=argparse.ArgumentParser();p.add_argument('--input-index',required=True);p.add_argument('--stage6r1-root',required=True);p.add_argument('--portable-output',required=True);p.add_argument('--omitted-output',required=True);a=p.parse_args(); d=yaml.safe_load(open(a.input_index)); paths=[]
 def walk(x):
  if isinstance(x,str) and os.path.isfile(x): paths.append(Path(x))
  elif isinstance(x,dict):
   for v in x.values():walk(v)
  elif isinstance(x,list):
   for v in x:walk(v)
 walk(d)
 paths=sorted(set(paths)); portable=[]; omitted=[]
 for pth in paths:
  size=pth.stat().st_size; row=f'{h(pth)}  {pth}\n'
  if size <= 200*1024*1024: portable.append(row)
  else: omitted.append((str(pth),size,h(pth),'large_file','required_for_full_recompute'))
 root=Path(a.stage6r1_root)
 for pth in sorted(root.rglob('*')):
  if pth.is_file() and pth.stat().st_size>200*1024*1024: omitted.append((str(pth),pth.stat().st_size,h(pth),'large_file','default_omit_from_zip'))
 Path(a.portable_output).parent.mkdir(parents=True,exist_ok=True);Path(a.portable_output).write_text(''.join(portable))
 with open(a.omitted_output,'w',newline='') as f:
  w=csv.writer(f,delimiter='\t');w.writerow(['path','size_bytes','sha256','reason','required_for_full_recompute']);w.writerows(omitted)
 print('portable_files=',len(portable),'omitted_large_files=',len(omitted))
if __name__=='__main__':main()
