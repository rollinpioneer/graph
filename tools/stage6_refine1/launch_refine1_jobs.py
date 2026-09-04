#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,os,subprocess,time
from pathlib import Path
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',type=Path,required=True);p.add_argument('--status',type=Path,required=True);p.add_argument('--max-parallel',type=int,default=8);a=p.parse_args();jobs=list(csv.DictReader(a.jobs.open(),delimiter='\t'));available=[str(i) for i in range(8)];running=[];done=[]
 while jobs or running:
  while jobs and len(running)<min(a.max_parallel,len(available)):
   j=jobs.pop(0);gpu=available[len(running)%len(available)];d=Path(j['output_dir']);d.mkdir(parents=True,exist_ok=True);f=(d/'stdout.log').open('w');env=dict(os.environ,CUDA_VISIBLE_DEVICES=gpu);started=time.time();proc=subprocess.Popen(j['command'],shell=True,stdout=f,stderr=subprocess.STDOUT,env=env);running.append((proc,j,gpu,f,started))
  remain=[]
  for proc,j,gpu,f,started in running:
   code=proc.poll()
   if code is None:remain.append((proc,j,gpu,f,started));continue
   f.close();done.append({**j,'gpu':gpu,'exit_code':code,'wall_seconds':round(time.time()-started,3),'status':'PASS' if code==0 else 'FAILED'})
  running=remain
  if running:time.sleep(.5)
 a.status.parent.mkdir(parents=True,exist_ok=True)
 with a.status.open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=done[0],delimiter='\t');w.writeheader();w.writerows(done)
 print(json.dumps({'jobs':len(done),'failed':sum(x['status']=='FAILED' for x in done)},indent=2));return 0 if all(x['status']=='PASS' for x in done) else 2
if __name__=='__main__':raise SystemExit(main())
