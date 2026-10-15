#!/usr/bin/env python3
import argparse,json,os,subprocess,csv
def main():
 p=argparse.ArgumentParser(); p.add_argument('--jobs',required=True); p.add_argument('--gpu-ids-file',required=True); p.add_argument('--logs-dir',required=True); p.add_argument('--status-csv',required=True); p.add_argument('--max-jobs-per-gpu',type=int,default=1); a=p.parse_args(); os.makedirs(a.logs_dir,exist_ok=True)
 jobs=[json.loads(x) for x in open(a.jobs) if x.strip()]; g=[x.strip() for x in open(a.gpu_ids_file) if x.strip()] or ['0']; rows=[]
 for i,j in enumerate(jobs):
  gpu=g[i%len(g)]; env=os.environ.copy(); env['CUDA_VISIBLE_DEVICES']=gpu; log=open(os.path.join(a.logs_dir,j['job_id']+'.log'),'w'); r=subprocess.run(['bash','-lc',j['command']],env=env,stdout=log,stderr=subprocess.STDOUT); log.close(); rows.append({'job_id':j['job_id'],'physical_gpu_id':gpu,'exit_code':r.returncode,'output_dir':j['output_dir'],'log':os.path.join(a.logs_dir,j['job_id']+'.log')})
 with open(a.status_csv,'w',newline='') as f: w=csv.DictWriter(f,fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)
 if any(r['exit_code'] for r in rows): raise SystemExit(1)
if __name__=='__main__': main()
