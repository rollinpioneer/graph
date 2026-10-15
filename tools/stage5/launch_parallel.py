#!/usr/bin/env python3
import argparse,json,os,subprocess,csv,time
def main():
 p=argparse.ArgumentParser();p.add_argument('--jobs',required=True);p.add_argument('--gpu-ids-file');p.add_argument('--status',required=True);p.add_argument('--logs-dir',required=True);p.add_argument('--max-jobs-per-gpu',type=int,default=1);a=p.parse_args();jobs=json.load(open(a.jobs));g=[x.strip() for x in open(a.gpu_ids_file)] if a.gpu_ids_file else ['0'];os.makedirs(a.logs_dir,exist_ok=True);rows=[]
 for i,j in enumerate(jobs):
  gpu=g[i%len(g)];env=os.environ.copy();env['CUDA_VISIBLE_DEVICES']=gpu;start=time.time();lp=open(os.path.join(a.logs_dir,j['job_id']+'.log'),'w');r=subprocess.run(['bash','-lc',j['command']],env=env,stdout=lp,stderr=subprocess.STDOUT);lp.close();rows.append({'job_id':j['job_id'],'gpu':gpu,'start':start,'end':time.time(),'exit_code':r.returncode,'output_dir':j['output_dir']})
 with open(a.status,'w',newline='') as f:w=csv.DictWriter(f,fieldnames=rows[0].keys());w.writeheader();w.writerows(rows)
 if any(x['exit_code'] for x in rows):raise SystemExit(1)
if __name__=='__main__':main()
