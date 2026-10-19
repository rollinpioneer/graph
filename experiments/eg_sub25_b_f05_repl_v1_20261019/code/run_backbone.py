import os,json,time,subprocess,resource,hashlib
from pathlib import Path
R=Path(os.environ['REPL_ROOT']);G=R.parent;C=G/'eg_sub25_v1_20261015/runtime/EconomicGrasp';start=time.time();used=0

def write(x):(R/'results/backbone_status.json').write_text(json.dumps(x,indent=2)+'\n')
def run(name,args,limit):
 global used
 t=time.time()
 with (R/f'logs/{name}.log').open('w') as f:
  p=subprocess.Popen(args,cwd=C,stdout=f,stderr=subprocess.STDOUT)
  write(dict(status='RUNNING',stage=name,pid=p.pid,controller_pid=os.getpid(),GPU=os.environ['CUDA_VISIBLE_DEVICES'],started=t,gpu_seconds_used=used))
  try:code=p.wait(timeout=limit)
  except subprocess.TimeoutExpired:
   p.terminate();p.wait(timeout=60);raise RuntimeError('Own job exceeded fixed GPU budget; stopped')
 dt=time.time()-t;used+=dt
 with (R/'results/resource_jobs.jsonl').open('a') as f:f.write(json.dumps(dict(name=name,kind='GPU',pid=p.pid,seconds=dt,returncode=code))+'\n')
 assert code==0,(name,code)
try:
 assert json.loads((R/'results/data_validation.json').read_text())['status']=='PASS'
 assert json.loads((R/'results/C3_exposure_audit.json').read_text())['status']=='PASS_KNOWN_RECORD_AUDIT'
 run('smoke',[os.sys.executable,str(R/'code/smoke.py')],600)
 assert json.loads((R/'results/smoke_status.json').read_text())['status']=='PASS'
 run('train_B',[os.sys.executable,str(C/'train.py'),'--dataset_root',str(G/'eg_sub25_v1_20261015/data/graspnet'),'--camera','kinect','--model','economicgrasp','--batch_size','4','--max_epoch','10','--learning_rate','0.001','--weight_decay','0','--num_point','20000','--voxel_size','0.005','--split_manifest',str(R/'B_frozen_split.json'),'--log_dir',str(R/'runs/B/train')],8*3600-used)
 import torch
 path=R/'runs/B/train/economicgrasp_epoch10.tar';ck=torch.load(path,map_location='cpu',weights_only=False)
 assert ck['epoch']==10 and ck['optimizer_updates']==16000 and ck['seed']==20261015
 assert all(torch.isfinite(v).all() for v in ck['model_state_dict'].values())
 rows=[json.loads(x) for x in (R/'runs/B/train/epoch_metrics.jsonl').read_text().splitlines()];assert len(rows)==10
 write(dict(status='PASS',epochs=10,optimizer_updates=16000,checkpoint=str(path),sha256=hashlib.sha256(path.read_bytes()).hexdigest(),gpu_seconds_used=used,seconds=time.time()-start))
except Exception as e:
 write(dict(status='TECHNICAL_STOP',error=repr(e),gpu_seconds_used=used,seconds=time.time()-start));raise
