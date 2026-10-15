import os,sys,json,time,argparse,hashlib,concurrent.futures,contextlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1'
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OUT=R/'runs/EG_SUB25_V1/evaluation'
def task(spec):
 import numpy as np
 from graspnetAPI import GraspNetEval
 role,part,scene=spec;target=OUT/'results'/role/part/f'scene_{scene:04d}';target.mkdir(parents=True,exist_ok=True)
 done=target/'COMPLETE.json'
 if done.exists():
  x=json.loads(done.read_text());assert x['status']=='PASS' and hashlib.sha256((target/'ap.npy').read_bytes()).hexdigest()==x['ap_sha256'];return x
 assert not (target/'ap.npy').exists(),'Preserve unfinished output for diagnosis'
 started=time.time();dump=OUT/'predictions'/role/part;assert (dump/'INFERENCE_COMPLETE.json').exists()
 ge=GraspNetEval(root=str(R/'data/graspnet'),camera='kinect',split='all')
 with (target/'official_api.log').open('w') as log,contextlib.redirect_stdout(log):res,grasps,quality,collision=ge.eval_scene(scene,str(dump),TOP_K=50,return_list=True,vis=False,max_width=.1)
 ap=np.asarray(res,dtype=np.float64);assert ap.shape==(256,50,6) and np.isfinite(ap).all() and np.all((ap>=0)&(ap<=1))
 assert len(grasps)==len(quality)==len(collision)==256
 evaluated=target/'evaluated';evaluated.mkdir(exist_ok=True);identities=[]
 for f in range(256):
  g=np.asarray(grasps[f],dtype=np.float64).reshape(-1,17);q=np.asarray(quality[f],dtype=np.float64);c=np.asarray(collision[f],dtype=bool);assert len(g)==len(q)==len(c)
  valid=(q[:,None]>0)&(q[:,None]<=np.array([.2,.4,.6,.8,1.,1.2])[None,:]);cum=np.cumsum(valid,axis=0);precision=np.zeros((50,6))
  if len(q):precision[:min(50,len(q))]=cum[:50];precision[min(50,len(q)):]=cum[-1]
  precision/=np.arange(1,51)[:,None];assert np.array_equal(precision,ap[f]),(scene,f,'serialized quality AP mismatch')
  p=evaluated/f'{f:04d}.npz';assert not p.exists();np.savez(p,grasps=g,quality=q,collision=c);identities.append(hashlib.sha256(p.read_bytes()).hexdigest())
 np.save(target/'ap.npy',ap)
 x={'status':'PASS','model':role,'partition':part,'scene':scene,'frames':256,'ap_percent':float(ap.mean()*100),'ap04_percent':float(ap[:,:,1].mean()*100),'ap08_percent':float(ap[:,:,3].mean()*100),'wall_seconds':time.time()-started,'ap_sha256':hashlib.sha256((target/'ap.npy').read_bytes()).hexdigest(),'evaluated_manifest_sha256':hashlib.sha256(''.join(identities).encode()).hexdigest(),'all_frame_AP_recomputed_from_official_returned_quality_exact':True}
 done.write_text(json.dumps(x,indent=2));return x
if __name__=='__main__':
 import inspect
 from graspnetAPI import GraspNetEval
 parser=argparse.ArgumentParser();parser.add_argument('--partition');parser.add_argument('--role');args=parser.parse_args();os.nice(10)
 cfg=json.loads((D/'evaluation_run_config.json').read_text());specs=[(args.role,args.partition,s) for s in cfg['partitions'][args.partition]]
 with concurrent.futures.ProcessPoolExecutor(max_workers=cfg['api_eval_processes']) as pool:
  futures=[pool.submit(task,s) for s in specs]
  for future in concurrent.futures.as_completed(futures):print('SCENE_COMPLETE',json.dumps(future.result()),flush=True)
 print('PARTITION_COMPLETE',args.role,args.partition,flush=True)
