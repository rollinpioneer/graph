"""Same-host fixed-input paired timing; no evaluator/quality truth in scorer."""
import os,json,time
from pathlib import Path
import numpy as np
from scoring import FrozenScorer,cache_inputs
from infer import EconomicExtractor

G=Path(os.environ['GRASP_ROOT']);R=Path(os.environ['LQR_ROOT']);C=json.loads((R/'frozen_config.json').read_text());old=G/'eg_sub25_v1_20261015';N=G/'eg_nqs_pilot_v2_20261018';head=C['heads'][0]
scorer=FrozenScorer(G/f'eg_pose_select_pilot_v1_20261018/heads/{head["seed"]}/step1000.pt',head['sha256'])
ex=EconomicExtractor(old/'runtime/EconomicGrasp',old/'data/graspnet',old/'runs/EG_SUB25_V1/train/economicgrasp_epoch10.tar',C['backbone_sha256'],'cuda:0');torch=ex.torch
from utils.collision_detector import ModelFreeCollisionDetector
from graspnetAPI import GraspGroup
sample=ex.sample(37,0,N/'inputs/scene_0037/0000.npz');pred,features,action,_=cache_inputs(N/'exports/subset/scene_0037/0000.npz');cloud,_=ex.data.get_data(37*256,return_raw_cloud=True)
samples={k:[] for k in ['scoring_only','EG','EG_score','EG_score_CD']}
def plain():
 batch={k:v.to(ex.device) for k,v in ex.collate_fn([sample]).items()}
 with torch.no_grad():return ex.decode(ex.net(dict(batch)))[0].cpu().numpy()
for iteration in range(25):
 t=time.perf_counter();scorer.apply(pred,features,action,'S');elapsed=time.perf_counter()-t
 if iteration>=5:samples['scoring_only'].append(elapsed)
 for handle in ex.handles:handle.remove()
 torch.cuda.synchronize();t=time.perf_counter();baseline=plain();torch.cuda.synchronize();elapsed=time.perf_counter()-t
 if iteration>=5:samples['EG'].append(elapsed)
 ex.install_hooks()
 for name in ['EG_score','EG_score_CD']:
  torch.cuda.synchronize();t=time.perf_counter();original,f,a,_=ex.extract(sample);updated,_,_=scorer.apply(original,f,a,'S')
  if name.endswith('_CD'):
   detector=ModelFreeCollisionDetector(cloud,voxel_size=.005);cd=detector.detect(GraspGroup(updated.copy()),approach_dist=.05,collision_thresh=.01);updated=updated[~cd]
  torch.cuda.synchronize();elapsed=time.perf_counter()-t
  if iteration>=5:samples[name].append(elapsed)
 print('LATENCY_ITERATION',iteration,flush=True)
result=dict(status='PASS',seed=head['seed'],batch=1,scene=37,frame=0,warmup=5,measured=20,gpu=os.environ.get('CUDA_VISIBLE_DEVICES'),head_device='CPU single thread',EG_device=torch.cuda.get_device_name(),IO_included=False,input='same frozen 20000-point sample and raw cloud; no evaluator',EG_scope='collate, transfer, original forward, decode and output transfer; no hooks',EG_score_scope='same input plus feature hooks, CPU feature transfer, normalization, head and score writeback',EG_score_CD_scope='EG_score plus full raw-cloud detector construction and filtering; excludes NMS/evaluation',gpu_synchronized=True,shared_hardware=True,seconds=samples,summary_ms={k:dict(median=float(np.median(v)*1000),p95=float(np.quantile(v,.95)*1000)) for k,v in samples.items()})
(R/'results/latency.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result['summary_ms']),flush=True)
