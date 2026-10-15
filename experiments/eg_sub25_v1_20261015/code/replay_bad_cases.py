import os,sys,json,random,time,hashlib
from pathlib import Path
import numpy as np
import torch
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OUT=R/'runs/EG_SUB25_V1/evaluation';BAD=R/'runs/EG_SUB25_V1/bad_cases';BAD.mkdir(exist_ok=True)
cfg=json.loads((D/'evaluation_run_config.json').read_text());cases=json.loads((D/'BAD_CASE_SELECTION.json').read_text())['cases'];assert len(cases)==12
sys.argv=[sys.argv[0],'--dataset_root',str(R/'data/graspnet'),'--camera','kinect','--model','economicgrasp','--batch_size','4','--split_manifest',str(D/'frozen_split.json')]
from models.economicgrasp import economicgrasp,pred_decode
from dataset.graspnet_dataset import collate_fn
os.nice(10);records=[]
keys=['objectness_score','graspness_score','xyz_graspable','view_score','grasp_top_view_xyz','grasp_top_view_inds','grasp_angle_pred','grasp_depth_pred','grasp_score_pred','grasp_width_pred']
for role in ['subset','official']:
 random.seed(20261015);np.random.seed(20261015);torch.manual_seed(20261015);torch.cuda.manual_seed_all(20261015)
 net=economicgrasp(seed_feat_dim=512,is_training=False);checkpoint=torch.load(cfg['models'][role]['path'],map_location='cpu',weights_only=False);net.load_state_dict(checkpoint['model_state_dict'],strict=True);del checkpoint;net.cuda().eval()
 groups={}
 for case in cases:groups.setdefault((case['partition'],case['scene'],case['frame']//4*4),[]).append(case)
 for (part,scene,start),selected in groups.items():
  source=[];hashes=[]
  for frame in range(start,start+4):
   p=OUT/'inputs'/part/f'scene_{scene:04d}/kinect/{frame:04d}.npz'
   hashes.append(hashlib.sha256(p.read_bytes()).hexdigest())
   with np.load(p) as z:source.append({k:z[k] for k in z.files})
  batch={k:v.cuda() for k,v in collate_fn(source).items()}
  with torch.no_grad():end=net(batch);predictions=pred_decode(end)
  for case in selected:
   j=case['frame']-start;dest=BAD/f'scene_{scene:04d}_frame_{case["frame"]:04d}';dest.mkdir(exist_ok=True)
   original=np.load(OUT/'predictions'/role/part/f'scene_{scene:04d}/kinect/{case["frame"]:04d}.npy');replay=predictions[j].cpu().numpy();same=bool(np.allclose(original,replay,rtol=2e-4,atol=2e-5));delta=float(np.max(np.abs(original-replay)))
   tensors={k:end[k][j].detach().cpu().numpy() for k in keys};tensors.update(source[j]);tensors['replay_prediction']=replay
   p=dest/(role+'_intermediates.npz');assert not p.exists();np.savez(p,**tensors)
   records.append({'scene':scene,'frame':case['frame'],'model':role,'original_batch_frames':list(range(start,start+4)),'input_sha256':hashes[j],'same_input_and_batch_as_inference':True,'replay_prediction_allclose':same,'replay_max_abs_difference':delta,'intermediate_sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'intermediate_shapes':{k:list(v.shape) for k,v in tensors.items()}})
  del batch,end,predictions
 del net;torch.cuda.empty_cache()
result={'status':'PASS','records':records,'all24_replays_match':all(x['replay_prediction_allclose'] for x in records),'rule':'Original saved predictions and official evaluations remain authoritative; nonmatching replays cannot prove the original internal mechanism'}
(D/'BAD_CASE_REPLAY.json').write_text(json.dumps(result,indent=2));print('REPLAY_COMPLETE',len(records),'ALL_MATCH',result['all24_replays_match'],flush=True)
