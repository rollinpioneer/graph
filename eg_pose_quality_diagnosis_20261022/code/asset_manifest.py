from pathlib import Path
import json,hashlib,subprocess,importlib.metadata,time,os
import numpy as np
import scipy.io as sio
B=Path.home()/'work/grasp';ROOT=B/'eg_pose_quality_diagnosis_v1_20261022';D=B/'eg_sub25_v1_20261015/data/graspnet';SRC=B/'eg_full_repro_v1_20261015/runtime/EconomicGrasp'
plan=json.loads((ROOT/'study_plan.json').read_text());assets={};start=time.time()
def add(p,kind,source=None):
 p=Path(p)
 if str(p) in assets:return
 if not p.exists():assets[str(p)]={'path':str(p),'kind':kind,'missing':True};return
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 assets[str(p)]=dict(path=str(p),kind=kind,bytes=p.stat().st_size,sha256=h.hexdigest(),source=source)
objs=set();rawobjs=set()
for split,ss,ff in [('C3',plan['C3_scenes'],plan['C3_frames']),('train',plan['training_scenes'],plan['training_frames'])]:
 for s in ss:
  meta=sio.loadmat(D/f'scenes/scene_{s:04d}/kinect/meta/0000.mat');ids=meta['cls_indexes'].reshape(-1).astype(int)-1;objs.update(map(int,ids))
  if split=='train':
   rawobjs.update(map(int,ids[:2]));add(D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz','economic_labels','GraspNet raw labels + official generate_economic.py')
   add(D/f'collision_label/scene_{s:04d}/collision_labels.npz','scene_collision_grid','https://graspnet.net/datasets.html')
  for name in ['camera_poses.npy','cam0_wrt_table.npy']:add(D/f'scenes/scene_{s:04d}/kinect/{name}','camera_pose','https://graspnet.net/datasets.html')
  for f in ff:
   receipt=json.loads((ROOT/f'cache/{split}_{s:04d}_{f:04d}.json').read_text())
   for x in receipt['inputs']:assets[x['path']]=dict(**x,kind='frame_input',bytes=Path(x['path']).stat().st_size,source='https://graspnet.net/datasets.html')
for o in sorted(objs):
 add(D/f'models/{o:03d}/nontextured.ply','object_geometry','https://graspnet.net/datasets.html');add(D/f'dex_models/{o:03d}.pkl','dexnet_mesh_SDF_cache','Existing GraspNet evaluation assets')
for o in sorted(rawobjs):add(D/f'grasp_label/{o:03d}_labels.npz','raw_grasp_labels_sampled_for_preflight','https://graspnet.net/datasets.html')
add(B/'eg_full_repro_v1_20261015/downloads/economicgrasp_kinect.tar','official_weight','https://github.com/iSEE-Laboratory/EconomicGrasp/releases/download/v1/economicgrasp_kinect.tar')
for rel in ['models/economicgrasp.py','models/modules_economicgrasp.py','models/loss_economicgrasp.py','models/backbone.py','dataset/graspnet_dataset.py','dataset/generate_economic.py','utils/label_generation.py','utils/loss_utils.py','utils/collision_detector.py','utils/arguments.py','libs/knn/knn_modules.py']:
 add(SRC/rel,'source','https://github.com/iSEE-Laboratory/EconomicGrasp/blob/4119bdcd6bf5d3712a110f78ca87504dd359eec0/'+rel)
import graspnetAPI.utils.eval_utils as EU,graspnetAPI.graspnet_eval as GE
for p in [EU.__file__,GE.__file__]:add(p,'official_evaluation_source','https://github.com/graspnet/graspnetAPI')
versions={}
for pkg in ['torch','numpy','scipy','graspnetAPI','MinkowskiEngine','open3d']:
 try:versions[pkg]=importlib.metadata.version(pkg)
 except Exception:versions[pkg]='not_in_package_metadata'
out=dict(study_plan=plan,assets=list(assets.values()),missing=[x for x in assets.values() if x.get('missing')],source_commit='4119bdcd6bf5d3712a110f78ca87504dd359eec0',inference_source_git_diff='',versions=versions,official_training_config=dict(source='official README + utils/arguments.py; checkpoint contains epoch=10, not independent training logs',camera='kinect',epoch=10,batch_size=4,learning_rate=.001,training_scene_ids='0-99',num_points=20000,num_views=300,num_angles=12,num_depths=4),runtime=dict(model_is_training=False,tf32_inference=True,tf32_label_probe=False,GPU_count=1,training_updates=0,C4_C5_read=False),seconds=time.time()-start)
(ROOT/'asset_manifest.json').write_text(json.dumps(out,indent=2));print('COMPLETE',len(assets),'assets',len(out['missing']),'missing',out['seconds'],flush=True)
