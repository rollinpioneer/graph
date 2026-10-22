import os,sys,json,time,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial import cKDTree
import scipy.io as sio
B=Path.home()/'work/grasp';ROOT=B/'eg_pose_quality_diagnosis_v1_20261022';D=B/'eg_sub25_v1_20261015/data/graspnet'
plan=json.loads((ROOT/'study_plan.json').read_text());mode=sys.argv[1];scenes=plan['training_scenes'][:1] if mode=='pilot' else plan['training_scenes']
rows=[]
for s in scenes:
 t=time.time();meta=sio.loadmat(D/f'scenes/scene_{s:04d}/kinect/meta/0000.mat');objs=meta['cls_indexes'].reshape(-1).astype(int)-1
 with np.load(D/f'economic_grasp_label_300views/scene_{s:04d}_labels.npz') as z:a={k:z[k] for k in z.files}
 coll=np.load(D/f'collision_label/scene_{s:04d}/collision_labels.npz')
 for oi in range(min(2,len(objs))):
  ids=np.flatnonzero(a['pointid']==oi);ids=ids[np.linspace(0,len(ids)-1,min(8,len(ids))).astype(int)]
  o=objs[oi]
  with np.load(D/f'grasp_label/{o:03d}_labels.npz') as raw:
   dist,idx=cKDTree(raw['points']).query(a['points'][ids]);assert dist.max()<1e-8
   scores=raw['scores'][idx].copy();width=raw['offsets'][idx,:,:,:,2].copy();cm=coll[f'arr_{oi}'][idx]
  scores[cm]=0;scores[scores<0]=0
  valid=(scores>0)&(width<=.1);scores[~valid]=0;po=scores>0;scores[po]=1.1-scores[po]
  maxd=scores.max(-1);dep=scores.argmax(-1);ang=maxd.argmax(-1);dep=dep[np.arange(len(ids))[:,None],np.arange(300)[None,:],ang]
  value=maxd.max(-1);ww=width[np.arange(len(ids))[:,None],np.arange(300)[None,:],ang,dep]
  slot=np.argsort(a['topview'][ids],axis=1)
  actual={k:np.take_along_axis(a[k][ids],slot,axis=1) for k in ['rotations','depth','scores','widths']}
  expected={'rotations':ang,'depth':dep,'scores':(value*10).astype(np.uint8),'widths':(ww*1000).astype(np.uint8)}
  mismatch={k:int((actual[k]!=expected[k]).sum()) for k in expected}
  assert not any(mismatch.values()),(s,int(o),mismatch)
  rows.append(dict(scene=s,object_id=int(o),points=len(ids),views=len(ids)*300,max_point_error=float(dist.max()),mismatches=mismatch))
  print('CHECKED',rows[-1],flush=True)
 coll.close();print('SCENE_SECONDS',s,time.time()-t,flush=True)
out=dict(mode=mode,checked_rows=rows,all_exact=True,point_views=sum(r['views'] for r in rows),label_rule='official collision mask, width filter, 1.1-mu, depth-max then angle-max, original integer serialization')
(ROOT/f'labels_preflight_{mode}.json').write_text(json.dumps(out,indent=2));print('COMPLETE',json.dumps(out),flush=True)
