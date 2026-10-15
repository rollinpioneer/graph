from pathlib import Path
import json,hashlib,math,collections,time
import numpy as np
from PIL import Image
from scipy.io import loadmat
R=Path(__file__).resolve().parents[1];ROOT=R/'data/graspnet';D=R/'data_recovery'
assert not (D/'frozen_split.json').exists()
rows=[]
for scene in list(range(100))+list(range(100,130))+list(range(160,190)):
 camera=ROOT/'scenes'/f'scene_{scene:04d}'/'kinect'
 obj=loadmat(camera/'meta/0000.mat')['cls_indexes'].reshape(-1).astype(int)-1
 foreground=[];imbalance=[];small=[]
 for frame in [0,85,170,255]:
  seg=np.asarray(Image.open(camera/'label'/f'{frame:04d}.png'));counts=np.bincount(seg.reshape(-1).astype(int),minlength=89)[obj+1];visible=counts[counts>0]
  foreground.append(float(counts.sum()/seg.size));imbalance.append(float(visible.std()/(visible.mean()+1e-12)))
  small.append(float(np.mean(counts<.005*seg.size)))
 rows.append({'scene':scene,'object_ids':sorted(set(map(int,obj))),'object_instances':len(obj),'foreground_fraction':float(np.mean(foreground)),'visible_area_imbalance':float(np.mean(imbalance)),'small_visible_object_fraction':float(np.mean(small))})
byid={x['scene']:x for x in rows}
def choose(pool,n):
 candidates=[byid[i] for i in pool];keys=['object_instances','foreground_fraction','visible_area_imbalance','small_visible_object_fraction']
 cuts={k:np.quantile([x[k] for x in candidates],[1/3,2/3]) for k in keys}
 features={x['scene']:['object:'+str(o) for o in x['object_ids']]+[k+':'+str(int(np.searchsorted(cuts[k],x[k],side='right'))) for k in keys] for x in candidates}
 freq=collections.Counter(f for fs in features.values() for f in fs);objweights=[1/math.sqrt(v) for k,v in freq.items() if k.startswith('object:')];scale=float(np.mean(objweights))
 weights={k:(1/math.sqrt(v)/scale if k.startswith('object:') else 2.0) for k,v in freq.items()};count=collections.Counter();selected=[];trace=[]
 while len(selected)<n:
  scored=[(sum(weights[f]*math.log((count[f]+2)/(count[f]+1)) for f in features[i]),-i,i) for i in pool if i not in selected]
  score,_,i=max(scored);selected.append(i);count.update(features[i]);trace.append({'scene':i,'gain':score})
 return sorted(selected),trace,{k:list(map(float,v)) for k,v in cuts.items()}
train,tt,tc=choose(list(range(100)),50);remaining=sorted(set(range(100))-set(train));dev,dt,dc=choose(remaining,10);reserve=sorted(set(remaining)-set(dev))
seen,_,_=choose(list(range(100,130)),2);novel,_,_=choose(list(range(160,190)),2)
frames=np.rint(np.linspace(0,255,128)).astype(int).tolist();smoke_scenes=[train[i] for i in [0,12,24,36,49]];smoke_frames=[frames[i] for i in np.rint(np.linspace(0,127,16)).astype(int)]
assert len(set(frames))==128 and len(train)==50 and len(dev)==10 and len(reserve)==40 and not set(train)&set(dev)
assert set(o for i in train for o in byid[i]['object_ids'])==set(o for i in range(100) for o in byid[i]['object_ids'])
manifest={'experiment_id':'EG-SUB25-V1','seed':20261015,'camera':'kinect','train_scene_ids':train,'train_frame_ids':frames,'dev_scene_ids':dev,'dev_frame_ids':list(range(256)),'reserve_scene_ids':reserve,'diagnostic_seen_scene_ids':seen,'diagnostic_novel_scene_ids':novel,'diagnostic_frame_ids':list(range(256)),'smoke_pairs':[[i,f] for i in smoke_scenes for f in smoke_frames],'training_frames':6400,'dev_frames':2560,'selection':'Deterministic weighted logarithmic diminishing-return coverage; object rarity and four metadata/segmentation tertile strata; lowest scene ID breaks ties','selection_uses_model_outputs':False,'dev_is_official_training_partition':True,'official_pretrained_not_independent_on_dev':True,'diagnostic_test_scenes_are_exploratory_development_material':True,'metadata_frames':[0,85,170,255],'training_object_ids':sorted(set(o for i in train for o in byid[i]['object_ids']))}
metadata=json.dumps({'scenes':rows,'train_trace':tt,'dev_trace':dt,'train_tertiles':tc,'dev_tertiles':dc},indent=2);(D/'metadata_selection.json').write_text(metadata);manifest['metadata_sha256']=hashlib.sha256(metadata.encode()).hexdigest()
s=json.dumps(manifest,indent=2);(D/'frozen_split.json').write_text(s);(D/'FROZEN_SPLIT_SHA256').write_text(hashlib.sha256(s.encode()).hexdigest()+'\n');print('TRAIN_SCENES',train);print('DEV_SCENES',dev);print('OBJECT_COVERAGE',len(manifest['training_object_ids']));print('DIAGNOSTIC',seen,novel);print('SPLIT_SHA',hashlib.sha256(s.encode()).hexdigest())
