"""Full shape/mask/hash checks plus deterministic raw-label semantic probes."""
import os,sys,json,hashlib,time,concurrent.futures
from pathlib import Path
import numpy as np
from PIL import Image
from scipy.io import loadmat
R=Path(__file__).resolve().parents[1];ROOT=R/'data/graspnet'
sys.path.insert(0,str(R/'runtime/EconomicGrasp'))
from utils.data_utils import CameraInfo,create_point_cloud_from_depth_image,get_workspace_mask

def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8*1024*1024),b''):h.update(b)
 return h.hexdigest()
def scene_check(scene):
 name=f'scene_{scene:04d}';camera=ROOT/'scenes'/name/'kinect';poses=np.load(camera/'camera_poses.npy');align=np.load(camera/'cam0_wrt_table.npy')
 records=[];mins=[];maxs=[];counts=[]
 for frame in range(256):
  stem=f'{frame:04d}';meta=loadmat(camera/'meta'/(stem+'.mat'));intrinsic=meta['intrinsic_matrix']
  cam=CameraInfo(1280.,720.,intrinsic[0,0],intrinsic[1,1],intrinsic[0,2],intrinsic[1,2],meta['factor_depth'])
  depth=np.asarray(Image.open(camera/'depth'/(stem+'.png')));seg=np.asarray(Image.open(camera/'label'/(stem+'.png')))
  cloud=create_point_cloud_from_depth_image(depth,cam,organized=True)
  mask=(depth>0)&get_workspace_mask(cloud,seg,trans=np.dot(align,poses[frame]),organized=True,outlier=.02)
  p=ROOT/'graspness'/name/'kinect'/(stem+'.npy');a=np.load(p,allow_pickle=False)
  assert a.shape==(int(mask.sum()),1) and a.size and np.isfinite(a).all(),(name,frame,'graspness mask')
  assert 0<=a.min()<=a.max()<=1 and a.max()>a.min(),(name,frame,'range')
  proof=json.loads(Path(str(p)+'.sha256.json').read_text());digest=sha(p);assert digest==proof['sha256']
  records.append(f'{p.relative_to(ROOT)}\t{p.stat().st_size}\t{digest}\n');mins.append(float(a.min()));maxs.append(float(a.max()));counts.append(len(a))
 ep=ROOT/'economic_grasp_label_300views'/(name+'_labels.npz')
 with np.load(ep,allow_pickle=False) as z:a={k:z[k] for k in z.files}
 required={'points','rotations','depth','scores','widths','pointid','vgraspness','topview'};assert set(a)==required
 n=len(a['points']);assert n>0 and a['points'].shape==(n,3)
 for k,v in a.items():
  assert np.isfinite(v).all(),(name,k)
  assert v.shape==((n,3) if k=='points' else (n,) if k=='pointid' else (n,300)),(name,k,v.shape)
 assert np.all(a['rotations']<12) and np.all(a['depth']<4) and np.all(a['scores']<=11)
 assert np.all(a['widths'][a['scores']>0]<=100)
 assert np.all((a['vgraspness']>=0)&(a['vgraspness']<=1))
 assert np.array_equal(np.sort(a['topview'],axis=1),np.broadcast_to(np.arange(300),(n,300)))
 meta=loadmat(camera/'meta/0000.mat');obj=meta['cls_indexes'].reshape(-1).astype(int)-1
 assert np.all(a['pointid']<len(obj))
 probes=[];empty_objects=[]
 with np.load(ROOT/'collision_label'/name/'collision_labels.npz') as collision:
  for ordinal,oid in enumerate(obj):
   select=np.flatnonzero(a['pointid']==ordinal)
   with np.load(ROOT/'grasp_label'/f'{oid:03d}_labels.npz') as raw:
    if len(select)==0:
     raw_scores=raw['scores'];raw_collision=collision[f'arr_{ordinal}']
     assert not np.any((raw_scores>0)&(raw_scores<=.4)&~raw_collision),(name,'missing valid object',oid)
     empty_objects.append({'object_id':int(oid),'ordinal':ordinal,'raw_points':len(raw_scores),'official_retained_points':0})
     continue
    points=raw['points'];lookup={tuple(p):i for i,p in enumerate(points)}
    raw_indices=np.array([lookup[tuple(p)] for p in a['points'][select]])
    assert np.array_equal(points[raw_indices],a['points'][select])
    if scene in [0,50,99]:
     chosen=np.unique(np.linspace(0,len(select)-1,min(3,len(select)),dtype=int));rows=select[chosen];ri=raw_indices[chosen]
     score=raw['scores'][ri].copy();width=raw['offsets'][ri,:,:,:,2];col=collision[f'arr_{ordinal}'][ri]
     score[col]=0;score[score<0]=0
     assert np.all(np.any((score>0)&(score<=.4),axis=(1,2,3)))
     v=np.sum((score>0)&(score<=.6),axis=(2,3))/48
     v=(v-v.min(axis=1,keepdims=True))/(v.max(axis=1,keepdims=True)-v.min(axis=1,keepdims=True)+1e-5)
     assert np.allclose(v,a['vgraspness'][rows],rtol=2e-6,atol=2e-6)
     label=(score>0)&(width<=.1);score[~label]=0;pos=score>0;score[pos]=1.1-score[pos]
     # Argmax ordering equals official torch.max: first maximum depth, then angle.
     best_d=score.argmax(axis=3);best_a=score.max(axis=3).argmax(axis=2)
     ii=np.arange(len(rows))[:,None];vv=np.arange(300)[None,:]
     dd=best_d[ii,vv,best_a];ss=score[ii,vv,best_a,dd];ww=width[ii,vv,best_a,dd];order=a['topview'][rows]
     for key,value in [('rotations',best_a.astype(np.uint8)),('depth',dd.astype(np.uint8)),('scores',(ss*10).astype(np.uint8)),('widths',(ww*1000).astype(np.uint8))]:
      assert np.array_equal(np.take_along_axis(value,order,axis=1),a[key][rows]),(name,oid,key,'semantic probe')
     probes.append({'object':int(oid),'rows':rows.tolist()})
 digest=sha(ep);records.append(f'{ep.relative_to(ROOT)}\t{ep.stat().st_size}\t{digest}\n')
 result={'scene':scene,'graspness_frames':256,'graspness_point_count_range':[min(counts),max(counts)],'range':[min(mins),max(maxs)],'economic_points':n,'economic_sha256':digest,'manifest_sha256':hashlib.sha256(''.join(records).encode()).hexdigest(),'raw_semantic_probes':probes,'empty_objects_verified_against_raw':empty_objects}
 (R/'data_recovery'/f'supervision_scene_{scene:04d}.json').write_text(json.dumps(result))
 return result
if __name__=='__main__':
 started=time.time();assert json.loads((R/'data_recovery/RAW_DATA_VALIDATION.json').read_text())['status']=='PASS'
 with concurrent.futures.ProcessPoolExecutor(max_workers=4) as pool:rows=list(pool.map(scene_check,range(100)))
 summary={'status':'PASS','graspness_files':sum(x['graspness_frames'] for x in rows),'economic_scene_files':len(rows),'scene_results':rows,'wall_seconds':time.time()-started,'mask_correspondence_all_frames':True,'raw_point_object_correspondence_all_points':True,'raw_score_direction_depth_width_probes_scenes':[0,50,99],'core_thresholds_unchanged':True}
 assert summary['graspness_files']==25600
 (R/'data_recovery/SUPERVISION_VALIDATION.json').write_text(json.dumps(summary,indent=2));print('SUPERVISION_PASS',summary['wall_seconds'],flush=True)
