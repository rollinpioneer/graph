import os,json,csv,math,hashlib
from pathlib import Path
os.environ['OMP_NUM_THREADS']='1';os.environ['OPENBLAS_NUM_THREADS']='1';os.environ['MKL_NUM_THREADS']='1'
import numpy as np
from PIL import Image
from scipy.io import loadmat
from scipy.spatial import cKDTree
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from graspnetAPI import Grasp
R=Path(__file__).resolve().parents[1];D=R/'data_recovery';OUT=R/'runs/EG_SUB25_V1/evaluation';BAD=R/'runs/EG_SUB25_V1/bad_cases';FIG=BAD/'figures';FIG.mkdir(exist_ok=True)
FRICTIONS=np.array([.2,.4,.6,.8,1.,1.2]);os.nice(10)
def load_evaluated(role,part,scene,frame):
 with np.load(OUT/'results'/role/part/f'scene_{scene:04d}/evaluated/{frame:04d}.npz') as z:return {k:z[k] for k in z.files}
def good04(x):return (x['quality']>0)&(x['quality']<=.4)&~x['collision']
def quality_color(q,col):
 return '#d62728' if col else '#2ca02c' if 0<q<=.4 else '#ff9900' if q>0 else '#9467bd'
def point_stats(z):
 pc=z['point_clouds'];obj=z['objectness_score'].argmax(axis=0)==1;gn=z['graspness_score'].reshape(-1);mask=obj&(gn>.1);xyz=z['xyz_graspable'];dist,idx=cKDTree(pc).query(xyz);assert dist.max()<1e-6
 labels=z['seg'].reshape(-1).astype(int);visible=sorted(set(labels[labels>0].tolist()));selected=labels[idx]
 return {'sampled_points':len(pc),'objectness_positive':int(obj.sum()),'graspable_points':int(mask.sum()),'graspness_quantiles':np.quantile(gn,[0,.1,.5,.9,1]).tolist(),'fps_centers':len(xyz),'fps_foreground_fraction':float(np.mean(selected>0)),'fps_background_count':int(np.sum(selected==0)),'visible_sampled_classes':visible,'classes_with_fps_centers':sorted(set(selected[selected>0].tolist())),'fps_counts_by_semantic_class':{str(i):int(np.sum(selected==i)) for i in sorted(set(selected.tolist()))}}
def pose_pair_probe(a,b):
 ia=np.flatnonzero(~good04(a));ib=np.flatnonzero(good04(b))
 if not len(ia) or not len(ib):return None
 ga=a['grasps'][ia];gb=b['grasps'][ib];dist=np.linalg.norm(ga[:,None,13:16]-gb[None,:,13:16],axis=2);i,j=np.unravel_index(np.argmin(dist),dist.shape)
 if dist[i,j]>.01:return None
 x,y=ga[i],gb[j];ra=x[4:13].reshape(3,3);rb=y[4:13].reshape(3,3);angle=np.degrees(np.arccos(np.clip((np.trace(ra.T@rb)-1)/2,-1,1)))
 return {'subset_evaluated_rank':int(ia[i]+1),'official_evaluated_rank':int(ib[j]+1),'center_distance_m':float(dist[i,j]),'rotation_geodesic_degrees':float(angle),'width_difference_m':float(abs(x[1]-y[1])),'depth_difference_m':float(abs(x[3]-y[3])),'subset_quality':float(a['quality'][ia[i]]),'subset_collision':bool(a['collision'][ia[i]]),'official_quality':float(b['quality'][ib[j]]),'interpretation':'Nearby centers and different pose/quality are an observation;point/contact/width/depth confounding remains,no head-specific causal proof'}
def project(points,K):
 points=np.asarray(points);uv=points@K.T;return uv[:,:2]/uv[:,2:3]
def draw_grasps(ax,rgb,evaluated,K,title):
 ax.imshow(rgb)
 for rank,(g,q,c) in enumerate(zip(evaluated['grasps'][:5],evaluated['quality'][:5],evaluated['collision'][:5]),1):
  mesh=Grasp(g).to_open3d_geometry(color=[.2,.7,.2]);vertices=np.asarray(mesh.vertices);tri=np.asarray(mesh.triangles);edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0);valid=(vertices[edges,2]>.001).all(axis=1);segments=project(vertices,K)[edges[valid]];color=quality_color(q,c);ax.add_collection(LineCollection(segments,colors=color,linewidths=.8));center=project(g[13:16][None],K)[0];ax.text(center[0],center[1],str(rank),color='white',fontsize=8,bbox={'facecolor':color,'alpha':.85,'pad':1})
 ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.set_title(title,fontsize=11);ax.axis('off')
def draw_graspness(ax,rgb,z,K,title):
 ax.imshow(rgb);uv=project(z['point_clouds'][::10],K);score=z['graspness_score'].reshape(-1)[::10];ax.scatter(uv[:,0],uv[:,1],c=score,cmap='viridis',vmin=0,vmax=1,s=2,alpha=.65,rasterized=True);ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.set_title(title+' (display1/10points)',fontsize=10);ax.axis('off')

def ranking_stats(x):
 good=good04(x);n=len(good);ids=np.flatnonzero(good)
 return {'evaluated_grasps':n,'top1_valid04':bool(good[0]) if n else False,'top1_quality':float(x['quality'][0]) if n else None,'top1_collision':bool(x['collision'][0]) if n else None,'first_valid04_rank':int(ids[0]+1) if len(ids) else None,'top1_bad_but_later_good04':bool(n and not good[0] and len(ids)),'top10_collision_fraction':float(x['collision'][:10].mean()) if n else None,'top50_valid04_count':int(good[:50].sum())}

def main():
 cases=json.loads((D/'BAD_CASE_SELECTION.json').read_text())['cases'];replay=json.loads((D/'BAD_CASE_REPLAY.json').read_text())
 frame_rows=list(csv.DictReader((OUT/'per_frame_results.csv').open()));lookup={(x['model'],x['partition'],int(x['scene']),int(x['frame'])):x for x in frame_rows}
 categories={};aggregate={};records=[];compact=[];grasp_rows=[]
 for part in ['diagnostic_seen','diagnostic_novel']:
  counts={'both_valid04':0,'subset_invalid_official_valid04':0,'both_invalid04':0,'subset_valid_official_invalid04':0};rank_counts={role:{'frames':0,'top1_collision_frames':0,'top1_bad_but_later_good04_frames':0,'any_valid04_candidate_frames':0} for role in ['subset','official']}
  for a in [x for x in frame_rows if x['model']=='subset' and x['partition']==part]:
   scene,frame=int(a['scene']),int(a['frame']);b=lookup[('official',part,scene,frame)];sa=a['top1_valid04']=='True';sb=b['top1_valid04']=='True'
   key='both_valid04' if sa and sb else 'subset_invalid_official_valid04' if sb else 'subset_valid_official_invalid04' if sa else 'both_invalid04';counts[key]+=1
   for role in ['subset','official']:
    x=load_evaluated(role,part,scene,frame);stats=ranking_stats(x);v=rank_counts[role];v['frames']+=1;v['top1_collision_frames']+=int(bool(stats['top1_collision']));v['top1_bad_but_later_good04_frames']+=int(stats['top1_bad_but_later_good04']);v['any_valid04_candidate_frames']+=int(stats['first_valid04_rank'] is not None)
  categories[part]=counts;aggregate[part]=rank_counts
 for case in cases:
  scene,frame,part=case['scene'],case['frame'],case['partition'];dest=BAD/f'scene_{scene:04d}_frame_{frame:04d}';base=R/'data/graspnet/scenes'/f'scene_{scene:04d}'/'kinect'
  rgb=np.array(Image.open(base/'rgb'/f'{frame:04d}.png').convert('RGB'));depth=np.array(Image.open(base/'depth'/f'{frame:04d}.png'));meta=loadmat(base/'meta'/f'{frame:04d}.mat');K=meta['intrinsic_matrix'];factor=float(meta['factor_depth'].reshape(-1)[0]);seg=np.array(Image.open(base/'label'/f'{frame:04d}.png'));labels,counts=np.unique(seg[seg>0],return_counts=True)
  np.savez(dest/'observations.npz',rgb=rgb,depth=depth,intrinsic_matrix=K,factor_depth=factor,object_ids=meta['cls_indexes'].reshape(-1))
  obs={'metadata_object_count':int(len(meta['cls_indexes'].reshape(-1))),'visible_segmentation_classes':int(len(labels)),'foreground_image_fraction':float(np.mean(seg>0)),'valid_depth_fraction':float(np.mean(depth>0)),'min_visible_class_image_fraction':float(counts.min()/seg.size) if len(counts) else 0.,'limitation':'Visible area and object count are complexity proxies;occlusion is not directly measured'}
  models={};inter={};evaluated={}
  for role in ['subset','official']:
   with np.load(dest/(role+'_intermediates.npz')) as z:inter[role]={k:z[k] for k in z.files}
   x=load_evaluated(role,part,scene,frame);evaluated[role]=x;v=lookup[(role,part,scene,frame)];rp=next(q for q in replay['records'] if q['model']==role and q['scene']==scene and q['frame']==frame)
   models[role]={'ap_percent':float(v['ap_percent']),'ap04_percent':float(v['ap04_percent']),'ap08_percent':float(v['ap08_percent']),'ranking':ranking_stats(x),'point_selection_replay':point_stats(inter[role]),'replay_prediction_matches':rp['replay_prediction_allclose'],'replay_max_abs_difference':rp['replay_max_abs_difference']}
   for rank,(g,q,c) in enumerate(zip(x['grasps'][:50],x['quality'][:50],x['collision'][:50]),1):
    row={'scene':scene,'frame':frame,'partition':part,'model':role,'evaluated_rank':rank,'quality_friction':float(q),'collision':bool(c)};row.update({f'grasp_{j}':float(t) for j,t in enumerate(g)});grasp_rows.append(row)
  probe=pose_pair_probe(evaluated['subset'],evaluated['official']);rec={**case,'scene_observations':obs,'models':models,'nearby_center_pose_probe':probe};records.append(rec)
  compact.append({'scene':scene,'frame':frame,'partition':part,'selection':case['selection'],'subset_ap_percent':models['subset']['ap_percent'],'official_ap_percent':models['official']['ap_percent'],'subset_first_valid04_rank':models['subset']['ranking']['first_valid04_rank'],'official_first_valid04_rank':models['official']['ranking']['first_valid04_rank'],'subset_top1_collision':models['subset']['ranking']['top1_collision'],'official_top1_collision':models['official']['ranking']['top1_collision'],'subset_fps_foreground_fraction':models['subset']['point_selection_replay']['fps_foreground_fraction'],'official_fps_foreground_fraction':models['official']['point_selection_replay']['fps_foreground_fraction'],'nearby_center_probe':probe is not None})

  fig,axs=plt.subplots(2,3,figsize=(18,10));fig.suptitle(f"Exploratory {part}: scene{scene:04d}/frame{frame:04d} | {case['selection']}",fontsize=15)
  for j,role in enumerate(['subset','official']):
   draw_grasps(axs[0,j],rgb,evaluated[role],K,f"{role}: native evaluated top5 | frame AP {models[role]['ap_percent']:.2f}%")
   draw_graspness(axs[1,j],rgb,inter[role],K,role+' graspness replay')
  dm=depth.astype(float)/factor;dm[depth==0]=np.nan;im=axs[0,2].imshow(dm,cmap='magma');axs[0,2].set_title('Depth (meters;invalid pixels blank)');axs[0,2].axis('off');fig.colorbar(im,ax=axs[0,2],fraction=.045)
  ax=axs[1,2]
  for j,role in enumerate(['subset','official']):
   x=evaluated[role];n=min(50,len(x['quality']));colors=[quality_color(q,c) for q,c in zip(x['quality'][:n],x['collision'][:n])];ax.scatter(np.arange(1,n+1),np.full(n,j),c=colors,s=50,marker='s')
  ax.set_xlim(.3,50.7);ax.set_ylim(1.7,-.7);ax.set_yticks([0,1],['subset','official']);ax.set_xlabel('Native evaluated confidence rank');ax.set_title('Quality/collision of evaluated grasps');ax.grid(axis='x',alpha=.2)
  fig.text(.5,.02,'Green: noncollision and 0<friction<=0.4 | Orange: positive friction>0.4 | Red: collision | Purple: invalid force closure. Raw scores are not probabilities.',ha='center',fontsize=10)
  fig.tight_layout(rect=[0,.045,1,.95]);stem=f'scene_{scene:04d}_frame_{frame:04d}';fig.savefig(FIG/(stem+'.png'),dpi=120);plt.close(fig)
  with Image.open(FIG/(stem+'.png')) as im:im.convert('RGB').save(FIG/(stem+'.jpg'),quality=82,optimize=True)
  rec['figure']=str(FIG/(stem+'.jpg'));print('BAD_CASE_RENDERED',stem,flush=True)
 result={'status':'PASS','selected_case_count':12,'paired_diagnostic_frames':1024,'top1_validity_criterion':'Native ranked evaluated top1:0<friction<=0.4 and no collision;not robot success','paired_top1_categories':categories,'ranking_summary':aggregate,'all24_replays_match':replay['all24_replays_match'],'cases':records,'causal_status':'Exploratory observations only;no module-level cause has been proven'}
 (BAD/'bad_case_metrics.json').write_text(json.dumps(result,indent=2))
 for name,rows in [('bad_case_summary.csv',compact),('bad_case_grasps.csv',grasp_rows)]:
  with (BAD/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
 lines=['# EG-SUB25-V1 exploratory Bad Case report','','## Protocol','','All four frozen diagnostic scenes use all256Kinect frames and the unmodified official GraspNetAPI. The paired models received identical cached point clouds. AP is the mean of official precision tensors,not robotic success. Candidate friction/collision records are the exact native results used for AP. Unevaluated raw predictions do not have known grasp quality.','','For each scene,the two lowest subset-frame APs and the median frame were selected after evaluation:12 paired cases. These are exploratory,posthoc cases and not unbiased causal evidence. The official model has full-data training and the subset model has25% training;this is a training-sufficiency diagnostic,not a controlled architecture comparison.','','## Paired observations over all1024diagnostic frames','','Top1valid means noncolliding official-evaluated first grasp with required friction in(0,0.4]. This operational summary supplements,does not replace,continuous AP.','','|Partition|Both valid|Subset invalid / official valid|Both invalid|Subset valid / official invalid|','|---|---:|---:|---:|---:|']
 for part,c in categories.items():lines.append(f"|{part}|{c['both_valid04']}|{c['subset_invalid_official_valid04']}|{c['both_invalid04']}|{c['subset_valid_official_invalid04']}|")
 lines+=['','|Partition/model|Top1collisions|Bad top1 with a later valid candidate|Frames with any valid candidate|','|---|---:|---:|---:|']
 for part,roles in aggregate.items():
  for role,c in roles.items():lines.append(f"|{part}/{role}|{c['top1_collision_frames']}/{c['frames']}|{c['top1_bad_but_later_good04_frames']}/{c['frames']}|{c['any_valid04_candidate_frames']}/{c['frames']}|")
 lines+=['','## Selected cases','','|Scene/frame|Selection|Subset AP|Official AP|Subset first valid rank|Official first valid rank|Near-center pose contrast|','|---|---|---:|---:|---:|---:|---|']
 for c in compact:lines.append(f"|{c['scene']:04d}/{c['frame']:04d}|{c['selection']}|{c['subset_ap_percent']:.2f}|{c['official_ap_percent']:.2f}|{c['subset_first_valid04_rank']}|{c['official_first_valid04_rank']}|{c['nearby_center_probe']}|")
 lines+=['','## Interpretation and limits','','Observed top1 failures with later valid candidates identify a ranking problem in the final candidate list. They do not isolate the scoring head,because point selection,pose generation,NMS and candidate coverage also affect ranking. The paired counts where only the official model succeeds are consistent with training data/optimization/distribution effects;none is individually proven.','','Nearby-center pose probes compare an invalid subset candidate and a valid official candidate within1cm. Rotation,width and depth can differ together;this establishes a local pose/quality contrast,not a proven view/angle/depth-head mechanism. No nearby valid candidate is also not evidence of a particular module failure.','','Point-selection diagnostics report sampled semantic classes,FPS foreground coverage and graspness/objectness masks. Segmentation classes do not describe grasp feasibility. Missing a class cannot by itself prove a missed feasible grasp. Intermediate outputs are replays from the same cached inputs and batch composition. Replay agreement is recorded per model/frame;nonmatching replays cannot prove the exact original internal path. All24replays match: '+str(replay['all24_replays_match'])+'.','','Object counts,visible area and valid depth are stored as scene-complexity proxies. These12selected cases cannot establish that occlusion or clutter causes failures. Novel observations are limited to scenes188and189;they are not the full official Novel AP. Raw confidence is the original logit-weighted decoder output,not a calibrated probability.','','No network change,new seed or retraining was performed. A useful next study would hold data and candidate generation fixed to test ranking/pose hypotheses,after deciding whether the subset baseline is sufficiently trained.','','## Artifacts','','bad_case_metrics.json contains all measured observations and replay identities;bad_case_summary.csv provides paired summaries;bad_case_grasps.csv stores native evaluated top50poses,raw scores,friction and collision. RGB/depth,full-resolution figures and intermediate tensors remain on the server. Derived JPEGs are in figures/.']
 (BAD/'bad_case_report.md').write_text('\n'.join(lines)+'\n');(D/'BAD_CASE_COMPLETE.json').write_text(json.dumps({k:v for k,v in result.items() if k!='cases'},indent=2));print('BAD_CASE_ANALYSIS_COMPLETE',flush=True)
if __name__=='__main__':main()
