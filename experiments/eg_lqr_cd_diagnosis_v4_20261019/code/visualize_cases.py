"""Posthoc C1 illustrations. Mesh projection is not a collision judgement."""
from common import *
import pandas as pd
from PIL import Image
from scipy.io import loadmat
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from graspnetAPI import Grasp

os.nice(10)
choices=pd.read_csv(R/'results/exploratory_bad_cases.csv')
selected=[choices[(choices.scene==136)&(choices.seed==SEEDS[0])].iloc[0],choices[(choices.scene==136)&(choices.seed==SEEDS[2])].iloc[0],choices[choices.scene==102].iloc[0]]
rows=[];files=[]
for example in selected:
 scene,frame,seed=int(example.scene),int(example.frame),int(example.seed)
 folder=DATA/f'scenes/scene_{scene:04d}/kinect';K=loadmat(folder/f'meta/{frame:04d}.mat')['intrinsic_matrix'];factor=float(loadmat(folder/f'meta/{frame:04d}.mat')['factor_depth'].ravel()[0])
 with Image.open(folder/f'rgb/{frame:04d}.png') as im:rgb=np.asarray(im)
 with Image.open(folder/f'depth/{frame:04d}.png') as im:depth=np.asarray(im).astype(float)/factor
 with np.load(V/f'exports/subset/scene_{scene:04d}/{frame:04d}.npz') as z:pred=z['pred'].copy()
 with np.load(R/f'evaluation/exploration/{scene:04d}/{frame:04d}.npz') as z:quality=z['quality'];collision=z['collision'];empty=z['empty'];q=z[f'Q_{seed}'];fusion,_=fused_score(pred[:,0],q)
 records=json.loads((R/f'evaluation/exploration/{scene:04d}/{frame:04d}.json').read_text());names=[('EG','R0_CD','tab:blue'),('S',f'S_{seed}_CD','tab:orange'),('F05',f'F05_{seed}_CD','tab:green')]
 fig,axes=plt.subplots(1,2,figsize=(13,5.5));axes[0].imshow(rgb);masked=np.ma.masked_where(depth<=0,depth);valid=depth[depth>0];vmin,vmax=np.quantile(valid,[.02,.98]);im=axes[1].imshow(masked,cmap='viridis_r',vmin=vmin,vmax=vmax);fig.colorbar(im,ax=axes[1],label='Observed depth (m)',fraction=.04);titles=[];table=[]
 for label,name,color in names:
  rec=next(x for x in records if x['configuration']==name);i=rec['top_candidate'];titles.append(f'{label} AP={rec["ap"]:.2f}')
  if i is None:continue
  assert np.isfinite(quality[i]);g=pred[i].astype(float).copy();g[1]=np.clip(g[1],0,.1);mesh=Grasp(g).to_open3d_geometry(color=(.2,.4,.8));meshes=mesh if isinstance(mesh,list) else [mesh]
  for geom in meshes:
   vertices=np.asarray(geom.vertices);tri=np.asarray(geom.triangles);edges=np.unique(np.sort(np.concatenate([tri[:,[0,1]],tri[:,[1,2]],tri[:,[2,0]]]),axis=1),axis=0);proj=vertices@K.T;uv=proj[:,:2]/proj[:,2:]
   lines=uv[edges[(vertices[edges,2]>.01).all(1)]]
   for ax in axes:ax.add_collection(LineCollection(lines,colors=color,linewidths=1.2))
  center=g[13:16]@K.T;center=center[:2]/center[2]
  for ax in axes:ax.scatter(*center,s=20,color=color,label=f'{label} id{i}',edgecolors='white',linewidth=.4)
  row=dict(scene=scene,frame=frame,seed=seed,method=label,candidate_id=i,AP=rec['ap'],official_mu=float(quality[i]),official_collision=bool(collision[i]),official_empty=bool(empty[i]),width_cm=float(g[1]*100),depth_cm=float(g[3]*100),EG_score=float(pred[i,0]),Q_original=float(q[i]),F05_score=float(fusion[i]),selection='First already-fixed worst-AP illustrative row for scene0136/head18, scene0136/head20, scene0102; exploratory posthoc')
  rows.append(row);table.append(f'{label:>3} id{i:4d} | mu={quality[i]:.2f} reject={int(collision[i])} empty={int(empty[i])} | width={g[1]*100:.1f}cm depth={g[3]*100:.1f}cm | EG={pred[i,0]:.3f} Q={q[i]:.3f} F05={fusion[i]:+.3f}')
 for ax in axes:ax.set_xlim(0,rgb.shape[1]);ax.set_ylim(rgb.shape[0],0);ax.axis('off')
 axes[0].legend(loc='upper left',fontsize=8);axes[0].set_title('RGB with projected official gripper meshes');axes[1].set_title('Depth with the same selected grasps')
 fig.suptitle(f'EXPLORATORY C1 | scene {scene:04d} frame {frame:04d} head {seed}\n'+', '.join(titles),fontsize=11)
 fig.text(.02,.02,'\n'.join(table)+'\nMesh outlines illustrate pose; collision/mu values come from full official geometry evaluation.',fontfamily='monospace',fontsize=8)
 fig.subplots_adjust(top=.82,bottom=.23,wspace=.08);name=f'results/bad_case_{scene:04d}_{frame:04d}_{seed}.jpg';fig.savefig(R/name,dpi=120,pil_kwargs={'quality':90});plt.close(fig);files.append(name)
csv_write('results/visualized_bad_cases.csv',rows)
write('results/visualized_bad_cases.json',dict(status='COMPLETE',files=files,selection='Already-fixed exploratory_bad_cases.csv rows; no new confirmation cases selected',not_causal_evidence=True,visual_geometry='Official Grasp.to_open3d_geometry projected with scene intrinsic K; illustrative only',quality='Actual candidate pose from official full geometric/friction evaluator cache'))
print('CASE_VISUALIZATIONS',files,flush=True)
