"""First scene/frame examples per stratum, first registered seed, no cherry-picking."""
import csv,shutil
from common import *
import scipy.io as sio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=R/'results';base=R/'confirmation_bad_cases';base.mkdir(exist_ok=True)
scene_rows=list(csv.DictReader(open(out/'confirmation_per_scene.csv')))
scene_ap={(x['role'],int(x['scene']),x['configuration']):float(x['ap']) for x in scene_rows if x['sensitivity']=='native256'}
cases=[];missing=[];seed_=20261018;categories=['R0_fail_A_success','A_fail_R0_success','both_fail_later_success']
for role in ['subset','official']:
    for split,ss in SPLIT['official_confirmation'].items():
        found=set()
        for s in sorted(ss):
            for f in sorted(frames(s)):
                if len(found)==3:break
                with np.load(R/f'evaluation/{role}/scene_{s:04d}/{f:04d}.npz') as z:
                    ri=z['R0_ids'];ai=z[f'A_{seed_}_ids'];q=z['quality']
                    if not len(ri) or not len(ai):continue
                    rv=0<q[ri[0]]<=.4;av=0<q[ai[0]]<=.4;flags=[]
                    if not rv and av:flags.append(categories[0])
                    if rv and not av:flags.append(categories[1])
                    if not rv and not av and (np.any((q[ai[1:]]>0)&(q[ai[1:]]<=.4)) or np.any((q[ri[1:]]>0)&(q[ri[1:]]<=.4))):flags.append(categories[2])
                    for category in flags:
                        if category in found:continue
                        found.add(category);folder=base/role/split/category;folder.mkdir(parents=True,exist_ok=True)
                        root=DATA/f'scenes/scene_{s:04d}/kinect'
                        shutil.copyfile(root/f'rgb/{f:04d}.png',folder/'rgb.png');shutil.copyfile(root/f'depth/{f:04d}.png',folder/'depth.png')
                        with np.load(R/f'exports/{role}/scene_{s:04d}/{f:04d}.npz') as e:pred=e['pred']
                        rows=[]
                        for name,ids in [('R0',ri),(f'A_{seed_}',ai)]:
                            ranks=list(range(min(10,len(ids))));good=np.flatnonzero((q[ids]>0)&(q[ids]<=.4))
                            if len(good) and int(good[0]) not in ranks:ranks.append(int(good[0]))
                            for rank in ranks:
                                idx=int(ids[rank]);rows.append(dict(configuration=name,rank=rank+1,candidate_id=idx,score=float(z[name+'_scores'][idx]),mu=float(q[idx]),collision=bool(z['collision'][idx]),empty=bool(z['empty'][idx]),grasp=pred[idx].tolist(),scene_ap=scene_ap[role,s,name]))
                        rec=dict(role=role,split=split,category=category,scene=s,frame=f,probe_seed=seed_,collision_filter=False,exploratory=True,selection='smallest scene then frame; first registered seed; no magnitude selection',rows=rows)
                        write_json(folder/'case.json',rec);cases.append(rec)
                        K=sio.loadmat(root/f'meta/{f:04d}.mat')['intrinsic_matrix'];rgb=np.array(Image.open(folder/'rgb.png'));depth=np.array(Image.open(folder/'depth.png'))
                        fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True);axs[0].imshow(rgb);axs[1].imshow(depth,cmap='viridis')
                        for name,idx,color in [('R0',int(ri[0]),'#ff5555'),('A',int(ai[0]),'#00d9ff')]:
                            g=pred[idx];rot=g[4:13].reshape(3,3);c=g[13:16];pts=np.stack([c,c+rot[:,0]*.04,c-rot[:,1]*g[1]/2,c+rot[:,1]*g[1]/2])@K.T;pts=pts[:,:2]/pts[:,2:]
                            for ax in axs:
                                ax.scatter(*pts[0],c=color,s=50);ax.plot(pts[2:,0],pts[2:,1],color=color,lw=2);ax.annotate('',xy=pts[1],xytext=pts[0],arrowprops=dict(color=color,arrowstyle='->',lw=2))
                            axs[0].plot([],[],color=color,label=f'{name}: id {idx}, mu {q[idx]:.1f}, width {g[1]*1000:.1f} mm')
                        axs[0].legend(fontsize=8,loc='lower left');axs[0].set_title('RGB: center, approach arrow, opening span');axs[1].set_title('Raw depth with projected poses')
                        for ax in axs:ax.axis('off')
                        fig.suptitle(f'{role} / {split} / {category} | scene {s:04d} / frame {f:04d}\nExploratory illustration; not causal evidence',fontsize=11);fig.savefig(folder/'pose_comparison.png',dpi=130);plt.close(fig)
            if len(found)==3:break
        missing.extend(dict(role=role,split=split,category=c) for c in categories if c not in found)
write_json(out/'confirmation_bad_case_details.json',cases)
write_json(out/'confirmation_bad_case_index.json',dict(cases=[{k:v for k,v in x.items() if k!='rows'} for x in cases],missing=missing,selection_seed=seed_,scope='exploratory examples from completed frozen confirmation; not unbiased mechanism evidence'))
print(json.dumps(dict(cases=len(cases),missing=missing)))
