"""Bounded exploratory illustrations, fixed first probe seed, no model selection."""
import csv, shutil, time
from common import *
import scipy.io as sio
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
out=R/'results';cases=R/'bad_cases';cases.mkdir(exist_ok=True)
decision=json.loads((out/'subset_development_decision.json').read_text())
per_scene=list(csv.DictReader(open(out/'subset_per_scene.csv')))
scene_ap={(int(x['scene']),x['configuration']):float(x['ap']) for x in per_scene}
selected={};case_records=[]
for s in sorted(SPLIT['probe_development']):
    for f in sorted(frames(s)):
        with np.load(R/f'evaluation/subset/scene_{s:04d}/{f:04d}.npz') as z,np.load(R/f'exports/subset/scene_{s:04d}/{f:04d}.npz') as e:
            for suffix in ['', '_CD']:
                rname='R0'+suffix;aname='A_20261018'+suffix;ri=z[rname+'_ids'];ai=z[aname+'_ids'];q=z['quality']
                if not len(ri) or not len(ai):continue
                rv=0<q[ri[0]]<=.4;av=0<q[ai[0]]<=.4
                categories=[]
                if not rv and av:categories.append('R0_fail_A_success')
                if rv and not av:categories.append('A_fail_R0_success')
                if not rv and not av and (np.any((q[ai[1:]]>0)&(q[ai[1:]]<=.4)) or np.any((q[ri[1:]]>0)&(q[ri[1:]]<=.4))):categories.append('both_fail_later_success')
                for category in categories:
                    key=category+suffix
                    if key in selected:continue
                    selected[key]=(s,f);folder=cases/key;folder.mkdir(exist_ok=True)
                    root=DATA/f'scenes/scene_{s:04d}/kinect'
                    shutil.copyfile(root/f'rgb/{f:04d}.png',folder/'rgb.png');shutil.copyfile(root/f'depth/{f:04d}.png',folder/'depth.png')
                    K=sio.loadmat(root/f'meta/{f:04d}.mat')['intrinsic_matrix'];pred=e['pred'];rows=[]
                    for name,ids in [(rname,ri),(aname,ai)]:
                        ranks=list(range(min(10,len(ids))))
                        valid_ranks=np.flatnonzero((q[ids]>0)&(q[ids]<=.4))
                        if len(valid_ranks) and int(valid_ranks[0]) not in ranks:ranks.append(int(valid_ranks[0]))
                        for rank in ranks:
                            idx=ids[rank]
                            rows.append(dict(configuration=name,rank=rank+1,candidate_id=int(idx),score=float(z[name.removesuffix('_CD')+'_scores'][idx]),mu=float(q[idx]),collision=bool(z['collision'][idx]),empty=bool(z['empty'][idx]),grasp=pred[idx].tolist(),scene_ap=scene_ap[s,name]))
                    rec=dict(category=key,split='sampled_development_exploratory',scene=s,frame=f,seed=20261018,selection='smallest_scene_then_frame; first_registered_seed; no selection by effect magnitude',rows=rows)
                    write_json(folder/'case.json',rec);case_records.append(rec)
                    rgb=np.array(Image.open(folder/'rgb.png'));depth=np.array(Image.open(folder/'depth.png'))
                    fig,axs=plt.subplots(1,2,figsize=(12,5),constrained_layout=True);axs[0].imshow(rgb);axs[1].imshow(depth,cmap='viridis')
                    def project(points):
                        pp=points@K.T;return pp[:,:2]/pp[:,2:]
                    for name,idx,color in [(rname,int(ri[0]),'#ff5555'),(aname,int(ai[0]),'#00d9ff')]:
                        g=pred[idx];rot=g[4:13].reshape(3,3);c=g[13:16];pts=project(np.stack([c,c+rot[:,0]*.04,c-rot[:,1]*g[1]/2,c+rot[:,1]*g[1]/2]))
                        for ax in axs:
                            ax.scatter(*pts[0],c=color,s=55);ax.plot(pts[2:,0],pts[2:,1],color=color,lw=2)
                            ax.annotate('',xy=pts[1],xytext=pts[0],arrowprops=dict(color=color,arrowstyle='->',lw=2))
                        axs[0].plot([],[],color=color,label=f'{name}: id {idx}, mu {q[idx]:.1f}, width {g[1]*1000:.1f} mm')
                    axs[0].legend(fontsize=8,loc='lower left');axs[0].set_title('RGB: center, approach arrow, opening span');axs[1].set_title('Raw depth with the same projected poses')
                    for ax in axs:ax.axis('off')
                    fig.suptitle(f'Exploratory {key} | scene {s:04d} / frame {f:04d}\nProjected pose cues; collisions are from native evaluation, not the drawing',fontsize=11)
                    fig.savefig(folder/'pose_comparison.png',dpi=130);plt.close(fig)
write_json(out/'bad_case_index.json',dict(scope='development only; not independent mechanism evidence',illustration_seed=20261018,cases=[{k:v for k,v in x.items() if k!='rows'} for x in case_records],missing=[c+s for c in ['R0_fail_A_success','A_fail_R0_success','both_fail_later_success'] for s in ['', '_CD'] if c+s not in selected]))
write_json(out/'bad_case_details.json',case_records)
# All configurations and all seeds, without selecting a favorable run.
summary=decision['summary'];lookup={x['configuration']:x for x in summary}
fig,axs=plt.subplots(1,2,figsize=(10,4),constrained_layout=True)
for ax,suffix in zip(axs,['','_CD']):
    ax.axhline(lookup['R0'+suffix]['ap'],color='black',ls='--',label='R0')
    for i,arm in enumerate(['E','B6','A']):
        vals=[lookup[f'{arm}_{seed_}{suffix}']['ap'] for seed_ in [20261018,20261019,20261020]]
        ax.scatter([i]*3,vals,s=35);ax.plot([i-.18,i+.18],[np.mean(vals)]*2,c='black')
    ax.set_xticks(range(3),['E','B6','A']);ax.set_ylabel('Sampled development native AP (%)');ax.set_title('With CD' if suffix else 'Without CD');ax.legend();ax.grid(axis='y',alpha=.2)
fig.savefig(out/'development_all_seeds.png',dpi=160);fig.savefig(out/'development_all_seeds.svg');plt.close(fig)
print(json.dumps({'cases':len(case_records),'missing':[c for c in selected if selected[c] is None]}))
