"""Actual recorded closed-loop curves and tip-box geometry, CPU only."""
from common import *
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
src=(R/'geometry.py').read_text();exec(src[src.index('def rot('):src.index('rows=[]')])
stats=[];figdir=R/'figures';figdir.mkdir(exist_ok=True)
for folder in sorted((R/'runs').iterdir()):
 cp=folder/'data/completed.json'
 if not cp.exists():continue
 c=json.loads(cp.read_text())
 if c.get('role') not in ['supplement','validation'] or c.get('full_episodes')!=1:continue
 z=np.load(folder/'data/trajectory.npz');body=z['body_state'];dd=z['distances'];st=z['states'];ee=z['eef'];ac=z['actions'];ctrl=z['controller_goal'];cfg=json.loads((folder/'data/config.json').read_text());bm={q['name']:q['index_env'] for q in cfg['body_map']};tips=[bm[k] for k in ['panda_leftfinger_tip','panda_rightfinger_tip']];cubes=[q['index_env'] for q in cfg['body_map'] if q['actor']>=3];gaps=np.stack([np.stack([sat(body[:,tip],np.array([.013,.004,.013]),body[:,cube],np.full(3,.035)) for cube in cubes],1) for tip in tips],1)
 pending=dd>=.04;mg=gaps.min(1);masked=np.where(pending,mg,np.inf);minpending=masked.min(-1);minpending[~np.isfinite(minpending)]=np.nan
 out={'run':folder.name,'min_all_tip_cube_signed_SAT_gap_m':float(gaps.min()),'min_pending_tip_cube_signed_SAT_gap_m':float(np.nanmin(minpending)),'control_samples_with_pending_tip_box_gap_le0p005':int((minpending<=.005).sum()),'negative_gap_control_samples':int((minpending<0).sum()),'limitation':'actual two tip boxes at control endpoints; no contacts, no counterpart, no otherlinks or interstep exclusion'};stats.append(out)
 rows=[]
 for t in range(len(ee)):rows.append({'step':t,'EEF_x':ee[t,0],'EEF_y':ee[t,1],'EEF_z':ee[t,2],'min_pending_tip_cube_SAT_gap':minpending[t],'pending_count':int(pending[t].sum()),'mean_goal_error':dd[t].mean()})
 table(folder.name+'_geometry.csv',rows)
 fig,axs=plt.subplots(4,1,figsize=(10,10),sharex=True);tt=np.arange(len(ee));axs[0].plot(tt,dd);axs[0].axhline(.04,color='k',ls=':');axs[0].set_ylabel('object goal error m')
 for k in [0,1]:axs[1].plot(tt,ee[:,k],label='eef '+['x','y'][k])
 axs[1].plot(tt,ee[:,2]-1.025,label='height/table');axs[1].legend();axs[1].set_ylabel('EEF m')
 axs[2].plot(tt[:-1],ac[:,:2]);axs[2].set_ylabel('executed XY action')
 axs[3].plot(tt,minpending);axs[3].axhline(.005,color='r',ls=':',label='contact offset, not contact evidence');axs[3].axhline(0,color='k',ls=':');axs[3].legend();axs[3].set_ylabel('tip box SAT gap m');axs[3].set_xlabel('closed-loop control step')
 fig.suptitle(folder.name+' all200 controls; fall retained; remaining H5 actions not executed');fig.tight_layout();fig.savefig(figdir/(folder.name+'_full.png'),dpi=110);plt.close(fig);z.close()
dump('new_collision_geometry_summaries.json',stats)
summ=json.loads((R/'results/new_run_summaries.json').read_text())
for n in [3,6]:
 runs=[q for q in summ if q['role']=='validation' and q['n']==n]
 if len(runs)!=2:continue
 fig,axs=plt.subplots(2,2,figsize=(12,7))
 for q in runs:
  z=np.load(R/'runs'/q['run']/'data/trajectory.npz');dd=z['distances'];ee=z['eef'];ac=z['actions'];tt=np.arange(len(ee));axs[0,0].plot(tt,dd.mean(-1),label=q['arm']);axs[0,1].plot(tt,(dd<.04).mean(-1),label=q['arm']);axs[1,0].plot(ee[:,0],ee[:,1],label=q['arm']);axs[1,1].plot(tt[:-1],np.linalg.norm(ac[:,:2],axis=-1),label=q['arm']);z.close()
 for ax in axs.flat:ax.legend()
 axs[0,0].set_title('mean per-object goal error m');axs[0,1].set_title('goal fraction, falls retained');axs[1,0].set_title('EEF XY path m');axs[1,1].set_title('executed XY action norm')
 fig.suptitle('%dC one matched pair, same noise multiset / different temporal order'%n);fig.tight_layout();fig.savefig(figdir/('validation_pair%d.png'%n),dpi=110);plt.close(fig)
print('NEW_FIGURES_DONE',len(stats))

