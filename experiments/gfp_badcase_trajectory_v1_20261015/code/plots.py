import json,pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=pathlib.Path(__file__).resolve().parents[1];M=json.loads((R/'reports/environment.json').read_text())
rows=json.loads((R/'reports/case_metrics.json').read_text());pairs=json.loads((R/'reports/paired_comparisons.json').read_text())
maze=np.array(M['maze_map']);unit=M['maze_unit'];off=M['offset'];ext=[-off[0]-unit/2,(maze.shape[1]-.5)*unit-off[0],-off[1]-unit/2,(maze.shape[0]-.5)*unit-off[1]]
def mapax(ax):
 ax.imshow(maze,cmap='Greys',origin='lower',extent=ext,vmin=0,vmax=1,alpha=.8)
 ax.scatter([20],[20],marker='*',c='#21a366',s=110,zorder=5)
 ax.add_patch(plt.Circle((20,20),.5,fill=False,color='#21a366',lw=1.5))
 ax.set_aspect('equal');ax.set_xlabel('x (m)');ax.set_ylabel('y (m)')
fig,axes=plt.subplots(2,2,figsize=(12,12))
for s in (0,1):
 for j,kind in enumerate(('failure','success')):
  ax=axes[j,s];mapax(ax);subset=[r for r in rows if r['model_seed']==s and r['kind']==kind]
  for r in subset:
   a=np.load(R/'trajectories'/f"{r['id']}.npz");xy=a['xy']
   ax.plot(*xy.T,alpha=.65,lw=.7);ax.scatter(*xy[-1],c='#d23c3c' if kind=='failure' else '#188456',s=18)
  ax.set_title(f'Seed {s}: selected 1M {kind}s (n={len(subset)})')
fig.suptitle('Original GFP / HumanoidMaze-medium task1\nOutcome-enriched cases; line crossings and contacts are not causal diagnoses')
fig.tight_layout();fig.savefig(R/'reports/trajectory_overview.png',dpi=140);plt.close(fig)
early=[p for p in pairs if p['kind']=='earlier']
fig,axes=plt.subplots(2,3,figsize=(15,10))
for ax,p in zip(axes.flat,early):
 mapax(ax)
 for key,color,label in [('reference_1m','#cc4444','1M failure'),('control','#00866a','earlier success')]:
  a=np.load(R/'trajectories'/f"{p[key]}.npz");ax.plot(*a['xy'].T,color=color,lw=1,label=label)
  t=p['first_sustained_2m_path_divergence_step'];ax.scatter(*a['xy'][t],color=color,s=32,marker='x')
 ax.set_title(p['reference_1m'].replace('_failure','')+'\nfirst sustained path separation: '+str(p['first_sustained_2m_path_divergence_step']))
 ax.legend(fontsize=8,loc='upper left')
fig.suptitle('Same initial state and action RNG, different frozen checkpoints\nMarkers: >2 m separation for 40 steps; not the first causal mistake')
fig.tight_layout();fig.savefig(R/'reports/checkpoint_pairs.png',dpi=140);plt.close(fig)
videos=[
 ('s0_k1000000_e007_failure','Near-goal miss followed by long low posture'),
 ('s0_k1000000_e023_failure','Junction-region timeout; earlier successful counterpart'),
 ('s1_k1000000_e048_failure','Repeated cell returns, wall contact and final low posture'),
 ('s1_k1000000_e038_failure','Mid-route persistent low posture'),
 ('s0_k1000000_e082_failure','Goal-region timeout without sustained final low posture'),
 ('s0_k900000_e023_earlier','Same scene as seed0 episode23, earlier success'),
 ('s1_k1000000_e048_rng','Same scene and weight as seed1 episode48, alternate RNG success'),
 ('s0_k800000_e082_earlier','Same scene as seed0 episode82, earlier success')]
(R/'configs/video_selection.json').write_text(json.dumps([dict(id=i,reason=t) for i,t in videos],indent=2))
print('PLOTS_AND_VIDEO_SELECTION',len(videos))

