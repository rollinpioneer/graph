import json,pathlib
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
r=pathlib.Path(__file__).resolve().parents[1];rows=json.load(open(r/'reports/branch_results.json'));ss=json.load(open(r/'configs/states.json'))['states']
ids=['near_goal_s0_e007_t964','near_goal_s0_e021_t1227','near_goal_s1_e033_t1292','pre_stall_s0_e027_t1287'];colors=['#333333','#1674b8','#d45d00'];labels=['Original Actor','Resampled Actor','VaBC']
fig,axes=plt.subplots(2,2,figsize=(11,7.5))
for ax,sid in zip(axes.flat,ids):
 c=next(c for c in ss if c['id']==sid);goal=np.load(c['source_npz'])['goal_xy'][c['step']]
 for x,col,lab in zip([x for x in rows if x['state']==sid],colors,labels):
  a=np.load(x['path']);d=np.linalg.norm(a['xy']-goal,axis=-1);ax.plot(np.arange(len(d)),d,color=col,label=lab+f' (Q={x["q_mean"]:.2f})')
  if x['success']:ax.scatter([len(d)-1],[d[-1]],color=col,s=36,marker='*',zorder=5)
 ax.axhline(.5,color='#888888',linestyle=':',label='Success boundary');ax.set_title(sid,fontsize=10);ax.set_xlabel('Steps after one-action intervention');ax.set_ylabel('Euclidean goal distance (m)');ax.set_xlim(0,300);ax.grid(alpha=.2);ax.legend(fontsize=8)
fig.suptitle('Frozen GFP: four local rescues under a shared subsequent noise sequence',fontsize=12)
fig.text(.5,.015,'Distance is descriptive, not a navigation reward or long-horizon Q estimate. No optimizer updates.',ha='center',fontsize=9)
fig.tight_layout(rect=[0,.04,1,.95]);fig.savefig(r/'reports/local_rescues.png',dpi=180);plt.close(fig)
