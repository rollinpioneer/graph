import json,csv,math,hashlib
from pathlib import Path
from collections import Counter
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path(__file__).resolve().parent;ROOT=OUT.parent
def save(n,x):(OUT/n).write_text(json.dumps(x,indent=2,allow_nan=False))
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
geometry_path=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015/reports/geometry.json')
geo=json.loads(geometry_path.read_text());maze=np.array(geo['maze_map']);goal=np.array(geo['tasks'][0]['goal_xy'])
eps={it:{e['episode']:e for e in json.loads((ROOT/f'runs/A/eval_{it}/results.json').read_text())['episodes']} for it in (25000,50000)}
def measure(e):
 xy=np.array(e['path']);t=np.r_[0,np.arange(20,e['length']+1,20)]
 if e['length']%20:t=np.r_[t,e['length']]
 assert len(xy)==len(t) and np.allclose(xy[-1],e['final_xy'])
 inside=((xy>=-2)&(xy<6)).all(1);outside=np.where(~inside)[0];dist=np.linalg.norm(xy-goal,axis=1)
 row=dict(success=e['success'],length=e['length'],return_=e['return_'],end_start=e['final_in_start'],no_sampled_exit=bool(inside.all()),first_sampled_exit=int(t[outside[0]]) if len(outside) else None,exited_and_ended_start=bool(len(outside) and e['final_in_start']),min_goal_distance=float(dist.min()),final_goal_distance=float(dist[-1]),near_1m=bool(dist.min()<=1),near_2m=bool(dist.min()<=2),near_3m=bool(dist.min()<=3))
 for horizon in (100,200,400,800):
  take=t<=min(horizon,e['length']);row[f'displacement_{horizon}']=float(np.linalg.norm(xy[take][-1]-xy[0]));row[f'outside_by_{horizon}']=bool((~inside[take]).any())
 tail=xy[t>=max(0,e['length']-400)];row['last400_sampled_position_diameter']=float(np.sqrt(((tail[:,None,:]-tail[None,:,:])**2).sum(-1)).max())
 row['last400_low_spatial_range_1m']=row['last400_sampled_position_diameter']<=1
 ij=np.floor((xy[-1]+2)/4).astype(int)+1;row['final_cell_row_col']=str((int(ij[1]),int(ij[0])))
 return row,t,xy
rows=[];measurements={};groups={name:[] for name in ('lost','gained','both_success','both_failure')}
for eid in range(100):
 a,b=eps[25000][eid],eps[50000][eid];assert a['initial_obs_sha256']==b['initial_obs_sha256'] and np.array_equal(a['path'][0],b['path'][0])
 group={ (1,0):'lost',(0,1):'gained',(1,1):'both_success',(0,0):'both_failure'}[(a['success'],b['success'])];groups[group].append(eid)
 aa,ta,xa=measure(a);bb,tb,xb=measure(b);measurements[eid]={25000:aa,50000:bb}
 ma=(len(xa)-1 if a['length']%20 else len(xa));mb=(len(xb)-1 if b['length']%20 else len(xb));n=min(ma,mb);delta=np.linalg.norm(xa[:n]-xb[:n],axis=1)
 sustained=next((int(ta[j]) for j in range(max(0,n-4)) if np.all(delta[j:j+5]>1)),None)
 for it,m in [(25000,aa),(50000,bb)]:rows.append(dict(episode=eid,group=group,checkpoint=it,first_5_samples_spatial_difference_gt1m=sustained,**m))
with (OUT/'episode_comparison.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
summary=[]
for group,ids in groups.items():
 for it in (25000,50000):
  mm=[measurements[i][it] for i in ids];exits=[x['first_sampled_exit'] for x in mm if x['first_sampled_exit'] is not None]
  s=dict(group=group,checkpoint=it,n=len(ids),ids=ids)
  for key in ['end_start','no_sampled_exit','exited_and_ended_start','near_1m','near_2m','near_3m','outside_by_200','outside_by_400','outside_by_800','last400_low_spatial_range_1m']:s[key]=sum(x[key] for x in mm)
  for key in ['min_goal_distance','final_goal_distance','displacement_100','displacement_200','displacement_400','last400_sampled_position_diameter']:s[key+'_median']=float(np.median([x[key] for x in mm]))
  s['first_exit_median_among_exited']=float(np.median(exits)) if exits else None
  s['end_cells']=dict(Counter(x['final_cell_row_col'] for x in mm));summary.append(s)
lost=groups['lost'];classes={}
for i in lost:
 m=measurements[i][50000];classes[i]='No sampled exit' if m['no_sampled_exit'] else ('Within 2m of goal' if m['near_2m'] else 'Exited, not within 2m')
trans=Counter((eps[25000][i]['final_in_start'],eps[50000][i]['final_in_start']) for i in range(100))
save('path_summary.json',dict(groups=summary,lost_classes={k:[i for i,v in classes.items() if v==k] for k in sorted(set(classes.values()))},start_transition={str(k):v for k,v in trans.items()},path_sample_spacing=20,new_environment_steps=0,parameter_updates=0,initial_observations_match=True))
fig,axs=plt.subplots(2,2,figsize=(12,10));colors={'No sampled exit':'#c0392b','Within 2m of goal':'#8e44ad','Exited, not within 2m':'#2471a3'}
def draw(ax):
 for y,x in np.argwhere(maze):ax.add_patch(plt.Rectangle((4*(x-1)-2,4*(y-1)-2),4,4,color='#dddddd'))
 ax.scatter([0],[0],c='black',marker='o',s=35);ax.scatter([20],[20],c='green',marker='*',s=110);ax.set(xlim=(-2,26),ylim=(-2,26),xlabel='x (m)',ylabel='y (m)');ax.set_aspect('equal')
for ax,it in zip(axs[0],(25000,50000)):
 draw(ax)
 for i in lost:
  xy=np.array(eps[it][i]['path']);ax.plot(xy[:,0],xy[:,1],color=colors[classes[i]],alpha=.45,lw=.8)
 ax.set_title(f'Same 29 lost-success episodes: +{it//1000}k')
for k,c in colors.items():axs[0,1].plot([],[],color=c,label=f'{k}: {list(classes.values()).count(k)}')
axs[0,1].legend(fontsize=8,loc='upper left')
labels=['Lost (29)','Gained (16)','Both success (8)','Both failure (47)'];xx=np.arange(4)
for it,shift,c in [(25000,-.18,'#2471a3'),(50000,.18,'#d35400')]:
 yy=[next(x for x in summary if x['group']==g and x['checkpoint']==it)['no_sampled_exit'] for g in groups]
 axs[1,0].bar(xx+shift,yy,.36,label=f'+{it//1000}k',color=c)
axs[1,0].set(xticks=xx,xticklabels=labels,ylabel='Episodes',title='No exit in saved position samples');axs[1,0].tick_params(axis='x',labelsize=8);axs[1,0].legend()
for it,c in [(25000,'#2471a3'),(50000,'#d35400')]:
 vals=np.sort([measurements[i][it]['min_goal_distance'] for i in lost]);axs[1,1].step(vals,np.arange(1,30)/29,where='post',label=f'+{it//1000}k',color=c)
axs[1,1].set(xlabel='Minimum sampled goal distance (m)',ylabel='Fraction of 29 episodes',title='How far did lost episodes get?');axs[1,1].legend();axs[1,1].grid(alpha=.2)
fig.tight_layout();fig.savefig(OUT/'paired_paths.png',dpi=160);plt.close(fig)
save('path_receipt.json',dict(inputs={str(ROOT/f'runs/A/eval_{it}/results.json'):sha(ROOT/f'runs/A/eval_{it}/results.json') for it in (25000,50000)},geometry_sha256=sha(geometry_path),episodes=200,new_environment_steps=0,parameter_updates=0,limitations=['Only x/y every 20 steps and final point','No full observation, posture, action or simulator state saved for these episodes','Spatial separation is not causal first error','Near-goal 2m is exploratory, not success threshold']))
print(json.dumps(dict(lost_classes={k:list(classes.values()).count(k) for k in colors},summary=summary)))
