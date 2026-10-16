"""One coarse check of the main spatial result, provenance and scientific plots."""
from pathlib import Path
import json,hashlib,importlib.metadata as md
import numpy as np
import matplotlib;matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[1];P=Path('/home/xushijie3/work/gfp_task4_baseline_pilot_v1_20261015');Z=Path('/home/xushijie3/work/gfp_guidance_distance_z01_20261015');C=Path('/home/xushijie3/work/flow_critic_comparative_train_v1_20261014')
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def save(n,x):(R/'reports'/n).write_text(json.dumps(x,indent=2,allow_nan=False))
check=[]
for split in ['train','validation']:
 s=dict(np.load(P/'arrays'/f'{split}_samples.npz'));o=s['observations'];ij=((o[:,:2]+6)/4).astype(int)[:,::-1];cid=ij[:,0]*8+ij[:,1];stable=(o[:,23]>=1.2)&(o[:,38]>=.7)
 for seed in [0,1]:
  for cp in [200000,600000,1000000]:
   a=dict(np.load(P/'arrays'/f's{seed}_{cp}.npz'));q=a['qdata'][(a['repeat']==0)&(a['split']==(split=='validation'))];vals=[];eps=[]
   for e in np.unique(s['trajectory']):
    u=(s['trajectory']==e)&stable&(cid==53);v=(s['trajectory']==e)&stable&(cid==54)
    if u.any() and v.any():vals.append(q[u].mean(0)-q[v].mean(0));eps.append(int(e))
   vals=np.array(vals);m=vals.mean(1);rng=np.random.default_rng(20261016);boot=m[rng.integers(len(m),size=(400,len(m)))].mean(1)
   check.append(dict(task=4,seed=seed,checkpoint=cp,split=split,episodes=len(m),episode_ids=eps,far_minus_near=float(m.mean()),ci95_trajectory_bootstrap=np.quantile(boot,[.025,.975]).tolist(),member_mean_differences=vals.mean(0).tolist(),posthoc_sensitivity=True,restriction='same original episode, head>=1.2, upright>=0.7, each cell represented; no matching of all physical state variables'))
save('first_edge_robustness.json',check)
identities=[]
for src in [P,Z]:
 for s in ['train','validation']:
  p=src/'arrays'/f'{s}_samples.npz';identities.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
 for seed in [0,1]:
  for cp in [200000,600000,1000000]:
   p=src/'arrays'/f's{seed}_{cp}.npz';identities.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
for p in [P/'reports/matched_results.json',P/'reports/start_failure/data_start_coverage.json',P/'reports/start_failure/START_FAILURE_REPORT.md',C/'reports/final_checkpoint_results.json',C/'reports/final_seed_results.json',P/'reports/geometry.json']:
 identities.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)))
configs=[]
for task in [1,4]:
 for seed in [0,1]:
  p=P/'runs'/f'GFP_seed{seed}'/'runtime.json' if task==4 else Z/'configs'/f'G0_seed{seed}_runtime.json'
  rt=json.loads(p.read_text());configs.append(dict(task=task,seed=seed,runtime_path=str(p),sha256=sha(p),agent_config=rt['agent_config'],output_dir=rt['output_dir']))
save('reused_asset_manifest.json',dict(files=identities,runtimes=configs,versions={x:md.version(x) for x in ['jax','jaxlib','flax','optax','numpy','mujoco','ogbench','gymnasium']},inference='float32; jax_enable_x64=False; default matmul precision preserved'))
geo=json.loads((P/'reports/geometry.json').read_text());data=json.loads((R/'reports/data_support.json').read_text());top=json.loads((R/'reports/value_topology.json').read_text());con=json.loads((R/'reports/controlled_contrasts.json').read_text())
fig,axs=plt.subplots(2,2,figsize=(13,11),layout='constrained');maze=np.array(geo['maze_map'])
for ax,task in zip(axs[0],[1,4]):
 ax.imshow(maze,cmap='Greys',vmin=0,vmax=1,origin='upper');d=next(x for x in data if x['task']==task and x['split']=='train');t=next(x for x in geo['tasks'] if x['task']==task)
 for e in d['shortest_edges']:
  u=np.array(e['source']);v=np.array(e['destination']);mid=(u+v)/2
  ax.annotate('',xy=v[::-1],xytext=u[::-1],arrowprops=dict(arrowstyle='->',color='#007c91',lw=2,shrinkA=4,shrinkB=4));ax.text(mid[1]+.11,mid[0]+.08,str(e['episodes']),fontsize=8,color='#b43b19',bbox=dict(fc='white',alpha=.8,pad=.8,ec='none'))
 ax.scatter(t['start_ij'][1],t['start_ij'][0],s=80,c='#e69f00',marker='s',zorder=4,label='Start');ax.scatter(t['goal_ij'][1],t['goal_ij'][0],s=100,c='#009e73',marker='*',zorder=4,label='Goal');ax.set_title(f"Task {task}: {d['complete_routes']} complete observed paths\nEdge labels = distinct training episodes");ax.set_xticks(range(8));ax.set_yticks(range(8));ax.set_xlabel('Column');ax.set_ylabel('Row');ax.legend(loc='upper right',fontsize=8)
ax=axs[1,0]
for seed,col in [(0,'#0072b2'),(1,'#d55e00')]:
 d=[x for x in top if x['task']==4 and x['seed']==seed and x['split']=='validation' and x['kind']=='shortest_edge' and x['far']==[6,5] and x['near']==[6,6]];d.sort(key=lambda x:x['checkpoint']);xx=np.array([x['checkpoint']/1000 for x in d]);yy=np.array([x['far_minus_near'] for x in d]);ci=np.array([x['ci95_cluster_normal'] for x in d]);ax.errorbar(xx,yy,yerr=np.stack([yy-ci[:,0],ci[:,1]-yy]),marker='o',capsize=4,c=col,label=f'Seed {seed}')
ax.axhline(0,color='grey',ls='--');ax.set_xlabel('Training checkpoint (k updates)');ax.set_ylabel('Q(start cell) - Q(east neighbor)');ax.set_title('Task4: first route edge, validation\nPosture/velocity adjusted, trajectory-cluster intervals');ax.legend()
ax=axs[1,1]
for task,col in [(1,'#0072b2'),(4,'#d55e00')]:
 for seed,mark in [(0,'o'),(1,'s')]:
  d=[x for x in con if x['task']==task and x['seed']==seed and x['split']=='validation' and x['metric']=='vabc_flow_mse'];d.sort(key=lambda x:x['checkpoint']);xx=np.array([x['checkpoint']/1000 for x in d])+(seed*2-1)*9;yy=np.array([x['detour_minus_regress'] for x in d]);ci=np.array([x['ci95_cluster_normal'] for x in d]);ax.errorbar(xx,yy,yerr=np.stack([yy-ci[:,0],ci[:,1]-yy]),marker=mark,capsize=3,c=col,ls='-' if seed==0 else '--',label=f'Task{task} seed{seed}')
ax.axhline(0,color='grey',ls='--');ax.set_xlabel('Training checkpoint (k updates)');ax.set_ylabel('VaBC flow-fit MSE: detour minus regress');ax.set_title('Behavior fit difficulty also occurs on task1\nCell/posture/velocity adjusted, validation');ax.legend(fontsize=8)
fig.suptitle('Global data, value and policy diagnosis — frozen assets, zero interaction',fontsize=15)
fig.savefig(R/'reports/joint_evidence.png',dpi=160);fig.savefig(R/'reports/joint_evidence.pdf');plt.close(fig)
# Lightweight exact provenance for all new arrays; full arrays remain remote.
save('array_index.json',[dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p)) for p in sorted((R/'arrays').glob('*.npz'))])
print('FINALIZE_DONE',[(x['seed'],x['checkpoint'],x['episodes'],round(x['far_minus_near'],2)) for x in check if x['split']=='validation'])
