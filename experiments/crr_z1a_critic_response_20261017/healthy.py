from common import *
import time,resource,random
seed=int(sys.argv[1]);start=600000 if seed==1 else 300000;end=660000 if seed==1 else 320000
out=R/f'runs/healthy_s{seed}';out.mkdir(exist_ok=False);np.random.seed(2026101700+seed);random.seed(2026101700+seed)
assert json.loads((R/'healthy_generation_budget.json').read_text())['status']=='AUTHORIZED_BY_USER'
asset=next(r for r in json.loads((R/'asset_manifest.json').read_text()) if r['seed']==seed and r['step']==start and r['role'].startswith('proposed_healthy_generation_source'))
assert sha(asset['path'])==asset['sha256'];p=read(asset['path'],start);g=restore(p,seed);initial=digest(p);data=dataset();N=len(data['observations']);t=time.monotonic()
probe_data=dict(np.load(R/'arrays/probes.npz'));fixed=[]
for j in range(2):
 aa=[];va=[]
 for k in range(0,4096,256):
  a,v=fixed_actions(g.network.params,probe_data['observations'][k:k+256],probe_data['next_observations'][k:k+256],probe_data[f'z{j}'][k:k+256]);aa.append(np.asarray(a));va.append(np.asarray(v))
 fixed.append([np.concatenate(aa),np.concatenate(va)])
fixed=np.array(fixed);logs=[]
def diagnostics(step):
 o=[]
 for j in range(2):
  qq=[];gg=[]
  for k in range(0,4096,256):
   q,gr=probe(g.network.params['modules_critic'],g.network.params['modules_target_critic'],probe_data['next_observations'][k:k+256],fixed[j,0,k:k+256],fixed[j,1,k:k+256]);qq.append(np.asarray(q));gg.append(np.linalg.norm(np.asarray(gr),axis=-1))
  q=np.concatenate(qq);gr=np.concatenate(gg);m=probe_data['start'];o.append(dict(latent=j,Q_start=float(q[m].mean()),Q_nonstart=float(q[~m].mean()),G_start=float(gr[m].mean()),G_nonstart=float(gr[~m].mean())))
 logs.append(dict(step=step,probes=o,elapsed_seconds=time.monotonic()-t));save(out/'progress.json',dict(step=step,source_step=start,updates=step-start,budget=end-start,seconds=time.monotonic()-t,latest=logs[-1]))
diagnostics(start)
for step in range(start+4,end+1,4):
 g,info=g.multi_sample_and_update(4,data,N,256)
 if step==start+4 or step%1000==0:
  loss={k:float(v) for k,v in info.items()};assert all(np.isfinite(list(loss.values())));assert int(g.network.step)==step+1 and int(g.network.opt_state[0].count)==step
  if step==start+4:
   path=out/'admission_four_updates.pkl';state=flax.serialization.to_state_dict(g);atomic_pickle(path,dict(agent=state,source=asset['path'],updates=4));rest=restore(read(path,step),seed);assert digest(flax.serialization.to_state_dict(rest))==digest(state)
   save(out/'admission.json',dict(status='PASS',updates=4,counted_in_budget=True,source_Adam_Agent_rng_exactly_restored=True,save_restore_exact=True,ordinary_official_fused4=True))
 if step%10000==0 or step==end:
  diagnostics(step);print('HEALTH',seed,step,round(time.monotonic()-t,1),flush=True)
 if step in ([650000,660000] if seed==1 else [310000,320000]):
  atomic_pickle(out/f'params_{step}.pkl',dict(agent=flax.serialization.to_state_dict(g),source=asset['path'],generated_for='CRR-Z1a',ordinary_official_GFP=True,step=step))
assert digest(p)==initial and sha(asset['path'])==asset['sha256'];save(out/'probe_history.json',logs)
save(out/'complete.json',dict(status='COMPLETE',seed=seed,updates=end-start,step=end,seconds=time.monotonic()-t,source_sha256=asset['sha256'],environment_operations=0,online_updates=0,actor_and_vabc_updates=end-start,cpu_user_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_utime,cpu_system_seconds=resource.getrusage(resource.RUSAGE_SELF).ru_stime,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,new_checkpoints=[dict(path=str(x),sha256=sha(x)) for x in out.glob('params_*.pkl')],global_numpy_python_rng_sidecar_missing_at_original_source=True,not_claimed_bitwise_original_unsaved_trajectory=True))
print('HEALTH_COMPLETE',seed,flush=True)
