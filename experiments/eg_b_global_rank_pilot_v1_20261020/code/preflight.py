from common import *
import concurrent.futures,subprocess,resource,torch
torch.set_num_threads(1);start=time.time();os.nice(10)
split=json.loads((P/'B_head_split.json').read_text())

def load_scene(task):
 role,s=task;items=[];manifest=[]
 for frame in split['frame_ids']:
  p=P/f'B/labels/{s:04d}_{frame:04d}.npz'
  with np.load(p) as z:a={k:z[k].copy() for k in z.files}
  mu=a['q'];assert mu.shape==(128,6) and np.isfinite(mu).all()
  assert np.isclose(mu[...,None],np.r_[-1.,FR],atol=1e-7,rtol=0).any(-1).all()
  assert len(np.unique(a['ids']))==128 and ((a['ids']>=0)&(a['ids']<1024)).all()
  assert (a['collision'][a['empty']]).all() and (mu[a['collision']]==-1).all()
  assert a['feature'].dtype==a['action'].dtype==np.float32
  ep=P/f'B/exports/subset/scene_{s:04d}/{frame:04d}.npz';eh=sha(ep);assert eh==str(a['export_sha256'])
  with np.load(ep) as z:
   ids=a['ids'];assert np.array_equal(a['feature'],z['feature320'][ids])
   assert np.array_equal(a['action'],z['actions'][ids])
   assert np.array_equal(z['hypotheses'][ids,0],z['pred'][ids])
   assert np.array_equal(a['action'][:,0],z['original_action'][ids])
   order=np.lexsort((np.arange(1024),-z['pred'][:,0]));assert np.array_equal(ids[:64],order[:64])
  g,w=masks(mu[:,0],a['collision'][:,0],a['empty'][:,0]);pairs=int(g.sum()*w.sum())
  items.append(dict(role=role,scene=s,frame=frame,ids=ids,feature=a['feature'],action=a['action'],mu=mu,collision=a['collision'],empty=a['empty'],g=g,w=w,pairs=pairs))
  manifest.append(dict(label=str(p),label_sha256=sha(p),export=str(ep),export_sha256=eh))
 return items,manifest

tasks=[(role,s) for role in ['fit','model_val','gate_cal'] for s in split[role]]
def cpu_sample():
 v=list(map(int,Path('/proc/stat').read_text().splitlines()[0].split()[1:]));return sum(v[:8]),v[3]+v[4]
a=cpu_sample();time.sleep(1);b=cpu_sample();busy=1-(b[1]-a[1])/(b[0]-a[0]);workers=min(32,max(1,int((1-busy)*os.cpu_count())))
write(R/'results/P0_assignment.json',dict(workers=workers,cpu_busy_percent=busy*100))
frames=[];manifest=[]
with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as pool:
 for items,mi in pool.map(load_scene,tasks):frames.extend(items);manifest.extend(mi)
assert len(frames)==256
fit=np.concatenate([f['feature'] for f in frames if f['role']=='fit']);mean=fit.mean(0);std=fit.std(0).clip(1e-6)
lock=json.loads((P/'C3_method_lock.json').read_text());source_ids=[];old=[]
for h in lock['heads']['B']:
 assert sha(h['path'])==h['sha256'];ck=torch.load(h['path'],map_location='cpu',weights_only=False)
 assert np.array_equal(mean,ck['mean']) and np.array_equal(std,ck['std'])
 m=model();m.load_state_dict(ck['model']);m.eval();old.append(m);source_ids.append(h)
back=P/'runs/B/train/economicgrasp_epoch10.tar';assert sha(back)==lock['backbones']['B']
rows=[];strata=[]
for f in frames:
 x=np.concatenate([(f['feature']-mean)/std,f['action'][:,0]],-1);g,w=f['g'],f['w']
 row={k:f[k] for k in ['role','scene','frame','pairs']};row.update(strong=int(g.sum()),weak=int(w.sum()))
 for seed,m in zip(SEEDS,old):
  with torch.no_grad():q=m(torch.from_numpy(x)).squeeze(-1).numpy()
  diff=q[g,None]-q[None,w];row[f'O_{seed}_inversion']=float(((diff<0)+.5*(diff==0)).mean()) if diff.size else None
 rows.append(row)
 for zone,ix in [('top64',slice(0,64)),('random64',slice(64,128))]:
  mu,c,e=f['mu'][ix,0],f['collision'][ix,0],f['empty'][ix,0];gg,ww=masks(mu,c,e)
  strata.append(dict(role=f['role'],scene=f['scene'],frame=f['frame'],zone=zone,N=64,strong=int(gg.sum()),weak=int(ww.sum()),empty=int(e.sum()),nonempty_collision=int((c&~e).sum()),weak_mu06=int(((~c)&(~e)&np.isclose(mu,.6)).sum()),weak_no_closure=int(((~c)&(~e)&(mu==-1)).sum())))
coverage={}
for role in ['fit','model_val','gate_cal']:
 rr=[v for v in rows if v['role']==role and v['pairs']>0];coverage[role]=dict(total_frames=sum(f['role']==role for f in frames),supported_scenes=len({v['scene'] for v in rr}),supported_frames=len(rr),distinct_strong=sum(v['strong'] for v in rr),distinct_weak=sum(v['weak'] for v in rr),pairs=sum(v['pairs'] for v in rr))
passed=coverage['fit']['supported_scenes']>=12 and coverage['fit']['supported_frames']>=48 and min(coverage['fit']['distinct_strong'],coverage['fit']['distinct_weak'])>=256 and all(coverage[r]['supported_scenes']>=2 and coverage[r]['supported_frames']>=8 for r in ['model_val','gate_cal'])
tests={};g=torch.tensor([True,False]);w=~g
tests['inversion_penalized']=bool(rank_loss(torch.tensor([.1,.8]),g,w)>0)
tests['margin_satisfied_zero']=bool(rank_loss(torch.tensor([.8,.2]),g,w)==0)
tests['no_pairs_is_MSE']=bool(rank_loss(torch.tensor([.1,.8]),g,torch.zeros(2,dtype=torch.bool))==0)
assert all(tests.values());write(R/'results/loss_microtests.json',tests)
np.savez_compressed(R/'prepared.npz',feature=np.stack([f['feature'] for f in frames]),action=np.stack([f['action'] for f in frames]),mu=np.stack([f['mu'] for f in frames]),collision=np.stack([f['collision'] for f in frames]),empty=np.stack([f['empty'] for f in frames]),g=np.stack([f['g'] for f in frames]),w=np.stack([f['w'] for f in frames]),scene=np.array([f['scene'] for f in frames]),frame=np.array([f['frame'] for f in frames]),ids=np.stack([f['ids'] for f in frames]),role=np.array([f['role'] for f in frames]),mean=mean,std=std)
csv_write(R/'results/pair_coverage_per_frame.csv',rows);csv_write(R/'results/sample_strata.csv',strata);write(R/'results/source_manifest.json',dict(labels=manifest,heads=source_ids,backbone_sha256=lock['backbones']['B'],prepared_sha256=sha(R/'prepared.npz')))
write(R/'results/pair_coverage.json',dict(status='PASS' if passed else 'INSUFFICIENT_PAIR_SUPPORT',roles=coverage,gradient_fit_actions=147456,cached_actions=196608,original_mapping_all_passed=True,normalization_matches_all_O=True,seconds=time.time()-start,workers=workers,cpu_seconds_including_children=time.process_time()+resource.getrusage(resource.RUSAGE_CHILDREN).ru_utime+resource.getrusage(resource.RUSAGE_CHILDREN).ru_stime))
print(json.dumps(coverage),passed,flush=True)
