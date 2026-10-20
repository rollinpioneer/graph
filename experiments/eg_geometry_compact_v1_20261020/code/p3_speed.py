from compact import *
import geometry_fast as gf
import resource,subprocess

start=time.time();ct=time.process_time();os.nice(10);torch.set_num_threads(1);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
summary=json.load(open(R/'results/C3_summary.json'));families=['G']+summary['P3_eligible'];cfg=json.load(open(R/'frozen_config.json'));heads={};norms={};seed=20261018
if len(families)==1:
 write(R/'results/speed_summary.json',dict(status='NOT_RUN_NO_ELIGIBLE_METHOD',eligible=[],GPU_process_seconds_upper=0,CPU_worker_seconds_upper=0));sys.exit(0)
for family in families:
 if family=='G':
  identity=next(v for v in cfg['protected_heads'] if v.get('family')=='G' and v['seed']==seed);assert sha(identity['path'])==identity['sha256'];ck=torch.load(identity['path'],map_location='cpu',weights_only=False);net=network(390)
 else:
  identity=json.load(open(R/f'results/train_{family}_{seed}.json'));assert sha(identity['checkpoint'])==identity['checkpoint_sha256'];ck=torch.load(identity['checkpoint'],map_location='cpu',weights_only=False);net=initialize(family,seed)
 net.load_state_dict(ck['model']);heads[family]=net.cuda().eval();norms[family]=ck.get('geom_norm');feature_norm=(ck['mean'],ck['std'])

def run(family,scene,frame,pred,feat,action):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter();cloud,_=visible_fast(scene,frame,pred);t1=time.perf_counter();tree=(RegionTree if family=='S' else CachedTree)(cKDTree(cloud),pred);t2=time.perf_counter()
 raw=[];pts=[];valid=[]
 for pose in pred:
  if family=='G':raw.append(gf.geom_cov(cloud,tree,pose));continue
  local=(cloud[tree.query_ball_point(pose[13:16],.16)]-pose[13:16])@pose[4:13].reshape(3,3)
  if family=='C':raw.append(compact_fast(local,pose))
  else:
   pp,mm=point_values(local,pose,regions(local,pose));pts.append(pp);valid.append(mm)
 x=np.concatenate([(feat-feature_norm[0])/feature_norm[1],action],-1).astype(np.float32)
 if family in ['G','C']:raw=np.asarray(raw,np.float32);gg=geom_input(raw,norms[family])
 else:pts=np.asarray(pts,np.float32);valid=np.asarray(valid,bool)
 t3=time.perf_counter();xx=torch.from_numpy(x).cuda()
 if family in ['G','C']:geo=torch.from_numpy(gg).cuda()
 else:pp=torch.from_numpy(pts).cuda();mm=torch.from_numpy(valid).cuda()
 torch.cuda.synchronize();t4=time.perf_counter()
 with torch.no_grad():
  if family=='S':encoded=heads[family].encode(pp,mm);torch.cuda.synchronize();t5=time.perf_counter();q=heads[family].score(torch.cat([xx,encoded],-1)).squeeze(-1)
  else:
   t5=t4;inputs=torch.cat([xx,geo],-1);q=(heads[family](inputs) if family=='G' else heads[family].score(inputs)).squeeze(-1)
 torch.cuda.synchronize();t6=time.perf_counter();q=q.cpu().numpy();torch.cuda.synchronize();t7=time.perf_counter();score,info=fused_score(pred[:,0],q);t8=time.perf_counter()
 timing=dict(preprocess=t1-t,neighborhood=t2-t1,geometry_sampling=t3-t2,H2D=t4-t3,encoding=t5-t4,score_network=t6-t5,D2H=t7-t6,fusion=t8-t7,total=t8-t,peak_gpu_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved())
 return timing,raw if family!='S' else (pts,valid),q

scenes=sum(cfg['evaluation']['C3'].values(),[]);pairs=[(s,f) for s in scenes for f in cfg['data']['frames']];results=[];snapshots=[]
for k,(scene,frame) in enumerate(pairs):
 with np.load(P/f'B/exports/subset/scene_{scene:04d}/{frame:04d}.npz') as z:pred=z['pred'];feat=z['feature320'];action=z['original_action']
 if k==0:
  for family in families:run(family,scene,frame,pred,feat,action)
 order=families[k%len(families):]+families[:k%len(families)];outputs={}
 for family in order:
  timing,raw,q=run(family,scene,frame,pred,feat,action);outputs[family]=(raw,q)
  results.append(dict(scene=scene,frame=frame,family=family,order='-'.join(order),**timing))
 # Untimed equivalence check on every timing frame; all S tokens/masks against canonical .16.
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/geometry/C3/{scene:04d}/{frame:04d}.npz') as z:gold=z['raw'][:,0]
 assert np.array_equal(outputs['G'][0],gold,equal_nan=True)
 if 'C' in families:assert np.array_equal(outputs['C'][0],gold[:,COLUMNS],equal_nan=True)
 if 'S' in families:
  cc,canonical,canonical_mask,_=extract(scene,frame,pred,mode='both');points,mask=outputs['S'][0];assert np.array_equal(points,canonical) and np.array_equal(mask,canonical_mask)
 with np.load(R/f'scores/{scene:04d}/{frame:04d}.npz') as z:
  for family in families:
   if family!='G':assert np.allclose(outputs[family][1],z[f'{family}_{seed}'],rtol=1e-6,atol=1e-6)
 if k%8==0:
  snapshots.append(dict(frame_index=k,time=time.time(),load=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True)));print('SPEED',k+1,96,round(time.time()-start,2),flush=True)
csv_write(R/'results/speed_per_frame.csv',results);statistics=[]
for family in families:
 selected=[v for v in results if v['family']==family];stat=dict(family=family,frames=96)
 for name in ['preprocess','neighborhood','geometry_sampling','H2D','encoding','score_network','D2H','fusion','total']:
  values=[v[name] for v in selected];stat[name]=dict(median=float(np.median(values)),p95=float(np.quantile(values,.95)),mean=float(np.mean(values)))
 stat['peak_gpu_bytes']=max(v['peak_gpu_bytes'] for v in selected);stat['peak_reserved_bytes']=max(v['peak_reserved_bytes'] for v in selected);statistics.append(stat)
g=next(v for v in statistics if v['family']=='G')
for stat in statistics:stat['median_ratio_to_G']=stat['total']['median']/g['total']['median'];stat['p95_ratio_to_G']=stat['total']['p95']/g['total']['p95']
result=dict(status='PASS',seed=seed,frames=96,eligible=families[1:],measurements=statistics,all_geometry_equivalence_pass=True,S_acceleration_exact_all96='S' in families,primary='one head deployment',three_head_secondary='Measured network cost can be multiplied by3 while CPU extraction/input transfer are shared; extrapolation only, not separately timed.',GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],load_snapshots=snapshots,seconds=time.time()-start,cpu_seconds=time.process_time()-ct,GPU_process_seconds_upper=time.time()-start,CPU_worker_seconds_upper=time.time()-start,peak_rss_kib=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,shared_environment=True)
write(R/'results/speed_summary.json',result);print('SPEED_COMPLETE',json.dumps(statistics),flush=True)
