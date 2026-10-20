from compact import *
import geometry_fast as gf
import subprocess
start=time.time();torch.set_num_threads(1);os.nice(10)
torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False
cfg=json.load(open(R/'frozen_config.json'));summary=json.load(open(R/'results/C3_summary.json'))
eligible=summary['P3_eligible'];families=['G','C']+eligible;seed=20261018
if not eligible:
 write(R/'results/speed_summary.json',dict(status='NOT_RUN_NO_ELIGIBLE_METHOD',eligible=[]));sys.exit(0)
heads={};norms={}
for fam in families:
 if fam in ['G','C']:
  v=next(v for v in cfg['protected_G_C_heads'] if v['family']==fam and v['seed']==seed)
  assert sha(v['path'])==v['sha256'];ck=torch.load(v['path'],map_location='cpu',weights_only=False)
  net=network(390) if fam=='G' else CompactHead()
 else:
  v=json.load(open(R/f'results/train_{fam}_{seed}.json'))
  assert sha(v['checkpoint'])==v['checkpoint_sha256'];ck=torch.load(v['checkpoint'],map_location='cpu',weights_only=False);net=initialize(fam,seed)
 net.load_state_dict(ck['model']);heads[fam]=net.cuda().eval();norms[fam]=ck['geom_norm'];feat_norm=(ck['mean'],ck['std'])
def run(fam,scene,frame,pred,feat,act):
 torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();t=time.perf_counter()
 cloud,_=visible_fast(scene,frame,pred);t1=time.perf_counter()
 tree=(RegionTree if fam=='SF' else CachedTree)(cKDTree(cloud),pred);t2=time.perf_counter()
 raw=[];points=[];mask=[]
 for pose in pred:
  if fam=='G':raw.append(gf.geom_cov(cloud,tree,pose));continue
  local=(cloud[tree.query_ball_point(pose[13:16],.16)]-pose[13:16])@pose[4:13].reshape(3,3)
  if fam=='C':raw.append(compact_fast(local,pose));continue
  masks=regions(local,pose)
  if fam=='H':raw.append(compact_values(local,pose,masks))
  pp,mm=point_values(local,pose,masks);points.append(pp);mask.append(mm)
 raw=np.asarray(raw,np.float32);points=np.asarray(points,np.float32);mask=np.asarray(mask,bool)
 x=np.concatenate([(feat-feat_norm[0])/feat_norm[1],act],-1).astype(np.float32)
 if len(raw):gg=geom_input(raw,norms[fam])
 t3=time.perf_counter();xx=torch.from_numpy(x).cuda()
 if len(raw):cc=torch.from_numpy(gg).cuda()
 if len(points):pp=torch.from_numpy(points).cuda();mm=torch.from_numpy(mask).cuda()
 torch.cuda.synchronize();t4=time.perf_counter()
 with torch.no_grad():
  if fam in ['SF','H']:
   encoded=heads[fam].encode(pp,mm);torch.cuda.synchronize();t5=time.perf_counter()
   q=heads[fam].score(torch.cat([xx,cc,encoded] if fam=='H' else [xx,encoded],-1)).squeeze(-1)
  else:
   t5=t4;q=(heads[fam](torch.cat([xx,cc],-1)) if fam=='G' else heads[fam](xx,cc)).squeeze()
 torch.cuda.synchronize();t6=time.perf_counter();q=q.cpu().numpy();torch.cuda.synchronize();t7=time.perf_counter()
 fused_score(pred[:,0],q);t8=time.perf_counter()
 return dict(preprocess=t1-t,neighborhood=t2-t1,geometry_sampling=t3-t2,H2D=t4-t3,encoding=t5-t4,score_network=t6-t5,D2H=t7-t6,fusion=t8-t7,total=t8-t,peak_gpu_bytes=torch.cuda.max_memory_allocated()),raw,points,mask,q
results=[];snapshots=[]
for i,(sc,fr) in enumerate((s,f) for s in sum(cfg['evaluation']['C3'].values(),[]) for f in cfg['data']['frames']):
 with np.load(P/f'B/exports/subset/scene_{sc:04d}/{fr:04d}.npz') as z:pred=z['pred'];feat=z['feature320'];act=z['original_action']
 if i==0:
  for fam in families:run(fam,sc,fr,pred,feat,act)
 order=families[i%len(families):]+families[:i%len(families)];outputs={}
 for fam in order:
  timing,*out=run(fam,sc,fr,pred,feat,act);outputs[fam]=out;results.append(dict(scene=sc,frame=fr,family=fam,order='-'.join(order),**timing))
 with np.load(G/f'eg_lg1_local_geometry_v1_20261020/geometry/C3/{sc:04d}/{fr:04d}.npz') as z:gold=z['raw'][:,0]
 assert np.array_equal(outputs['G'][0],gold,equal_nan=True)
 assert np.array_equal(outputs['C'][0],gold[:,COLUMNS],equal_nan=True)
 cc,canon,cm,_=extract(sc,fr,pred,mode='both')
 with np.load(R/f'scores/{sc:04d}/{fr:04d}.npz') as z:
  for fam in eligible:
   raw,pp,mm,q=outputs[fam]
   assert np.array_equal(pp,canon) and np.array_equal(mm,cm)
   if fam=='H':assert np.array_equal(raw,cc,equal_nan=True)
   assert np.allclose(q,z[f'{fam}_{seed}'],rtol=1e-6,atol=1e-6)
 if i%8==0:
  snapshots.append(dict(frame_index=i,time=time.time(),load=subprocess.check_output(['nvidia-smi','--query-gpu=index,utilization.gpu,memory.used,memory.free','--format=csv,noheader'],text=True)))
  print('TIMING',i+1,96,round(time.time()-start,1),flush=True)
csv_write(R/'results/speed_per_frame.csv',results);stats=[]
for fam in families:
 rows=[v for v in results if v['family']==fam];a=dict(family=fam,frames=96,peak_gpu_bytes=max(v['peak_gpu_bytes'] for v in rows))
 for key in ['preprocess','neighborhood','geometry_sampling','H2D','encoding','score_network','D2H','fusion','total']:
  v=[r[key] for r in rows];a[key]=dict(median=float(np.median(v)),mean=float(np.mean(v)),p95=float(np.quantile(v,.95)))
 stats.append(a)
g=stats[0]
for a in stats:a['median_ratio_to_G']=a['total']['median']/g['total']['median'];a['p95_ratio_to_G']=a['total']['p95']/g['total']['p95']
write(R/'results/speed_summary.json',dict(status='PASS',eligible=eligible,measurements=stats,frames=96,load_snapshots=snapshots,seconds=time.time()-start,GPU_slot=os.environ['CUDA_VISIBLE_DEVICES'],all_geometry_equivalence_pass=True))
print('TIMING_DONE',json.dumps(stats),flush=True)
