import os,sys,json,time,subprocess,concurrent.futures
from pathlib import Path
from validate import Geometry,E,N,np,rx
if len(sys.argv)>1:
 os.nice(10);scene=int(sys.argv[1]);t=time.time();ctx=Geometry(scene)
 p=N/f'results/subset_dev_{scene:04d}/records.json'
 rows=[r for r in json.loads(p.read_text()) if r['rank']==0 and not (0<r['actual']<=.4000001)];out=[]
 for r in rows:
  f=r['frame'];ctx.frame(f);z=np.load(E/f'results/subset/dev/scene_{scene:04d}/evaluated/{f:04d}.npz');g=z['grasps'][0];owner=int(ctx.owners(g[None])[0])
  grid=np.tile(g,(48,1));grid[:,3]=np.tile(np.array([.01,.02,.03,.04],np.float32),12)
  for k in range(12):grid[k*4:(k+1)*4,4:13]=(g[4:13].reshape(3,3)@rx(k*np.pi/12)).ravel()
  q,c,_=ctx.score(grid,owner);di=int(np.abs(grid[:4,3]-g[3]).argmin());assert grid[di,3]==g[3]
  assert q[di]==r['actual'] and bool(c[di])==r['collision']
  good=(q>0)&(q<=.4000001);out.append(dict(scene=scene,frame=f,label_frame_pose_pass=bool(0<r['pose_only']<=.4000001),decoder_pose_pass=bool(good.any()),angle_only_pass=bool(good[di::4].any()),depth_only_pass=bool(good[:4].any()),q=q.tolist()))
 target=N/'decoder_control';target.mkdir(exist_ok=True);(target/f'{scene:04d}.json').write_text(json.dumps(dict(rows=out,elapsed=time.time()-t,quality_calls=ctx.calls)))
 print(scene,len(rows),sum(r['decoder_pose_pass'] for r in out),flush=True)
else:
 scenes=json.loads((N/'config.json').read_text())['dev_scenes'];out=[]
 def run(s):
  with (N/f'logs/decoder_{s}.log').open('w') as f:return subprocess.run([sys.executable,__file__,str(s)],stdout=f,stderr=subprocess.STDOUT).returncode
 with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:exits=list(pool.map(run,scenes))
 assert not any(exits),exits
 for p in sorted((N/'decoder_control').glob('*.json')):out+=json.loads(p.read_text())['rows']
 d=dict(failed_top1=len(out),original_label_frame_pose_repaired=sum(r['label_frame_pose_pass'] for r in out),decoder_pose_repaired=sum(r['decoder_pose_pass'] for r in out),angle_only=sum(r['angle_only_pass'] for r in out),depth_only=sum(r['depth_only_pass'] for r in out),both_single=sum(r['angle_only_pass'] and r['depth_only_pass'] for r in out),joint_only=sum(r['decoder_pose_pass'] and not r['angle_only_pass'] and not r['depth_only_pass'] for r in out),quality_calls=48*len(out),notes='Supplementary check prompted by action-grid definition: original decoded rotation plus k*15 degrees, original four depths, original width/center/view; only frozen development Top-1 failures. Not a causal learning ablation.')
 (N/'decoder_control_summary.json').write_text(json.dumps(d,indent=2));print(d,flush=True)
