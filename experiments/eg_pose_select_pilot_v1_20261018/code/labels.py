from core import *
import argparse
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--micro',action='store_true');a=p.parse_args();os.nice(10);start=time.time();cpu=time.process_time();ctx=Native(a.scene);nnew=0;checks=[]
for f in CFG['fit_frames'][:2] if a.micro else CFG['fit_frames']:
 pred,hyp,feat,act,actions=export(a.scene,f);ids=sample_ids(pred,a.scene,f);path=R/f'labels/{a.scene:04d}_{f:04d}.npz';ctx.frame(f)
 q=np.full((128,6),np.nan);c=np.zeros((128,6),bool);e=c.copy()
 if path.exists():
  z=np.load(path);assert np.array_equal(z['ids'],ids);q[:]=z['q'];c[:]=z['collision'];e[:]=z['empty']
 limit=16 if a.micro else 128;missing=np.flatnonzero(~np.isfinite(q[:limit]).all(1))
 if len(missing):
  vals,cs,es,_=ctx.score(hyp[ids[missing]].reshape(-1,17));q[missing]=vals.reshape(-1,6);c[missing]=cs.reshape(-1,6);e[missing]=es.reshape(-1,6);nnew+=len(vals)
 assert np.isfinite(q[:limit]).all();u=utility(q[:limit]);assert (u.max(1)>=u[:,0]).all()
 if a.micro:
  oq,oc,oe,_=ctx.score(pred[ids[:limit]]);assert np.array_equal(oq,q[:limit,0]) and np.array_equal(oc,c[:limit,0]) and np.array_equal(oe,e[:limit,0]);checks.append(dict(frame=f,original_quality_exact=True))
 np.savez(path,ids=ids,q=q,collision=c,empty=e,feature=feat[ids],action=act[ids],actions=actions[ids],export_sha256=np.array(sha(N/f'exports/subset/scene_{a.scene:04d}/{f:04d}.npz')))
print('LABEL',a.scene,nnew,flush=True);write(R/f'results/labels_{a.scene:04d}_{"micro" if a.micro else "full"}.json',dict(status='PASS',scene=a.scene,micro=a.micro,new_calls=nnew,seconds=time.time()-start,cpu_seconds=time.process_time()-cpu,checks=checks))
