import os,argparse,time,resource,json
import numpy as np
from common import *
p=argparse.ArgumentParser();p.add_argument('--scene',type=int,required=True);p.add_argument('--role',default='subset');p.add_argument('--micro',action='store_true');a=p.parse_args()
os.nice(10);start=time.time();setup=time.perf_counter();geo=Native(a.scene);setup=time.perf_counter()-setup
records=[];calls=0
for frame in (frames(a.scene)[:2] if a.micro else frames(a.scene)):
    name='micro_labels' if a.micro else 'labels'
    out=R/f'{name}/{a.role}/scene_{a.scene:04d}/{frame:04d}.npz'
    if out.exists():continue
    z=np.load(R/f'exports/{a.role}/scene_{a.scene:04d}/{frame:04d}.npz')
    ids=z['sample_ids'][:64] if a.micro else z['sample_ids']
    ids=ids[z['M'][ids]]
    geo.frame(frame);hy=z['hypotheses'][ids];t=time.perf_counter()
    if len(ids):
        q,c,e,o=geo.score(hy.reshape(-1,17));calls+=len(q)
        q=q.reshape(-1,6);c=c.reshape(-1,6);e=e.reshape(-1,6);o=o.reshape(-1,6)
    else:q=np.zeros((0,6));c=e=np.zeros((0,6),bool);o=np.zeros((0,6),int)
    elapsed=time.perf_counter()-t;u=utility(q);A=u[:,0];B=u.max(1) if len(ids) else np.zeros(0)
    assert np.all(B>=A)
    check=None
    if a.micro:
        selected,oq,oc=geo.official(z['pred'])
        nq,nc,ne,no=geo.score(selected);calls+=len(nq)
        assert np.array_equal(nq,oq) and np.array_equal(nc,oc)
        check=dict(native_original_candidates=len(nq),quality_exact=True,collision_exact=True,ap=float(ap(oq).mean()*100))
    save_npz(out,ids=ids,mu=q,collision=c,empty=e,owner=o,A=A,B6=B,E=z['E'][ids],grasp=hy,export_sha256=np.array(sha(R/f'exports/{a.role}/scene_{a.scene:04d}/{frame:04d}.npz')))
    rec=dict(scene=a.scene,frame=frame,matched=len(ids),requested=64 if a.micro else 256,quality_calls=len(ids)*6,seconds=elapsed,B6_greater_A=int((B>A).sum()),native_check=check)
    records.append(rec);print(json.dumps(rec),flush=True)
write_json(R/f'receipts/label_{a.role}_{a.scene}_{"micro" if a.micro else "full"}.json',dict(status='PASS',rows=records,quality_calls=calls,setup_seconds=setup,wall_seconds=time.time()-start,rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss))
