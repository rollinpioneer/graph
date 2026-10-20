from visibility import *
from concurrent.futures import ProcessPoolExecutor
import pandas as pd
def check(row):
 start=time.time();s=int(row['scene']);f=int(row['frame']);idx=int(row['candidate']);obs=Observation(s,f)
 # Backprojection/projection roundtrip on actual valid depth pixels.
 points=obs.points[obs.depth>0][::97];st,res,_,_=obs.query(points)
 assert np.all(st==1) and np.max(np.abs(res))<1e-9
 n=native(s);n.frame(f);p=np.load(export(s,f))['pred'][idx].astype(float);p[1]=np.clip(p[1],0,.1)
 owner=int(n.owners(p[None])[0]);model=n.mt[owner];lo=model.min(0)-.05;hi=model.max(0)+.05
 allpts=n.allpts;inside=((allpts>lo)&(allpts<hi)).all(1);pts=allpts[inside]
 local=(pts-p[13:16])@p[4:13].reshape(3,3);x,y,z=local.T;w=p[1]
 h=(z>-.01)&(z<.01);finger=h&(x>-.02)&(x<p[3])&(np.abs(y)>w/2)&(np.abs(y)<w/2+.01)
 palm=h&(x>-.03)&(x<-.02)&(np.abs(y)<w/2+.01);hit=finger|palm
 st,res,_,_=obs.query(pts[hit]);table=(np.flatnonzero(inside)>=len(n.objpts))[hit]
 free=res[st==0]
 return dict(scene=s,frame=f,candidate=idx,roundtrip_points=len(points),roundtrip_max_error_m=float(np.max(np.abs(obs.query(points)[1]))),collision_points=int(hit.sum()),table_fraction=float(table.mean()),free_points=len(free),free_residual_mm_median=float(np.median(free)*1000) if len(free) else None,free_residual_mm_p10=float(np.quantile(free,.1)*1000) if len(free) else None,surface_fraction=float((st==1).mean()),occluded_fraction=float((st==2).mean()),seconds=time.time()-start)
if __name__=='__main__':
 os.nice(10);d=pd.read_csv(R/'results/Top1_collision_visibility.csv');rows=d.sort_values(['scene','frame','head']).groupby('scene').first().reset_index().to_dict('records')
 with ProcessPoolExecutor(max_workers=12) as pool:out=list(pool.map(check,rows))
 write(R/'results/projection_sanity.json',dict(status='PASS',selection='first nonempty-collision frame per C3 scene, descriptive only',examples=out,worker_seconds=sum(x['seconds'] for x in out)))
