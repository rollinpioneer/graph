import json,numpy as np
import evaluate_angles as A
receipts=[]
for s,f,cid in [(100,0,760),(100,64,752)]:
 a=np.load(A.ROOT/f'evaluation/{s:04d}_{f:04d}.npz');j=int(np.flatnonzero(a['candidate_id']==cid)[0]);direct=A.E.Direct(s);direct.frame(f)
 r=direct.score(a['pred'][j]);checks={k:bool(np.array_equal(r[k],a[k][j],equal_nan=True)) for k in ['mu','collision','empty']}
 assert all(checks.values()),checks
 receipts.append(dict(scene=s,frame=f,candidate_id=cid,n=12,checks=checks))
(A.ROOT/'coarse_recheck.json').write_text(json.dumps(receipts));print(json.dumps(receipts))
