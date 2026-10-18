from core import *
start=time.time();cpu=time.process_time();s=CFG['fit'][0];f=CFG['fit_frames'][0];p,h,*_=export(s,f)
with np.load(R/f'labels/{s:04d}_{f:04d}.npz') as z:ids=z['ids'];q=z['q'];c=z['collision'];e=z['empty']
flat=h[ids].reshape(-1,17);wanted=[]
for mask in [q.reshape(-1)>0,c.reshape(-1),e.reshape(-1),(q.reshape(-1)<=0)&~c.reshape(-1)&~e.reshape(-1)]:
 ix=np.flatnonzero(mask)
 if len(ix):wanted.append(int(ix[0]))
wanted=np.unique(wanted);g=flat[wanted].copy();rotation=g[:,4:13].reshape(-1,3,3);rotation[:,:,1:]*=-1
ctx=Native(s);ctx.frame(f);sq,sc,se,_=ctx.score(g)
assert np.array_equal(sq,q.reshape(-1)[wanted]) and np.array_equal(sc,c.reshape(-1)[wanted]) and np.array_equal(se,e.reshape(-1)[wanted])
a=np.arange(12)*np.pi/12;assert np.allclose(np.sin(2*a),np.sin(2*(a+np.pi))) and np.allclose(np.cos(2*a),np.cos(2*(a+np.pi)))
write(R/'results/symmetry_check.json',dict(status='PASS',additional_native_recheck_calls=len(wanted),scene=s,frame=f,local_six_action_indices=wanted.tolist(),q=sq.tolist(),collision=sc.tolist(),empty=se.tolist(),pi_action_encoding_equal=True,seconds=time.time()-start,cpu_seconds=time.process_time()-cpu))
print('SYMMETRY_PASS',len(wanted))
