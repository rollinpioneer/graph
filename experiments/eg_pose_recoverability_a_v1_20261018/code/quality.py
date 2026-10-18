from core import *
import importlib.util
spec=importlib.util.spec_from_file_location('native_reference',G/'eg_nqs_pilot_v2_20261018/code/common.py');ref=importlib.util.module_from_spec(spec);spec.loader.exec_module(ref)
s=int(sys.argv[1]);os.nice(10);start=time.time();ctx=ref.Native(s);cohort=[x for x in json.loads((R/'results/cohort.json').read_text()) if x['scene']==s];cache={x['frame']:x for x in json.loads((PRIOR/f'decoder_control/{s:04d}.json').read_text())['rows']};new=0;reused=0
for row in cohort:
 f=row['frame'];g=np.load(E/f'results/subset/dev/scene_{s:04d}/evaluated/{f:04d}.npz')['grasps'][0];di=int(np.argmin(abs(np.arange(1,5)*.01-g[3])))
 if row['success']:
  ctx.frame(f);v=grid(g);v[di]=g;q,c,empty,owner=ctx.score(v);assert q[di]==row['original_quality'] and bool(c[di])==row['original_collision'],(s,f,q[di],row['original_quality']);new+=48
 else:
  q=np.array(cache[f]['q']);assert q[di]==row['original_quality'];c=np.full(48,False);empty=c.copy();reused+=48
 np.savez(R/f'quality/{s:04d}_{f:04d}.npz',q=q,collision=c,empty=empty,collision_available=row['success'])
write(R/f'results/quality_{s:04d}.json',dict(status='PASS',scene=s,frames=len(cohort),new_native_calls=new,reused_native_calls=reused,seconds=time.time()-start))
print('QUALITY',s,new,reused,flush=True)
