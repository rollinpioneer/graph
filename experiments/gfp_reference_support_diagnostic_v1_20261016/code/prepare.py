from util import *
import collections
t0=time.time();rng=np.random.default_rng(20261016);queries=[];records=[]
for split in ['train','val']:
 o,a,t,tid,p=raw(split);valid=~t;reg=region(o)
 records.append(dict(path=str(p),sha256=sha(p),rows=len(o),valid=int(valid.sum()),trajectories=int(t.sum())))
 if split=='train':
  normidx=rng.choice(np.flatnonzero(valid),100000,replace=False)
  mean=o[normidx].mean(0);std=np.maximum(o[normidx].std(0),.01)
  np.savez(ROOT/'arrays/normalizer.npz',mean=mean,std=std,indices=normidx)
 else:
  for role,parity,n in [('calibration',0,8192),('test',1,16384)]:
   idx=rng.choice(np.flatnonzero(valid&(tid%2==parity)),n,replace=False)
   np.savez(ROOT/'arrays'/f'{role}.npz',obs=o[idx],actions=a[idx],trajectory=tid[idx],region=reg[idx],indices=idx)
  # Independent positive controls: observed detours/routes and geometric turns.
  lab={k:dict(np.load(GLOBAL/'arrays'/f't{k}_validation_labels.npz')) for k in [1,4]}
  turns=np.zeros(len(o),bool)
  for ep in range(int(t.sum())):
   ids=np.flatnonzero(tid==ep);cells=((o[ids,:2]+6)/4).astype(int)
   changes=np.flatnonzero(np.r_[True,np.any(cells[1:]!=cells[:-1],axis=1)])
   for j in range(1,len(changes)-1):
    u,v,w=cells[changes[j-1]],cells[changes[j]],cells[changes[j+1]]
    if np.dot(v-u,w-v)==0 and np.sum(abs(v-u))==1 and np.sum(abs(w-v))==1:
     turns[ids[max(0,changes[j]-20):min(len(ids),changes[j+1]+20)]]=True
  rare=[]
  masks={'turn':turns}
  for k,l in lab.items():
   masks[f't{k}_detour']=l['valid_window']&l['corridor']&(l['progress100']>=1)&(l['euclidean_change100']>.5)
   masks[f't{k}_route']=l['route_member']
  for name,m in masks.items():
   ids=np.flatnonzero(m&valid&(tid%2==1));rng.shuffle(ids);ids=ids[:2048]
   np.savez(ROOT/'arrays'/f'positive_{name}.npz',obs=o[ids],actions=a[ids],trajectory=tid[ids],region=reg[ids],indices=ids)
   rare.append(dict(group=name,n=len(ids),trajectories=len(np.unique(tid[ids]))))
  write('reports/positive_controls.json',rare)
 # Actual next states: exclude first/terminal observation; actual a(s') available.
 eligible=valid&np.r_[False,~t[:-1]]
 if split=='val':eligible&=tid%2==1
 chosen=[]
 for rr in [0,1]:
  ids=np.flatnonzero(eligible&(reg==rr));rng.shuffle(ids);count=collections.Counter();take=[]
  for i in ids:
   ep=int(tid[i])
   if count[ep]>=32:continue
   take.append(i);count[ep]+=1
   if len(take)==1024:break
  assert len(take)==1024,(split,rr,len(take))
  chosen.extend(take)
 ids=np.array(chosen)
 queries.append(dict(obs=o[ids],actions=a[ids],trajectory=tid[ids]+(10000 if split=='val' else 0),region=reg[ids],indices=ids,split=np.full(len(ids),split)))
q={k:np.concatenate([d[k] for d in queries]) for k in queries[0]}
np.savez(ROOT/'arrays/queries.npz',**q)
write('receipts/data.json',dict(inputs=records,queries=4096,query_sha256=sha(ROOT/'arrays/queries.npz'),normalizer_sha256=sha(ROOT/'arrays/normalizer.npz'),seconds=time.time()-t0,gfp_updates=0,environment_steps=0))
print('PREPARED',time.time()-t0,flush=True)
