"""SAW07: fresh student-only optimizer, immutable official reference."""
import os
assert len(os.sched_getaffinity(0)) <= 4
import copy, hashlib, json, pickle, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parent
OLD = Path('/home/xushijie3/work/graph_saw_rd01_20261012/experiments/saw_rd01_20261012')
MECH = Path('/home/xushijie3/work/graph_saw_mech04_20261013/experiments/saw_mech04_20261013')
D06 = Path('/home/xushijie3/work/graph_saw_distance06_20261013/experiments/saw_distance06_20261013')
sys.path.insert(0, str(OLD/'vendor/saw/impls'))
import jax, jax.numpy as jnp, numpy as np, optax, flax.serialization, ogbench
from agents.saw import SAWAgent, get_config
from utils.datasets import Dataset, HGCDataset
KEYS = ('modules_actor', 'modules_actor_goal_rep_def')
def write(p,x):
    p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
    t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(x,indent=2,allow_nan=False,default=lambda v:v.tolist() if hasattr(v,'tolist') else str(v))+'\n');t.replace(p)
def sha(p):
    h=hashlib.sha256()
    with open(p,'rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def tree_sha(x):
    h=hashlib.sha256()
    for path,a in jax.tree_util.tree_flatten_with_path(x)[0]:
        a=np.asarray(a);h.update(str(path).encode());h.update(str(a.shape).encode());h.update(a.dtype.str.encode());h.update(a.tobytes())
    return h.hexdigest()
def equal(a,b):
    return jax.tree_util.tree_structure(a)==jax.tree_util.tree_structure(b) and all(np.array_equal(np.asarray(x),np.asarray(y)) for x,y in zip(jax.tree_util.tree_leaves(a),jax.tree_util.tree_leaves(b)))
def dump(p,x):
    with open(p,'xb') as f:pickle.dump(jax.device_get(x),f,4)
def references():
    out=[]
    for r in range(3):
        d=OLD if r==0 else MECH
        ix=d/'results'/('stitch_checkpoints.json' if r==0 else f'C_stitch_A_s{r}_checkpoints.json')
        item=next(x for x in json.loads(ix.read_text()) if x['step']==1000000 and Path(x['path']).name=='params_1000000.pkl')
        assert sha(item['path'])==item['sha256']
        cp=d/'runs'/('formal_stitch' if r==0 else f'C_stitch_A_s{r}')/'config.json'
        cfg=json.loads(cp.read_text())['config']; assert json.loads(json.dumps(get_config().to_dict()))==cfg
        out.append(dict(**item,block=r,source_index=str(ix),source_index_sha256=sha(ix),config_path=str(cp),config_sha256=sha(cp)))
    return out
def load_ref(item):
    assert sha(item['path'])==item['sha256']; cfg=get_config().to_dict()
    a=SAWAgent.create(item['block'],np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),cfg)
    with open(item['path'],'rb') as f:s=pickle.load(f)
    assert s['step']==1000000
    a=flax.serialization.from_state_dict(a,s['agent']);assert int(a.network.step)==1000001
    assert cfg['share_goal_rep'] is False and cfg['const_std'] is True
    assert KEYS[0] in a.network.params and KEYS[1] in a.network.params
    return jax.device_put(a,jax.devices()[0])
def fresh(r):
    a=SAWAgent.create(71001300+r,np.zeros((1,2),np.float32),np.zeros((1,2),np.float32),get_config().to_dict())
    p={k:a.network.params[k] for k in KEYS}; o=optax.adam(3e-4).init(p)
    assert int(o[0].count)==0
    assert all(np.count_nonzero(np.asarray(x))==0 for x in jax.tree_util.tree_leaves((o[0].mu,o[0].nu)))
    return p,o
def data_load():
    env,raw,_=ogbench.make_env_and_datasets('pointmaze-large-stitch-v0',dataset_dir=str(OLD/'vendor/data'),compact_dataset=True)
    env.close();return HGCDataset(Dataset.create(**raw),get_config().to_dict())
def policy(ref,p):
    full=dict(ref.network.params);full.update(p);return ref.replace(network=ref.network.replace(params=full))
def loss_parts(ref,p,b):
    s,g,n,w=(b[k] for k in ('observations','high_actor_goals','next_observations','high_actor_targets'))
    v1,v2=ref.network.select('value')(s,g);n1,n2=ref.network.select('value')(n,g);w1,w2=ref.network.select('value')(w,g)
    v=(v1+v2)/2;nval=(n1+n2)/2;wv=(w1+w2)/2
    rho=jnp.minimum(jnp.exp(3*(nval-v)),100.);omega=jnp.minimum(jnp.exp(3*(wv-v)),100.)
    full=dict(ref.network.params);full.update(p)
    dist=ref.network.select('actor')(s,g,params=full);teacher=jax.lax.stop_gradient(ref.network.select('low_actor')(s,w).mode())
    awr=-rho*dist.log_prob(b['actions']);wp=omega*jnp.sum((dist.mode()-teacher)**2,axis=-1)
    return awr,wp,rho,omega,teacher,dist.mode()
def loss(ref,p,b):
    a,w,*_=loss_parts(ref,p,b);return a.mean()+w.mean(),dict(awr=a.mean(),waypoint=w.mean())
def finite(x):return jnp.all(jnp.stack([jnp.all(jnp.isfinite(v)) for v in jax.tree_util.tree_leaves(x)]))
@jax.jit
def update(ref,p,o,b):
    (l,info),g=jax.value_and_grad(loss,argnums=1,has_aux=True)(ref,p,b)
    d,o=optax.adam(3e-4).update(g,o,p);p=optax.apply_updates(p,d)
    return p,o,dict(info,loss=l,finite=finite((l,g,p)))
@jax.jit
def cost(ref,s,g):
    v1,v2=ref.network.select('value')(s,g);return -(v1+v2)/2
def cost_fixed(ref,s,g):
    """Fixed 1024 forward shape avoids compiling each rejection population size."""
    chunks=[]
    for n in range(0,len(s),1024):
        count=min(1024,len(s)-n);ss=s[n:n+count];gg=g[n:n+count]
        if count<1024:
            ss=np.concatenate([ss,np.repeat(ss[:1],1024-count,axis=0)]);gg=np.concatenate([gg,np.repeat(gg[:1],1024-count,axis=0)])
        chunks.append(np.asarray(cost(ref,ss,gg))[:count])
    return np.concatenate(chunks)

class Sampler:
    """Independent saved NumPy Generators; logical data independent of arm scheduling."""
    def __init__(self,data,r,roots=None):
        self.d=data;self.r=r;self.valid=np.asarray(data.dataset.valid_idxs);self.term=data.terminal_locs
        self.obs=np.asarray(data.dataset['observations'])
        roots=roots or (74001300+r,75001300+r,76001300+r,77001300+r)
        self.roots=roots;self.base,self.future,self.original,self.extra=(np.random.default_rng(x) for x in roots)
        self.cursor=0
    def state(self):return dict(roots=self.roots,cursor=self.cursor,generators=[copy.deepcopy(x.bit_generator.state) for x in (self.base,self.future,self.original,self.extra)])
    def random(self,rng,n):return self.valid[rng.integers(len(self.valid),size=n)]
    def indices(self,n=1024):
        i=self.random(self.base,n);end=self.term[np.searchsorted(self.term,i)];u=self.future.random(n)
        fg=np.round(np.minimum(i+1,end)*u+end*(1-u)).astype(np.int64)
        rg=self.random(self.original,n);self.cursor+=1
        return i,fg,np.minimum(i+25,fg),rg,np.minimum(i+25,end)
    def reject(self,ref,i,g0,tau,high):
        """Only goal indices change; each rejected slot gets uniform independent candidates."""
        out=g0.copy();used=np.ones(len(i),np.int32)
        def accepts(c):return c<=tau if high else c>tau
        c=cost_fixed(ref,self.obs[i],self.obs[out]);assert np.isfinite(c).all()
        pending=np.flatnonzero(~accepts(c));forward=len(i)
        # Fixed chunks of four per pending slot, take earliest accepted; consumption is saved.
        while len(pending):
            assert np.all(used[pending]<4096), 'R_SLOT_UNFILLED'
            chunk=min(4,int(4096-used[pending].max()))
            gi=self.random(self.extra,len(pending)*chunk).reshape(len(pending),chunk)
            cs=cost_fixed(ref,np.repeat(self.obs[i[pending]],chunk,axis=0),self.obs[gi.ravel()]).reshape(len(pending),chunk)
            assert np.isfinite(cs).all();ok=accepts(cs);got=ok.any(axis=1);first=np.argmax(ok,axis=1)
            out[pending[got]]=gi[np.flatnonzero(got),first[got]]
            used[pending[got]]+=first[got]+1;used[pending[~got]]+=chunk
            forward+=gi.size;pending=pending[~got]
        final=cost_fixed(ref,self.obs[i],self.obs[out]);assert accepts(final).all()
        return out,dict(candidate_mean=float(used.mean()),candidate_p99=float(np.quantile(used,.99)),candidate_max=int(used.max()),candidate_count=int(used.sum()),forward_candidates=forward+len(i),acceptance_rate=float(len(i)/used.sum()))
    def batch(self,arm,ref=None,tau=None,n=1024):
        ix=self.indices(n);i,fg,fw,rg,rw=ix;g=fg.copy();w=fw.copy();stats={}
        if arm:
            g[n//2:]=rg[n//2:];w[n//2:]=rw[n//2:]
        if arm in (2,3):
            g[n//2:],stats=self.reject(ref,i[n//2:],rg[n//2:],tau,arm==2)
        native=self.d.dataset.sample(n,i);b={k:native[k] for k in ('observations','actions','next_observations')}
        b.update(high_actor_goals=self.obs[g],high_actor_targets=self.obs[w])
        return b,dict(i=i,goal=g,waypoint=w,future=fg,original_random=rg,random_waypoint=rw),stats
