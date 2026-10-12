"""CPU-only admission; compute gradients, never call an optimizer update."""
from frozen import *
from r_supervision import Trace
from t_core import *

def main():
    start=time.time();a,identity=load_agent('stitch','A',500000);assert not a.config['share_goal_rep'] and a.config['const_std']
    env,data,val,cfg=load('pointmaze-large-stitch-v0','A');F=Trace(data.dataset,dict(cfg,actor_p_trajgoal=1.,actor_p_randomgoal=0.));R=Trace(data.dataset,dict(cfg,actor_p_trajgoal=0.,actor_p_randomgoal=1.))
    with open(identity['path'],'rb') as f:source=pickle.load(f)
    np.random.set_state(source['numpy_rng']);params=subset(a.network.params);opt=actor_optimizer(a)
    assert int(opt[0].count)==500000
    chunks=[];directory=ROOT/'runs/T_SHARED_STREAM';directory.mkdir(exist_ok=False)
    first=None
    for block in range(100):
        out=np.empty((1000,5,1024),dtype=np.int32)
        for j in range(1000):
            i=data.dataset.get_random_idxs(1024);begin=np.random.get_state()
            F.indices=[];fb=F.sample(1024,i,evaluation=True);after=np.random.get_state();fi=F.indices
            np.random.set_state(begin);R.indices=[];rb=R.sample(1024,i,evaluation=True);ri=R.indices;assert equal(after,np.random.get_state())
            for key in ('observations','actions','next_observations','value_goals','low_actor_goals','rewards','masks'):assert np.array_equal(fb[key],rb[key])
            out[j]=np.array([i,fi[2],fi[3],ri[2],ri[3]],np.int32)
            if first is None:first=out[j].copy()
        p=directory/f'indices_{block:03d}.npy';np.save(p,out);rngfile=directory/f'rng_{block:03d}.pkl';dump(rngfile,np.random.get_state())
        chunks.append(dict(block=block,path=str(p),sha256=sha(p),rng_path=str(rngfile),rng_sha256=sha(rngfile)))
        if block%10==0:print('T_STREAM',block+1,'/100',flush=True)
    bs=[make_batch(data,first,k) for k in range(4)]
    gs=[];ls=[]
    for arm in range(4):
        val,grad=jax.jit(jax.value_and_grad(lambda p:loss(a,p,bs[arm],arm)[0]))(params);sync(grad);assert bool(all_finite(grad));gs.append(grad);ls.append(float(val))
    native= float(a.actor_loss(bs[0],a.network.params)[0]+a.waypoint_loss(bs[0],a.network.params)[0])
    nativegrad=jax.grad(lambda p:a.actor_loss(bs[0],p)[0]+a.waypoint_loss(bs[0],p)[0])(a.network.params)
    random_half={k:v.copy() for k,v in bs[3].items()};random_half={k:v[512:] for k,v in random_half.items()}
    rgrad=jax.grad(lambda p:(a.actor_loss(random_half,p)[0]+a.waypoint_loss(random_half,p)[0])*.5)(a.network.params)
    def maxerr(x,y):return max(float(np.max(np.abs(np.asarray(u)-np.asarray(v)))) for u,v in zip(jax.tree_util.tree_leaves(x),jax.tree_util.tree_leaves(y)))
    def passerr(x,y):
        err=maxerr(x,y);scale=max(float(np.max(np.abs(np.asarray(v)))) for v in jax.tree_util.tree_leaves(y));assert err<=1e-5+1e-5*scale;return err
    errs=dict(T0_native_gradient=passerr(gs[0],subset(nativegrad)),T1_half_T2=passerr(gs[1],jax.tree_util.tree_map(lambda x:x*.5,gs[2])),T3_minus_T1_randomhalf=passerr(jax.tree_util.tree_map(lambda x,y:x-y,gs[3],gs[1]),subset(rgrad)))
    assert abs(ls[0]-native)<=1e-5+1e-5*abs(native)
    write(ROOT/'results/T_SHARED_STREAM.json',dict(source=identity,updates=100000,batch_slots=1024,index_columns=['base','future_goal','future_waypoint','random_goal','random_waypoint'],chunks=chunks,initial_numpy_sha256=digest(source['numpy_rng']),final_numpy_sha256=digest(np.random.get_state()),condition_goal_proportions=[[1,0],[.5,0],[.5,0],[.5,.5]],note='T1/T2 second half masked; all four consume every identical base and latent candidate slot'))
    write(ROOT/'results/T_CPU_ADMISSION.json',dict(status='PASS',new_parameter_updates=0,source=identity,trainable_keys=ACTOR_KEYS,frozen_keys=[k for k in a.network.params if k not in ACTOR_KEYS],optimizer_inherited_count=int(opt[0].count),shared_initial_actor_optimizer_rng=True,initial_losses=ls,native_loss=native,gradient_max_abs_errors=errs,tolerance='1e-5 + 1e-5 * max_abs_reference',seconds=time.time()-start))
    env.close();print('T_CPU_ADMISSION_PASS',flush=True)
if __name__=='__main__':main()
