"""P: no policy scores; locked Q95, independent manipulation, four discarded pilots."""
from common07 import *
from collections import deque
import csv, traceback
assert json.loads((ROOT/'receipts/AUTHORIZATION.json').read_text())['P']
assert all(d.platform=='gpu' for d in jax.devices())
begin=time.monotonic();data=data_load();items=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text())
env=ogbench.make_env_and_datasets('pointmaze-large-stitch-v0',env_only=True);u=env.unwrapped
cells=sorted(tuple(x) for x in np.argwhere(u.maze_map==0));free=set(cells);dist={}
for c in cells:
    d={c:0};q=deque([c])
    while q:
        a=q.popleft()
        for di,dj in [(-1,0),(0,-1),(1,0),(0,1)]:
            b=(a[0]+di,a[1]+dj)
            if b in free and b not in d:d[b]=d[a]+1;q.append(b)
    dist[c]=d
obs=np.asarray(data.dataset['observations']);obs_cells=[tuple(u.xy_to_ij(s)) for s in obs];env.close()
def distances(i,g):return np.asarray([dist.get(obs_cells[a],{}).get(obs_cells[b],np.nan) for a,b in zip(i,g)])
def region(c):return ('N' if c[0]<=4 else 'S')+('W' if c[1]<=5 else 'E')
traj=np.searchsorted(data.terminal_locs,np.arange(len(obs)))
# All future pairs have cell membership in suffix of the same trajectory. Enumerate
# distinct suffix cells once per observed cell change, preserving temporal direction.
max_f=0;invalid=0;start=0
for end in data.terminal_locs:
    suffix=set()
    for idx in range(int(end),start-1,-1):
        c=obs_cells[idx]
        if c not in free:invalid+=1
        else:
            for g in suffix:max_f=max(max_f,dist[c][g])
        suffix.add(c)
    start=int(end)+1
write(ROOT/'FUTURE_SUPPORT_AUDIT.json',dict(maximum_bfs_among_all_temporally_legal_future_pairs=max_f,invalid_state_count=invalid,note='Exact cell-set suffix enumeration; no new model probe. Includes terminal state self-goal allowed by native sampler.'))
cal=[];man=[];pairing=[];pilot=[];checks=[];refs=[];taus=[];sampling=[]
try:
    for r,item in enumerate(items):
        ref=load_ref(item['reference']);refs.append(ref);before=tree_sha(flax.serialization.to_state_dict(ref))
        rng=np.random.default_rng(73001300+r);valid=np.asarray(data.dataset.valid_idxs)
        i=valid[rng.integers(len(valid),size=16384)];end=data.terminal_locs[np.searchsorted(data.terminal_locs,i)];z=rng.random(16384)
        fg=np.round(np.minimum(i+1,end)*z+end*(1-z)).astype(np.int64);scores=[];t=time.monotonic()
        for n in range(0,16384,1024):scores.extend(np.asarray(cost(ref,obs[i[n:n+1024]],obs[fg[n:n+1024]])))
        scores=np.asarray(scores);assert np.isfinite(scores).all();tau=float(np.sort(scores)[int(np.ceil(.95*len(scores)))-1]);taus.append(tau)
        raw=ROOT/'runs'/f'CALIBRATION_b{r}.npz';np.savez(raw,indices=i,goal_indices=fg,cost=scores)
        cal.append(dict(block=r,seed=73001300+r,n=16384,q=.95,rule='nearest-rank ceil(.95N)',cost='negative mean of two frozen value heads',tau=tau,path=str(raw),sha256=sha(raw),seconds=time.monotonic()-t))
        write(ROOT/'GATE_CALIBRATION.json',cal)
        # Frozen audit batch comes from training data before any student update.
        audit_sampler=Sampler(data,r);ab,ai,_=audit_sampler.batch(1,n=256);dump(ROOT/'runs'/f'TRAINING_AUDIT_b{r}.pkl',dict(batch=ab,indices=ai,n=256))
        roots=(79001300+r,79011300+r,79021300+r,79031300+r)
        template=Sampler(data,r,roots);idx=template.indices(8192);mi,ff,fw,g0,rw=idx;arms=[];timing={}
        for arm in range(4):
            sampler=Sampler(data,r,roots);same=sampler.indices(8192);assert all(np.array_equal(a,b) for a,b in zip(idx,same))
            t=time.monotonic();g=ff.copy() if arm==0 else g0.copy();stat={}
            if arm in (2,3):g,stat=sampler.reject(ref,mi,g0,tau,arm==2)
            dd=distances(mi,g);legal=np.isfinite(dd);dmedian=float(np.nanmedian(dd));far=float(np.sum(dd>=8)/len(dd))
            c=np.asarray(cost(ref,obs[mi],obs[g]));assert np.isfinite(c).all()
            rec=dict(block=r,arm=arm,n=8192,roots=list(roots),median_bfs=dmedian,far_ge8_fraction=far,legal_fraction=float(legal.mean()),same_trajectory_fraction=float(np.mean(traj[mi]==traj[g])),same_index_fraction=float(np.mean(mi==g)),unique_target_fraction=float(len(np.unique(g))/len(g)),unique_goal_cells=len(set(obs_cells[k] for k in g)),target_regions={z:sum(region(obs_cells[k])==z for k in g)/len(g) for z in ['NW','NE','SW','SE']},cost_mean=float(c.mean()),cost_q05=float(np.quantile(c,.05)),cost_q95=float(np.quantile(c,.95)),seconds=time.monotonic()-t,**stat)
            # Weight summaries use the exact training definitions, without a student outcome.
            wi=fw if arm==0 else rw
            weights=[]
            for n in range(0,8192,1024):
                ix=mi[n:n+1024];gg=g[n:n+1024];ww=wi[n:n+1024]
                sv=np.asarray(cost(ref,obs[ix],obs[gg]));nv=np.asarray(cost(ref,obs[ix+1],obs[gg]));wv=np.asarray(cost(ref,obs[ww],obs[gg]))
                weights.append(np.stack([np.minimum(np.exp(3*(sv-nv)),100.),np.minimum(np.exp(3*(sv-wv)),100.)],axis=1))
            weights=np.concatenate(weights);rec['awr_weight_mean']=float(weights[:,0].mean());rec['waypoint_weight_mean']=float(weights[:,1].mean());rec['awr_weight_median']=float(np.median(weights[:,0]));rec['waypoint_weight_median']=float(np.median(weights[:,1]))
            arms.append(rec);man.append(rec)
            out=ROOT/'runs'/f'MANIPULATION_b{r}_a{arm}.npz';np.savez(out,indices=mi,goal_indices=g,bfs=dd,cost=c,waypoint_indices=wi,weights=weights)
            rec['raw_path']=str(out);rec['raw_sha256']=sha(out)
            if arm in (2,3):timing[arm]=stat
        passed=arms[3]['median_bfs']>=arms[2]['median_bfs']+2 and arms[3]['far_ge8_fraction']>=arms[2]['far_ge8_fraction']+.30 and arms[2]['median_bfs']<=arms[1]['median_bfs']<=arms[3]['median_bfs'] and arms[2]['far_ge8_fraction']<=arms[1]['far_ge8_fraction']<=arms[3]['far_ge8_fraction'] and min(x['legal_fraction'] for x in arms)>=.95
        for row in arms:row['block_manipulation_pass']=passed
        write(ROOT/'GATE_MANIPULATION.json',man)
        # Same-base/same-F audit and original random waypoint invariance, before pilots.
        samplers=[Sampler(data,r) for _ in range(4)];batches=[s.batch(a,ref,tau) for a,s in enumerate(samplers)]
        for a in range(1,4):
            assert np.array_equal(batches[0][1]['i'],batches[a][1]['i'])
            for key in ('observations','actions','next_observations','high_actor_goals','high_actor_targets'):assert np.array_equal(batches[1][0][key][:512],batches[a][0][key][:512])
        for a in (2,3):assert np.array_equal(batches[a][1]['waypoint'][512:],batches[a][1]['random_waypoint'][512:])
        nextix=[s.indices() for s in samplers];assert all(all(np.array_equal(x,y) for x,y in zip(nextix[0],n)) for n in nextix[1:])
        pairing.append(dict(block=r,same_base=True,same_F512=True,R512_present=True,waypoint_and_successor_unchanged=True,replenishment_does_not_advance_next_base_F_original_R=True,full_next_indices_sha256=[tree_sha(x) for x in nextix]))
        # Loss/gradient comparison: original methods called with same fresh actor params.
        p,o=fresh(r);bb=batches[1][0]
        def original(pp):
            full=dict(ref.network.params);full.update(pp)
            a,_=ref.actor_loss(bb,full);w,_=ref.waypoint_loss(bb,full);return a+w
        ll,gg=jax.jit(jax.value_and_grad(original))(p);(l,_),g=jax.jit(jax.value_and_grad(loss,argnums=1,has_aux=True))(ref,p,bb)
        lerr=abs(float(ll)-float(l));gerr=max(float(np.max(np.abs(np.asarray(a)-np.asarray(b)))) for a,b in zip(jax.tree_util.tree_leaves(gg),jax.tree_util.tree_leaves(g)))
        gmax=max(float(np.max(np.abs(np.asarray(x)))) for x in jax.tree_util.tree_leaves(gg));assert lerr<=1e-5+1e-5*abs(float(ll));assert gerr<=1e-5+1e-5*gmax
        # Native full gradient tree proves no value/teacher or shared-representation path.
        def native_full(pp):return ref.actor_loss(bb,pp)[0]+ref.waypoint_loss(bb,pp)[0]
        fullgrad=jax.jit(jax.grad(native_full))(ref.network.params)
        frozen_zero=all(np.count_nonzero(np.asarray(x))==0 for k in fullgrad if k not in KEYS for x in jax.tree_util.tree_leaves(fullgrad[k]));assert frozen_zero
        assert before==tree_sha(flax.serialization.to_state_dict(ref))
        checks.append(dict(block=r,loss_error=lerr,loss_tolerance=1e-5+1e-5*abs(float(ll)),gradient_max_error=gerr,gradient_tolerance=1e-5+1e-5*gmax,frozen_native_gradients_zero=frozen_zero,reference_full_state_unchanged=True))
        # Each block's near/far sampler cost on a fixed 16 batches, no policy scores.
        for a in (2,3):
            s=Sampler(data,r);t=time.monotonic();stats=[]
            for k in range(16):stats.append(s.batch(a,ref,tau)[2])
            elapsed=time.monotonic()-t
            sampling.append(dict(block=r,arm=a,batches=16,sampler_seconds=elapsed,seconds_per_batch=elapsed/16,candidate_mean=float(np.mean([x['candidate_mean'] for x in stats])),candidate_p99_max=max(x['candidate_p99'] for x in stats),candidate_max=max(x['candidate_max'] for x in stats)))
        print('P_BLOCK',r,'tau',tau,'manipulation',passed,flush=True)
    write(ROOT/'SAMPLER_PAIRING_AUDIT.json',pairing);write(ROOT/'LOSS_AND_GRADIENT_AUDIT.json',checks)
    # Four global pilots, one shared discarded block0 initialization, ordered a0->a3.
    for a in range(4):
        ref=refs[0];before=tree_sha(flax.serialization.to_state_dict(ref));p,o=fresh(0);s=Sampler(data,0);t=time.monotonic();times=[];st=[]
        for k in range(1000):
            tick=time.monotonic();b,ix,stat=s.batch(a,ref,taus[0]);p,o,info=update(ref,p,o,b);jax.block_until_ready(info)
            assert bool(info['finite']);times.append(time.monotonic()-tick)
            if stat:st.append(stat)
            if k==0:assert before==tree_sha(flax.serialization.to_state_dict(ref))
        assert int(o[0].count)==1000 and before==tree_sha(flax.serialization.to_state_dict(ref))
        steady=float(np.mean(times[100:]));pilot.append(dict(arm=a,block=0,updates=1000,seconds=time.monotonic()-t,first_update_seconds=times[0],steady_seconds_per_update=steady,projected_500k_seconds=steady*500000+times[0],parameter_finite=True,reference_full_state_unchanged_after_1_and_1000=True,initial_parameters_sha256=items[0]['four_arm_parameter_sha256'][a],fresh_zero_adam=True,weights_discarded=True))
        write(ROOT/'P_PILOTS.json',pilot);print('P_PILOT',a,pilot[-1]['projected_500k_seconds'],flush=True)
    pred=[]
    for r in range(3):
        for a in range(4):
            sec=pilot[a]['projected_500k_seconds']
            if a>=2:
                b0=next(x for x in sampling if x['block']==0 and x['arm']==a);br=next(x for x in sampling if x['block']==r and x['arm']==a)
                sec+=max(0,br['seconds_per_batch']-b0['seconds_per_batch'])*500000
            # Conservative billing includes observed compile/load and a 10% runtime guard.
            sec=sec*1.10+30;pred.append(dict(block=r,arm=a,projected_seconds=sec,hard_cap_seconds=1800 if a<2 else 2700))
    iok=all(x['projected_seconds']<=x['hard_cap_seconds'] for x in pred if x['arm']<2) and sum(x['projected_seconds'] for x in pred if x['arm']<2)<=10800
    iiok=all(x['projected_seconds']<=x['hard_cap_seconds'] for x in pred if x['arm']>=2) and sum(x['projected_seconds'] for x in pred if x['arm']>=2)<=16200
    status='PASS' if iok else 'STOP_BEFORE_FORMAL_PROJECTED_I_BUDGET_EXCEEDED'
    write(ROOT/'P_ADMISSION.json',dict(status=status,technical_gates_pass=True,manipulation_all_blocks_pass=all(x['block_manipulation_pass'] for x in man),I_budget_prediction_pass=iok,II_budget_prediction_pass=iiok,projections=pred,sampler_measurements=sampling,P_seconds=time.monotonic()-begin,no_student_success_rates_queried=True,formal_updates=0,pilots_discarded=True))
except Exception as exc:
    write(ROOT/'P_ADMISSION.json',dict(status='P_TECHNICAL_FAILURE',error=repr(exc),traceback=traceback.format_exc(),P_seconds=time.monotonic()-begin,formal_updates=0,no_student_success_rates_queried=True,pilot_updates=sum(x['updates'] for x in pilot)))
    raise
