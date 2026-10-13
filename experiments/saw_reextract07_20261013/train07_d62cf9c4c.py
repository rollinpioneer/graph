"""Formal SAW07 student extraction, immutable reference, 500k exact updates."""
from common07 import *
import argparse, csv, signal, traceback

@jax.jit
def exposure(ref,p,b):return loss_parts(ref,p,b)
@jax.jit
def audit_gradients(ref,p,b):
    ga=jax.grad(lambda pp:loss_parts(ref,pp,b)[0].mean())(p)
    gw=jax.grad(lambda pp:loss_parts(ref,pp,b)[1].mean())(p)
    return ga,gw
def main(r,arm):
    auth=json.loads((ROOT/'receipts/AUTHORIZATION.json').read_text());assert auth['time_cost_limits_removed']
    adm=json.loads((ROOT/'results/P_continuation/P_ADMISSION.json').read_text());assert adm['technical_gates_pass']
    assert auth['I' if arm<2 else 'II']
    if arm>=2:
        verdict=json.loads((ROOT/'results/I_GATE.json').read_text());assert verdict['H_R_pass'] and adm['manipulation_all_blocks_pass']
    ident=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text())[r];ref=load_ref(ident['reference'])
    assert sha(ident['reference']['config_path'])==ident['reference']['config_sha256']
    source_commit=__import__('subprocess').check_output(['git','-C',str(ROOT.parent.parent),'rev-parse','HEAD'],text=True).strip()
    source_files={str(q):sha(q) for q in (ROOT/'common07.py',ROOT/'train07.py')}
    frozen_code=json.loads((ROOT/'TRAINING_PROGRAM_IDENTITY.json').read_text());assert source_files==frozen_code['files']
    before=tree_sha(flax.serialization.to_state_dict(ref));assert before==ident['reference_full_state_sha256']
    data=data_load();p,o=fresh(r);assert tree_sha(p)==ident['four_arm_parameter_sha256'][arm]
    tau=json.loads((ROOT/'results/P_continuation/GATE_CALIBRATION.json').read_text())[r]['tau'];sampler=Sampler(data,r)
    bank=json.loads((ROOT/'EVALUATION_BANK.json').read_text());assert sha(bank['path'])==bank['sha256']
    ap=ROOT/'results/P_continuation/runs'/f'TRAINING_AUDIT_b{r}.pkl'
    with open(ap,'rb') as f:audit=pickle.load(f)
    assert audit['n']==256
    label=f'b{r}_a{arm}';directory=ROOT/'runs'/label;directory.mkdir(exist_ok=False)
    start=time.monotonic();step=0;cps=[];status='RUNNING';stop=[False];error=None
    rng=jax.random.PRNGKey(78001300+r)
    signal.signal(signal.SIGTERM,lambda *_:stop.__setitem__(0,True))
    def save(u,formal=True):
        jax.block_until_ready((p,o));assert int(o[0].count)==u
        assert before==tree_sha(flax.serialization.to_state_dict(ref));assert sha(ident['reference']['path'])==ident['reference']['sha256']
        assert bool(finite((p,o)))
        ga,gw=audit_gradients(ref,p,audit['batch']);jax.block_until_ready((ga,gw));assert bool(finite((ga,gw)))
        aa=[np.asarray(x,dtype=np.float64) for x in jax.tree_util.tree_leaves(ga)];ww=[np.asarray(x,dtype=np.float64) for x in jax.tree_util.tree_leaves(gw)]
        an=float(np.sqrt(sum(np.sum(x*x) for x in aa)));wn=float(np.sqrt(sum(np.sum(x*x) for x in ww)));dot=float(sum(np.sum(x*y) for x,y in zip(aa,ww)))
        path=directory/f'actor_{u}.pkl';dump(path,dict(params=p,actor_opt=o,student_step=u,reference=ident['reference'],reference_native_step=int(ref.network.step),student_rng=np.asarray(rng),sampler_state=sampler.state(),bank_sha256=bank['sha256'],tau=tau,matmul_precision='highest',source_commit=source_commit,source_files=source_files))
        cp=dict(block=r,arm=arm,step=u,path=str(path),sha256=sha(path),bytes=path.stat().st_size,formal_savepoint=formal,adam_count=int(o[0].count),reference_full_state_sha256=before,frozen_reference_exactly_unchanged=True,audit_tuple_count=256,audit_bank_sha256=sha(ap),awr_gradient_norm=an,waypoint_gradient_norm=wn,gradient_inner_product=dot,student_parameter_sha256=tree_sha(p),base_F_original_R_rng_state_sha256=hashlib.sha256(json.dumps(sampler.state()['generators'][:3],sort_keys=True).encode()).hexdigest())
        cps.append(cp);write(ROOT/'results'/f'{label}_checkpoints.json',cps)
    log=open(ROOT/'results'/f'{label}_training.csv','x',newline='');fields=['block','arm','step','elapsed_seconds','loss','awr','waypoint','window_F_samples','window_R_samples','window_candidate_mean','window_max_batch_candidate_p99','window_candidate_max','window_unpadded_forward_pairs','F_awr_contribution','R_awr_contribution','F_wp_contribution','R_wp_contribution','F_awr_q05','F_awr_q50','F_awr_q95','R_awr_q05','R_awr_q50','R_awr_q95','F_wp_q05','F_wp_q50','F_wp_q95','R_wp_q05','R_wp_q50','R_wp_q95','data_action_norm_mean','teacher_action_norm_mean','student_action_norm_mean','goal_unique_fraction','goal_same_trajectory_fraction','goal_region_NW','goal_region_NE','goal_region_SW','goal_region_SE']
    writer=csv.DictWriter(log,fieldnames=fields,lineterminator='\n');writer.writeheader()
    window_candidates=window_forward=0;window_batches=0;window_p99=0.;window_max=0
    try:
        save(0)
        for u in range(1,500001):
            if stop[0]:status='OWNED_MONITOR_STOP';break
            b,ix,stat=sampler.batch(arm,ref,tau)
            pp,oo,info=update(ref,p,o,b);jax.block_until_ready(info)
            if not bool(info['finite']):
                dump(directory/f'NONFINITE_attempt_{u}.pkl',dict(attempt=u,prior_valid_step=step,params=pp,opt=oo,batch=b,indices=ix,info=info));raise RuntimeError('NONFINITE_LOSS_GRADIENT_OR_PARAMETER')
            p,o=pp,oo;step=u;rng=jax.random.split(rng)[0]
            if stat:
                window_candidates+=stat['candidate_count'];window_forward+=stat['forward_candidates'];window_p99=max(window_p99,stat['candidate_p99']);window_max=max(window_max,stat['candidate_max'])
            elif arm:window_candidates+=512;window_max=max(window_max,1);window_p99=max(window_p99,1)
            window_batches+=1
            if u%1000==0:
                write(ROOT/'receipts'/f'{label}_progress.json',dict(block=r,arm=arm,step=step,elapsed_seconds=time.monotonic()-start,loss=float(info['loss']),status='RUNNING'));print(label,'TRAIN',u,flush=True)
            if u%25000==0:
                vals=[np.asarray(x) for x in exposure(ref,p,b)];aw,wp,rho,omega,teacher,student=vals;assert all(np.isfinite(x).all() for x in vals)
                row=dict(block=r,arm=arm,step=u,elapsed_seconds=time.monotonic()-start,**{k:float(info[k]) for k in ('loss','awr','waypoint')},window_F_samples=window_batches*(1024 if arm==0 else 512),window_R_samples=window_batches*(0 if arm==0 else 512),window_candidate_mean=window_candidates/(window_batches*512) if arm else 0,window_max_batch_candidate_p99=window_p99,window_candidate_max=window_max,window_unpadded_forward_pairs=window_forward,data_action_norm_mean=float(np.linalg.norm(b['actions'],axis=1).mean()),teacher_action_norm_mean=float(np.linalg.norm(teacher,axis=1).mean()),student_action_norm_mean=float(np.linalg.norm(student,axis=1).mean()),goal_unique_fraction=len(np.unique(ix['goal']))/1024,goal_same_trajectory_fraction=float(np.mean(np.searchsorted(data.terminal_locs,ix['i'])==np.searchsorted(data.terminal_locs,ix['goal']))))
                masks={'F':slice(None) if arm==0 else slice(0,512),'R':slice(0,0) if arm==0 else slice(512,None)}
                for branch,sl in masks.items():
                    row[f'{branch}_awr_contribution']=float(aw[sl].sum()/1024);row[f'{branch}_wp_contribution']=float(wp[sl].sum()/1024)
                    for name,wgt in [('awr',rho),('wp',omega)]:
                        for key,q in [('q05',.05),('q50',.5),('q95',.95)]:row[f'{branch}_{name}_{key}']=float(np.quantile(wgt[sl],q)) if len(wgt[sl]) else ''
                # Region boundaries fixed from the audited 9x12 maze, diagnostic only.
                cells=np.floor((b['high_actor_goals']+6)/4).astype(int)
                for reg in ('NW','NE','SW','SE'):row[f'goal_region_{reg}']=float(np.mean((cells[:,0]<=4 if reg[0]=='N' else cells[:,0]>4)&(cells[:,1]<=5 if reg[1]=='W' else cells[:,1]>5)))
                writer.writerow(row);log.flush();window_candidates=window_forward=window_batches=0;window_p99=0.;window_max=0
            if u%100000==0:save(u)
        if step==500000:status='COMPLETE_500K'
    except Exception as exc:status='FAILED';error=dict(error=repr(exc),traceback=traceback.format_exc());raise
    finally:
        if not any(c['step']==step for c in cps):
            try:save(step,False)
            except Exception as exc:error=dict(error=error,partial_save_error=repr(exc))
        log.close();write(ROOT/'results'/f'{label}_status.json',dict(status=status,block=r,arm=arm,student_updates=step,elapsed_seconds=time.monotonic()-start,reference_identity=ident['reference'],initial_student_sha256=ident['four_arm_parameter_sha256'][arm],new_adam_zero_start=True,matmul_precision='highest',checkpoint_count=len(cps),error=error))
    print('TRAIN_DONE',label,status,step,flush=True)
if __name__=='__main__':
    a=argparse.ArgumentParser();a.add_argument('--block',type=int,choices=range(3),required=True);a.add_argument('--arm',type=int,choices=range(4),required=True);x=a.parse_args();main(x.block,x.arm)
