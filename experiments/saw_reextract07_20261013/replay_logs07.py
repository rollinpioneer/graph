"""Reconstruct only actual legacy a0 logging tuples; never train or advance physics."""
from common07 import *
import csv, argparse, traceback, resource
assert all(x.platform=='cpu' for x in jax.devices())
assert os.sched_getaffinity(0)=={75}

def main():
    start=time.time();records=[];checks=[]
    oldcode=json.loads((ROOT/'results/TRAINING_PROGRAM_IDENTITY_PRE_LOG_FIX.json').read_text())
    assert sha(ROOT/'train07_d62cf9c4c.py')==oldcode['files'][str(ROOT/'train07.py')]
    data=data_load();ident=json.loads((ROOT/'REFERENCE_AND_INIT_IDENTITY.json').read_text())
    cal=json.loads((ROOT/'results/P_continuation/GATE_CALIBRATION.json').read_text())
    rawdir=ROOT/'runs/LEGACY_LOG_REPLAY';rawdir.mkdir(exist_ok=False)
    for r in range(3):
        status=json.loads((ROOT/'results'/f'b{r}_a0_status.json').read_text())
        assert status['status']=='COMPLETE_500K'
        cps=json.loads((ROOT/'results'/f'b{r}_a0_checkpoints.json').read_text())
        oldrows=list(csv.DictReader(open(ROOT/'results'/f'b{r}_a0_training.csv')))
        assert [int(x['step']) for x in oldrows]==list(range(25000,500001,25000))
        bystep={int(x['step']):x for x in oldrows};sampler=Sampler(data,r);ref=load_ref(ident[r]['reference'])
        tau=cal[r]['tau'];checkpoint_checks=[]
        for u in range(1,500001):
            i,fg,fw,rg,rw=sampler.indices()
            if u%25000==0:
                # Exactly the originally executed training batch, with corrected row/column.
                cells=np.floor((sampler.obs[fg]+6)/4).astype(int)[:,::-1]
                cc=cost_fixed(ref,sampler.obs[i],sampler.obs[fg]);assert np.isfinite(cc).all()
                row=dict(bystep[u]);row['tau']=tau
                for reg in ('NW','NE','SW','SE'):
                    row['original_uncorrected_goal_region_'+reg]=row['goal_region_'+reg]
                    row['goal_region_'+reg]=float(np.mean((cells[:,0]<=4 if reg[0]=='N' else cells[:,0]>4)&(cells[:,1]<=5 if reg[1]=='W' else cells[:,1]>5)))
                for key,q in [('q05',.05),('q50',.5),('q95',.95)]:
                    row['F_cost_'+key]=float(np.quantile(cc,q));row['R_cost_'+key]=''
                raw=rawdir/f'b{r}_a0_SUPERVISION_{u}.pkl'
                dump(raw,dict(step=u,indices=dict(i=i,goal=fg,waypoint=fw,future=fg,original_random=rg,random_waypoint=rw),cost=cc,tau=tau,source_trajectory_ids=np.searchsorted(data.terminal_locs,i),goal_trajectory_ids=np.searchsorted(data.terminal_locs,fg),goal_cells=cells,sampler_state=sampler.state(),cost_numeric_device='CPU highest precision; descriptive logging only, not gate acceptance',source_files=oldcode['files'],loss_contributions_timing='original same executed batch, post-update student; loss CSV pre-update'))
                row.update(supervision_raw_path=str(raw),supervision_raw_sha256=sha(raw),logging_origin='OFFLINE_EXACT_INPUT_STREAM_REPLAY',cost_numeric_device='CPU highest')
                assert float(row['goal_unique_fraction'])==len(np.unique(fg))/1024
                assert float(row['goal_same_trajectory_fraction'])==float(np.mean(np.searchsorted(data.terminal_locs,i)==np.searchsorted(data.terminal_locs,fg)))
                records.append(row)
            if u%100000==0:
                cp=next(x for x in cps if x['step']==u);assert sha(cp['path'])==cp['sha256']
                with open(cp['path'],'rb') as f:s=pickle.load(f)
                state=sampler.state();assert state==s['sampler_state']
                state_sha=hashlib.sha256(json.dumps(state['generators'][:3],sort_keys=True).encode()).hexdigest()
                assert state_sha==cp['base_F_original_R_rng_state_sha256']
                assert s['source_files']==oldcode['files']
                checkpoint_checks.append(dict(step=u,checkpoint_sha256=cp['sha256'],all_four_generator_states_exact=True,base_F_original_R_rng_state_sha256=state_sha))
                print('REPLAY',r,u,flush=True)
        checks.append(dict(block=r,sampler_checks=checkpoint_checks,actual_logged_batches=20,source_CSV_sha256=sha(ROOT/'results'/f'b{r}_a0_training.csv'),original_CSV_retained=True))
    fields=list(dict.fromkeys(k for row in records for k in row))
    with open(ROOT/'results/LEGACY_A0_CORRECTED_SUPERVISION.csv','x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(records)
    write(ROOT/'results/LEGACY_LOG_REPLAY_AUDIT.json',dict(status='COMPLETE',blocks=checks,formal_training_updates=0,physics_steps=0,additional_student_model_queries=0,additional_frozen_value_tuple_queries=60*1024,only_logged_actual_training_batches_reconstructed=True,original_teacher_student_norms_and_loss_contributions_retained=True,cost_device='CPU highest; costs are descriptive, not a new sampling gate',elapsed_seconds=time.time()-start,process_peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss*1024,CPU_affinity=[75]))
    print('REPLAY_COMPLETE',flush=True)

if __name__=='__main__':main()
