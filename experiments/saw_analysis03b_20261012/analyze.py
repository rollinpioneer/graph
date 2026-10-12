"""Paired whole-episode uncertainty; missing checkpoints remain missing."""
import csv,json,math
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parent
def read(p):
    with open(p) as f:return list(csv.DictReader(f))
def write(p,x):p.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n')
def save(p,rows):
    if not rows:return
    with open(p,'w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]),lineterminator='\n');w.writeheader();w.writerows(rows)
def wilson(k,n):
    z=1.959963984540054;p=k/n;d=1+z*z/n;c=(p+z*z/(2*n))/d;h=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/d
    return [max(0,c-h),min(1,c+h)]
def paired(a,b,rng):
    delta=np.asarray(b)-np.asarray(a);draw=rng.integers(0,len(delta),size=(10000,len(delta)));ci=np.quantile(delta[draw].mean(1),[.025,.975])
    return dict(n=len(delta),first_success=int(sum(a)),second_success=int(sum(b)),success_to_failure=int(((np.asarray(a)==1)&(np.asarray(b)==0)).sum()),
        failure_to_success=int(((np.asarray(a)==0)&(np.asarray(b)==1)).sum()),paired_difference=float(delta.mean()),paired_ci_low=float(ci[0]),paired_ci_high=float(ci[1]),
        first_wilson_low=wilson(sum(a),len(a))[0],first_wilson_high=wilson(sum(a),len(a))[1],second_wilson_low=wilson(sum(b),len(b))[0],second_wilson_high=wilson(sum(b),len(b))[1])
def main():
    rng=np.random.default_rng(203001);summaries=[];transitions=[];final_comparisons=[];groups=[];coverage=[]
    for kind in ('navigate','stitch'):
        stage={};final={}
        for variant in ('A','B'):
            p=ROOT/'results'/f'E1_{kind}_{variant}.csv'
            rows=read(p) if p.exists() else []
            stage[variant]={int(s):{r['scene_id']:r for r in rows if int(r['checkpoint_step'])==int(s)} for s in (100000,300000,500000,1000000)}
            for step,entries in stage[variant].items():
                coverage.append(dict(kind=kind,variant=variant,checkpoint_step=step,executed_episodes=len(entries),expected_episodes=100,status='COMPLETE' if len(entries)==100 else 'NOT_COMPLETE'))
                for task in range(1,6):
                    tr=[r for r in entries.values() if int(r['task'])==task]
                    if not tr:continue
                    k=sum(float(r['success']) for r in tr);ci=wilson(k,len(tr));summaries.append(dict(kind=kind,variant=variant,checkpoint_step=step,task=task,n=len(tr),successes=int(k),success_rate=k/len(tr),ci_low=ci[0],ci_high=ci[1]))
            for early,late in ((100000,300000),(300000,500000),(500000,1000000),(300000,1000000),(100000,1000000)):
                for task in range(1,6):
                    aa=stage[variant][early];bb=stage[variant][late];ids=sorted(k for k in aa if k in bb and int(aa[k]['task'])==task)
                    if len(ids)!=20:continue
                    assert all(aa[k]['full_scene_sha256']==bb[k]['full_scene_sha256'] for k in ids)
                    transitions.append(dict(kind=kind,variant=variant,task=task,early_step=early,late_step=late,**paired([float(aa[k]['success']) for k in ids],[float(bb[k]['success']) for k in ids],rng)))
            p=ROOT/'results'/f'final_{kind}_{variant}.csv';rr=read(p) if p.exists() else [];final[variant]={r['scene_id']:r for r in rr}
            for task in range(1,6):
                for group in (0,1,-1):
                    tr=[r for r in rr if int(r['task'])==task and (group==-1 or int(r['group'])==group)]
                    if not tr:continue
                    k=sum(float(r['success']) for r in tr);ci=wilson(k,len(tr));groups.append(dict(kind=kind,variant=variant,task=task,group=group,n=len(tr),successes=int(k),success_rate=k/len(tr),ci_low=ci[0],ci_high=ci[1]))
        for task in range(1,6):
            aa,bb=final['A'],final['B'];ids=sorted(k for k in aa if k in bb and int(aa[k]['task'])==task)
            if len(ids)!=100:continue
            assert all(aa[k]['full_scene_sha256']==bb[k]['full_scene_sha256'] for k in ids)
            final_comparisons.append(dict(kind=kind,task=task,**paired([float(aa[k]['success']) for k in ids],[float(bb[k]['success']) for k in ids],rng),
                A_ineffective_steps_mean=float(np.mean([float(aa[k]['ineffective_steps']) for k in ids])),B_ineffective_steps_mean=float(np.mean([float(bb[k]['ineffective_steps']) for k in ids])),
                A_final_distance_mean=float(np.mean([float(aa[k]['final_distance']) for k in ids])),B_final_distance_mean=float(np.mean([float(bb[k]['final_distance']) for k in ids]))))
    save(ROOT/'results/E1_SCORES.csv',summaries);save(ROOT/'results/E1_TRANSITIONS.csv',transitions);save(ROOT/'results/FINAL_SCORES.csv',groups);save(ROOT/'results/FINAL_PAIRED.csv',final_comparisons);write(ROOT/'results/EVALUATION_COVERAGE.json',coverage)
    p=ROOT/'results/E2_BRANCHES.csv';rr=read(p) if p.exists() else [];e2=[]
    for kind in ('navigate','stitch'):
        for success in ('True','False'):
            for condition in ('C0','C1','C2','C3','C4'):
                tr=[r for r in rr if r['kind']==kind and r['source_success']==success and r['condition']==condition]
                if not tr:continue
                e2.append(dict(kind=kind,source_success=success,condition=condition,n=len(tr),local_reached=sum(r['local_reached']=='True' for r in tr),far_success=sum(r['original_far_success']=='True' for r in tr),
                    far_progress_mean=float(np.mean([float(r['far_progress']) for r in tr])),displacement_mean=float(np.mean([float(r['net_displacement']) for r in tr])),ineffective_steps_mean=float(np.mean([float(r['ineffective_steps']) for r in tr]))))
    save(ROOT/'results/E2_SUMMARY.csv',e2)
    evidence=[]
    support=read(ROOT/'results/E3_SUPPORT.csv') if (ROOT/'results/E3_SUPPORT.csv').exists() else []
    gates=json.loads((ROOT/'results/E2_GATES.json').read_text()) if (ROOT/'results/E2_GATES.json').exists() else []
    verified={v:read(ROOT/'results'/f'E3_{v}_VERIFIED_TARGETS.csv') if (ROOT/'results'/f'E3_{v}_VERIFIED_TARGETS.csv').exists() else [] for v in ('A','B')}
    joint=read(ROOT/'results/E3_JOINT_SAMPLING_SUPPORT.csv') if (ROOT/'results/E3_JOINT_SAMPLING_SUPPORT.csv').exists() else []
    for x in support:
        sid=x['snapshot_id'];g=next((g for g in gates if g['snapshot_id']==sid),{});branches={r['condition']:r for r in rr if r['snapshot_id']==sid}
        item=dict(snapshot_id=sid,kind=x['kind'],task=x['task'],source_success=x['source_success'],E2_gate=g.get('status','NOT_EXECUTED'),gate_reason=g.get('reason'),
            qpos_qvel_supported_neighbors=int(x['nearby_qpos_qvel_supported_count']),other_trajectory_neighbors=int(x['nearby_other_trajectory_supported_count']),
            A_goal_region_probability=float(x['future_goal_region_probability_A']),B_goal_region_probability=float(x['goal_region_probability_B']),
            C0_far_progress=float(branches['C0']['far_progress']) if branches else None,
            C1_near_success=branches.get('C1',{}).get('local_reached'),C2_near_success=branches.get('C2',{}).get('local_reached'),
            C3_control_success=branches.get('C3',{}).get('local_reached'),C4_far_progress=float(branches['C4']['far_progress']) if branches else None,
            causal_scope='50-step local branch only; complete far task success and exact data state stitching not implied')
        for variant in ('A','B'):
            v=next((r for r in verified[variant] if r['snapshot_id']==sid and r['target']=='near' and r['checkpoint_step']=='1000000'),None)
            item.update({f'{variant}_1M_verified_near_log_rho':float(v['log_rho']) if v else None,
                f'{variant}_1M_verified_near_rho':float(v['rho']) if v else None,
                f'{variant}_1M_far_actor_norm':float(math.hypot(float(v['actor_0']),float(v['actor_1']))) if v else None,
                f'{variant}_1M_near_teacher_norm':float(math.hypot(float(v['teacher_0']),float(v['teacher_1']))) if v else None,
                f'{variant}_1M_heads_prefer_different_targets':v['head1_preferred']!=v['head2_preferred'] if v else None,
                f'{variant}_1M_actor_teacher_mse':float(v['actor_teacher_mse']) if v else None})
        j=next((r for r in joint if r['snapshot_id']==sid and r['group']=='E2_verified_near'),None)
        item['A_joint_far_region_verified_waypoint_probability']=float(j['joint_far_region_waypoint_probability_A']) if j else None
        item['B_joint_far_region_verified_waypoint_probability']=float(j['joint_far_region_waypoint_probability_B']) if j else None
        evidence.append(item)
    save(ROOT/'results/EVIDENCE_MATRIX.csv',evidence)
    write(ROOT/'results/STATISTICAL_METHOD.json',dict(bootstrap_samples=10000,seed=203001,unit='whole paired episode',score_interval='Wilson 95%',difference_interval='paired percentile bootstrap 95%',
        limitations='single training seed; two scene groups; saturated all-success/all-failure paired bootstrap may be degenerate; Wilson score intervals retained; no cross-dataset causal comparison'))
    print('ANALYSIS_WRITTEN',len(summaries),len(transitions),len(final_comparisons),flush=True)
if __name__=='__main__':main()
