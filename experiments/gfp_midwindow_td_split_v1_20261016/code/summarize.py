import json, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
OLD=Path('/home/xushijie3/work/gfp_midwindow_td_freeze_v1_20261016')
NEW_ARMS=('L1a','L1b')
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def classify(points,e):
    streak=longest=0;first=None
    for p in points:
        hit=p['historical_gap_increase']>=15 and p['historical_gradient_ratio']>=2
        assert hit==p['joint_anomaly']
        streak=streak+1 if hit else 0;longest=max(longest,streak)
        if streak==2 and first is None:first=p['updates']-5000
    numerical=longest>=2
    collapse=e['success_percent']<=5 and e['start_stay_percent']>=90
    strong=e['success_percent']>=25 and e['start_stay_percent']<=30 and not numerical
    return dict(joint_anomaly=numerical,first_anomaly_start=first,longest_consecutive_points=longest,
      behavior_collapse=collapse,strong_benefit=strong,mixed_numerical_behavior=numerical!=collapse)
def main():
    all_arms={};episodes={};control=external=physics=0;hashes={};checks={}
    assets=read(ROOT/'receipts/assets.json')
    assert all(sha(Path(p))==v for p,v in assets['historical_sha256'].items())
    assert sha(Path(read(ROOT/'protocol.json')['source']))==read(ROOT/'protocol.json')['source_sha256']
    for arm in ('L0','L1','L1a','L1b'):
        root=ROOT if arm in NEW_ARMS else OLD;run=root/f'runs/s3_{arm}'
        assert read(run/'exit.json')['exit_code']==0
        points=read(run/'probes.json');assert [p['updates'] for p in points]==list(range(0,50001,5000))
        metadata=[read(run/f'checkpoint_{it}.json') for it in range(0,50001,5000)]
        assert all(m['full_restore']=='PASS' and m['step']==350000+m['updates'] for m in metadata)
        ev={};cost={}
        for it in (25000,50000):
            d=run/(f'eval_{it}_repair1' if arm in NEW_ARMS and it==25000 else f'eval_{it}');e=read(d/'results.json')
            assert read(d/'exit.json')['exit_code']==0 and len(e['episodes'])==100 and e['optimizer_updates']==0
            assert e['checkpoint_sha256']==read(run/f'checkpoint_{it}.json')['sha256']
            hashes[f'{arm}_{it}']={x['episode']:x['initial_obs_sha256'] for x in e['episodes']}
            episodes[f'{arm}_{it}']={x['episode']:x for x in e['episodes']}
            ev[str(it)]={k:v for k,v in e.items() if k!='episodes'}
            if arm in NEW_ARMS:
                control+=e['control_calls'];external+=e['external_steps'];physics+=e['physics_substeps']
                cost[str(it)]=read(d/'cost.json');assert cost[str(it)]['workers_stopped']
        c=classify(points,ev['50000'])
        all_arms[arm]=dict(historical=arm not in NEW_ARMS,probes=points,evaluation=ev,classification=c,
          training=read(run/'complete.json'),evaluation_cost=cost,checkpoint_index=metadata)
        if arm=='L1a':assert len({m['reference_sha256'] for m in metadata})==1
        if arm=='L1b':
            assert len({m['frozen_vabc_sha256'] for m in metadata})==1
            assert metadata[0]['reference_sha256']!=metadata[-1]['reference_sha256']
        if arm in NEW_ARMS:
            assert read(run/'admission.json')['passed'] and read(run/'zero_update_admission.json')['passed']
            assert all(m['agent_rng']==n['agent_rng'] for m,n in zip(metadata,all_arms['L0']['checkpoint_index']))
    assert sum(all_arms[a]['training']['updates'] for a in NEW_ARMS)==100000
    assert control+800<=833332 and external<=800000 and physics+4000<=4166660
    hashes_match=all(h==hashes['L0_25000'] for h in hashes.values())
    a,b=(all_arms[x]['classification'] for x in NEW_ARMS)
    if a['strong_benefit'] and b['joint_anomaly'] and b['behavior_collapse']:decision='SUPPORT_ACTOR_CONNECTION'
    elif b['strong_benefit'] and a['joint_anomaly'] and a['behavior_collapse']:decision='SUPPORT_VABC_CONNECTION'
    elif a['strong_benefit'] and b['strong_benefit']:decision='EITHER_FREEZE_SUFFICIENT_IN_WINDOW'
    elif all(x['joint_anomaly'] and x['behavior_collapse'] for x in [a,b]):decision='SINGLE_FREEZE_INSUFFICIENT'
    else:decision='MIXED_OR_INTERMEDIATE'
    paired={}
    if hashes_match:
        for it in (25000,50000):
            for left,right in [('L1a','L1b'),('L1a','L1'),('L1b','L0')]:
                l,r=episodes[f'{left}_{it}'],episodes[f'{right}_{it}']
                paired[f'{left}_vs_{right}_{it}']=dict(both_success=sum(l[k]['success']>0 and r[k]['success']>0 for k in l),
                  only_left=sum(l[k]['success']>0 and r[k]['success']==0 for k in l),
                  only_right=sum(l[k]['success']==0 and r[k]['success']>0 for k in l),
                  neither_success=sum(l[k]['success']==0 and r[k]['success']==0 for k in l))
    samples=[json.loads(x) for x in (ROOT/'logs/own_gpu_memory.jsonl').read_text().splitlines()]
    gpu={}
    for row in samples:
        for p in row['processes']:
            pid=str(p['pid']);s=gpu.setdefault(pid,dict(first_observed=row['time'],last_observed=row['time'],sampled_peak_mib=0))
            s['last_observed']=row['time'];s['sampled_peak_mib']=max(s['sampled_peak_mib'],p['used_mib'])
    for s in gpu.values():s['observed_residence_span_seconds']=s['last_observed']-s['first_observed']
    complete=read(ROOT/'receipts/manager_complete.json');complete['time']=read(ROOT/'receipts/repair_manager_complete.json')['time'];launch=read(ROOT/'receipts/launch.json')
    process_bounds={}
    for arm in NEW_ARMS:
        run=ROOT/f'runs/s3_{arm}';t=all_arms[arm]['training'];q=complete['jobs'][arm]
        process_bounds[arm+'_training']=dict(pid=q['pid'],internal_measured_seconds=t['seconds'],dispatch_to_poll_seconds=q['ended']-q['started'],cpu_seconds=t['cpu_seconds'],max_rss_kib=t['max_rss_kib'])
        for it in (25000,50000):
            c=all_arms[arm]['evaluation_cost'][str(it)];x=read(run/(f'eval_{it}_repair1' if it==25000 else f'eval_{it}')/'exit.json')
            process_bounds[f'{arm}_eval_{it}']=dict(pid=x['pid'],internal_measured_seconds=c['process_wall_seconds'],dispatch_to_poll_seconds=x['time']-x['dispatch_time'],cpu_seconds=c['cpu_seconds_self']+c['cpu_seconds_workers'],max_rss_kib=c['max_rss_self_kib'])
    cost=dict(new_updates=100000,new_evaluation_episodes=400,external_steps=external,control_calls=control,physics_substeps=physics,
      failed_evaluation_control_calls_upper_bound=800,failed_evaluation_physics_substeps_upper_bound=4000,
      total_control_calls_upper_bound=control+800,total_physics_substeps_upper_bound=physics+4000,
      failed_evaluation_cpu_seconds=None,failed_evaluation_note='Startup counters lost; retain code-path conservative upper bound. Failed parent/worker wall and CPU cost are not included in measured successful-evaluation totals.',
      wall_seconds=complete['time']-launch['time'],processes=process_bounds,gpu_process_samples=gpu,
      allocated_process_hours_measured=sum(x['internal_measured_seconds'] for x in process_bounds.values())/3600,
      allocated_process_hours_dispatch_to_poll=sum(x['dispatch_to_poll_seconds'] for x in process_bounds.values())/3600,
      total_cpu_seconds=sum(x['cpu_seconds'] for x in process_bounds.values()),
      gpu_note='Process-duration accounting on allocated GPUs, not GPU busy-time integration. GPU residence is sampled; first-to-last spans and sampled peaks are not exact totals/maxima.')
    failed=[]
    for arm in NEW_ARMS:
        x=read(ROOT/f'runs/s3_{arm}/eval_25000/exit.json')
        failed.append(dict(arm=arm,**x,dispatch_to_exit_receipt_seconds=x['time']-x['dispatch_time']))
    cost['failed_evaluation_processes']=failed
    cost['failed_parent_process_hours_dispatch_to_poll']=sum(x['dispatch_to_exit_receipt_seconds'] for x in failed)/3600
    result=dict(decision=decision,arms=all_arms,initial_observation_hashes_match_all_eight_conditions=hashes_match,
      strict_paired_evaluation_allowed=hashes_match,paired=paired,cost=cost,historical_files_unchanged=True,
      RNG_matches_historical_control_all_saved_points=True,source_checkpoint_unchanged=True,
      no_claim_of_bitwise_replay=True,scope='One training state; two new arms; historical controls reused without rerunning')
    (ROOT/'reports/results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    (ROOT/'reports/cost.json').write_text(json.dumps(cost,indent=2))
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    labels={'L0':'L0: EMA (historical)','L1':'L1: both frozen (historical)','L1a':'L1a: reference actor frozen','L1b':'L1b: TD VaBC frozen'}
    for arm in all_arms:
        p=all_arms[arm]['probes'];x=[v['updates']/1000 for v in p]
        for ax,key,title in zip(axes.flat,['historical_gap_increase','historical_gradient_ratio','dqda_cos_start','V_target_start'],['Start-to-global Q gap increase','Action-gradient norm / historical norm','Gradient cosine vs. 300k','Target Q at start']):
            ax.plot(x,[v[key] for v in p],ls='--' if arm in ('L0','L1') else '-',marker='.',label=labels[arm]);ax.set_title(title);ax.set_xlabel('New updates (k)');ax.grid(alpha=.2)
    axes[0,0].axhline(15,ls=':',color='gray');axes[0,1].axhline(2,ls=':',color='gray')
    for ax in axes.flat:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'reports/split_curves.png',dpi=160);plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for arm in all_arms:
        p=all_arms[arm]['probes'];style='--' if arm in ('L0','L1') else '-'
        axes[0].plot([v['updates']/1000 for v in p],[v['action_shift_start'] for v in p],ls=style,marker='.',label=labels[arm])
        axes[1].plot([25,50],[all_arms[arm]['evaluation'][str(it)]['success_percent'] for it in (25000,50000)],ls=style,marker='o',label=labels[arm])
    axes[0].set_title('Actor action shift at start vs. 300k');axes[1].set_title('Online Actor success (100 episodes / point)')
    axes[1].axhline(25,color='gray',ls=':')
    for ax in axes:ax.set_xlabel('New updates (k)');ax.grid(alpha=.2);ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(ROOT/'reports/action_and_success.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(decision=decision,hashes_match=hashes_match,new={a:dict(evaluation=all_arms[a]['evaluation'],classification=all_arms[a]['classification']) for a in NEW_ARMS},cost=cost),indent=2))
if __name__=='__main__':main()
