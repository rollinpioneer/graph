import json, math, hashlib
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1]
SOURCE=Path('/home/xushijie3/work/gfp_bfc_dev_screen_v1_20261016')
def read(p):return json.loads(p.read_text())
def main():
    report={}; initial_hashes=None;external=control=physics=0
    hist=read(SOURCE/'runs/s3_R3/probes.json')
    for arm in ('L0','L1'):
        run=ROOT/f'runs/s3_{arm}'
        assert read(run/'exit.json')['exit_code']==0
        complete=read(run/'complete.json');assert complete['updates']==50000
        points=read(run/'probes.json');assert [p['updates'] for p in points]==list(range(0,50001,5000))
        checkpoints=[read(run/f'checkpoint_{it}.json') for it in range(0,50001,5000)]
        assert all(p['full_restore']=='PASS' and p['step']==350000+p['updates'] for p in checkpoints)
        if arm=='L1':
            assert len({p['reference_sha256'] for p in checkpoints})==1
            assert len({p['frozen_vabc_sha256'] for p in checkpoints})==1
        streak=longest=0;first=None
        for p in points:
            streak=streak+1 if p['joint_anomaly'] else 0;longest=max(streak,longest)
            if streak==2 and first is None:first=p['updates']-5000
        ev={}
        for it in (25000,50000):
            d=run/f'eval_{it}';assert read(d/'exit.json')['exit_code']==0
            e=read(d/'results.json');assert len(e['episodes'])==100 and e['optimizer_updates']==0
            assert e['checkpoint_sha256']==read(run/f'checkpoint_{it}.json')['sha256']
            hashes={str(x['episode']):x['initial_obs_sha256'] for x in e['episodes']}
            if initial_hashes is None:initial_hashes=hashes
            assert hashes==initial_hashes
            external+=e['external_steps'];control+=e['control_calls'];physics+=e['physics_substeps']
            ev[str(it)]={k:v for k,v in e.items() if k!='episodes'}
        e=ev['50000'];behavior=e['success_percent']<=5 and e['start_stay_percent']>=90
        report[arm]=dict(probes=points,evaluation=ev,training=complete,joint_anomaly_confirmed=longest>=2,
          first_confirmed_anomaly_start=first,longest_consecutive_points=longest,behavior_collapse=behavior,
          combined_collapse=longest>=2 and behavior,checkpoint_index=checkpoints)
    assert external<=800000 and control<=833332
    r0,r1=report['L0'],report['L1']
    if not r0['combined_collapse']:decision='C: control did not reproduce both preregistered endpoints; intervention efficacy not established'
    elif r1['combined_collapse']:decision='B: freezing TD action sources was insufficient in this window'
    elif not r1['joint_anomaly_confirmed'] and not r1['behavior_collapse']:decision='A: continuing TD source changes mattered in this window'
    else:decision='MIXED: numerical and behavioral endpoints disagree'
    # Compare new control with the original continuation; do not assume exact replay.
    fields=['V_online_start','V_online_all','V_target_start','dqda_norm_start','dqda_cos_start']
    differences=[]
    for p in r0['probes']:
        h=next(x for x in hist if x['updates']==50000+p['updates'])
        differences.append(dict(updates=p['updates'],**{k:p[k]-h[k] for k in fields}))
    rng_equal=all(a['agent_rng']==b['agent_rng'] for a,b in zip(r0['checkpoint_index'],r1['checkpoint_index']))
    assert rng_equal
    paired={}
    for it in (25000,50000):
        a={e['episode']:e for e in read(ROOT/f'runs/s3_L0/eval_{it}/results.json')['episodes']}
        b={e['episode']:e for e in read(ROOT/f'runs/s3_L1/eval_{it}/results.json')['episodes']}
        paired[str(it)]=dict(both_success=sum(a[k]['success']>0 and b[k]['success']>0 for k in a),
          only_L0_success=sum(a[k]['success']>0 and b[k]['success']==0 for k in a),
          only_L1_success=sum(b[k]['success']>0 and a[k]['success']==0 for k in a),
          neither_success=sum(b[k]['success']==0 and a[k]['success']==0 for k in a))
    result=dict(arms=report,decision=decision,paired_evaluation=paired,all_initial_observation_hashes_match=True,
      paired_agent_rng_all_saved_points_match=rng_equal,control_minus_historical=differences,
      total_new_optimizer_updates=100000,external_steps=external,control_calls=control,physics_substeps=physics,
      limitations=['One selected training state; no cross-seed claim','Actor and VaBC TD sources frozen jointly',
        'Both action generators still receive fresh random inputs','Current-Actor Q values and fixed-action gradients are distinct measurements',
        'Evaluation episode pairing does not create independent training replicates'])
    (ROOT/'reports/results.json').write_text(json.dumps(result,indent=2,allow_nan=False))
    fig,axes=plt.subplots(2,2,figsize=(11,8))
    for arm in ('L0','L1'):
        p=report[arm]['probes'];x=[v['updates']/1000 for v in p]
        for ax,key in zip(axes.flat,['historical_gap_increase','historical_gradient_ratio','dqda_cos_start','V_target_start']):
            ax.plot(x,[v[key] for v in p],marker='.',label=arm);ax.set_title(key);ax.set_xlabel('New updates (k)');ax.grid(alpha=.2)
    axes[0,0].axhline(15,ls='--',color='gray');axes[0,1].axhline(2,ls='--',color='gray')
    for ax in axes.flat:ax.legend()
    fig.tight_layout();fig.savefig(ROOT/'reports/paired_curves.png',dpi=160);plt.close(fig)
    print(json.dumps(dict(decision=decision,eval={a:report[a]['evaluation'] for a in report},control_calls=control,paired=paired),indent=2))
if __name__=='__main__':main()
